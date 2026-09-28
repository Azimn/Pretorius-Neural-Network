from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.battery import load_items, score_results
from persona_net.encoding import ACTIONS, SCALAR_KEYS

ROOT = Path(__file__).resolve().parents[1]


def normalize_target(raw):
    target = {a: max(float(raw.get(a, 0.0)), 0.0) for a in ACTIONS}
    total = sum(target.values())
    if total <= 0:
        raise ValueError("Target distribution has no positive mass")
    return {a: v / total for a, v in target.items()}


def train_recurrent_only(net, encoder, items, total_steps):
    done = 0
    cursor = 0
    zero_action = slice(encoder.action_offset, encoder.input_dim)
    while done < total_steps:
        item = items[cursor % len(items)]
        cursor += 1
        scalars = {k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
        target = normalize_target(item["target_actions"])

        context = encoder.encode(
            item["scenario"], scalars=scalars, action=None, outcome=0.0
        ).vector
        x_context = context.copy()
        x_context[zero_action] = 0.0
        net.step(x_context, reward=0.0, learn=False)
        done += 1
        if done >= total_steps:
            break

        teaching = context.copy()
        for ai, action in enumerate(ACTIONS):
            teaching[encoder.action_offset + ai] = target[action]
        net.step(teaching, reward=1.0, learn=True)
        done += 1
    return done


def population_probs(net):
    raw = {
        action: max(float(net.rate[idx].mean()), 1e-12)
        for action, idx in net.action_populations.items()
    }
    total = sum(raw.values())
    return {a: v / total for a, v in raw.items()}


def evaluate(net, encoder, items, settle=40, probe=60):
    zero = np.zeros(encoder.input_dim, dtype=np.float32)
    rows = []
    for item in items:
        net.reset_fast_state(noise=0.01)
        for _ in range(settle):
            net.step(zero, reward=0.0, learn=False)

        scalars = {k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
        x = encoder.encode(
            item["scenario"], scalars=scalars, action=None, outcome=0.0
        ).vector
        accum = {a: 0.0 for a in ACTIONS}
        for _ in range(probe):
            net.step(x, reward=0.0, learn=False)
            for action, value in population_probs(net).items():
                accum[action] += value
        probs = {a: value / probe for a, value in accum.items()}
        row = {
            k: item.get(k)
            for k in ("id", "domain", "split", "variant", "scenario", "contrast_group")
            if k in item
        }
        row["action_probabilities"] = probs
        row["top_actions"] = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        row["target_actions"] = item["target_actions"]
        row["anti_actions"] = item.get("anti_actions", [])
        rows.append(row)
    return score_results(rows)


def compact(score):
    return {
        k: score[k]
        for k in (
            "js_similarity",
            "domain_macro_js_similarity",
            "target_mass",
            "top1_agreement",
            "top3_coverage",
            "anti_mass",
        )
    }


def same_topology(a, b):
    return np.array_equal(a.W.indices, b.W.indices) and np.array_equal(a.W.indptr, b.W.indptr)


def copy_founder_state(founder, template):
    out = copy.deepcopy(template)
    out.W.data[:] = founder.W.data
    out.bias[:] = founder.bias
    out.eligibility[:] = founder.eligibility
    out.reset_fast_state()
    return out


def graft(founder, trained, fraction):
    out = copy_founder_state(founder, trained)
    out.W.data[:] = founder.W.data + fraction * (trained.W.data - founder.W.data)
    out.bias[:] = founder.bias + fraction * (trained.bias - founder.bias)
    return out


def weight_only(founder, trained):
    out = copy_founder_state(founder, trained)
    out.W.data[:] = trained.W.data
    return out


def bias_only(founder, trained):
    out = copy_founder_state(founder, trained)
    out.bias[:] = trained.bias
    return out


def scrambled_delta(founder, trained, rng):
    out = copy_founder_state(founder, trained)
    delta_w = trained.W.data - founder.W.data
    delta_b = trained.bias - founder.bias

    exc_edges = founder.excitatory[founder.pre_idx]
    shuffled = delta_w.copy()
    for mask in (exc_edges, ~exc_edges):
        values = shuffled[mask].copy()
        rng.shuffle(values)
        shuffled[mask] = values

    shuffled_b = delta_b.copy()
    rng.shuffle(shuffled_b)
    out.W.data[:] = founder.W.data + shuffled
    out.bias[:] = founder.bias + shuffled_b

    out.W.data[exc_edges] = np.clip(out.W.data[exc_edges], 0.0, out.max_abs_weight)
    out.W.data[~exc_edges] = np.clip(out.W.data[~exc_edges], -out.max_abs_weight, 0.0)
    return out


def inverted_delta(founder, trained):
    out = copy_founder_state(founder, trained)
    delta_w = trained.W.data - founder.W.data
    delta_b = trained.bias - founder.bias
    exc_edges = founder.excitatory[founder.pre_idx]
    out.W.data[:] = founder.W.data - delta_w
    out.bias[:] = founder.bias - delta_b
    out.W.data[exc_edges] = np.clip(out.W.data[exc_edges], 0.0, out.max_abs_weight)
    out.W.data[~exc_edges] = np.clip(out.W.data[~exc_edges], -out.max_abs_weight, 0.0)
    return out


def lesion_synapses(trained, fraction, rng):
    out = copy.deepcopy(trained)
    count = max(1, int(round(fraction * out.W.data.size)))
    idx = rng.choice(out.W.data.size, size=count, replace=False)
    out.W.data[idx] = 0.0
    return out, count


def lesion_by_delta(founder, trained, fraction):
    out = copy.deepcopy(trained)
    count = max(1, int(round(fraction * out.W.data.size)))
    delta = np.abs(trained.W.data - founder.W.data)
    idx = np.argpartition(delta, -count)[-count:]
    out.W.data[idx] = 0.0
    return out, count


def score_condition(net, encoder, validation, adversarial):
    return {
        "validation": compact(evaluate(net, encoder, validation)),
        "adversarial": compact(evaluate(net, encoder, adversarial)),
    }


def main():
    p = argparse.ArgumentParser(description="v0.4 recurrent persona engram stress test")
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--steps", type=int, default=24000)
    p.add_argument("--seed", type=int, default=1842)
    p.add_argument("--plasticity", type=float, default=16.0)
    args = p.parse_args()

    cfg = json.loads((ROOT / "config" / "default.json").read_text())
    cfg["network"]["neurons"] = args.neurons
    cfg["network"]["seed"] = args.seed
    cfg["network"]["avg_recurrent_degree"] = min(24, max(12, args.neurons // 64))
    cfg["network"]["action_population_size"] = min(48, max(16, args.neurons // 24))
    cfg["network"]["hebb_lr"] *= args.plasticity
    cfg["network"]["reward_lr"] *= args.plasticity

    encoder = ExperienceEncoder(cfg["network"]["sensory_dim"])
    founder = PlasticRecurrentPersonaNet(cfg["network"], encoder)
    trained = PlasticRecurrentPersonaNet(cfg["network"], encoder)
    if not same_topology(founder, trained):
        raise RuntimeError("Matched founder and trained networks do not share topology")

    payload = json.loads(
        (ROOT / "data" / "phenotype_battery" / "pretorius_phenotype_train_battery_v1.json").read_text()
    )
    training = payload["training_items"]
    validation = load_items(ROOT / "data" / "phenotype_battery" / "pretorius_validation_v1.json")
    adversarial = load_items(ROOT / "data" / "phenotype_battery" / "pretorius_adversarial_v1.json")

    founder_w = founder.W.data.copy()
    founder_b = founder.bias.copy()
    train_recurrent_only(trained, encoder, training, args.steps)

    rng = np.random.default_rng(args.seed + 9001)
    conditions = {}
    conditions["founder"] = score_condition(copy.deepcopy(founder), encoder, validation, adversarial)
    conditions["trained_intact"] = score_condition(copy.deepcopy(trained), encoder, validation, adversarial)
    conditions["trained_weights_only"] = score_condition(weight_only(founder, trained), encoder, validation, adversarial)
    conditions["trained_bias_only"] = score_condition(bias_only(founder, trained), encoder, validation, adversarial)

    for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
        conditions[f"graft_{fraction:.2f}"] = score_condition(
            graft(founder, trained, fraction), encoder, validation, adversarial
        )

    conditions["scrambled_delta"] = score_condition(
        scrambled_delta(founder, trained, rng), encoder, validation, adversarial
    )
    conditions["inverted_delta"] = score_condition(
        inverted_delta(founder, trained), encoder, validation, adversarial
    )

    lesion_meta = {}
    for fraction in (0.10, 0.25, 0.50):
        random_lesioned, count = lesion_synapses(trained, fraction, rng)
        targeted_lesioned, target_count = lesion_by_delta(founder, trained, fraction)
        conditions[f"random_lesion_{fraction:.2f}"] = score_condition(
            random_lesioned, encoder, validation, adversarial
        )
        conditions[f"delta_targeted_lesion_{fraction:.2f}"] = score_condition(
            targeted_lesioned, encoder, validation, adversarial
        )
        lesion_meta[f"{fraction:.2f}"] = {
            "random_zeroed_synapses": count,
            "targeted_zeroed_synapses": target_count,
        }

    out = {
        "experiment": "005_engram_stress",
        "version": "v0.4-dev",
        "question": "Is the recurrent Pretorius signal carried by topologically specific learned changes, and is it dose-dependent and lesion-robust?",
        "terminal_battery_touched": False,
        "neurons": args.neurons,
        "neural_training_steps": args.steps,
        "seed": args.seed,
        "plasticity_multiplier": args.plasticity,
        "mean_abs_weight_delta": float(np.mean(np.abs(trained.W.data - founder_w))),
        "mean_abs_bias_delta": float(np.mean(np.abs(trained.bias - founder_b))),
        "lesion_meta": lesion_meta,
        "conditions": conditions,
    }

    outdir = ROOT / "results" / "experiment_005_engram_stress"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"report_n{args.neurons}_steps{args.steps}_s{args.seed}.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"seed={args.seed} n={args.neurons} steps={args.steps} plasticity={args.plasticity:g}x")
    for name, result in conditions.items():
        v = result["validation"]
        a = result["adversarial"]
        print(
            f"{name:30s} valJS={v['js_similarity']:.4f} valTop1={v['top1_agreement']:.3f} "
            f"advJS={a['js_similarity']:.4f} advTop1={a['top1_agreement']:.3f}"
        )
    print("saved", path)


if __name__ == "__main__":
    main()
