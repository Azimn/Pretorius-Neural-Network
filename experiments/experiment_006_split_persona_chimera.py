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


def axis_of(item_id: str) -> str | None:
    if "_train_" not in item_id:
        return None
    return item_id.rsplit("_train_", 1)[0]


def normalize_target(raw):
    d = {a: max(float(raw.get(a, 0.0)), 0.0) for a in ACTIONS}
    s = sum(d.values())
    if s <= 0:
        raise ValueError("empty target")
    return {a: v / s for a, v in d.items()}


def train_recurrent_only(net, encoder, items, total_steps):
    done = 0
    cursor = 0
    zero_action = slice(encoder.action_offset, encoder.input_dim)
    while done < total_steps:
        item = items[cursor % len(items)]
        cursor += 1
        scalars = {k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
        target = normalize_target(item["target_actions"])
        context = encoder.encode(item["scenario"], scalars=scalars, action=None, outcome=0.0).vector
        context[zero_action] = 0.0
        net.step(context, reward=0.0, learn=False)
        done += 1
        if done >= total_steps:
            break
        teaching = encoder.encode(item["scenario"], scalars=scalars, action=None, outcome=0.0).vector
        for ai, action in enumerate(ACTIONS):
            teaching[encoder.action_offset + ai] = target[action]
        net.step(teaching, reward=1.0, learn=True)
        done += 1
    return done


def population_probs(net):
    raw = {a: max(float(net.rate[idx].mean()), 1e-12) for a, idx in net.action_populations.items()}
    s = sum(raw.values())
    return {a: v / s for a, v in raw.items()}


def evaluate(net, encoder, items, settle=40, probe=60):
    zero = np.zeros(encoder.input_dim, dtype=np.float32)
    rows = []
    for item in items:
        net.reset_fast_state(noise=0.01)
        for _ in range(settle):
            net.step(zero, reward=0.0, learn=False)
        scalars = {k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
        x = encoder.encode(item["scenario"], scalars=scalars, action=None, outcome=0.0).vector
        acc = {a: 0.0 for a in ACTIONS}
        for _ in range(probe):
            net.step(x, reward=0.0, learn=False)
            for action, value in population_probs(net).items():
                acc[action] += value
        probs = {a: value / probe for a, value in acc.items()}
        rows.append({
            "id": item["id"],
            "domain": item.get("domain", "unknown"),
            "action_probabilities": probs,
            "target_actions": item["target_actions"],
            "anti_actions": item.get("anti_actions", []),
        })
    return score_results(rows)


def compact(score):
    return {
        "js_similarity": score["js_similarity"],
        "domain_macro_js_similarity": score["domain_macro_js_similarity"],
        "target_mass": score["target_mass"],
        "top1_agreement": score["top1_agreement"],
        "top3_coverage": score["top3_coverage"],
        "anti_mass": score["anti_mass"],
        "domains": score["domains"],
    }


def merge(founder, a, b, scale_a=1.0, scale_b=1.0):
    out = copy.deepcopy(founder)
    da = a.W.data - founder.W.data
    db = b.W.data - founder.W.data
    dba = a.bias - founder.bias
    dbb = b.bias - founder.bias
    out.W.data[:] = founder.W.data + scale_a * da + scale_b * db
    out.bias[:] = founder.bias + scale_a * dba + scale_b * dbb
    exc = founder.excitatory[founder.pre_idx]
    out.W.data[exc] = np.clip(out.W.data[exc], 0.0, out.max_abs_weight)
    out.W.data[~exc] = np.clip(out.W.data[~exc], -out.max_abs_weight, 0.0)
    return out


def subset_mean(score, domains):
    vals = [score["domains"][d]["js_similarity"] for d in domains]
    tops = [score["domains"][d]["top1_agreement"] for d in domains]
    return {"js_similarity": float(np.mean(vals)), "top1_agreement": float(np.mean(tops))}


def main():
    p = argparse.ArgumentParser(description="Split-persona recurrent chimera experiment")
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--half-steps", type=int, default=24000)
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
    half_a = PlasticRecurrentPersonaNet(cfg["network"], encoder)
    half_b = PlasticRecurrentPersonaNet(cfg["network"], encoder)
    full = PlasticRecurrentPersonaNet(cfg["network"], encoder)

    payload = json.loads((ROOT / "data" / "phenotype_battery" / "pretorius_phenotype_train_battery_v1.json").read_text())
    grounded = [item for item in payload["training_items"] if axis_of(item["id"]) is not None]
    axes = sorted({axis_of(item["id"]) for item in grounded})
    if len(axes) != 20:
        raise RuntimeError(f"Expected 20 grounded phenotype axes, found {len(axes)}")
    axes_a = axes[::2]
    axes_b = axes[1::2]
    items_a = [item for item in grounded if axis_of(item["id"]) in axes_a]
    items_b = [item for item in grounded if axis_of(item["id"]) in axes_b]

    train_recurrent_only(half_a, encoder, items_a, args.half_steps)
    train_recurrent_only(half_b, encoder, items_b, args.half_steps)
    train_recurrent_only(full, encoder, grounded, args.half_steps * 2)

    validation = load_items(ROOT / "data" / "phenotype_battery" / "pretorius_validation_v1.json")
    adversarial = load_items(ROOT / "data" / "phenotype_battery" / "pretorius_adversarial_v1.json")

    networks = {
        "founder": copy.deepcopy(founder),
        "half_a": half_a,
        "half_b": half_b,
        "merged_additive": merge(founder, half_a, half_b, 1.0, 1.0),
        "merged_average": merge(founder, half_a, half_b, 0.5, 0.5),
        "full_matched_exposure": full,
    }

    conditions = {}
    for name, net in networks.items():
        val = compact(evaluate(net, encoder, validation))
        adv = compact(evaluate(net, encoder, adversarial))
        conditions[name] = {
            "validation": val,
            "adversarial": adv,
            "validation_axes_a": subset_mean(val, axes_a),
            "validation_axes_b": subset_mean(val, axes_b),
            "adversarial_axes_a": subset_mean(adv, axes_a),
            "adversarial_axes_b": subset_mean(adv, axes_b),
        }

    out = {
        "experiment": "006_split_persona_chimera",
        "version": "v0.4-dev",
        "question": "Can separately learned halves of a recurrent persona be recombined by adding their synaptic deltas, and do half-trained brains preferentially improve their trained phenotype domains?",
        "terminal_battery_touched": False,
        "neurons": args.neurons,
        "half_training_steps": args.half_steps,
        "full_training_steps": args.half_steps * 2,
        "seed": args.seed,
        "plasticity_multiplier": args.plasticity,
        "axes_a": axes_a,
        "axes_b": axes_b,
        "items_a": len(items_a),
        "items_b": len(items_b),
        "conditions": conditions,
    }

    outdir = ROOT / "results" / "experiment_006_split_persona_chimera"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"report_n{args.neurons}_halfsteps{args.half_steps}_s{args.seed}.json"
    path.write_text(json.dumps(out, indent=2))

    print("axes A:", ", ".join(axes_a))
    print("axes B:", ", ".join(axes_b))
    for name, r in conditions.items():
        print(
            f"{name:22s} valJS={r['validation']['js_similarity']:.4f} "
            f"A={r['validation_axes_a']['js_similarity']:.4f} B={r['validation_axes_b']['js_similarity']:.4f} "
            f"valTop1={r['validation']['top1_agreement']:.3f} advJS={r['adversarial']['js_similarity']:.4f}"
        )
    print("saved", path)


if __name__ == "__main__":
    main()
