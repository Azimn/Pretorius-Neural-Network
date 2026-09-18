from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.battery import load_items, score_results
from persona_net.encoding import ACTIONS, SCALAR_KEYS

ROOT = Path(__file__).resolve().parents[1]


def axis_of(item_id):
    return item_id.rsplit("_train_", 1)[0] if "_train_" in item_id else None


def normalize(raw):
    d = {a: max(float(raw.get(a, 0)), 0.0) for a in ACTIONS}
    s = sum(d.values())
    return {a: v / s for a, v in d.items()}


def compile_items(enc, items):
    z = slice(enc.action_offset, enc.input_dim)
    out = []
    for item in items:
        scalars = {k: float(item.get("scalars", {}).get(k, 0)) for k in SCALAR_KEYS}
        target = normalize(item["target_actions"])
        base = enc.encode(item["scenario"], scalars=scalars).vector
        context = base.copy()
        context[z] = 0
        teaching = base.copy()
        for i, action in enumerate(ACTIONS):
            teaching[enc.action_offset + i] = target[action]
        out.append((context, teaching))
    return out


def train_steps(net, compiled, steps, start_cursor=0):
    done = 0
    cursor = start_cursor
    while done < steps:
        context, teaching = compiled[cursor % len(compiled)]
        cursor += 1
        net.step(context, reward=0, learn=False)
        done += 1
        if done >= steps:
            break
        net.step(teaching, reward=1, learn=True)
        done += 1
    return cursor


def population_probs(net):
    raw = {
        a: max(float(net.rate[idx].mean()), 1e-12)
        for a, idx in net.action_populations.items()
    }
    total = sum(raw.values())
    return {a: v / total for a, v in raw.items()}


def evaluate(net, enc, items, settle=40, probe=60):
    zero = np.zeros(enc.input_dim, dtype=np.float32)
    rows = []
    for item in items:
        net.reset_fast_state(noise=0.01)
        for _ in range(settle):
            net.step(zero, reward=0, learn=False)
        scalars = {k: float(item.get("scalars", {}).get(k, 0)) for k in SCALAR_KEYS}
        x = enc.encode(item["scenario"], scalars=scalars).vector
        acc = {a: 0.0 for a in ACTIONS}
        for _ in range(probe):
            net.step(x, reward=0, learn=False)
            for a, v in population_probs(net).items():
                acc[a] += v
        p = {a: v / probe for a, v in acc.items()}
        rows.append(
            {
                "id": item["id"],
                "domain": item.get("domain", "unknown"),
                "action_probabilities": p,
                "target_actions": item["target_actions"],
                "anti_actions": item.get("anti_actions", []),
            }
        )
    score = score_results(rows)
    return {
        k: score[k]
        for k in (
            "js_similarity",
            "top1_agreement",
            "domain_macro_js_similarity",
            "target_mass",
            "anti_mass",
        )
    }


def targeted_lesion(founder, mature, fraction):
    out = copy.deepcopy(mature)
    delta = np.abs(mature.W.data - founder.W.data)
    count = max(1, int(round(fraction * len(delta))))
    idx = np.argpartition(delta, -count)[-count:]
    out.W.data[idx] = 0.0
    out.eligibility.fill(0)
    out.reset_fast_state()
    return out, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=1024)
    ap.add_argument("--mature-steps", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=1842)
    ap.add_argument("--plasticity", type=float, default=16.0)
    ap.add_argument("--lesion", type=float, default=0.10)
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config/default.json").read_text())
    ncfg = cfg["network"]
    ncfg["neurons"] = args.neurons
    ncfg["seed"] = args.seed
    ncfg["avg_recurrent_degree"] = min(24, max(12, args.neurons // 64))
    ncfg["action_population_size"] = min(48, max(16, args.neurons // 24))
    ncfg["hebb_lr"] *= args.plasticity
    ncfg["reward_lr"] *= args.plasticity

    enc = ExperienceEncoder(ncfg["sensory_dim"])
    founder = PlasticRecurrentPersonaNet(ncfg, enc)
    mature = PlasticRecurrentPersonaNet(ncfg, enc)

    payload = json.loads(
        (
            ROOT
            / "data/phenotype_battery/pretorius_phenotype_train_battery_v1.json"
        ).read_text()
    )
    items = [x for x in payload["training_items"] if axis_of(x["id"])]
    compiled = compile_items(enc, items)

    validation = load_items(
        ROOT / "data/phenotype_battery/pretorius_validation_v1.json"
    )
    adversarial = load_items(
        ROOT / "data/phenotype_battery/pretorius_adversarial_v1.json"
    )

    train_steps(mature, compiled, args.mature_steps)
    lesioned, count = targeted_lesion(founder, mature, args.lesion)
    relearn = copy.deepcopy(lesioned)
    naive = copy.deepcopy(founder)

    checkpoints = [0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 24000]
    previous = 0
    cursor_relearn = 0
    cursor_naive = 0
    curve = []

    for checkpoint in checkpoints:
        additional = checkpoint - previous
        if additional:
            cursor_relearn = train_steps(
                relearn, compiled, additional, cursor_relearn
            )
            cursor_naive = train_steps(naive, compiled, additional, cursor_naive)

        curve.append(
            {
                "additional_steps": checkpoint,
                "lesioned_relearn": {
                    "validation": evaluate(
                        copy.deepcopy(relearn), enc, validation
                    ),
                    "adversarial": evaluate(
                        copy.deepcopy(relearn), enc, adversarial
                    ),
                },
                "naive_learn": {
                    "validation": evaluate(copy.deepcopy(naive), enc, validation),
                    "adversarial": evaluate(
                        copy.deepcopy(naive), enc, adversarial
                    ),
                },
            }
        )
        previous = checkpoint

    baseline = {
        "founder": {
            "validation": evaluate(copy.deepcopy(founder), enc, validation),
            "adversarial": evaluate(copy.deepcopy(founder), enc, adversarial),
        },
        "mature": {
            "validation": evaluate(copy.deepcopy(mature), enc, validation),
            "adversarial": evaluate(copy.deepcopy(mature), enc, adversarial),
        },
        "post_lesion": {
            "validation": evaluate(copy.deepcopy(lesioned), enc, validation),
            "adversarial": evaluate(copy.deepcopy(lesioned), enc, adversarial),
        },
    }

    out = {
        "experiment": "008_relearning_savings",
        "version": "v0.4-dev",
        "question": "After targeted destruction of the strongest learned synapses, does the damaged recurrent persona relearn faster than a virgin founder?",
        "terminal_battery_touched": False,
        "neurons": args.neurons,
        "mature_training_steps": args.mature_steps,
        "seed": args.seed,
        "plasticity_multiplier": args.plasticity,
        "targeted_lesion_fraction": args.lesion,
        "zeroed_synapses": count,
        "baseline": baseline,
        "relearning_curve": curve,
    }

    outdir = ROOT / "results/experiment_008_relearning_savings"
    outdir.mkdir(parents=True, exist_ok=True)
    path = (
        outdir
        / f"report_n{args.neurons}_mature{args.mature_steps}_s{args.seed}.json"
    )
    path.write_text(json.dumps(out, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
