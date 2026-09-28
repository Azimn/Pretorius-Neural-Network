from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.v04_data import load_v04_split
from experiment_010_minimal_transferable_core import (
    aggregate as _unused_aggregate,
    compile_items,
    evaluate_both,
    train_recurrent_only,
)

ROOT = Path(__file__).resolve().parents[1]


def restore_to_founder(mature, founder, indices):
    """Erase learned values at selected recurrent edges without creating zero holes."""
    out = copy.deepcopy(mature)
    out.W.data[indices] = founder.W.data[indices]
    out.eligibility.fill(0.0)
    out.reset_fast_state()
    return out


def one_seed(seed, neurons, training_steps, fractions):
    config = json.loads((ROOT / "config" / "default.json").read_text())
    cfg = config["network"]
    cfg["neurons"] = neurons
    cfg["seed"] = seed
    cfg["avg_recurrent_degree"] = min(24, max(12, neurons // 64))
    cfg["action_population_size"] = min(48, max(16, neurons // 24))
    cfg["hebb_lr"] *= 16.0
    cfg["reward_lr"] *= 16.0

    encoder = ExperienceEncoder(cfg["sensory_dim"])
    founder = PlasticRecurrentPersonaNet(cfg, encoder)
    mature = PlasticRecurrentPersonaNet(cfg, encoder)

    data_root = ROOT / "data" / "v0_4"
    training = load_v04_split(data_root, "train")
    validation = load_v04_split(data_root, "validation")
    adversarial = load_v04_split(data_root, "adversarial")

    actual_steps = train_recurrent_only(
        mature, compile_items(encoder, training), training_steps
    )

    baseline = {
        "founder": evaluate_both(founder, encoder, validation, adversarial),
        "mature": evaluate_both(mature, encoder, validation, adversarial),
    }

    delta = np.abs(mature.W.data - founder.W.data)
    order = np.argsort(delta)[::-1]
    edge_count = len(delta)
    rng = np.random.default_rng(seed + 1111)

    fraction_results = {}
    for fraction in fractions:
        count = max(1, int(round(fraction * edge_count)))
        top_indices = order[:count]
        bottom_indices = order[-count:]
        random_indices = rng.choice(edge_count, size=count, replace=False)

        conditions = {
            "top_restored_to_founder": restore_to_founder(
                mature, founder, top_indices
            ),
            "random_restored_to_founder": restore_to_founder(
                mature, founder, random_indices
            ),
            "bottom_restored_to_founder": restore_to_founder(
                mature, founder, bottom_indices
            ),
        }
        fraction_results[str(fraction)] = {
            "count": count,
            "conditions": {
                name: evaluate_both(net, encoder, validation, adversarial)
                for name, net in conditions.items()
            },
        }

    return {
        "seed": seed,
        "actual_training_neural_steps": actual_steps,
        "recurrent_edges": edge_count,
        "baseline": baseline,
        "fractions": fraction_results,
    }


def aggregate(runs):
    metrics = ("js_similarity", "top1_agreement")
    output = {"baseline": {}, "fractions": {}}
    for name in ("founder", "mature"):
        output["baseline"][name] = {}
        for split in ("validation", "adversarial"):
            output["baseline"][name][split] = {
                metric: float(np.mean([
                    r["baseline"][name][split][metric] for r in runs
                ]))
                for metric in metrics
            }

    for fraction in runs[0]["fractions"]:
        output["fractions"][fraction] = {
            "count_mean": float(np.mean([
                r["fractions"][fraction]["count"] for r in runs
            ])),
            "conditions": {},
        }
        for condition in runs[0]["fractions"][fraction]["conditions"]:
            output["fractions"][fraction]["conditions"][condition] = {}
            for split in ("validation", "adversarial"):
                output["fractions"][fraction]["conditions"][condition][split] = {
                    metric: float(np.mean([
                        r["fractions"][fraction]["conditions"][condition][split][metric]
                        for r in runs
                    ]))
                    for metric in metrics
                }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 011: necessity of the minimal transferable core"
    )
    parser.add_argument("--seeds", default="1842,101,202")
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--training-neural-steps", type=int, default=24000)
    parser.add_argument("--fractions", default="0.0025,0.01,0.05,0.10")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    fractions = [float(value) for value in args.fractions.split(",")]
    runs = [
        one_seed(seed, args.neurons, args.training_neural_steps, fractions)
        for seed in seeds
    ]

    report = {
        "experiment": "011_core_necessity",
        "question": (
            "Are the high-change recurrent synapses that are sufficient to transfer "
            "Pretorius also disproportionately necessary for maintaining the mature phenotype?"
        ),
        "design": (
            "Restore selected mature recurrent weights to their matched founder values, "
            "avoiding the zero-hole lesion artifact isolated in Experiment 009. Compare "
            "high-change edges with equal-size random and lowest-change controls."
        ),
        "terminal_battery_touched": False,
        "neurons": args.neurons,
        "training_neural_steps": args.training_neural_steps,
        "plasticity_multiplier": 16.0,
        "evaluation": {"settle_steps": 40, "probe_steps": 60},
        "seeds": seeds,
        "fractions": fractions,
        "runs": runs,
        "means": aggregate(runs),
    }

    outdir = ROOT / "results" / "experiment_011_core_necessity"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"report_n{args.neurons}_steps{args.training_neural_steps}.json"
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
