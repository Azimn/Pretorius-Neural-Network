from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.battery import score_results
from persona_net.encoding import ACTIONS, SCALAR_KEYS
from persona_net.v04_data import load_v04_split

ROOT = Path(__file__).resolve().parents[1]


def normalize_target(raw):
    values = {a: max(float(raw.get(a, 0.0)), 0.0) for a in ACTIONS}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Target action distribution has no positive mass.")
    return {a: v / total for a, v in values.items()}


def compile_items(encoder, items):
    zero_action = slice(encoder.action_offset, encoder.input_dim)
    compiled = []
    for item in items:
        scalars = {
            k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS
        }
        target = normalize_target(item["target_actions"])
        base = encoder.encode(item["scenario"], scalars=scalars).vector

        context = base.copy()
        context[zero_action] = 0.0

        teaching = base.copy()
        for i, action in enumerate(ACTIONS):
            teaching[encoder.action_offset + i] = target[action]

        compiled.append((context, teaching))
    return compiled


def train_recurrent_only(net, compiled, neural_steps):
    start = int(net.tick)
    cursor = 0
    while int(net.tick) - start < neural_steps:
        context, teaching = compiled[cursor % len(compiled)]
        cursor += 1

        net.step(context, reward=0.0, learn=False)
        if int(net.tick) - start >= neural_steps:
            break

        net.step(teaching, reward=1.0, learn=True)

    return int(net.tick) - start


def population_probabilities(net):
    raw = {
        action: max(float(net.rate[idx].mean()), 1e-12)
        for action, idx in net.action_populations.items()
    }
    total = sum(raw.values())
    return {a: value / total for a, value in raw.items()}


def evaluate(net, encoder, items, settle_steps=40, probe_steps=60):
    zero = np.zeros(encoder.input_dim, dtype=np.float32)
    rows = []

    for item in items:
        net.reset_fast_state(noise=0.01)
        for _ in range(settle_steps):
            net.step(zero, reward=0.0, learn=False)

        scalars = {
            k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS
        }
        x = encoder.encode(item["scenario"], scalars=scalars).vector

        accum = {a: 0.0 for a in ACTIONS}
        for _ in range(probe_steps):
            net.step(x, reward=0.0, learn=False)
            for action, value in population_probabilities(net).items():
                accum[action] += value

        probabilities = {a: value / probe_steps for a, value in accum.items()}
        rows.append(
            {
                "id": item["id"],
                "domain": item.get("domain", "unknown"),
                "action_probabilities": probabilities,
                "target_actions": item["target_actions"],
                "anti_actions": item.get("anti_actions", []),
            }
        )

    score = score_results(rows)
    return {
        key: score[key]
        for key in (
            "js_similarity",
            "top1_agreement",
            "domain_macro_js_similarity",
            "target_mass",
            "anti_mass",
        )
    }


def evaluate_both(net, encoder, validation, adversarial):
    return {
        "validation": evaluate(copy.deepcopy(net), encoder, validation),
        "adversarial": evaluate(copy.deepcopy(net), encoder, adversarial),
    }


def graft(founder, source, indices):
    out = copy.deepcopy(founder)
    out.W.data[indices] = source.W.data[indices]
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
    donor = PlasticRecurrentPersonaNet(cfg, encoder)

    data_root = ROOT / "data" / "v0_4"
    training = load_v04_split(data_root, "train")
    validation = load_v04_split(data_root, "validation")
    adversarial = load_v04_split(data_root, "adversarial")

    mature_steps = train_recurrent_only(
        mature, compile_items(encoder, training), training_steps
    )
    donor_steps = train_recurrent_only(
        donor, compile_items(encoder, list(reversed(training))), training_steps
    )

    baseline = {
        "founder": evaluate_both(founder, encoder, validation, adversarial),
        "mature": evaluate_both(mature, encoder, validation, adversarial),
        "donor": evaluate_both(donor, encoder, validation, adversarial),
    }

    delta = np.abs(mature.W.data - founder.W.data)
    order = np.argsort(delta)[::-1]
    edge_count = len(delta)
    rng = np.random.default_rng(seed + 1010)

    fraction_results = {}
    for fraction in fractions:
        count = max(1, int(round(fraction * edge_count)))
        top_indices = order[:count]
        random_indices = rng.choice(edge_count, size=count, replace=False)

        conditions = {
            "self_top": graft(founder, mature, top_indices),
            "donor_top": graft(founder, donor, top_indices),
            "random_location": graft(founder, mature, random_indices),
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
        "mature_actual_neural_steps": mature_steps,
        "donor_actual_neural_steps": donor_steps,
        "recurrent_edges": edge_count,
        "baseline": baseline,
        "fractions": fraction_results,
    }


def aggregate(runs):
    output = {"baseline": {}, "fractions": {}}

    for name in ("founder", "mature", "donor"):
        output["baseline"][name] = {}
        for split in ("validation", "adversarial"):
            output["baseline"][name][split] = {
                metric: float(
                    np.mean([r["baseline"][name][split][metric] for r in runs])
                )
                for metric in ("js_similarity", "top1_agreement")
            }

    for fraction in runs[0]["fractions"]:
        output["fractions"][fraction] = {
            "count_mean": float(
                np.mean([r["fractions"][fraction]["count"] for r in runs])
            ),
            "conditions": {},
        }
        for condition in runs[0]["fractions"][fraction]["conditions"]:
            output["fractions"][fraction]["conditions"][condition] = {}
            for split in ("validation", "adversarial"):
                output["fractions"][fraction]["conditions"][condition][split] = {
                    metric: float(
                        np.mean(
                            [
                                r["fractions"][fraction]["conditions"][condition][
                                    split
                                ][metric]
                                for r in runs
                            ]
                        )
                    )
                    for metric in ("js_similarity", "top1_agreement")
                }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 010: minimal transferable recurrent persona core"
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
        "experiment": "010_minimal_transferable_core",
        "question": (
            "How small a subset of the most changed recurrent synapses is "
            "sufficient to transfer the Pretorius phenotype into a matched "
            "virgin founder?"
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

    outdir = ROOT / "results" / "experiment_010_minimal_transferable_core"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (
        f"report_n{args.neurons}_steps{args.training_neural_steps}.json"
    )
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
