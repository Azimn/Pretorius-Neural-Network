from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.battery import score_results
from persona_net.encoding import ACTIONS, SCALAR_KEYS
from persona_net.v04_data import load_v04_split

ROOT = Path(__file__).resolve().parents[1]


def norm_target(raw):
    values = {a: max(float(raw.get(a, 0.0)), 0.0) for a in ACTIONS}
    total = sum(values.values())
    return {a: v / total for a, v in values.items()}


def train_recurrent_only(net, encoder, items, neural_steps):
    start = int(net.tick)
    cursor = 0
    zero_action = slice(encoder.action_offset, encoder.input_dim)

    while int(net.tick) - start < neural_steps:
        item = items[cursor % len(items)]
        cursor += 1
        scalars = {
            k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS
        }
        target = norm_target(item["target_actions"])
        context = encoder.encode(
            item["scenario"], scalars=scalars, action=None, outcome=0.0
        ).vector

        x = context.copy()
        x[zero_action] = 0.0
        net.step(x, reward=0.0, learn=False)
        if int(net.tick) - start >= neural_steps:
            break

        teaching = context.copy()
        for ai, action in enumerate(ACTIONS):
            teaching[encoder.action_offset + ai] = target[action]
        net.step(teaching, reward=1.0, learn=True)

    return int(net.tick) - start


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

        scalars = {
            k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS
        }
        x = encoder.encode(
            item["scenario"], scalars=scalars, action=None, outcome=0.0
        ).vector
        accum = {a: 0.0 for a in ACTIONS}

        for _ in range(probe):
            net.step(x, reward=0.0, learn=False)
            for action, value in population_probs(net).items():
                accum[action] += value

        probs = {a: value / probe for a, value in accum.items()}
        rows.append(
            {
                "id": item["id"],
                "domain": item.get("domain", "unknown"),
                "action_probabilities": probs,
                "target_actions": item["target_actions"],
                "anti_actions": item.get("anti_actions", []),
            }
        )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--neural-steps", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=1842)
    args = parser.parse_args()

    base = json.loads((ROOT / "config" / "default.json").read_text())
    base["network"]["neurons"] = args.neurons
    base["network"]["seed"] = args.seed
    base["network"]["avg_recurrent_degree"] = min(
        24, max(12, args.neurons // 64)
    )
    base["network"]["action_population_size"] = min(
        48, max(16, args.neurons // 24)
    )

    data_root = ROOT / "data" / "v0_4"
    train = load_v04_split(data_root, "train")
    validation = load_v04_split(data_root, "validation")
    adversarial = load_v04_split(data_root, "adversarial")

    conditions = []
    for multiplier in (1.0, 4.0, 16.0, 64.0):
        cfg = json.loads(json.dumps(base))
        cfg["network"]["hebb_lr"] *= multiplier
        cfg["network"]["reward_lr"] *= multiplier

        encoder = ExperienceEncoder(cfg["network"]["sensory_dim"])
        net = PlasticRecurrentPersonaNet(cfg["network"], encoder)
        w0 = net.W.data.copy()
        b0 = net.bias.copy()

        before_validation = evaluate(net, encoder, validation)
        before_adversarial = evaluate(net, encoder, adversarial)

        net.reset_fast_state(noise=0.01)
        started = time.time()
        actual_steps = train_recurrent_only(
            net, encoder, train, args.neural_steps
        )
        elapsed = time.time() - started

        after_validation = evaluate(net, encoder, validation)
        after_adversarial = evaluate(net, encoder, adversarial)

        conditions.append(
            {
                "plasticity_multiplier": multiplier,
                "requested_neural_steps": args.neural_steps,
                "actual_neural_steps": actual_steps,
                "elapsed_seconds": elapsed,
                "mean_abs_W_change": float(
                    np.mean(np.abs(net.W.data - w0))
                ),
                "mean_abs_bias_change": float(
                    np.mean(np.abs(net.bias - b0))
                ),
                "before_validation": compact(before_validation),
                "after_validation": compact(after_validation),
                "before_adversarial": compact(before_adversarial),
                "after_adversarial": compact(after_adversarial),
            }
        )

    report = {
        "experiment": "004_recurrent_embedding_sweep_reproducible",
        "question": (
            "Can stronger recurrent plasticity embed the phenotype so it is "
            "recoverable directly from neural action populations with no "
            "learned motor decoder?"
        ),
        "terminal_battery_touched": False,
        "seed": args.seed,
        "neurons": args.neurons,
        "requested_neural_steps": args.neural_steps,
        "conditions": conditions,
    }

    outdir = ROOT / "results" / "experiment_004_recurrent_embedding_sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (
        f"repro_n{args.neurons}_steps{args.neural_steps}_s{args.seed}.json"
    )
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
