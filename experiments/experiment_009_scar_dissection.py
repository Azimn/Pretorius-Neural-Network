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


def train_recurrent_only(net, compiled, neural_steps, cursor=0):
    start = int(net.tick)
    while int(net.tick) - start < neural_steps:
        context, teaching = compiled[cursor % len(compiled)]
        cursor += 1

        net.step(context, reward=0.0, learn=False)
        if int(net.tick) - start >= neural_steps:
            break

        net.step(teaching, reward=1.0, learn=True)

    return cursor, int(net.tick) - start


def population_probabilities(net):
    raw = {
        action: max(float(net.rate[idx].mean()), 1e-12)
        for action, idx in net.action_populations.items()
    }
    total = sum(raw.values())
    return {a: v / total for a, v in raw.items()}


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

        probs = {a: v / probe_steps for a, v in accum.items()}
        rows.append(
            {
                "id": item["id"],
                "domain": item.get("domain", "unknown"),
                "action_probabilities": probs,
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


def evaluate_both(net, encoder, validation, adversarial):
    return {
        "validation": evaluate(copy.deepcopy(net), encoder, validation),
        "adversarial": evaluate(copy.deepcopy(net), encoder, adversarial),
    }


def targeted_indices(founder, mature, fraction):
    delta = np.abs(mature.W.data - founder.W.data)
    count = max(1, int(round(len(delta) * fraction)))
    return np.argpartition(delta, -count)[-count:]


def make_damage_conditions(founder, mature, donor, idx, rng):
    conditions = {}

    zero = copy.deepcopy(mature)
    zero.W.data[idx] = 0.0
    zero.eligibility.fill(0.0)
    zero.reset_fast_state()
    conditions["zero"] = zero

    restored = copy.deepcopy(mature)
    restored.W.data[idx] = founder.W.data[idx]
    restored.eligibility.fill(0.0)
    restored.reset_fast_state()
    conditions["founder_restore"] = restored

    random_same_sign = copy.deepcopy(mature)
    excitatory_edges = founder.excitatory[founder.pre_idx]
    for sign in (True, False):
        target = idx[excitatory_edges[idx] == sign]
        pool = np.flatnonzero(excitatory_edges == sign)
        random_same_sign.W.data[target] = founder.W.data[
            rng.choice(pool, size=len(target), replace=True)
        ]
    random_same_sign.eligibility.fill(0.0)
    random_same_sign.reset_fast_state()
    conditions["same_sign_random"] = random_same_sign

    donor_transplant = copy.deepcopy(mature)
    donor_transplant.W.data[idx] = donor.W.data[idx]
    donor_transplant.eligibility.fill(0.0)
    donor_transplant.reset_fast_state()
    conditions["trained_donor"] = donor_transplant

    return conditions


def one_seed(
    seed,
    neurons,
    mature_neural_steps,
    relearning_neural_steps,
    lesion_fraction,
):
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

    compiled = compile_items(encoder, training)
    donor_compiled = compile_items(encoder, list(reversed(training)))

    _, mature_actual = train_recurrent_only(
        mature, compiled, mature_neural_steps
    )
    _, donor_actual = train_recurrent_only(
        donor, donor_compiled, mature_neural_steps
    )

    idx = targeted_indices(founder, mature, lesion_fraction)
    conditions = make_damage_conditions(
        founder,
        mature,
        donor,
        idx,
        np.random.default_rng(seed + 909),
    )
    conditions["virgin"] = copy.deepcopy(founder)
    conditions["mature"] = copy.deepcopy(mature)

    baseline = {
        name: evaluate_both(net, encoder, validation, adversarial)
        for name, net in conditions.items()
    }

    recovery = {
        name: copy.deepcopy(net)
        for name, net in conditions.items()
        if name != "mature"
    }
    actual_relearning = {}
    for name, net in recovery.items():
        _, actual = train_recurrent_only(
            net, compiled, relearning_neural_steps
        )
        actual_relearning[name] = actual

    after_relearning = {
        name: evaluate_both(net, encoder, validation, adversarial)
        for name, net in recovery.items()
    }

    return {
        "seed": seed,
        "targeted_synapses": int(len(idx)),
        "mature_actual_neural_steps": mature_actual,
        "donor_actual_neural_steps": donor_actual,
        "relearning_actual_neural_steps": actual_relearning,
        "baseline": baseline,
        "after_relearning": after_relearning,
    }


def aggregate(runs):
    output = {}
    for phase in ("baseline", "after_relearning"):
        output[phase] = {}
        for condition in runs[0][phase]:
            output[phase][condition] = {}
            for split in ("validation", "adversarial"):
                output[phase][condition][split] = {
                    "js_similarity": float(
                        np.mean(
                            [
                                r[phase][condition][split]["js_similarity"]
                                for r in runs
                            ]
                        )
                    ),
                    "top1_agreement": float(
                        np.mean(
                            [
                                r[phase][condition][split]["top1_agreement"]
                                for r in runs
                            ]
                        )
                    ),
                }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1842,101,202")
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--mature-neural-steps", type=int, default=24000)
    parser.add_argument("--relearning-neural-steps", type=int, default=12000)
    parser.add_argument("--lesion-fraction", type=float, default=0.10)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    runs = [
        one_seed(
            seed,
            args.neurons,
            args.mature_neural_steps,
            args.relearning_neural_steps,
            args.lesion_fraction,
        )
        for seed in seeds
    ]

    report = {
        "experiment": "009_scar_dissection",
        "question": (
            "Is the relearning deficit caused by loss of learned information, "
            "by zero-valued holes in a developed recurrent topology, or both?"
        ),
        "terminal_battery_touched": False,
        "neurons": args.neurons,
        "mature_neural_steps": args.mature_neural_steps,
        "relearning_neural_steps": args.relearning_neural_steps,
        "plasticity_multiplier": 16.0,
        "targeted_lesion_fraction": args.lesion_fraction,
        "evaluation": {"settle_steps": 40, "probe_steps": 60},
        "seeds": seeds,
        "runs": runs,
        "means": aggregate(runs),
    }

    outdir = ROOT / "results" / "experiment_009_scar_dissection"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (
        f"report_n{args.neurons}_mature{args.mature_neural_steps}_"
        f"relearn{args.relearning_neural_steps}.json"
    )
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
