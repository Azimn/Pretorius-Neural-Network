from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.v04_data import load_v04_split
from experiment_010_minimal_transferable_core import compile_items, evaluate_both, graft, train_recurrent_only

ROOT = Path(__file__).resolve().parents[1]


def configured(seed, neurons):
    config = json.loads((ROOT / "config" / "default.json").read_text())
    cfg = config["network"]
    cfg["neurons"] = neurons
    cfg["seed"] = seed
    cfg["avg_recurrent_degree"] = min(24, max(12, neurons // 64))
    cfg["action_population_size"] = min(48, max(16, neurons // 24))
    cfg["hebb_lr"] *= 16.0
    cfg["reward_lr"] *= 16.0
    return cfg


def one_seed(seed, neurons, training_steps, fractions):
    cfg = configured(seed, neurons)
    encoder = ExperienceEncoder(cfg["sensory_dim"])
    founder = PlasticRecurrentPersonaNet(cfg, encoder)
    data_root = ROOT / "data" / "v0_4"
    training = load_v04_split(data_root, "train")
    validation = load_v04_split(data_root, "validation")
    adversarial = load_v04_split(data_root, "adversarial")

    rng = np.random.default_rng(seed + 1212)
    shuffled = list(training)
    rng.shuffle(shuffled)
    curricula = {"canonical": training, "reversed": list(reversed(training)), "shuffled": shuffled}
    mature = {}
    steps = {}
    for name, curriculum in curricula.items():
        net = PlasticRecurrentPersonaNet(cfg, encoder)
        steps[name] = train_recurrent_only(net, compile_items(encoder, curriculum), training_steps)
        mature[name] = net

    deltas = {name: np.abs(net.W.data - founder.W.data) for name, net in mature.items()}
    orders = {name: np.argsort(delta)[::-1] for name, delta in deltas.items()}
    edge_count = len(founder.W.data)
    baseline = {"founder": evaluate_both(founder, encoder, validation, adversarial)}
    baseline.update({name: evaluate_both(net, encoder, validation, adversarial) for name, net in mature.items()})

    pairs = (("canonical", "reversed"), ("canonical", "shuffled"), ("reversed", "shuffled"))
    results = {}
    for fraction in fractions:
        count = max(1, int(round(fraction * edge_count)))
        tops = {name: order[:count] for name, order in orders.items()}
        pair_stats = {}
        for a, b in pairs:
            aset, bset = set(map(int, tops[a])), set(map(int, tops[b]))
            union = aset | bset
            overlap = aset & bset
            av = mature[a].W.data[tops[a]] - founder.W.data[tops[a]]
            bv_at_a = mature[b].W.data[tops[a]] - founder.W.data[tops[a]]
            corr = float(np.corrcoef(av, bv_at_a)[0, 1]) if count > 1 and np.std(av) and np.std(bv_at_a) else 0.0
            pair_stats[f"{a}__{b}"] = {
                "overlap_count": len(overlap),
                "jaccard": len(overlap) / len(union),
                "delta_value_correlation_on_a_core": corr,
            }

        cross = {}
        for donor_name in ("reversed", "shuffled"):
            cross[f"{donor_name}_values_at_canonical_core"] = evaluate_both(
                graft(founder, mature[donor_name], tops["canonical"]), encoder, validation, adversarial
            )
            cross[f"{donor_name}_own_core"] = evaluate_both(
                graft(founder, mature[donor_name], tops[donor_name]), encoder, validation, adversarial
            )
        cross["canonical_own_core"] = evaluate_both(
            graft(founder, mature["canonical"], tops["canonical"]), encoder, validation, adversarial
        )
        results[str(fraction)] = {"count": count, "pair_stats": pair_stats, "grafts": cross}

    return {"seed": seed, "actual_training_neural_steps": steps, "recurrent_edges": edge_count, "baseline": baseline, "fractions": results}


def aggregate(runs):
    out = {"fractions": {}}
    for fraction in runs[0]["fractions"]:
        pair_names = runs[0]["fractions"][fraction]["pair_stats"]
        out["fractions"][fraction] = {"pair_stats": {}}
        for pair in pair_names:
            out["fractions"][fraction]["pair_stats"][pair] = {
                metric: float(np.mean([r["fractions"][fraction]["pair_stats"][pair][metric] for r in runs]))
                for metric in ("jaccard", "delta_value_correlation_on_a_core")
            }
    return out


def main():
    p = argparse.ArgumentParser(description="Experiment 012: convergence of the causal recurrent core across developmental order")
    p.add_argument("--seeds", default="1842,101,202")
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--training-neural-steps", type=int, default=24000)
    p.add_argument("--fractions", default="0.0025,0.01,0.05,0.10")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]
    runs = [one_seed(seed, args.neurons, args.training_neural_steps, fractions) for seed in seeds]
    report = {
        "experiment": "012_core_convergence",
        "question": "Does the causal recurrent Pretorius core converge to the same locations and values when the same founder topology develops under different curriculum orders?",
        "design": "Train canonical, reversed, and deterministically shuffled curricula on identical founder topology. Compare high-change core overlap and value correlation, then cross-graft independently learned values and independently selected cores into the matched virgin founder.",
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
    outdir = ROOT / "results" / "experiment_012_core_convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"report_n{args.neurons}_steps{args.training_neural_steps}.json"
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
