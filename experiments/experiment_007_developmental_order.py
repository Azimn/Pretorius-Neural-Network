from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.battery import load_items, score_results
from persona_net.encoding import ACTIONS, SCALAR_KEYS

ROOT = Path(__file__).resolve().parents[1]


def axis_of(item_id: str):
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


def train_cycle(net, enc, items, total_steps):
    compiled = compile_items(enc, items)
    done = 0
    i = 0
    while done < total_steps:
        context, teaching = compiled[i % len(compiled)]
        i += 1
        net.step(context, reward=0, learn=False)
        done += 1
        if done >= total_steps:
            break
        net.step(teaching, reward=1, learn=True)
        done += 1
    return done


def train_block(net, enc, first, second, half_steps):
    train_cycle(net, enc, first, half_steps)
    train_cycle(net, enc, second, half_steps)


def train_alternating_epochs(net, enc, a, b, total_steps, epoch_steps):
    done = 0
    side = 0
    while done < total_steps:
        budget = min(epoch_steps, total_steps - done)
        train_cycle(net, enc, a if side % 2 == 0 else b, budget)
        done += budget
        side += 1


def population_probs(net):
    raw = {
        a: max(float(net.rate[idx].mean()), 1e-12)
        for a, idx in net.action_populations.items()
    }
    s = sum(raw.values())
    return {a: v / s for a, v in raw.items()}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=1842)
    ap.add_argument("--plasticity", type=float, default=16.0)
    ap.add_argument("--epoch", type=int, default=2000)
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
    payload = json.loads(
        (
            ROOT
            / "data/phenotype_battery/pretorius_phenotype_train_battery_v1.json"
        ).read_text()
    )
    grounded = [x for x in payload["training_items"] if axis_of(x["id"])]
    axes = sorted({axis_of(x["id"]) for x in grounded})
    aa = set(axes[::2])
    ab = set(axes[1::2])
    a_items = [x for x in grounded if axis_of(x["id"]) in aa]
    b_items = [x for x in grounded if axis_of(x["id"]) in ab]

    validation = load_items(
        ROOT / "data/phenotype_battery/pretorius_validation_v1.json"
    )
    adversarial = load_items(
        ROOT / "data/phenotype_battery/pretorius_adversarial_v1.json"
    )

    nets = {
        k: PlasticRecurrentPersonaNet(ncfg, enc)
        for k in ("founder", "interleaved", "A_then_B", "B_then_A", "short_epochs")
    }
    train_cycle(nets["interleaved"], enc, grounded, args.steps)
    half = args.steps // 2
    train_block(nets["A_then_B"], enc, a_items, b_items, half)
    train_block(nets["B_then_A"], enc, b_items, a_items, half)
    train_alternating_epochs(
        nets["short_epochs"], enc, a_items, b_items, args.steps, args.epoch
    )

    conditions = {}
    for name, net in nets.items():
        conditions[name] = {
            "validation": compact(evaluate(copy.deepcopy(net), enc, validation)),
            "adversarial": compact(evaluate(copy.deepcopy(net), enc, adversarial)),
        }

    base = nets["founder"].W.data.astype(np.float64)
    learned = {
        name: net.W.data.astype(np.float64) - base
        for name, net in nets.items()
        if name != "founder"
    }
    pairwise = {}
    names = sorted(learned)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            da, db = learned[a], learned[b]
            denom = float(np.linalg.norm(da) * np.linalg.norm(db))
            cosine = float(np.dot(da, db) / denom) if denom > 0 else 0.0
            rel = float(
                np.linalg.norm(da - db)
                / (0.5 * (np.linalg.norm(da) + np.linalg.norm(db)) + 1e-12)
            )
            pairwise[f"{a}__{b}"] = {
                "delta_cosine": cosine,
                "relative_l2_distance": rel,
            }

    out = {
        "experiment": "007_developmental_order",
        "version": "v0.4-dev",
        "question": "Does phenotype co-development order alter the recurrent persona under matched exposure?",
        "terminal_battery_touched": False,
        "neurons": args.neurons,
        "neural_training_steps": args.steps,
        "seed": args.seed,
        "plasticity_multiplier": args.plasticity,
        "epoch_steps": args.epoch,
        "axes_a": sorted(aa),
        "axes_b": sorted(ab),
        "conditions": conditions,
        "pairwise_weight_delta_similarity": pairwise,
    }

    outdir = ROOT / "results/experiment_007_developmental_order"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"report_n{args.neurons}_steps{args.steps}_s{args.seed}.json"
    path.write_text(json.dumps(out, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
