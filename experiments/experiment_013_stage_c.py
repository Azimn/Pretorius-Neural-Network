from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.v04_data import load_v04_split
from experiment_010_minimal_transferable_core import compile_items, evaluate_both, train_recurrent_only
from experiment_012_core_convergence import configured
from experiment_013_cross_topology import fingerprints

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1842, 101, 202, 303, 404, 505)
FRACTIONS = (0.0025, 0.01, 0.05, 0.10)


def build_pair(seed: int, neurons: int, compiled, training_steps: int):
    cfg = configured(seed, neurons)
    encoder = ExperienceEncoder(cfg["sensory_dim"])
    founder = PlasticRecurrentPersonaNet(cfg, encoder)
    mature = PlasticRecurrentPersonaNet(cfg, encoder)
    actual = train_recurrent_only(mature, compiled(encoder), training_steps)
    return encoder, founder, mature, actual


def core_indices(founder, mature, fraction: float):
    delta = np.abs(mature.W.data - founder.W.data)
    count = max(1, int(round(fraction * len(delta))))
    return np.argsort(delta)[::-1][:count]


def edge_signatures(net, neuron_fp, indices):
    idx = np.asarray(indices, dtype=np.int64)
    post = neuron_fp[net.post_idx[idx]]
    pre = neuron_fp[net.pre_idx[idx]]
    sig = np.concatenate([post, pre], axis=1)
    norm = np.linalg.norm(sig, axis=1, keepdims=True)
    return sig / np.where(norm > 1e-8, norm, 1.0)


def sign_class(net, indices):
    idx = np.asarray(indices, dtype=np.int64)
    return net.excitatory[net.pre_idx[idx]]


def symmetric_match_similarity(sig_a, sign_a, sig_b, sign_b):
    # Mean best compatible edge similarity in both directions. No raw edge indices are compared.
    sim = sig_a @ sig_b.T
    compatible = sign_a[:, None] == sign_b[None, :]
    masked = np.where(compatible, sim, -np.inf)
    a_best = np.max(masked, axis=1)
    b_best = np.max(masked, axis=0)
    if not np.all(np.isfinite(a_best)) or not np.all(np.isfinite(b_best)):
        raise RuntimeError("compatible Dale-sign edge absent from comparison set")
    return float(0.5 * (a_best.mean() + b_best.mean()))


def sign_matched_random(net, count, target_signs, rng):
    all_idx = np.arange(len(net.W.data), dtype=np.int64)
    all_sign = sign_class(net, all_idx)
    n_exc = int(np.sum(target_signs))
    n_inh = count - n_exc
    exc = all_idx[all_sign]
    inh = all_idx[~all_sign]
    return np.concatenate([
        rng.choice(exc, size=n_exc, replace=False),
        rng.choice(inh, size=n_inh, replace=False),
    ])


def run(neurons: int, training_steps: int, seeds=SEEDS, fractions=FRACTIONS):
    data_root = ROOT / "data" / "v0_4"
    training = load_v04_split(data_root, "train")
    validation = load_v04_split(data_root, "validation")
    adversarial = load_v04_split(data_root, "adversarial")

    # Each topology gets its own encoder/configured network. Training situations are public only.
    nets = {}
    for seed in seeds:
        cfg = configured(seed, neurons)
        encoder = ExperienceEncoder(cfg["sensory_dim"])
        compiled = compile_items(encoder, training)
        founder = PlasticRecurrentPersonaNet(cfg, encoder)
        mature = PlasticRecurrentPersonaNet(cfg, encoder)
        actual = train_recurrent_only(mature, compiled, training_steps)
        if actual != training_steps:
            raise RuntimeError((seed, actual, training_steps))
        nets[seed] = {
            "encoder": encoder,
            "compiled": compiled,
            "founder": founder,
            "mature": mature,
            "fingerprints": fingerprints(copy.deepcopy(mature), compiled),
            "scores": evaluate_both(mature, encoder, validation, adversarial),
            "actual_training_neural_steps": actual,
        }

    pair_results = []
    for i, a_seed in enumerate(seeds):
        for b_seed in seeds[i + 1:]:
            a = nets[a_seed]
            b = nets[b_seed]
            scales = {}
            for fraction in fractions:
                ai = core_indices(a["founder"], a["mature"], fraction)
                bi = core_indices(b["founder"], b["mature"], fraction)
                asig = edge_signatures(a["mature"], a["fingerprints"], ai)
                bsig = edge_signatures(b["mature"], b["fingerprints"], bi)
                asign = sign_class(a["mature"], ai)
                bsign = sign_class(b["mature"], bi)
                core_similarity = symmetric_match_similarity(asig, asign, bsig, bsign)

                # Fixed pair/fraction RNG, same-size and Dale-sign-matched random-edge null.
                token = f"013C:{a_seed}:{b_seed}:{fraction:.4f}".encode()
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
                ar = sign_matched_random(a["mature"], len(ai), asign, rng)
                br = sign_matched_random(b["mature"], len(bi), bsign, rng)
                random_similarity = symmetric_match_similarity(
                    edge_signatures(a["mature"], a["fingerprints"], ar), sign_class(a["mature"], ar),
                    edge_signatures(b["mature"], b["fingerprints"], br), sign_class(b["mature"], br),
                )
                scales[str(fraction)] = {
                    "edges_a": int(len(ai)), "edges_b": int(len(bi)),
                    "core_functional_similarity": core_similarity,
                    "random_sign_matched_similarity": random_similarity,
                    "core_minus_random": core_similarity - random_similarity,
                }
            pair_results.append({"seed_a": a_seed, "seed_b": b_seed, "scales": scales})

    aggregates = {}
    for fraction in fractions:
        key = str(fraction)
        diffs = np.array([p["scales"][key]["core_minus_random"] for p in pair_results])
        cores = np.array([p["scales"][key]["core_functional_similarity"] for p in pair_results])
        randoms = np.array([p["scales"][key]["random_sign_matched_similarity"] for p in pair_results])
        aggregates[key] = {
            "pair_count": len(pair_results),
            "mean_core_similarity": float(cores.mean()),
            "mean_random_similarity": float(randoms.mean()),
            "mean_core_minus_random": float(diffs.mean()),
            "median_core_minus_random": float(np.median(diffs)),
            "pairs_core_gt_random": int(np.sum(diffs > 0)),
        }

    return {
        "experiment": "013_cross_topology_functional_core_convergence",
        "stage": "C_cross_topology_core_comparison",
        "issue": 15,
        "stage_b_gate_required_and_passed": True,
        "terminal_battery_touched": False,
        "raw_recurrent_edge_indices_compared_across_topologies": False,
        "seeds": list(seeds),
        "neurons": neurons,
        "training_neural_steps_per_topology": training_steps,
        "total_training_neural_steps": training_steps * len(seeds),
        "alignment_inputs": "unlabeled public training situations only",
        "edge_similarity": "cosine of concatenated mature post/pre neuron response fingerprints; hard Dale-sign compatibility; symmetric mean-best match",
        "random_control": "same-size, Dale-sign-matched recurrent edges selected with deterministic SHA-256-derived pair/fraction seeds",
        "mature_phenotype": {str(seed): nets[seed]["scores"] for seed in seeds},
        "pair_results": pair_results,
        "aggregate": aggregates,
    }


def main():
    p = argparse.ArgumentParser(description="Experiment 013 Stage C independent-topology functional core comparison")
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--training-neural-steps", type=int, default=24000)
    args = p.parse_args()
    report = run(args.neurons, args.training_neural_steps)
    outdir = ROOT / "results" / "experiment_013_cross_topology"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"stage_c_n{args.neurons}_steps{args.training_neural_steps}.json"
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
