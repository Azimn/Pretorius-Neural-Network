from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.v04_data import load_v04_split
from experiment_010_minimal_transferable_core import compile_items, evaluate_both, train_recurrent_only
from experiment_012_core_convergence import configured
from experiment_013_cross_topology import fingerprints, run_gate
from experiment_013_stage_c import SEEDS, core_indices, edge_signatures, sign_class

ROOT = Path(__file__).resolve().parents[1]
CORE_FRACTION = 0.01
CYCLIC_PAIRS = tuple(zip(SEEDS, SEEDS[1:] + SEEDS[:1]))


def clip_for_edge(net, edge_idx: int, value: float) -> float:
    if net.excitatory[net.pre_idx[edge_idx]]:
        return float(np.clip(value, 0.0, net.max_abs_weight))
    return float(np.clip(value, -net.max_abs_weight, 0.0))


def aligned_recipient_edges(donor, donor_fp, donor_core, recipient, recipient_fp):
    """Unique maximum-cosine edge assignment with hard Dale-sign compatibility."""
    donor_core = np.asarray(donor_core, dtype=np.int64)
    recipient_edges = np.arange(len(recipient.W.data), dtype=np.int64)
    dsig = edge_signatures(donor, donor_fp, donor_core)
    rsig = edge_signatures(recipient, recipient_fp, recipient_edges)
    dsign = sign_class(donor, donor_core)
    rsign = sign_class(recipient, recipient_edges)
    mapped = np.empty(len(donor_core), dtype=np.int64)
    similarities = np.empty(len(donor_core), dtype=np.float64)
    for dale in (False, True):
        dr = np.flatnonzero(dsign == dale)
        rr = np.flatnonzero(rsign == dale)
        if len(dr) == 0:
            continue
        if len(rr) < len(dr):
            raise RuntimeError("recipient lacks enough compatible Dale-sign edges")
        sim = dsig[dr] @ rsig[rr].T
        rows, cols = linear_sum_assignment(-sim.astype(np.float64))
        mapped[dr[rows]] = recipient_edges[rr[cols]]
        similarities[dr[rows]] = sim[rows, cols]
    return mapped, similarities


def graft_deltas(recipient_founder, recipient_edges, deltas):
    out = copy.deepcopy(recipient_founder)
    recipient_edges = np.asarray(recipient_edges, dtype=np.int64)
    deltas = np.asarray(deltas, dtype=np.float64)
    if len(recipient_edges) != len(deltas) or len(set(recipient_edges.tolist())) != len(recipient_edges):
        raise ValueError("graft requires one delta per unique recipient edge")
    for edge, delta in zip(recipient_edges, deltas):
        value = float(out.W.data[edge]) + float(delta)
        out.W.data[edge] = clip_for_edge(out, int(edge), value)
    out.eligibility.fill(0.0)
    out.reset_fast_state()
    return out


def sign_matched_random_edges(net, target_signs, token: bytes):
    """Return unique random edges whose sign matches each donor position exactly."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
    all_edges = np.arange(len(net.W.data), dtype=np.int64)
    all_signs = sign_class(net, all_edges)
    selected = np.empty(len(target_signs), dtype=np.int64)
    for dale in (False, True):
        positions = np.flatnonzero(target_signs == dale)
        pool = all_edges[all_signs == dale]
        selected[positions] = rng.choice(pool, size=len(positions), replace=False)
    return selected


def recovery_fraction(founder_value, mature_value, graft_value):
    improvement = float(mature_value) - float(founder_value)
    if abs(improvement) < 1e-12:
        return None
    return (float(graft_value) - float(founder_value)) / improvement


def recovery(scores, founder_scores, mature_scores):
    out = {}
    for split in ("validation", "adversarial"):
        out[split] = {}
        for metric in ("js_similarity", "top1_agreement"):
            out[split][metric] = recovery_fraction(founder_scores[split][metric], mature_scores[split][metric], scores[split][metric])
    return out


def run(neurons: int, training_steps: int, seeds=SEEDS):
    if tuple(seeds) != SEEDS:
        raise ValueError("Stage D uses the preregistered six seeds and cyclic ordering")
    data_root = ROOT / "data" / "v0_4"
    training = load_v04_split(data_root, "train")
    validation = load_v04_split(data_root, "validation")
    adversarial = load_v04_split(data_root, "adversarial")
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
            "encoder": encoder, "founder": founder, "mature": mature,
            "fp": fingerprints(copy.deepcopy(mature), compiled), "actual": actual,
            "founder_scores": evaluate_both(founder, encoder, validation, adversarial),
            "mature_scores": evaluate_both(mature, encoder, validation, adversarial),
        }

    pair_results = []
    for donor_seed, recipient_seed in CYCLIC_PAIRS:
        donor, recipient = nets[donor_seed], nets[recipient_seed]
        donor_core = core_indices(donor["founder"], donor["mature"], CORE_FRACTION)
        recipient_core = core_indices(recipient["founder"], recipient["mature"], CORE_FRACTION)
        donor_delta = donor["mature"].W.data[donor_core] - donor["founder"].W.data[donor_core]
        recipient_delta = recipient["mature"].W.data[recipient_core] - recipient["founder"].W.data[recipient_core]
        matched_edges, match_similarity = aligned_recipient_edges(donor["mature"], donor["fp"], donor_core, recipient["mature"], recipient["fp"])
        donor_signs = sign_class(donor["mature"], donor_core)
        recipient_signs = sign_class(recipient["mature"], matched_edges)
        if not np.array_equal(donor_signs, recipient_signs):
            raise RuntimeError("Dale-sign compatibility violated by alignment")
        random_edges = sign_matched_random_edges(recipient["mature"], donor_signs, f"013D:{donor_seed}:{recipient_seed}:random".encode())
        if not np.array_equal(donor_signs, sign_class(recipient["mature"], random_edges)):
            raise RuntimeError("Dale-sign compatibility violated by random control")
        shuffle_rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"013D:{donor_seed}:{recipient_seed}:shuffle".encode()).digest()[:8], "little"))
        shuffled_delta = donor_delta.copy()
        for dale in (False, True):
            positions = np.flatnonzero(donor_signs == dale)
            shuffled_delta[positions] = shuffled_delta[shuffle_rng.permutation(positions)]

        conditions = {
            "cross_topology_aligned": graft_deltas(recipient["founder"], matched_edges, donor_delta),
            "within_topology_core": graft_deltas(recipient["founder"], recipient_core, recipient_delta),
            "random_edge_sign_matched": graft_deltas(recipient["founder"], random_edges, donor_delta),
            "shuffled_delta_same_aligned_edges": graft_deltas(recipient["founder"], matched_edges, shuffled_delta),
        }
        condition_scores = {name: evaluate_both(net, recipient["encoder"], validation, adversarial) for name, net in conditions.items()}
        pair_results.append({
            "donor_seed": donor_seed, "recipient_seed": recipient_seed,
            "core_edges": int(len(donor_core)), "mapped_recipient_edges_unique": int(len(set(matched_edges.tolist()))),
            "dale_sign_compatible": True, "mean_aligned_edge_cosine": float(np.mean(match_similarity)),
            "recipient_virgin": recipient["founder_scores"], "recipient_mature": recipient["mature_scores"],
            "conditions": condition_scores,
            "recovery_fraction_relative_to_virgin_to_mature": {
                name: recovery(scores, recipient["founder_scores"], recipient["mature_scores"]) for name, scores in condition_scores.items()
            },
        })

    permutation_control = run_gate(1842, neurons, training_steps, CORE_FRACTION)
    if not permutation_control["gate_passed"]:
        raise RuntimeError("permutation-isomorph positive control failed during Stage D")
    return {
        "experiment": "013_cross_topology_functional_core_convergence", "stage": "D_preregistered_cross_topology_delta_graft", "issue": 15,
        "terminal_battery_touched": False, "raw_recurrent_edge_indices_compared_across_topologies": False,
        "seeds": list(seeds), "cyclic_pairs": [list(pair) for pair in CYCLIC_PAIRS], "neurons": neurons, "core_fraction": CORE_FRACTION,
        "training_neural_steps_per_topology": training_steps,
        "independent_topology_training_neural_steps": training_steps * len(seeds),
        "permutation_control_training_neural_steps": training_steps,
        "total_training_neural_steps": training_steps * (len(seeds) + 1),
        "plasticity_multiplier": 16.0, "evaluation": {"settle_steps": 40, "probe_steps": 60},
        "alignment_inputs": "unlabeled public training situations only",
        "alignment_metric": "cosine of concatenated mature post/pre neuron response fingerprints with hard Dale-sign compatibility",
        "edge_assignment": "unique maximum-total-cosine assignment within each Dale class using scipy linear_sum_assignment",
        "graft_rule": "donor learned recurrent deltas added to virgin recipient founder edges, then existing Dale-sign/max_abs_weight clipping",
        "controls": ["recipient_virgin", "recipient_mature", "within_topology_core", "random_edge_sign_matched", "shuffled_delta_same_aligned_edges", "permutation_isomorph"],
        "permutation_isomorph_control": permutation_control, "pair_results": pair_results,
    }


def main():
    p = argparse.ArgumentParser(description="Experiment 013 Stage D preregistered cross-topology delta graft")
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--training-neural-steps", type=int, default=24000)
    args = p.parse_args()
    report = run(args.neurons, args.training_neural_steps)
    outdir = ROOT / "results" / "experiment_013_cross_topology"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"stage_d_n{args.neurons}_steps{args.training_neural_steps}.json"
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
