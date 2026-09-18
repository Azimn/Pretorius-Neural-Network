from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.v04_data import load_v04_split
from experiment_010_minimal_transferable_core import compile_items, evaluate_both, train_recurrent_only
from experiment_012_core_convergence import configured

ROOT = Path(__file__).resolve().parents[1]


def permute_network(net, permutation):
    """Return an exactly neuron-permuted copy. permutation[old] = new."""
    p = np.asarray(permutation, dtype=np.int64)
    if sorted(p.tolist()) != list(range(net.n)):
        raise ValueError("permutation must be a bijection")
    inv = np.argsort(p)
    out = copy.deepcopy(net)
    out.W = net.W[inv, :][:, inv].tocsr()
    out.Win = net.Win[inv, :].tocsr()
    out.excitatory = net.excitatory[inv].copy()
    out.bias = net.bias[inv].copy()
    out.v = net.v[inv].copy()
    out.rate = net.rate[inv].copy()
    out.motor_w = net.motor_w[:, inv].copy()
    out.action_populations = {a: np.sort(p[idx]).astype(np.int32) for a, idx in net.action_populations.items()}
    out.post_idx = np.repeat(np.arange(out.n, dtype=np.int32), np.diff(out.W.indptr))
    out.pre_idx = out.W.indices.astype(np.int32, copy=False)
    out.eligibility = np.zeros_like(out.W.data, dtype=np.float32)
    return out


def fingerprints(net, compiled, probe_steps=8):
    """Unlabeled public-training response signatures. No targets are used by alignment."""
    responses = np.empty((net.n, len(compiled)), dtype=np.float32)
    for j, (context, _teaching) in enumerate(compiled):
        net.reset_fast_state(noise=0.0)
        for _ in range(probe_steps):
            net.step(context, reward=0.0, learn=False)
        responses[:, j] = net.rate
    responses -= responses.mean(axis=0, keepdims=True)
    scale = responses.std(axis=0, keepdims=True)
    responses /= np.where(scale > 1e-8, scale, 1.0)
    norms = np.linalg.norm(responses, axis=1, keepdims=True)
    return responses / np.where(norms > 1e-8, norms, 1.0)


def align_neurons(reference, candidate, compiled, probe_steps=8):
    """Hungarian maximum-cosine alignment with hard Dale E/I compatibility."""
    a = fingerprints(copy.deepcopy(reference), compiled, probe_steps=probe_steps)
    b = fingerprints(copy.deepcopy(candidate), compiled, probe_steps=probe_steps)
    similarity = a @ b.T
    incompatible = reference.excitatory[:, None] != candidate.excitatory[None, :]
    cost = -similarity.astype(np.float64)
    cost[incompatible] = 1e6
    rows, cols = linear_sum_assignment(cost)
    mapping = np.empty(reference.n, dtype=np.int64)
    mapping[rows] = cols
    return mapping, float(np.mean(similarity[rows, cols]))


def edge_index(net):
    return {(int(post), int(pre)): i for i, (post, pre) in enumerate(zip(net.post_idx, net.pre_idx))}


def mapped_core_correspondence(founder, mature, mapped_founder, mapped_mature, mapping, fraction=0.01):
    delta = np.abs(mature.W.data - founder.W.data)
    count = max(1, int(round(fraction * len(delta))))
    core = np.argsort(delta)[::-1][:count]
    mapped_delta = np.abs(mapped_mature.W.data - mapped_founder.W.data)
    mapped_core = set(np.argsort(mapped_delta)[::-1][:count].tolist())
    target_edges = edge_index(mapped_founder)
    hits = 0
    for idx in core:
        key = (int(mapping[founder.post_idx[idx]]), int(mapping[founder.pre_idx[idx]]))
        j = target_edges.get(key)
        hits += int(j in mapped_core)
    return hits / count, core


def graft_mapped_deltas(founder, mature, recipient_founder, mapping, core):
    out = copy.deepcopy(recipient_founder)
    target_edges = edge_index(out)
    applied = 0
    for idx in core:
        post = int(mapping[founder.post_idx[idx]])
        pre = int(mapping[founder.pre_idx[idx]])
        j = target_edges.get((post, pre))
        if j is None:
            continue
        delta = float(mature.W.data[idx] - founder.W.data[idx])
        value = float(out.W.data[j]) + delta
        if out.excitatory[out.pre_idx[j]]:
            value = np.clip(value, 0.0, out.max_abs_weight)
        else:
            value = np.clip(value, -out.max_abs_weight, 0.0)
        out.W.data[j] = value
        applied += 1
    out.eligibility.fill(0.0)
    out.reset_fast_state()
    return out, applied


def run_gate(seed, neurons, training_steps, fraction):
    cfg = configured(seed, neurons)
    encoder = ExperienceEncoder(cfg["sensory_dim"])
    founder = PlasticRecurrentPersonaNet(cfg, encoder)
    mature = PlasticRecurrentPersonaNet(cfg, encoder)
    data_root = ROOT / "data" / "v0_4"
    training = load_v04_split(data_root, "train")
    validation = load_v04_split(data_root, "validation")
    adversarial = load_v04_split(data_root, "adversarial")
    compiled = compile_items(encoder, training)
    actual_steps = train_recurrent_only(mature, compiled, training_steps)

    rng = np.random.default_rng(seed + 13013)
    hidden = rng.permutation(neurons)
    perm_founder = permute_network(founder, hidden)
    perm_mature = permute_network(mature, hidden)
    mapping, mean_cosine = align_neurons(mature, perm_mature, compiled)
    neuron_accuracy = float(np.mean(mapping == hidden))
    correspondence, core = mapped_core_correspondence(founder, mature, perm_founder, perm_mature, mapping, fraction)
    grafted, applied = graft_mapped_deltas(founder, mature, perm_founder, mapping, core)

    mature_scores = evaluate_both(mature, encoder, validation, adversarial)
    graft_scores = evaluate_both(grafted, encoder, validation, adversarial)
    val_gap = abs(mature_scores["validation"]["top1_agreement"] - graft_scores["validation"]["top1_agreement"])
    adv_gap = abs(mature_scores["adversarial"]["top1_agreement"] - graft_scores["adversarial"]["top1_agreement"])
    passed = correspondence >= 0.90 and val_gap <= 0.01 and adv_gap <= 0.01
    return {
        "seed": seed,
        "actual_training_neural_steps": actual_steps,
        "training_items_for_alignment": len(training),
        "alignment": {"metric": "standardized training-response cosine + hard E/I; Hungarian assignment", "mean_cosine": mean_cosine, "hidden_permutation_neuron_accuracy": neuron_accuracy},
        "core_fraction": fraction,
        "mapped_core_correspondence": correspondence,
        "core_edges": len(core),
        "mapped_graft_edges_applied": applied,
        "mature": mature_scores,
        "mapped_graft": graft_scores,
        "top1_absolute_gaps": {"validation": val_gap, "adversarial": adv_gap},
        "gate_passed": passed,
    }


def main():
    p = argparse.ArgumentParser(description="Experiment 013 Stage B permutation-isomorph design gate")
    p.add_argument("--seed", type=int, default=1842)
    p.add_argument("--neurons", type=int, default=1024)
    p.add_argument("--training-neural-steps", type=int, default=24000)
    p.add_argument("--core-fraction", type=float, default=0.01)
    args = p.parse_args()
    result = run_gate(args.seed, args.neurons, args.training_neural_steps, args.core_fraction)
    report = {
        "experiment": "013_cross_topology_functional_core_convergence",
        "stage": "B_permutation_isomorph_gate",
        "issue": 15,
        "terminal_battery_touched": False,
        "independent_topology_interpretation_permitted": bool(result["gate_passed"]),
        "gate": {"mapped_core_correspondence_min": 0.90, "validation_top1_gap_max": 0.01, "adversarial_top1_gap_max": 0.01},
        "result": result,
    }
    outdir = ROOT / "results" / "experiment_013_cross_topology"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"stage_b_gate_n{args.neurons}_steps{args.training_neural_steps}.json"
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)
    if not result["gate_passed"]:
        raise SystemExit("Experiment 013 alignment gate failed; Stage C/D remain blocked.")


if __name__ == "__main__":
    main()
