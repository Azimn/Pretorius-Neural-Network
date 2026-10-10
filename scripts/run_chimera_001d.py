#!/usr/bin/env python3
"""001D controlled W/bias and motor weight/bias factorial on previously exposed data.

No historic 001 reproduction claim, new labels, original terminal, external data,
or parameter optimization. Inputs and neural design are reused from completed 001B.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from persona_net.phenotype_training import PhenotypeCurriculum
from scripts.run_chimera_001 import atomic_json, fingerprint, score
from scripts.run_chimera_001b import load_data, source_priors

REC_CONDS = {
    "r00_virgin_W_virgin_bias_trained_decoder": (False, False),
    "r10_trained_W_virgin_bias_trained_decoder": (True, False),
    "r01_virgin_W_trained_bias_trained_decoder": (False, True),
    "r11_trained_W_trained_bias_trained_decoder": (True, True),
}
MOTOR_CONDS = {
    "d00_virgin_motorW_virgin_motorB_trained_recurrence": (False, False),
    "d10_trained_motorW_virgin_motorB_trained_recurrence": (True, False),
    "d01_virgin_motorW_trained_motorB_trained_recurrence": (False, True),
    "d11_trained_motorW_trained_motorB_trained_recurrence": (True, True),
}


def static_hash(net, element: str) -> str:
    a = {
        "recurrent_W": (net.W.data, net.W.indices, net.W.indptr),
        "homeostatic_bias": (net.bias,),
        "motor_w": (net.motor_w,),
        "motor_b": (net.motor_b,),
    }[element]
    h = hashlib.sha256()
    for block in a:
        arr = np.asarray(block)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def factorial_graft(
    virgin: PlasticRecurrentPersonaNet,
    mature: PlasticRecurrentPersonaNet,
    *,
    trained_W: bool,
    trained_bias: bool,
    trained_motor_w: bool,
    trained_motor_b: bool,
) -> PlasticRecurrentPersonaNet:
    """Exactly isolate W, recurrent bias, motor weights, and motor biases.

    Topology and inputs are identical, all transplants are deep copies,
    fast states are virgin and each evaluation resets them.
    Eligibility remains virgin and is inert under learn=False evaluation.
    """
    if not np.array_equal(virgin.W.indices, mature.W.indices):
        raise AssertionError("Cannot graft across nonidentical recurrent topology")
    if not np.array_equal(virgin.W.indptr, mature.W.indptr):
        raise AssertionError("Cannot graft across nonidentical recurrent topology")
    if not np.array_equal(virgin.Win.data, mature.Win.data):
        raise AssertionError("Cannot graft mismatching input projections")
    out = copy.deepcopy(virgin)
    out.W.data[:] = mature.W.data if trained_W else virgin.W.data
    out.bias[:] = mature.bias if trained_bias else virgin.bias
    out.motor_w[:] = mature.motor_w if trained_motor_w else virgin.motor_w
    out.motor_b[:] = mature.motor_b if trained_motor_b else virgin.motor_b
    for key, source in (
        ("recurrent_W", mature if trained_W else virgin),
        ("homeostatic_bias", mature if trained_bias else virgin),
        ("motor_w", mature if trained_motor_w else virgin),
        ("motor_b", mature if trained_motor_b else virgin),
    ):
        assert static_hash(out, key) == static_hash(source, key)
    if np.shares_memory(out.W.data, mature.W.data):
        raise AssertionError("Recurrent graft aliased source network")
    if np.shares_memory(out.motor_w, mature.motor_w):
        raise AssertionError("Decoder graft aliased source network")
    return out


def permute_context(items: list[dict], seed: int, *, within_domain: bool) -> list[dict]:
    """Only scenario and environment scalars are reassigned; labels never move."""
    rng = np.random.default_rng(seed + 876523)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        domain = str(item.get("domain")) if within_domain else "__global__"
        groups[domain].append(i)
    source_idx = list(range(len(items)))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        # Derangement: a randomly rotated list of the original indices.
        offset = int(rng.integers(1, len(indices)))
        replacement = indices[offset:] + indices[:offset]
        for destination, src in zip(indices, replacement):
            source_idx[destination] = src
    result = []
    for i, item in enumerate(items):
        donor = items[source_idx[i]]
        updated = dict(item, scenario=donor["scenario"])
        updated["scalars"] = dict(donor.get("scalars", {}))
        result.append(updated)
    if [x["id"] for x in items] != [x["id"] for x in result]:
        raise AssertionError("No evaluation item ID may move")
    if [x["target_actions"] for x in items] != [x["target_actions"] for x in result]:
        raise AssertionError("No evaluation target may move")
    return result


def measure_one(seed: int, config: dict, sets: dict, hashes: dict) -> dict:
    ncfg = dict(config["network"], seed=seed)
    encoder = ExperienceEncoder(ncfg["sensory_dim"])
    virgin = PlasticRecurrentPersonaNet(ncfg, encoder)
    mature = copy.deepcopy(virgin)
    tr = PhenotypeCurriculum(sets["train"], encoder).run(
        mature, total_ticks=config["phenotype_training_ticks"], progress_every=0)
    if tr.ticks != config["phenotype_training_ticks"]:
        raise AssertionError("Training step count was not maintained")
    components = {
        name: {
            "virgin_sha256": static_hash(virgin, name),
            "trained_sha256": static_hash(mature, name),
        }
        for name in ("recurrent_W", "homeostatic_bias", "motor_w", "motor_b")
    }
    output = {}
    for label, (use_w, use_bias) in REC_CONDS.items():
        net = factorial_graft(virgin, mature, trained_W=use_w,
                              trained_bias=use_bias,
                              trained_motor_w=True, trained_motor_b=True)
        output[label] = {
            split: score(net, encoder, sets[split], config, seed, offset)
            for offset, split in enumerate(("validation", "adversarial"))
        }
    for label, (use_w, use_bias) in MOTOR_CONDS.items():
        net = factorial_graft(virgin, mature, trained_W=True,
                              trained_bias=True,
                              trained_motor_w=use_w, trained_motor_b=use_bias)
        output[label] = {
            split: score(net, encoder, sets[split], config, seed, offset)
            for offset, split in enumerate(("validation", "adversarial"))
        }
    paired_reference = output["r11_trained_W_trained_bias_trained_decoder"]
    duplicate_reference = output["d11_trained_motorW_trained_motorB_trained_recurrence"]
    for split in ("validation", "adversarial"):
        if paired_reference[split] != duplicate_reference[split]:
            raise AssertionError(f"Factorial intersection mismatch: {split}")
    # Context counterfactuals are pure evaluation probes on the intact model.
    intact = factorial_graft(virgin, mature, trained_W=True,
                             trained_bias=True, trained_motor_w=True,
                             trained_motor_b=True)
    context = {}
    for offset, split in enumerate(("validation", "adversarial")):
        changed_global = permute_context(sets[split], seed + offset * 1000,
                                         within_domain=False)
        context[split] = {
            "intact": paired_reference[split],
            "shuffled_context_global": score(
                intact, encoder, changed_global, config, seed, offset),
            "shuffled_context_within_domain": None,
        }
        if split == "validation":
            changed_domain = permute_context(sets[split], seed,
                                              within_domain=True)
            context[split]["shuffled_context_within_domain"] = score(
                intact, encoder, changed_domain, config, seed, offset)
    rec00 = output["r00_virgin_W_virgin_bias_trained_decoder"]
    rec10 = output["r10_trained_W_virgin_bias_trained_decoder"]
    rec01 = output["r01_virgin_W_trained_bias_trained_decoder"]
    rec11 = paired_reference
    d10 = output["d10_trained_motorW_virgin_motorB_trained_recurrence"]
    d01 = output["d01_virgin_motorW_trained_motorB_trained_recurrence"]
    deltas = {}
    for split in ("validation", "adversarial"):
        s = lambda x: float(x[split]["js_similarity"])
        deltas[split] = {
            "W_only_vs_virgin_W": s(rec10) - s(rec00),
            "bias_only_vs_virgin_bias": s(rec01) - s(rec00),
            "W_effect_at_trained_bias": s(rec11) - s(rec01),
            "bias_effect_at_trained_W": s(rec11) - s(rec10),
            "trained_motorW_only_minus_trained_motorB_only": s(d10) - s(d01),
            "intact_minus_context_global_shuffled": (
                s(rec11) - float(context[split]["shuffled_context_global"]["js_similarity"])),
        }
        if split == "validation":
            deltas[split]["intact_minus_context_within_domain_shuffled"] = (
                s(rec11) - float(context[split]["shuffled_context_within_domain"]["js_similarity"]))
    return {
        "seed":seed,"neural_training_steps":tr.ticks,"source_sha256":hashes,
        "component_hashes":components,
        "component_l2_shift": {
            "recurrent_W":float(np.linalg.norm(mature.W.data-virgin.W.data)),
            "homeostatic_bias":float(np.linalg.norm(mature.bias-virgin.bias)),
            "motor_w":float(np.linalg.norm(mature.motor_w-virgin.motor_w)),
            "motor_b":float(np.linalg.norm(mature.motor_b-virgin.motor_b)),
        },
        "conditions":output,"context_counterfactuals":context,
        "paired_deltas":deltas,
        "terminal_battery_touched":False,"historical_001_reproduced":False,
    }


def summarize(rows: list[dict], config: dict, provenance: dict) -> dict:
    result = {
        "protocol_id":config["protocol_id"],
        "seeds":[x["seed"] for x in rows],
        "seed_count":len(rows),
        "source_sha256":provenance,
        "evaluation_roles":config["evaluation_roles"],
        "conditions":{},"paired_deltas":{},"component_l2_shifts":{},
        "terminal_battery_touched":False,"historical_001_reproduced":False,
        "claim":"component ablation on previously exposed evaluation fixtures only",
    }
    for condition in (*REC_CONDS, *MOTOR_CONDS):
        result["conditions"][condition] = {}
        for split in ("validation", "adversarial"):
            vals=[x["conditions"][condition][split]["js_similarity"] for x in rows]
            result["conditions"][condition][split] = {
                "per_seed_js_similarity": vals,
                "mean_js_similarity":float(np.mean(vals)),
                "mean_top1_agreement":float(np.mean(
                    [x["conditions"][condition][split]["top1_agreement"] for x in rows])),
            }
    for split in ("validation", "adversarial"):
        result["paired_deltas"][split]={}
        for key in rows[0]["paired_deltas"][split]:
            vals=[x["paired_deltas"][split][key] for x in rows]
            result["paired_deltas"][split][key]={
                "per_seed": vals,"mean":float(np.mean(vals))
            }
    for name in rows[0]["component_l2_shift"]:
        vals=[x["component_l2_shift"][name] for x in rows]
        result["component_l2_shifts"][name]={"per_seed":vals,"mean":float(np.mean(vals))}
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/chimera_001d.json")
    parser.add_argument("--historical-v04",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=ROOT/"results/chimera_001d")
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--seeds",type=int,nargs="+")
    parser.add_argument("--training-ticks",type=int)
    parser.add_argument("--probe-ticks",type=int)
    args=parser.parse_args()
    raw=args.config.read_bytes()
    cfg=json.loads(raw)
    if cfg["protocol_id"]!="chimera-001d-isolated-synapse-bias-decoder-components-v1":
        raise ValueError("Unexpected 001D config")
    if cfg["source_v04_commit"]!="7b7eb19ad852dc016ee0d370528d903bab78a5cf":
        raise ValueError("V0.4 source version was changed")
    data,hashes=load_data(cfg,args.historical_v04,ROOT)
    if args.preflight_only:
        print("001D source-integrity preflight passed; no training performed")
        return 0
    seeds=cfg["seeds"] if args.seeds is None else args.seeds
    if len(set(seeds))!=len(seeds):
        raise ValueError("Duplicate seeds")
    if args.training_ticks is not None:
        if args.training_ticks<=0 or args.training_ticks%2:
            raise ValueError("Training ticks must be positive and even")
        cfg["phenotype_training_ticks"]=args.training_ticks
    if args.probe_ticks is not None:
        if args.probe_ticks<=0:
            raise ValueError("Probe ticks must be positive")
        cfg["eval"]["probe_ticks"]=args.probe_ticks
        cfg["eval"]["settle_ticks"]=args.probe_ticks
    full=(seeds==cfg["seeds"] and args.training_ticks is None
          and args.probe_ticks is None)
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    mode="full_six_seed_exploratory" if full else "smoke_not_full_evidence"
    metadata={
        "protocol_id":cfg["protocol_id"],
        "config_sha256":hashlib.sha256(raw).hexdigest(),
        "mode":mode,"source_sha256":hashes,
        "evaluation_roles":cfg["evaluation_roles"],
        "terminal_battery_touched":False,"historical_001_reproduced":False,
    }
    atomic_json(out/"RUN_METADATA.json",metadata)
    rows=[]
    for seed in seeds:
        row=measure_one(int(seed),cfg,data,hashes)
        row.update({"mode":mode,"config_sha256":metadata["config_sha256"]})
        atomic_json(out/f"seed_{seed}.json",row)
        rows.append(row)
        print(f"001D {mode} seed {seed}: W-only validation effect "
              f"{row['paired_deltas']['validation']['W_only_vs_virgin_W']:+.7f}, "
              f"intact context gap {row['paired_deltas']['validation']['intact_minus_context_global_shuffled']:+.7f}",
              flush=True)
    summary=summarize(rows,cfg,hashes)
    summary.update(metadata)
    summary["non_neural_priors"]={
        name:source_priors(data["train"],data[name]) for name in ("validation","adversarial")
    }
    atomic_json(out/"RUN_REPORT.json",summary)
    print("001D completed; source-exposed diagnostic evaluation only")
    return 0


if __name__=="__main__":
    sys.exit(main())
