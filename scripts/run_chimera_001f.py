#!/usr/bin/env python3
"""001F: unsupervised autobiographical replay with frozen, order and no-replay controls.

Reconstructed memories are unsupervised *textual experience*, not motor labels.
The original v1 40/20 evaluation sources are previously exposed diagnostics.
No terminal file, 001E draft, model-generated action truth or Hugging Face data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from persona_net.phenotype_training import PhenotypeCurriculum
from scripts.run_chimera_001 import atomic_json, fingerprint, score, sha256
from scripts.run_chimera_001b import load_data
from scripts.run_chimera_001d import static_hash


def source_preflight(config: dict, historical_v04: Path, connectome: Path) -> tuple[dict,list[dict],dict]:
    """Refuse source drift before beginning neural computation."""
    info=config["source_memory"]
    path=connectome/info["path"]
    digest=sha256(path)
    if digest != info["sha256"]:
        raise ValueError(f"Unpinned memory archive hash: {digest}, expected {info['sha256']}")
    if info["commit"]!="5364f43dfe3c192b6c13b7bf373405ee7a41b420":
        raise ValueError("Memory source commit mismatch")
    events=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(events)!=info["events"] or len(events)!=450:
        raise ValueError("Expected 450 autobiographical memories")
    orders=[e.get("chronological_order") for e in events]
    if set(orders)!=set(range(1,451)) or len({e["event_id"] for e in events})!=450:
        raise ValueError("Missing or duplicate order/event identifiers")
    if Counter(e.get("provenance") for e in events)!={"reconstructed":450}:
        raise ValueError("Source provenance changed: all currently classified reconstructed")
    fields=info["fields"]
    for row in events:
        if any(not isinstance(row.get(field),str) for field in fields):
            raise ValueError(f"Nontext memory field: {row['event_id']}")
        if "target_actions" in row:
            raise ValueError("Biography must not provide a scored motor target")
    events=sorted(events,key=lambda row:row["chronological_order"])
    old=dict(config["historical_training"])
    old.update(validation_file=config["validation_file"],
               validation_sha256=config["validation_sha256"],
               adversarial_file=config["adversarial_file"],
               adversarial_sha256=config["adversarial_sha256"])
    splits, hashes=load_data(old,historical_v04,ROOT)
    hashes[info["path"]]=digest
    return splits,events,hashes


def epoch_indices(condition: str, n: int, epoch: int, seed: int, offset: int) -> list[int]:
    if condition in ("plastic_chronological","frozen_chronological"):
        return list(range(n))
    if condition=="plastic_reverse":
        return list(range(n-1,-1,-1))
    if condition=="plastic_shuffled_each_epoch":
        return list(np.random.default_rng(seed+offset+epoch*7919).permutation(n).astype(int))
    raise ValueError("Unknown memory replay condition")


def encode_memory_vectors(events: list[dict],encoder: ExperienceEncoder, fields: list[str]) -> list[np.ndarray]:
    vectors=[]
    for row in events:
        narrative="\n".join(row[field] for field in fields if row[field].strip())
        if not narrative.strip():
            raise ValueError("Empty source narrative")
        # Neither efference-copy actions nor outcome/reward labels are provided.
        vec=encoder.encode(narrative,scalars={},action=None,outcome=0.0).vector
        assert not np.any(vec[encoder.action_offset:])
        vectors.append(vec)
    return vectors


def replay(net: PlasticRecurrentPersonaNet, vectors: list[np.ndarray],
           *, condition: str, epochs: int, seed: int, offset: int) -> dict:
    before_w=static_hash(net,"recurrent_W")
    before_bias=static_hash(net,"homeostatic_bias")
    before_motor=fingerprint(net,"decoder")
    w0=net.W.data.copy()
    bias0=net.bias.copy()
    net.reset_fast_state(noise=0.0)
    tick0=net.tick
    learn=condition!="frozen_chronological"
    for epoch in range(epochs):
        for i in epoch_indices(condition,len(vectors),epoch,seed,offset):
            net.step(vectors[i],reward=0.0,learn=learn)
    if net.tick-tick0 != len(vectors)*epochs:
        raise AssertionError("Incorrect autobiographical neural exposure budget")
    after_motor=fingerprint(net,"decoder")
    if after_motor!=before_motor:
        raise AssertionError("Unsupervised autobiography modified trained motor decoder")
    if not learn and (static_hash(net,"recurrent_W")!=before_w
                      or static_hash(net,"homeostatic_bias")!=before_bias):
        raise AssertionError("Frozen replay mutated synapses or bias")
    return {
        "condition":condition,
        "memory_presentations":len(vectors)*epochs,
        "full_epochs":epochs,
        "unsupervised_reward":0.0,
        "motor_supervision_present":False,
        "learn_recurrent":learn,
        "recurrent_W_l2_change":float(np.linalg.norm(net.W.data-w0)),
        "homeostatic_bias_l2_change":float(np.linalg.norm(net.bias-bias0)),
        "W_before_sha256":before_w,
        "W_after_sha256":static_hash(net,"recurrent_W"),
        "bias_before_sha256":before_bias,
        "bias_after_sha256":static_hash(net,"homeostatic_bias"),
        "motor_decoder_sha256_unchanged":before_motor,
        "actual_neural_steps":net.tick-tick0,
    }


def one_seed(config: dict,sets: dict,events: list[dict],source_hashes: dict,seed: int) -> dict:
    enc=ExperienceEncoder(config["network"]["sensory_dim"])
    ncfg=dict(config["network"],seed=int(seed))
    virgin=PlasticRecurrentPersonaNet(ncfg,enc)
    mature=copy.deepcopy(virgin)
    phenotype=PhenotypeCurriculum(sets["train"],enc).run(
        mature,total_ticks=config["phenotype_training_ticks"],progress_every=0)
    if phenotype.ticks!=config["phenotype_training_ticks"]:
        raise AssertionError("Phenotype foundation changed")
    vectors=encode_memory_vectors(events,enc,config["source_memory"]["fields"])
    if len(vectors)!=450:
        raise AssertionError("Wrong memory count")
    if config["replay_ticks"]!=len(vectors)*config["replay_full_epochs"]:
        raise AssertionError("Memory replay budget not a whole-number epoch count")
    out={}
    for condition in config["conditions"]:
        net=copy.deepcopy(mature)
        detail=(None if condition=="no_replay" else replay(
            net,vectors,condition=condition,epochs=config["replay_full_epochs"],
            seed=seed,offset=config["neural_replay_policy"]["randomization_seed_offset"]))
        out[condition]={
            "replay":detail,
            "recurrent_W_sha256":static_hash(net,"recurrent_W"),
            "homeostatic_bias_sha256":static_hash(net,"homeostatic_bias"),
            "motor_decoder_sha256":fingerprint(net,"decoder"),
            "validation":score(net,enc,sets["validation"],config,seed,0),
            "adversarial":score(net,enc,sets["adversarial"],config,seed,1),
        }
    origin=out["no_replay"]
    frozen=out["frozen_chronological"]
    for split in ("validation","adversarial"):
        if origin[split]!=frozen[split]:
            raise AssertionError("Frozen-exposure and no-replay diagnostics disagree")
    for cond in config["conditions"]:
        if out[cond]["motor_decoder_sha256"]!=origin["motor_decoder_sha256"]:
            raise AssertionError("Autobiography secretly retrained motor decoder")
    paired={}
    for split in ("validation","adversarial"):
        baseline=frozen[split]["js_similarity"]
        paired[split]={cond:float(out[cond][split]["js_similarity"]-baseline)
                       for cond in config["conditions"] if cond not in ("no_replay","frozen_chronological")}
        paired[split]["chronological_minus_reverse"]=float(
            out["plastic_chronological"][split]["js_similarity"]
            -out["plastic_reverse"][split]["js_similarity"])
        paired[split]["chronological_minus_shuffled"]=float(
            out["plastic_chronological"][split]["js_similarity"]
            -out["plastic_shuffled_each_epoch"][split]["js_similarity"])
    return {
        "seed":int(seed),
        "source_hashes":source_hashes,
        "initial_phenotype_training_ticks":phenotype.ticks,
        "initial_phenotype_presentations":phenotype.presentations,
        "conditions":out,"paired_delta_vs_frozen":paired,
        "terminal_battery_touched":False,
        "historical_001_reproduced":False,
        "new_behavioral_labels_used":False,
        "source_memory_is_reconstructed":True,
    }


def summarize(rows: list[dict],cfg: dict,hashes: dict,meta: dict) -> dict:
    result={
        **meta,
        "seeds":[x["seed"] for x in rows],
        "seed_count":len(rows),
        "source_hashes":hashes,
        "diagnostics":{},
        "paired_deltas":{},
        "claim_status":"autobiographical_replay_exploratory_not_generalization",
        "terminal_battery_touched":False,
        "historical_001_reproduced":False,
        "new_behavioral_labels_used":False,
    }
    for condition in cfg["conditions"]:
        result["diagnostics"][condition]={}
        for split in ("validation","adversarial"):
            vals=[r["conditions"][condition][split]["js_similarity"] for r in rows]
            result["diagnostics"][condition][split]={
                "per_seed":vals,"mean":float(np.mean(vals)),
                "top1_mean":float(np.mean([r["conditions"][condition][split]["top1_agreement"]
                                          for r in rows])),
            }
        if condition!="no_replay":
            result["diagnostics"][condition]["replay_w_l2_mean"]=float(np.mean([
                r["conditions"][condition]["replay"]["recurrent_W_l2_change"] for r in rows]))
            result["diagnostics"][condition]["replay_bias_l2_mean"]=float(np.mean([
                r["conditions"][condition]["replay"]["homeostatic_bias_l2_change"] for r in rows]))
    for split in ("validation","adversarial"):
        result["paired_deltas"][split]={}
        for name in rows[0]["paired_delta_vs_frozen"][split]:
            vals=[r["paired_delta_vs_frozen"][split][name] for r in rows]
            result["paired_deltas"][split][name]={"per_seed":vals,"mean":float(np.mean(vals))}
    return result


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"config/chimera_001f.json")
    p.add_argument("--historical-v04",type=Path,required=True)
    p.add_argument("--connectome",type=Path,required=True)
    p.add_argument("--out",type=Path,default=ROOT/"results/chimera_001f")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--seeds",type=int,nargs="+")
    p.add_argument("--phenotype-ticks",type=int)
    p.add_argument("--replay-epochs",type=int)
    p.add_argument("--probe-ticks",type=int)
    args=p.parse_args()
    raw=args.config.read_bytes()
    cfg=json.loads(raw)
    if cfg["protocol_id"]!="chimera-001f-autobiographical-unsupervised-replay-v1":
        raise ValueError("Invalid protocol")
    splits,events,hashes=source_preflight(cfg,args.historical_v04,args.connectome)
    if args.preflight_only:
        print("001F verified both original diagnostic sources, 100 v0.4 phenotype cards, 450 reconstructed memories")
        return 0
    seeds=cfg["seeds"] if args.seeds is None else args.seeds
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Empty or duplicate seed list")
    if args.phenotype_ticks is not None:
        if args.phenotype_ticks<=0 or args.phenotype_ticks%2:
            raise ValueError("Invalid smoke phenotype budget")
        cfg["phenotype_training_ticks"]=args.phenotype_ticks
    if args.replay_epochs is not None:
        if args.replay_epochs<=0:
            raise ValueError("Invalid memory replay epochs")
        cfg["replay_full_epochs"]=args.replay_epochs
        cfg["replay_ticks"]=450*args.replay_epochs
    if args.probe_ticks is not None:
        if args.probe_ticks<=0:
            raise ValueError("Invalid smoke evaluation ticks")
        cfg["eval"]["settle_ticks"]=cfg["eval"]["probe_ticks"]=args.probe_ticks
    full=(seeds==cfg["seeds"] and args.phenotype_ticks is None
          and args.replay_epochs is None and args.probe_ticks is None)
    meta={
        "protocol_id":cfg["protocol_id"],"config_sha256":hashlib.sha256(raw).hexdigest(),
        "run_mode":"full_six_seed_exploratory" if full else "smoke_not_full_evidence",
        "memory_source_role":"reconstructed_unlabeled_unsupervised_exposure",
        "evaluation_roles":cfg["eval_roles"],
        "terminal_battery_touched":False,
        "historical_001_reproduced":False,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out/"RUN_METADATA.json",dict(meta,source_hashes=hashes))
    rows=[]
    for seed in seeds:
        row=one_seed(cfg,splits,events,hashes,seed)
        atomic_json(args.out/f"seed_{seed}.json",dict(row,**meta))
        rows.append(row)
        print(f"001F {meta['run_mode']} seed {seed}: delta chronology vs frozen "
              f"{row['paired_delta_vs_frozen']['validation']['plastic_chronological']:+.6f}, "
              f"vs shuffled {row['paired_delta_vs_frozen']['validation']['chronological_minus_shuffled']:+.6f}",
              flush=True)
    atomic_json(args.out/"RUN_REPORT.json",summarize(rows,cfg,hashes,meta))
    print("001F complete. Previously exposed evaluation, no new labels, no terminal.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
