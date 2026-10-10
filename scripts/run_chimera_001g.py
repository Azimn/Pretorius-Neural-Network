#!/usr/bin/env python3
"""001G: source-owned cue discrimination in Pretorius recurrent activity.

This measures cue-to-reconstructed-source correspondence, NOT autonomous memory,
verified semantic entailment, or action/persona generalization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from biocircuit.bc01 import load_corpus, retrieve
from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from persona_net.phenotype_training import PhenotypeCurriculum
from scripts.run_chimera_001 import atomic_json, fingerprint
from scripts.run_chimera_001d import static_hash
from scripts.run_chimera_001f import source_preflight, encode_memory_vectors, replay

CONDITIONS=(
    "baseline_frozen_replay",
    "autobiography_plastic_replay",
    "autobiography_W_lesion_holding_bias",
    "autobiography_bias_lesion_holding_W",
)
EVENT_ID=re.compile(r"\bE\d{2}-\d{3}\b",re.I)


def candidate_cues(events:list[dict], count:int)->list[str]:
    """Reuse reconstructed source-authored recall_cues, never look up by ID."""
    cues=[]
    for e in events:
        choices=e["recall_cues"]
        if (not isinstance(choices,list) or len(choices)<count or
            any(not isinstance(x,str) or not x.strip() for x in choices[:count])):
            raise ValueError("Insufficient source-owned recall cues")
        cue=" ".join(choices[:count])
        if EVENT_ID.search(cue) or e["event_id"].lower() in cue.lower():
            raise ValueError("Event ID leakage in inference cue")
        cues.append(cue)
    return cues


def deranged_within_episode(events:list[dict],seed:int)->list[int]:
    """Target reassignment: all cues, documents and episode marginals unchanged."""
    rng=np.random.default_rng(seed+77019)
    per_episode=defaultdict(list)
    for i,event in enumerate(events):
        per_episode[event["episode_id"]].append(i)
    indices=list(range(len(events)))
    for ids in per_episode.values():
        if len(ids)<2:
            raise ValueError("At least two source events per episode required")
        perm=rng.permutation(len(ids))
        # Rotate the randomly ordered positions for a guaranteed derangement
        for position,src in zip(perm,np.roll(perm,-1)):
            indices[ids[int(position)]]=ids[int(src)]
    if any(i==j for i,j in enumerate(indices)):
        raise AssertionError("Permuted assignment must change every target")
    return indices


def embed_inputs(vectors:list[np.ndarray],dim:int)->np.ndarray:
    if not vectors:
        raise ValueError("No memory or query encodings")
    return np.stack([x[:dim] for x in vectors]).astype(np.float32)


def activation_states(net:PlasticRecurrentPersonaNet,inputs:np.ndarray,
                      blank_ticks:int,stimulus_ticks:int)->np.ndarray:
    """Rate response contrasted against matched blank-state with learning disabled."""
    z=np.zeros(net.encoder.input_dim,dtype=np.float32)
    net.reset_fast_state(noise=0.0)
    for _ in range(blank_ticks):
        net.step(z,reward=0.0,learn=False)
    background=net.rate.copy()
    states=[]
    for x in inputs:
        net.reset_fast_state(noise=0.0)
        for _ in range(blank_ticks):
            net.step(z,reward=0.0,learn=False)
        accum=np.zeros(net.n,dtype=np.float64)
        for _ in range(stimulus_ticks):
            net.step(x,reward=0.0,learn=False)
            accum+=net.rate
        states.append((accum/stimulus_ticks-background).astype(np.float32))
    return np.stack(states)


def cosine_table(queries:np.ndarray,memories:np.ndarray)->np.ndarray:
    """Independent input-level baseline and non-supervised neural rate comparison."""
    if queries.ndim!=2 or memories.ndim!=2 or queries.shape[1]!=memories.shape[1]:
        raise ValueError("Query/document feature dimension mismatch")
    def unit(arr):
        norms=np.linalg.norm(arr.astype(np.float64),axis=1)
        return arr/np.maximum(norms[:,None],1e-12)
    q=unit(queries.astype(np.float64))
    m=unit(memories.astype(np.float64))
    result=q@m.T
    if not np.all(np.isfinite(result)):
        raise AssertionError("Nonfinite neural or lexical affinity")
    return result


def rank_assay(scores:np.ndarray,events:list[dict],targets:list[int]|None=None,
               *,retain_per_case:bool=True)->dict:
    n=len(events)
    if scores.shape!=(n,n):
        raise ValueError("Scores must form exact 450-cue x 450-source square matrix")
    truth=targets if targets is not None else list(range(n))
    if len(truth)!=n or sorted(truth)!=list(range(n)):
        raise ValueError("Target assignment must be full one-to-one permutation")
    episodes=[e["episode_id"] for e in events]
    groups=defaultdict(list)
    for i,ep in enumerate(episodes):
        groups[ep].append(i)
    top1=top5=ep_top1=0
    rr=ep_rr=0.
    cases=[]
    for i in range(n):
        ranked=np.argsort(-scores[i],kind="stable").tolist()
        target=truth[i]
        full_rank=ranked.index(target)+1
        same=[j for j in ranked if episodes[j]==episodes[target]]
        ep_rank=same.index(target)+1
        top1+=full_rank==1
        top5+=full_rank<=5
        ep_top1+=ep_rank==1
        rr+=1/full_rank
        ep_rr+=1/ep_rank
        if retain_per_case:
            cases.append({
                "cue_event_id":events[i]["event_id"],
                "expected_event_id":events[target]["event_id"],
                "retrieved_event_id":events[ranked[0]]["event_id"],
                "rank":full_rank,"episode_rank":ep_rank,
                "source_kind":"reconstructed_unreviewed_cue_association",
                "source_provenance":events[target]["provenance"],
            })
    return {
        "n":n,"top1":top1/n,"top5":top5/n,"mean_reciprocal_rank":rr/n,
        "same_episode_top1":ep_top1/n,"same_episode_mrr":ep_rr/n,
        "per_case":cases if retain_per_case else None,
    }


def one_seed(cfg:dict,source:tuple[list[dict],list[dict],dict],seed:int,
             *,smoke:bool=False)->dict:
    sets,events,hashes=source
    encoder=ExperienceEncoder(cfg["network"]["sensory_dim"])
    founder=PlasticRecurrentPersonaNet(dict(cfg["network"],seed=seed),encoder)
    phenotype=PhenotypeCurriculum(sets["train"],encoder).run(
        founder,total_ticks=cfg["phenotype_training_ticks"],progress_every=0)
    if phenotype.ticks!=cfg["phenotype_training_ticks"]:
        raise AssertionError("Wrong phenotype foundation budget")
    memory_vectors=encode_memory_vectors(events,encoder,cfg["source_memory"]["fields"])
    cues=candidate_cues(events,cfg["probe"]["cue_count_per_event"])
    query_vectors=[encoder.encode(t,scalars={},action=None,outcome=0).vector for t in cues]
    source_inputs=np.stack(memory_vectors)
    cue_inputs=np.stack(query_vectors)
    dim=encoder.sensory_dim
    lexical=cosine_table(cue_inputs[:,:dim],source_inputs[:,:dim])
    frozen=copy.deepcopy(founder)
    neutral=copy.deepcopy(founder)
    replay_meta_frozen=replay(
        frozen,memory_vectors,condition="frozen_chronological",
        epochs=cfg["replay_full_epochs"],seed=seed,offset=93017)
    replay_meta_plastic=replay(
        neutral,memory_vectors,condition="plastic_chronological",
        epochs=cfg["replay_full_epochs"],seed=seed,offset=93017)
    if replay_meta_frozen["actual_neural_steps"]!=replay_meta_plastic["actual_neural_steps"]:
        raise AssertionError("Exposure budgets differ")
    if fingerprint(neutral,"decoder")!=fingerprint(frozen,"decoder"):
        raise AssertionError("Memory replay altered motor decoder")
    lesion_W=copy.deepcopy(neutral)
    lesion_W.W.data[:]=frozen.W.data
    lesion_bias=copy.deepcopy(neutral)
    lesion_bias.bias[:]=frozen.bias
    if (static_hash(lesion_W,"recurrent_W")!=static_hash(frozen,"recurrent_W")
        or static_hash(lesion_W,"homeostatic_bias")!=static_hash(neutral,"homeostatic_bias")
        or static_hash(lesion_bias,"recurrent_W")!=static_hash(neutral,"recurrent_W")
        or static_hash(lesion_bias,"homeostatic_bias")!=static_hash(frozen,"homeostatic_bias")):
        raise AssertionError("Causal W/bias isolation failed")
    networks={
        "baseline_frozen_replay":frozen,
        "autobiography_plastic_replay":neutral,
        "autobiography_W_lesion_holding_bias":lesion_W,
        "autobiography_bias_lesion_holding_W":lesion_bias,
    }
    dims=cfg["probe"]
    output={}
    wrong=deranged_within_episode(events,seed)
    for name,net in networks.items():
        src=activation_states(net,source_inputs,dims["blank_ticks"],dims["stimulus_ticks"])
        cue=activation_states(net,cue_inputs,dims["blank_ticks"],dims["stimulus_ticks"])
        affinities=cosine_table(cue,src)
        output[name]={
            "correct":rank_assay(affinities,events),
            "permuted_target_assignment":rank_assay(affinities,events,wrong,retain_per_case=False),
            "W_sha256":static_hash(net,"recurrent_W"),
            "bias_sha256":static_hash(net,"homeostatic_bias"),
            "decoder_sha256":fingerprint(net,"decoder"),
        }
    equal_key=encoder.encode("neutral archival key",scalars={},action=None,outcome=0).vector
    z=activation_states(frozen,np.stack([equal_key]),dims["blank_ticks"],dims["stimulus_ticks"])
    src=activation_states(frozen,source_inputs,dims["blank_ticks"],dims["stimulus_ticks"])
    equal_scores=np.repeat(cosine_table(z,src),len(events),axis=0)
    lexical_report=rank_assay(lexical,events)
    lexical_wrong=rank_assay(lexical,events,wrong,retain_per_case=False)
    equal_report=rank_assay(equal_scores,events,retain_per_case=False)
    paired={}
    for key in ("top1","top5","mean_reciprocal_rank","same_episode_top1","same_episode_mrr"):
        vals=lambda name:output[name]["correct"][key]
        paired[key]={
            "plastic_minus_frozen":vals("autobiography_plastic_replay")-vals("baseline_frozen_replay"),
            "plastic_minus_W_lesion":vals("autobiography_plastic_replay")-vals("autobiography_W_lesion_holding_bias"),
            "plastic_minus_bias_lesion":vals("autobiography_plastic_replay")-vals("autobiography_bias_lesion_holding_W"),
            "plastic_minus_raw_lexical":vals("autobiography_plastic_replay")-lexical_report[key],
        }
    return {
        "seed":seed,"source_hashes":hashes,"source_episodes":len(set(x["episode_id"] for x in events)),
        "phenotype_training_steps":phenotype.ticks,
        "replay_steps_each":replay_meta_plastic["actual_neural_steps"],
        "replay_frozen":replay_meta_frozen,"replay_plastic":replay_meta_plastic,
        "conditions":output,"raw_lexical_input_baseline":lexical_report,
        "raw_lexical_permuted":lexical_wrong,"neutral_equal_key":equal_report,
        "paired_deltas":paired,"source_authority":"source_editorial_recall_cues_not_independent_labels",
        "terminal_battery_touched":False,"historical_001_reproduced":False,
        "independent_semantic_recall_claim":False,
    }


def summarize(rows:list[dict],cfg:dict,meta:dict,lexical_external:dict)->dict:
    out={
        **meta,"seeds":[x["seed"] for x in rows],"seed_count":len(rows),
        "source_kind":"reconstructed_editorially_associated_cues_not_independent_validation",
        "baseline_lexical_existing_bc01":lexical_external,
        "conditions":{},"paired_effects":{},
        "interpretation_gate":"Neural improvement must outperform frozen, W lesion, and raw lexical in paired seeds; this still does not demonstrate unaided recall.",
        "historical_001_reproduced":False,"terminal_battery_touched":False,
        "independent_semantic_recall_claim":False,
    }
    keys=("top1","top5","mean_reciprocal_rank","same_episode_top1","same_episode_mrr")
    for condition in cfg["conditions"]:
        out["conditions"][condition]={}
        for key in keys:
            x=[r["conditions"][condition]["correct"][key] for r in rows]
            out["conditions"][condition][key]={"per_seed":x,"mean":float(np.mean(x))}
        out["conditions"][condition]["shuffled_label_top1_mean"]=float(np.mean([
            r["conditions"][condition]["permuted_target_assignment"]["top1"] for r in rows
        ]))
    for control,field in (("raw_lexical","raw_lexical_input_baseline"),("neutral_equal_key","neutral_equal_key")):
        out["conditions"][control]={}
        for key in keys:
            x=[r[field][key] for r in rows]
            out["conditions"][control][key]={"per_seed":x,"mean":float(np.mean(x))}
    for key in keys:
        out["paired_effects"][key]={}
        for name in rows[0]["paired_deltas"][key]:
            x=[r["paired_deltas"][key][name] for r in rows]
            out["paired_effects"][key][name]={"per_seed":x,"mean":float(np.mean(x))}
    return out


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"config/chimera_001g.json")
    p.add_argument("--historical-v04",type=Path,required=True)
    p.add_argument("--connectome",type=Path,required=True)
    p.add_argument("--out",type=Path,default=ROOT/"results/chimera_001g")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--seed",type=int,help="Single-seed reduced-budget smoke only")
    p.add_argument("--phenotype-ticks",type=int)
    p.add_argument("--replay-epochs",type=int)
    p.add_argument("--neural-probe-ticks",type=int)
    a=p.parse_args()
    raw=a.config.read_bytes()
    cfg=json.loads(raw)
    if cfg["protocol_id"]!="chimera-001g-cue-source-neural-discrimination-v1":
        raise ValueError("Unexpected 001G protocol")
    legacy=dict(cfg["phenotype"])
    # Bridge only existing source loader; no new serializer or corpus index.
    old={
        "source_memory":cfg["source_memory"],
        "historical_training":legacy,
        "validation_file":"pretorius_validation_v1.json",
        "validation_sha256":"54aeab2e4ab0bde7f9a4c1acdaad062731657f2a664b5e3cdf5409636b5d1246",
        "adversarial_file":"pretorius_adversarial_v1.json",
        "adversarial_sha256":"e1510589bbf468d4c4d94a9587ac5dba9cd3284370efe9dae6779d043ccbc5cf",
    }
    source=source_preflight(old,a.historical_v04,a.connectome)
    sets,events,hashes=source
    corpus=load_corpus(a.connectome/cfg["source_memory"]["path"])
    if [e["event_id"] for e in events]!=[e["event_id"] for e in corpus.records]:
        raise ValueError("BC01 canonical corpus identity/order mismatch")
    if len(events)!=cfg["probe"]["candidate_count"]:
        raise ValueError("Expected exactly 450 source targets")
    cues=candidate_cues(events,cfg["probe"]["cue_count_per_event"])
    if a.preflight_only:
        print("001G source preflight OK: 450 editorial cues, source-owned BC01 importer, source SHA and split pins")
        return 0
    if a.seed is not None:
        seeds=[a.seed]
        if a.phenotype_ticks is None or a.replay_epochs is None or a.neural_probe_ticks is None:
            raise ValueError("Smoke seed requires explicit reduced budgets")
    else:
        seeds=cfg["seeds"]
        if any(x is not None for x in (a.phenotype_ticks,a.replay_epochs,a.neural_probe_ticks)):
            raise ValueError("Only single-seed smoke may modify budgets")
    if a.phenotype_ticks is not None:
        cfg["phenotype_training_ticks"]=a.phenotype_ticks
        cfg["replay_full_epochs"]=a.replay_epochs
        cfg["probe"]["stimulus_ticks"]=a.neural_probe_ticks
        cfg["probe"]["blank_ticks"]=max(2,a.neural_probe_ticks)
    if cfg["phenotype_training_ticks"]<=0 or cfg["replay_full_epochs"]<=0:
        raise ValueError("Neural budget must be positive")
    modes="full_six_seed_editorial_cue_assay" if a.seed is None else "smoke_not_full_evidence"
    meta={
        "protocol_id":cfg["protocol_id"],"config_sha256":hashlib.sha256(raw).hexdigest(),
        "run_mode":modes,"source_hashes":hashes,
        "no_unreviewed_001e_scenario_scores":True,
        "no_external_holdout_labels":True,
    }
    a.out.mkdir(parents=True,exist_ok=True)
    atomic_json(a.out/"RUN_METADATA.json",meta)
    external={"n":450,"accepted":0,"abstained":0,"top1_candidate_matches":0}
    # This is a non-neural source-owned lexical search calibration, not truth verification.
    for e,cue in zip(events,cues):
        hit=retrieve(corpus,cue)
        proposed=hit["candidate"]
        if proposed is None:
            external["abstained"]+=1
        else:
            external["accepted"]+=1
            external["top1_candidate_matches"]+=proposed["event_id"]==e["event_id"]
    external["top1_source_candidate_accuracy"]=external["top1_candidate_matches"]/len(events)
    external["coverage"]=external["accepted"]/len(events)
    rows=[]
    for seed in seeds:
        row=one_seed(cfg,source,int(seed))
        atomic_json(a.out/f"seed_{seed}.json",dict(row,**meta))
        rows.append(row)
        print(f"001G {modes} seed={seed} frozen top1={row['conditions']['baseline_frozen_replay']['correct']['top1']:.4f} "
              f"plastic={row['conditions']['autobiography_plastic_replay']['correct']['top1']:.4f} "
              f"raw lexical={row['raw_lexical_input_baseline']['top1']:.4f}",flush=True)
    report=summarize(rows,cfg,meta,external)
    atomic_json(a.out/"RUN_REPORT.json",report)
    print("001G complete: only editorial cue association, not independent recall")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
