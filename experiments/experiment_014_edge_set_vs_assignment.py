from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.v04_data import load_v04_split
from experiment_010_minimal_transferable_core import compile_items, evaluate_both, train_recurrent_only
from experiment_012_core_convergence import configured
from experiment_013_cross_topology import fingerprints, run_gate
from experiment_013_stage_c import SEEDS, core_indices, sign_class
from experiment_013_stage_d import aligned_recipient_edges, recovery

ROOT=Path(__file__).resolve().parents[1]
CORE_FRACTION=0.01
CYCLIC_PAIRS=tuple(zip(SEEDS,SEEDS[1:]+SEEDS[:1]))
RANDOMIZATION_NAMESPACE="experiment014-v1-preregistered"

def fixed_seed(donor_seed,recipient_seed,purpose):
    token=f"{RANDOMIZATION_NAMESPACE}:{donor_seed}:{recipient_seed}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8],"little")

def dale_shuffle(values,signs,seed):
    values=np.asarray(values,dtype=np.float64); signs=np.asarray(signs,dtype=bool); out=values.copy(); rng=np.random.default_rng(seed)
    for dale in (False,True):
        pos=np.flatnonzero(signs==dale); out[pos]=values[rng.permutation(pos)]
    return out

def paired_random_edges(net,target_signs,excluded_edges,seed):
    target_signs=np.asarray(target_signs,dtype=bool); excluded=set(np.asarray(excluded_edges,dtype=np.int64).tolist())
    all_edges=np.arange(len(net.W.data),dtype=np.int64); all_signs=sign_class(net,all_edges); rng=np.random.default_rng(seed); selected=np.empty(len(target_signs),dtype=np.int64)
    for dale in (False,True):
        pos=np.flatnonzero(target_signs==dale); pool=np.asarray([e for e in all_edges[all_signs==dale] if int(e) not in excluded],dtype=np.int64)
        if len(pool)<len(pos): raise RuntimeError("recipient lacks enough eligible non-aligned Dale-compatible random edges")
        selected[pos]=rng.choice(pool,size=len(pos),replace=False)
    return selected

def graft_with_audit(founder,edges,deltas):
    out=copy.deepcopy(founder); edges=np.asarray(edges,dtype=np.int64); deltas=np.asarray(deltas,dtype=np.float64)
    if len(edges)!=len(deltas) or len(set(edges.tolist()))!=len(edges): raise ValueError("one delta is required per unique recipient edge")
    clipped=0
    for edge,delta in zip(edges,deltas):
        raw=float(out.W.data[edge])+float(delta)
        value=float(np.clip(raw,0.0,out.max_abs_weight)) if out.excitatory[out.pre_idx[edge]] else float(np.clip(raw,-out.max_abs_weight,0.0))
        clipped+=int(abs(value-raw)>1e-15); out.W.data[edge]=value
    out.eligibility.fill(0.0); out.reset_fast_state(); return out,clipped

def same_multiset(a,b):
    return np.array_equal(np.sort(np.asarray(a,dtype=np.float64)),np.sort(np.asarray(b,dtype=np.float64)))

def run(neurons,training_steps,seeds=SEEDS):
    if tuple(seeds)!=SEEDS: raise ValueError("Experiment 014 uses the frozen six topology seeds")
    if training_steps!=24000: raise ValueError("Experiment 014 preregisters exactly 24,000 training neural steps per topology")
    data_root=ROOT/"data"/"v0_4"; training=load_v04_split(data_root,"train"); validation=load_v04_split(data_root,"validation"); adversarial=load_v04_split(data_root,"adversarial")
    if (len(training),len(validation),len(adversarial))!=(100,40,20): raise RuntimeError("public split count contract violated")
    nets={}
    for seed in seeds:
        cfg=configured(seed,neurons); encoder=ExperienceEncoder(cfg["sensory_dim"]); compiled=compile_items(encoder,training); founder=PlasticRecurrentPersonaNet(cfg,encoder); mature=PlasticRecurrentPersonaNet(cfg,encoder)
        actual=train_recurrent_only(mature,compiled,training_steps)
        if actual!=training_steps: raise RuntimeError((seed,actual,training_steps))
        nets[seed]={"encoder":encoder,"founder":founder,"mature":mature,"fp":fingerprints(copy.deepcopy(mature),compiled),"actual":actual,"founder_scores":evaluate_both(founder,encoder,validation,adversarial),"mature_scores":evaluate_both(mature,encoder,validation,adversarial)}
    pair_results=[]
    for donor_seed,recipient_seed in CYCLIC_PAIRS:
        donor,recipient=nets[donor_seed],nets[recipient_seed]; donor_core=core_indices(donor["founder"],donor["mature"],CORE_FRACTION); recipient_core=core_indices(recipient["founder"],recipient["mature"],CORE_FRACTION)
        donor_delta=donor["mature"].W.data[donor_core]-donor["founder"].W.data[donor_core]; recipient_delta=recipient["mature"].W.data[recipient_core]-recipient["founder"].W.data[recipient_core]; donor_signs=sign_class(donor["mature"],donor_core)
        aligned_edges,aligned_similarity=aligned_recipient_edges(donor["mature"],donor["fp"],donor_core,recipient["mature"],recipient["fp"])
        if not np.array_equal(donor_signs,sign_class(recipient["mature"],aligned_edges)): raise RuntimeError("aligned mapping violated Dale compatibility")
        edge_seed=fixed_seed(donor_seed,recipient_seed,"random-edge-set"); b_seed=fixed_seed(donor_seed,recipient_seed,"aligned-shuffle"); d_seed=fixed_seed(donor_seed,recipient_seed,"random-shuffle")
        random_edges=paired_random_edges(recipient["mature"],donor_signs,aligned_edges,edge_seed)
        if set(random_edges.tolist())&set(aligned_edges.tolist()): raise RuntimeError("random recipient set overlaps aligned set")
        if not np.array_equal(donor_signs,sign_class(recipient["mature"],random_edges)): raise RuntimeError("random recipient set violated per-position Dale compatibility")
        b_delta=dale_shuffle(donor_delta,donor_signs,b_seed); d_delta=dale_shuffle(donor_delta,donor_signs,d_seed)
        if not all(same_multiset(donor_delta,x) for x in (b_delta,d_delta)): raise RuntimeError("A-D donor delta multiset preservation failed")
        specs={"A_aligned_exact":(aligned_edges,donor_delta),"B_aligned_shuffled":(aligned_edges,b_delta),"C_random_deterministic":(random_edges,donor_delta),"D_random_shuffled":(random_edges,d_delta),"within_topology_core":(recipient_core,recipient_delta),"zero_delta_bookkeeping":(aligned_edges,np.zeros_like(donor_delta))}
        scores={}; recoveries={}; clipping={}
        for name,(edges,deltas) in specs.items():
            graft,clipping[name]=graft_with_audit(recipient["founder"],edges,deltas); scores[name]=evaluate_both(graft,recipient["encoder"],validation,adversarial); recoveries[name]=recovery(scores[name],recipient["founder_scores"],recipient["mature_scores"])
        def contrast(left,right):
            out={}
            for split in ("validation","adversarial"):
                out[split]={}
                for metric in ("js_similarity","top1_agreement"):
                    lv,rv=recoveries[left][split][metric],recoveries[right][split][metric]; out[split][metric]=None if lv is None or rv is None else float(lv-rv)
            return out
        pair_results.append({"donor_seed":donor_seed,"recipient_seed":recipient_seed,"core_edges":int(len(donor_core)),"mean_aligned_edge_cosine":float(np.mean(aligned_similarity)),"randomization_seeds":{"random_edge_set":edge_seed,"aligned_shuffle":b_seed,"random_shuffle":d_seed},"paired_random_edge_set_reused_for_C_and_D":True,"random_set_excludes_aligned_edges":True,"same_donor_delta_multiset_A_through_D":True,"per_position_dale_compatible_A_through_D":True,"recipient_virgin":recipient["founder_scores"],"recipient_mature":recipient["mature_scores"],"conditions":scores,"recovery_fraction_relative_to_virgin_to_mature":recoveries,"contrasts":{"B_minus_D_edge_set":contrast("B_aligned_shuffled","D_random_shuffled"),"A_minus_B_exact_assignment":contrast("A_aligned_exact","B_aligned_shuffled"),"C_minus_D_negative_assignment":contrast("C_random_deterministic","D_random_shuffled")},"clipping_counts":clipping})
    permutation_control=run_gate(1842,neurons,training_steps,CORE_FRACTION)
    if not permutation_control["gate_passed"]: raise RuntimeError("permutation-isomorph positive control failed")
    condition_names=("A_aligned_exact","B_aligned_shuffled","C_random_deterministic","D_random_shuffled")
    aggregate_conditions={}
    for name in condition_names:
        aggregate_conditions[name]={}
        for split in ("validation","adversarial"):
            aggregate_conditions[name][split]={}
            for metric in ("js_similarity","top1_agreement"):
                raw=[p["conditions"][name][split][metric] for p in pair_results]; rec=[p["recovery_fraction_relative_to_virgin_to_mature"][name][split][metric] for p in pair_results]; rec=[v for v in rec if v is not None]
                aggregate_conditions[name][split][metric]={"mean_score":float(np.mean(raw)),"mean_recovery":None if not rec else float(np.mean(rec))}
    aggregate_contrasts={}
    for cname in ("B_minus_D_edge_set","A_minus_B_exact_assignment","C_minus_D_negative_assignment"):
        aggregate_contrasts[cname]={}
        for split in ("validation","adversarial"):
            aggregate_contrasts[cname][split]={}
            for metric in ("js_similarity","top1_agreement"):
                vals=[p["contrasts"][cname][split][metric] for p in pair_results]; vals=[v for v in vals if v is not None]; aggregate_contrasts[cname][split][metric]=None if not vals else float(np.mean(vals))
    return {"experiment":"014_edge_set_selection_vs_delta_assignment","issue":17,"protocol_version":"preregistered-v1","terminal_battery_touched":False,"raw_recurrent_edge_indices_compared_across_topologies":False,"neurons":neurons,"seeds":list(seeds),"cyclic_pairs":[list(p) for p in CYCLIC_PAIRS],"core_fraction":CORE_FRACTION,"training_neural_steps_per_topology":training_steps,"independent_topology_training_neural_steps":training_steps*len(seeds),"permutation_control_training_neural_steps":training_steps,"total_training_neural_steps":training_steps*(len(seeds)+1),"plasticity_multiplier":16.0,"evaluation":{"settle_steps":40,"probe_steps":60},"alignment_inputs":"unlabeled public training situations only","alignment_metric":"frozen Experiment 013 cosine response-fingerprint metric with hard Dale compatibility","randomization_namespace":RANDOMIZATION_NAMESPACE,"condition_definition":{"A":"aligned recipient set plus exact donor-delta assignment","B":"same aligned set plus Dale-class-preserving shuffled donor deltas","C":"paired non-aligned random set plus donor deltas in original deterministic donor-core order","D":"identical random set as C plus Dale-class-preserving shuffled donor deltas"},"permutation_isomorph_control":permutation_control,"pair_results":pair_results,"aggregate_condition_outcomes":aggregate_conditions,"aggregate_primary_contrasts_mean_recovery_difference":aggregate_contrasts}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--neurons",type=int,default=1024); p.add_argument("--training-neural-steps",type=int,default=24000); args=p.parse_args(); report=run(args.neurons,args.training_neural_steps)
    outdir=ROOT/"results"/"experiment_014_edge_set_vs_assignment"; outdir.mkdir(parents=True,exist_ok=True); path=outdir/f"report_n{args.neurons}_steps{args.training_neural_steps}.json"; path.write_text(json.dumps(report,indent=2)); print("saved",path)
if __name__=="__main__": main()
