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
from experiment_014_edge_set_vs_assignment import graft_with_audit

ROOT=Path(__file__).resolve().parents[1]
CORE_FRACTION=0.01
CYCLIC_PAIRS=tuple(zip(SEEDS,SEEDS[1:]+SEEDS[:1]))
RANDOMIZATION_NAMESPACE="experiment015-v1-preregistered"

def fixed_seed(donor_seed,recipient_seed,purpose):
    token=f"{RANDOMIZATION_NAMESPACE}:{donor_seed}:{recipient_seed}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8],"little")

def _sign_labels(values):
    values=np.asarray(values,dtype=np.float64)
    return np.sign(values).astype(np.int8)

def matched_M(values,dale_classes,donor_seed,recipient_seed):
    values=np.asarray(values,dtype=np.float64); dale_classes=np.asarray(dale_classes,dtype=bool); out=np.empty_like(values)
    provenance={}
    for dale in (False,True):
        pos=np.flatnonzero(dale_classes==dale); original=values[pos]; mags=np.abs(original); labels=_sign_labels(original)
        mag_seed=fixed_seed(donor_seed,recipient_seed,f"M-magnitude-dale-{int(dale)}")
        sign_seed=fixed_seed(donor_seed,recipient_seed,f"M-sign-dale-{int(dale)}")
        pm=np.random.default_rng(mag_seed).permutation(len(pos)); ps=np.random.default_rng(sign_seed).permutation(len(pos))
        candidate=mags[pm]*labels[ps]
        fallback="none"
        if len(pos)>1 and np.array_equal(candidate,original):
            # Frozen deterministic nonidentity fallback: rotate magnitude permutation only.
            # This preserves the exact magnitude multiset and sign-label counts.
            for shift in range(1,len(pos)):
                trial=mags[np.roll(pm,shift)]*labels[ps]
                if not np.array_equal(trial,original): candidate=trial; fallback=f"magnitude-roll-{shift}"; break
        out[pos]=candidate
        provenance[str(int(dale))]={"magnitude_seed":mag_seed,"sign_seed":sign_seed,"nonidentity_fallback":fallback}
    return out,provenance

def matched_R(values,dale_classes,donor_seed,recipient_seed):
    values=np.asarray(values,dtype=np.float64); dale_classes=np.asarray(dale_classes,dtype=bool); out=np.empty_like(values); provenance={}
    for dale in (False,True):
        pos=np.flatnonzero(dale_classes==dale); original=values[pos]; mags=np.abs(original); labels=_sign_labels(original)
        mag_seed=fixed_seed(donor_seed,recipient_seed,f"R-magnitude-dale-{int(dale)}")
        sign_seed=fixed_seed(donor_seed,recipient_seed,f"R-sign-dale-{int(dale)}")
        rm=np.random.default_rng(mag_seed); rs=np.random.default_rng(sign_seed)
        out[pos]=rm.choice(mags,size=len(pos),replace=True)*rs.choice(labels,size=len(pos),replace=True)
        provenance[str(int(dale))]={"magnitude_seed":mag_seed,"sign_seed":sign_seed,"sampling":"independent with replacement"}
    return out,provenance

def descriptors(values,dale_classes):
    values=np.asarray(values,dtype=np.float64); dale_classes=np.asarray(dale_classes,dtype=bool)
    def one(v):
        a=np.abs(v); s=_sign_labels(v)
        return {"count":int(len(v)),"positive":int(np.sum(s>0)),"negative":int(np.sum(s<0)),"zero":int(np.sum(s==0)),"mean_abs":float(np.mean(a)) if len(a) else 0.0,"median_abs":float(np.median(a)) if len(a) else 0.0,"l1":float(np.sum(a)),"l2":float(np.linalg.norm(v))}
    return {"all":one(values),"dale_0":one(values[~dale_classes]),"dale_1":one(values[dale_classes])}

def contrast(left,right,recoveries):
    out={}
    for split in ("validation","adversarial"):
        out[split]={}
        for metric in ("js_similarity","top1_agreement"):
            lv,rv=recoveries[left][split][metric],recoveries[right][split][metric]
            out[split][metric]=None if lv is None or rv is None else float(lv-rv)
    return out

def run(neurons,training_steps,seeds=SEEDS):
    if tuple(seeds)!=SEEDS: raise ValueError("Experiment 015 uses frozen topology seeds")
    if training_steps!=24000: raise ValueError("Experiment 015 preregisters exactly 24,000 neural updates per topology")
    root=ROOT/"data"/"v0_4"; training=load_v04_split(root,"train"); validation=load_v04_split(root,"validation"); adversarial=load_v04_split(root,"adversarial")
    if (len(training),len(validation),len(adversarial))!=(100,40,20): raise RuntimeError("public split count contract violated")
    nets={}
    for seed in seeds:
        cfg=configured(seed,neurons); encoder=ExperienceEncoder(cfg["sensory_dim"]); compiled=compile_items(encoder,training)
        founder=PlasticRecurrentPersonaNet(cfg,encoder); mature=PlasticRecurrentPersonaNet(cfg,encoder)
        actual=train_recurrent_only(mature,compiled,training_steps)
        if actual!=training_steps: raise RuntimeError((seed,actual,training_steps))
        nets[seed]={"encoder":encoder,"compiled":compiled,"founder":founder,"mature":mature,"fp":fingerprints(copy.deepcopy(mature),compiled),"actual":actual,"founder_scores":evaluate_both(founder,encoder,validation,adversarial),"mature_scores":evaluate_both(mature,encoder,validation,adversarial)}
    pairs=[]
    for donor_seed,recipient_seed in CYCLIC_PAIRS:
        donor,recipient=nets[donor_seed],nets[recipient_seed]
        donor_core=core_indices(donor["founder"],donor["mature"],CORE_FRACTION); recipient_core=core_indices(recipient["founder"],recipient["mature"],CORE_FRACTION)
        learned=donor["mature"].W.data[donor_core]-donor["founder"].W.data[donor_core]; dale=sign_class(donor["mature"],donor_core)
        aligned,similarity=aligned_recipient_edges(donor["mature"],donor["fp"],donor_core,recipient["mature"],recipient["fp"])
        if not np.array_equal(dale,sign_class(recipient["mature"],aligned)): raise RuntimeError("aligned edge set violated Dale compatibility")
        M,mprov=matched_M(learned,dale,donor_seed,recipient_seed); R,rprov=matched_R(learned,dale,donor_seed,recipient_seed); Z=np.zeros_like(learned)
        # M invariants are exact and checked before phenotype evaluation.
        for cls in (False,True):
            p=np.flatnonzero(dale==cls)
            if not np.array_equal(np.sort(np.abs(M[p])),np.sort(np.abs(learned[p]))): raise RuntimeError("M magnitude multiset invariant failed")
            if not np.array_equal(np.sort(_sign_labels(M[p])),np.sort(_sign_labels(learned[p]))): raise RuntimeError("M sign-count invariant failed")
        recipient_delta=recipient["mature"].W.data[recipient_core]-recipient["founder"].W.data[recipient_core]
        specs={"L_learned":(aligned,learned),"M_matched_permuted":(aligned,M),"R_matched_resampled":(aligned,R),"Z_zero":(aligned,Z),"within_topology_core":(recipient_core,recipient_delta)}
        scores={}; recoveries={}; clipping={}
        # Realized null descriptors are computed before any phenotype score.
        desc={"L_learned":descriptors(learned,dale),"M_matched_permuted":descriptors(M,dale),"R_matched_resampled":descriptors(R,dale)}
        for name,(edges,deltas) in specs.items():
            graft,clipping[name]=graft_with_audit(recipient["founder"],edges,deltas); scores[name]=evaluate_both(graft,recipient["encoder"],validation,adversarial); recoveries[name]=recovery(scores[name],recipient["founder_scores"],recipient["mature_scores"])
        pairs.append({"donor_seed":donor_seed,"recipient_seed":recipient_seed,"core_edges":int(len(donor_core)),"mean_aligned_edge_cosine":float(np.mean(similarity)),"identical_aligned_edge_set_and_assignment_order_LMRZ":True,"null_provenance":{"namespace":RANDOMIZATION_NAMESPACE,"M":mprov,"R":rprov},"realized_delta_descriptors_before_phenotype_interpretation":desc,"recipient_virgin":recipient["founder_scores"],"recipient_mature":recipient["mature_scores"],"conditions":scores,"recovery_fraction_relative_to_virgin_to_mature":recoveries,"contrasts":{"L_minus_M":contrast("L_learned","M_matched_permuted",recoveries),"L_minus_R":contrast("L_learned","R_matched_resampled",recoveries)},"clipping_counts":clipping})
    permutation=run_gate(1842,neurons,training_steps,CORE_FRACTION)
    if not permutation["gate_passed"]: raise RuntimeError("permutation-isomorph positive control failed")
    aggregate={}
    for cname in ("L_minus_M","L_minus_R"):
        aggregate[cname]={}
        for split in ("validation","adversarial"):
            aggregate[cname][split]={}
            for metric in ("js_similarity","top1_agreement"):
                vals=[p["contrasts"][cname][split][metric] for p in pairs]; vals=[v for v in vals if v is not None]
                aggregate[cname][split][metric]=None if not vals else float(np.mean(vals))
    return {"experiment":"015_learned_delta_multiset_necessity","issue":19,"protocol_version":"preregistered-v1","terminal_battery_touched":False,"raw_recurrent_edge_indices_compared_across_topologies":False,"neurons":neurons,"seeds":list(seeds),"cyclic_pairs":[list(p) for p in CYCLIC_PAIRS],"core_fraction":CORE_FRACTION,"training_neural_steps_per_topology":training_steps,"independent_topology_training_neural_steps":training_steps*len(seeds),"permutation_control_training_neural_steps":training_steps,"total_training_neural_steps":training_steps*(len(seeds)+1),"plasticity_multiplier":16.0,"evaluation":{"settle_steps":40,"probe_steps":60},"alignment_inputs":"unlabeled public training situations only","alignment_metric":"frozen Experiment 013 cosine response-fingerprint metric with hard Dale compatibility","randomization_namespace":RANDOMIZATION_NAMESPACE,"condition_definition":{"L":"learned donor delta multiset on frozen aligned recipient set/order","M":"per-Dale exact magnitude multiset and sign-count preserving independent deterministic permutations","R":"per-Dale independent with-replacement draws from learned magnitude and sign-label empirical distributions","Z":"zero deltas on identical aligned recipient set/order"},"permutation_isomorph_control":permutation,"pair_results":pairs,"aggregate_primary_contrasts_mean_recovery_difference":aggregate}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--neurons",type=int,default=1024); p.add_argument("--training-neural-steps",type=int,default=24000); a=p.parse_args(); report=run(a.neurons,a.training_neural_steps)
    out=ROOT/"results"/"experiment_015_learned_delta_necessity"; out.mkdir(parents=True,exist_ok=True); path=out/f"report_n{a.neurons}_steps{a.training_neural_steps}.json"; path.write_text(json.dumps(report,indent=2)); print("saved",path)
if __name__=="__main__": main()
