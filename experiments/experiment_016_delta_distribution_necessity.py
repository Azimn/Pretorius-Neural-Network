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
from experiment_015_learned_delta_necessity import matched_R, contrast

ROOT=Path(__file__).resolve().parents[1]
CORE_FRACTION=0.01
CYCLIC_PAIRS=tuple(zip(SEEDS,SEEDS[1:]+SEEDS[:1]))
RANDOMIZATION_NAMESPACE="experiment016-v1-preregistered"

def s_seed(donor_seed,recipient_seed,dale_class):
    token=f"{RANDOMIZATION_NAMESPACE}|S-halfnormal|{donor_seed}|{recipient_seed}|{dale_class}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8],"big",signed=False)

def synthetic_S(values,dale_classes,donor_seed,recipient_seed):
    values=np.asarray(values,dtype=np.float64); dale_classes=np.asarray(dale_classes,dtype=bool); out=np.empty_like(values); provenance={}
    for dale in (False,True):
        pos=np.flatnonzero(dale_classes==dale); a=np.abs(values[pos]); seed=s_seed(donor_seed,recipient_seed,int(dale))
        if len(pos)==0: provenance[str(int(dale))]={"seed":seed,"count":0}; continue
        q=float(np.sqrt(np.mean(a*a)))
        if q==0.0: s=np.zeros(len(pos),dtype=np.float64)
        else:
            u=np.abs(np.random.default_rng(seed).standard_normal(len(pos))); urms=float(np.sqrt(np.mean(u*u)))
            if urms==0.0: raise RuntimeError("degenerate half-normal realization")
            s=u*q/urms
        signs=np.sign(values[pos]); out[pos]=s*signs
        if not np.isclose(np.sqrt(np.mean(out[pos]*out[pos])),q,rtol=1e-12,atol=1e-15): raise RuntimeError("S RMS invariant failed")
        provenance[str(int(dale))]={"seed":seed,"count":int(len(pos)),"target_rms":q,"family":"RMS-normalized half-normal"}
    return out,provenance

def descriptors(values,dale_classes):
    values=np.asarray(values,dtype=np.float64); dale_classes=np.asarray(dale_classes,dtype=bool)
    def one(v):
        a=np.abs(v); s=np.sign(v)
        return {"count":int(len(v)),"positive":int(np.sum(s>0)),"negative":int(np.sum(s<0)),"zero":int(np.sum(s==0)),"mean_abs":float(np.mean(a)) if len(a) else 0.0,"median_abs":float(np.median(a)) if len(a) else 0.0,"l1":float(np.sum(a)),"l2":float(np.linalg.norm(v)),"rms":float(np.sqrt(np.mean(a*a))) if len(a) else 0.0,"q25_abs":float(np.quantile(a,.25)) if len(a) else 0.0,"q75_abs":float(np.quantile(a,.75)) if len(a) else 0.0,"q90_abs":float(np.quantile(a,.90)) if len(a) else 0.0}
    return {"all":one(values),"dale_0":one(values[~dale_classes]),"dale_1":one(values[dale_classes])}

def run(neurons,training_steps,seeds=SEEDS):
    if tuple(seeds)!=SEEDS: raise ValueError("Experiment 016 uses frozen topology seeds")
    if training_steps!=24000: raise ValueError("Experiment 016 preregisters exactly 24,000 neural updates per topology")
    root=ROOT/"data"/"v0_4"; training=load_v04_split(root,"train"); validation=load_v04_split(root,"validation"); adversarial=load_v04_split(root,"adversarial")
    if (len(training),len(validation),len(adversarial))!=(100,40,20): raise RuntimeError("public split count contract violated")
    nets={}
    for seed in seeds:
        cfg=configured(seed,neurons); encoder=ExperienceEncoder(cfg["sensory_dim"]); compiled=compile_items(encoder,training); founder=PlasticRecurrentPersonaNet(cfg,encoder); mature=PlasticRecurrentPersonaNet(cfg,encoder)
        actual=train_recurrent_only(mature,compiled,training_steps)
        if actual!=training_steps: raise RuntimeError((seed,actual,training_steps))
        nets[seed]={"encoder":encoder,"compiled":compiled,"founder":founder,"mature":mature,"fp":fingerprints(copy.deepcopy(mature),compiled),"actual":actual,"founder_scores":evaluate_both(founder,encoder,validation,adversarial),"mature_scores":evaluate_both(mature,encoder,validation,adversarial)}
    pairs=[]
    for donor_seed,recipient_seed in CYCLIC_PAIRS:
        donor,recipient=nets[donor_seed],nets[recipient_seed]; donor_core=core_indices(donor["founder"],donor["mature"],CORE_FRACTION); recipient_core=core_indices(recipient["founder"],recipient["mature"],CORE_FRACTION)
        L=donor["mature"].W.data[donor_core]-donor["founder"].W.data[donor_core]; dale=sign_class(donor["mature"],donor_core); aligned,similarity=aligned_recipient_edges(donor["mature"],donor["fp"],donor_core,recipient["mature"],recipient["fp"])
        if not np.array_equal(dale,sign_class(recipient["mature"],aligned)): raise RuntimeError("aligned edge set violated Dale compatibility")
        R,rprov=matched_R(L,dale,donor_seed,recipient_seed); S,sprov=synthetic_S(L,dale,donor_seed,recipient_seed); Z=np.zeros_like(L); within=recipient["mature"].W.data[recipient_core]-recipient["founder"].W.data[recipient_core]
        desc={n:descriptors(v,dale) for n,v in (("L_learned",L),("R_empirical_resampled",R),("S_halfnormal_rms_matched",S),("Z_zero",Z))}
        specs={"L_learned":(aligned,L),"R_empirical_resampled":(aligned,R),"S_halfnormal_rms_matched":(aligned,S),"Z_zero":(aligned,Z),"within_topology_core":(recipient_core,within)}; scores={}; recoveries={}; clipping={}
        for name,(edges,deltas) in specs.items():
            graft,clipping[name]=graft_with_audit(recipient["founder"],edges,deltas); scores[name]=evaluate_both(graft,recipient["encoder"],validation,adversarial); recoveries[name]=recovery(scores[name],recipient["founder_scores"],recipient["mature_scores"])
        pairs.append({"donor_seed":donor_seed,"recipient_seed":recipient_seed,"core_edges":int(len(donor_core)),"mean_aligned_edge_cosine":float(np.mean(similarity)),"identical_aligned_edge_set_and_assignment_order_LRSZ":True,"null_provenance":{"namespace":RANDOMIZATION_NAMESPACE,"R_experiment015":rprov,"S":sprov},"realized_delta_descriptors_before_phenotype_interpretation":desc,"recipient_virgin":recipient["founder_scores"],"recipient_mature":recipient["mature_scores"],"conditions":scores,"recovery_fraction_relative_to_virgin_to_mature":recoveries,"contrasts":{"L_minus_S":contrast("L_learned","S_halfnormal_rms_matched",recoveries),"R_minus_S":contrast("R_empirical_resampled","S_halfnormal_rms_matched",recoveries)},"clipping_counts":clipping})
    permutation=run_gate(1842,neurons,training_steps,CORE_FRACTION)
    if not permutation["gate_passed"]: raise RuntimeError("permutation-isomorph positive control failed")
    aggregate={}
    for cname in ("L_minus_S","R_minus_S"):
        aggregate[cname]={}
        for split in ("validation","adversarial"):
            aggregate[cname][split]={}
            for metric in ("js_similarity","top1_agreement"):
                vals=[p["contrasts"][cname][split][metric] for p in pairs]; vals=[v for v in vals if v is not None]; aggregate[cname][split][metric]=None if not vals else float(np.mean(vals))
    return {"experiment":"016_learned_delta_distribution_necessity","issue":21,"protocol_version":"preregistered-v1","terminal_battery_touched":False,"raw_recurrent_edge_indices_compared_across_topologies":False,"neurons":neurons,"seeds":list(seeds),"cyclic_pairs":[list(p) for p in CYCLIC_PAIRS],"core_fraction":CORE_FRACTION,"training_neural_steps_per_topology":training_steps,"independent_topology_training_neural_steps":training_steps*len(seeds),"permutation_control_training_neural_steps":training_steps,"total_training_neural_steps":training_steps*(len(seeds)+1),"plasticity_multiplier":16.0,"evaluation":{"settle_steps":40,"probe_steps":60},"alignment_metric":"frozen Experiment 013 cosine response-fingerprint metric with hard Dale compatibility","randomization_namespace":RANDOMIZATION_NAMESPACE,"condition_definition":{"L":"learned donor deltas","R":"Experiment 015 per-Dale empirical resampling","S":"per-Dale RMS-exact half-normal amplitudes using learned sign orientation","Z":"zero deltas"},"permutation_isomorph_control":permutation,"pair_results":pairs,"aggregate_primary_contrasts_mean_recovery_difference":aggregate}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--neurons",type=int,default=1024); p.add_argument("--training-neural-steps",type=int,default=24000); a=p.parse_args(); report=run(a.neurons,a.training_neural_steps); out=ROOT/"results"/"experiment_016_delta_distribution_necessity"; out.mkdir(parents=True,exist_ok=True); path=out/f"report_n{a.neurons}_steps{a.training_neural_steps}.json"; path.write_text(json.dumps(report,indent=2)); print("saved",path)
if __name__=="__main__": main()
