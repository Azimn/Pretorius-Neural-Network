from __future__ import annotations
import argparse, copy, json, math
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
from experiment_016_delta_distribution_necessity import synthetic_S, descriptors

ROOT=Path(__file__).resolve().parents[1]
CORE_FRACTION=0.01
CYCLIC_PAIRS=tuple(zip(SEEDS,SEEDS[1:]+SEEDS[:1]))
PROTOCOL_VERSION="experiment017-v1-preregistered"

def clipping_audit(founder,edges,deltas):
    edges=np.asarray(edges,dtype=np.int64); deltas=np.asarray(deltas,dtype=np.float64)
    count=0; magnitude=0.0
    for edge,delta in zip(edges,deltas):
        raw=float(founder.W.data[edge])+float(delta)
        clipped=float(np.clip(raw,0.0,founder.max_abs_weight)) if founder.excitatory[founder.pre_idx[edge]] else float(np.clip(raw,-founder.max_abs_weight,0.0))
        loss=abs(clipped-raw); count+=int(loss>1e-15); magnitude+=loss
    return {"count":int(count),"total_clipped_magnitude":float(magnitude)}

def _legal_ratio(founder,edge,delta):
    delta=float(delta)
    if delta==0.0: return math.inf
    w=float(founder.W.data[int(edge)])
    if founder.excitatory[founder.pre_idx[int(edge)]]:
        lower,upper=0.0,float(founder.max_abs_weight)
    else:
        lower,upper=-float(founder.max_abs_weight),0.0
    headroom=(upper-w) if delta>0 else (w-lower)
    if headroom<0.0 and headroom>-1e-14: headroom=0.0
    if headroom<0.0: raise RuntimeError("founder weight outside legal Dale bounds")
    return headroom/abs(delta)

def common_no_clip_alpha(founder,edges,R,S):
    edges=np.asarray(edges,dtype=np.int64); R=np.asarray(R,dtype=np.float64); S=np.asarray(S,dtype=np.float64)
    if not (len(edges)==len(R)==len(S)): raise ValueError("aligned edge and delta lengths differ")
    ratios=[_legal_ratio(founder,e,d) for values in (R,S) for e,d in zip(edges,values) if float(d)!=0.0]
    analytic=float(min([1.0]+[r for r in ratios if np.isfinite(r)]))
    alpha=max(0.0,min(1.0,analytic)); adjustments=0
    while True:
        ar=clipping_audit(founder,edges,R*alpha); ass=clipping_audit(founder,edges,S*alpha)
        if ar["count"]==0 and ass["count"]==0 and ar["total_clipped_magnitude"]==0.0 and ass["total_clipped_magnitude"]==0.0: break
        if alpha==0.0: raise RuntimeError("zero alpha still clips")
        alpha=float(np.nextafter(alpha,0.0)); adjustments+=1
        if adjustments>128: raise RuntimeError("floating boundary adjustment failed to establish no-clip arm")
    return {"analytic_alpha":analytic,"alpha_pair":alpha,"nextafter_adjustments":adjustments}

def run(neurons,training_steps,seeds=SEEDS):
    if tuple(seeds)!=SEEDS: raise ValueError("Experiment 017 uses frozen topology seeds")
    if training_steps!=24000: raise ValueError("Experiment 017 preregisters exactly 24,000 neural updates per topology")
    root=ROOT/"data"/"v0_4"; training=load_v04_split(root,"train"); validation=load_v04_split(root,"validation"); adversarial=load_v04_split(root,"adversarial")
    if (len(training),len(validation),len(adversarial))!=(100,40,20): raise RuntimeError("public split count contract violated")
    nets={}
    for seed in seeds:
        cfg=configured(seed,neurons); encoder=ExperienceEncoder(cfg["sensory_dim"]); compiled=compile_items(encoder,training); founder=PlasticRecurrentPersonaNet(cfg,encoder); mature=PlasticRecurrentPersonaNet(cfg,encoder)
        actual=train_recurrent_only(mature,compiled,training_steps)
        if actual!=training_steps: raise RuntimeError((seed,actual,training_steps))
        nets[seed]={"encoder":encoder,"compiled":compiled,"founder":founder,"mature":mature,"fp":fingerprints(copy.deepcopy(mature),compiled),"founder_scores":evaluate_both(founder,encoder,validation,adversarial),"mature_scores":evaluate_both(mature,encoder,validation,adversarial)}
    pairs=[]
    for donor_seed,recipient_seed in CYCLIC_PAIRS:
        donor,recipient=nets[donor_seed],nets[recipient_seed]
        donor_core=core_indices(donor["founder"],donor["mature"],CORE_FRACTION); recipient_core=core_indices(recipient["founder"],recipient["mature"],CORE_FRACTION)
        L=donor["mature"].W.data[donor_core]-donor["founder"].W.data[donor_core]; dale=sign_class(donor["mature"],donor_core)
        aligned,similarity=aligned_recipient_edges(donor["mature"],donor["fp"],donor_core,recipient["mature"],recipient["fp"])
        if not np.array_equal(dale,sign_class(recipient["mature"],aligned)): raise RuntimeError("aligned edge set violated Dale compatibility")
        R,rprov=matched_R(L,dale,donor_seed,recipient_seed); S,sprov=synthetic_S(L,dale,donor_seed,recipient_seed)
        alpha=common_no_clip_alpha(recipient["founder"],aligned,R,S); a=alpha["alpha_pair"]
        Rn=R*a; Sn=S*a; Z=np.zeros_like(L); within=recipient["mature"].W.data[recipient_core]-recipient["founder"].W.data[recipient_core]
        specs={"R_native":(aligned,R),"S_native":(aligned,S),"R_no_clip":(aligned,Rn),"S_no_clip":(aligned,Sn),"Z_zero":(aligned,Z),"within_topology_core":(recipient_core,within)}
        scores={}; recoveries={}; clipping={}
        for name,(edges,deltas) in specs.items():
            clipping[name]=clipping_audit(recipient["founder"],edges,deltas); graft,count=graft_with_audit(recipient["founder"],edges,deltas)
            if count!=clipping[name]["count"]: raise RuntimeError("clipping audit disagreement")
            scores[name]=evaluate_both(graft,recipient["encoder"],validation,adversarial); recoveries[name]=recovery(scores[name],recipient["founder_scores"],recipient["mature_scores"])
        for name in ("R_no_clip","S_no_clip"):
            if clipping[name]["count"] or clipping[name]["total_clipped_magnitude"]!=0.0: raise RuntimeError("no-clip arm clipped")
        native=contrast("R_native","S_native",recoveries); noclip=contrast("R_no_clip","S_no_clip",recoveries)
        did={split:{metric:(None if native[split][metric] is None or noclip[split][metric] is None else float(native[split][metric]-noclip[split][metric])) for metric in ("js_similarity","top1_agreement")} for split in ("validation","adversarial")}
        desc={"R_native":descriptors(R,dale),"S_native":descriptors(S,dale),"R_no_clip":descriptors(Rn,dale),"S_no_clip":descriptors(Sn,dale)}
        pairs.append({"donor_seed":donor_seed,"recipient_seed":recipient_seed,"core_edges":int(len(donor_core)),"mean_aligned_edge_cosine":float(np.mean(similarity)),"experiment016_vectors_reused_deterministically":True,"identical_aligned_edge_set_and_assignment_order":True,"alpha":alpha,"null_provenance":{"R_experiment015":rprov,"S_experiment016":sprov},"delta_descriptors_before_phenotype_interpretation":desc,"recipient_virgin":recipient["founder_scores"],"recipient_mature":recipient["mature_scores"],"conditions":scores,"recovery_fraction_relative_to_virgin_to_mature":recoveries,"contrasts":{"R_no_clip_minus_S_no_clip":noclip,"R_native_minus_S_native":native,"native_minus_no_clip_difference_of_differences":did},"clipping":clipping})
    permutation=run_gate(1842,neurons,training_steps,CORE_FRACTION)
    if not permutation["gate_passed"]: raise RuntimeError("permutation-isomorph positive control failed")
    aggregate={}
    for cname in ("R_no_clip_minus_S_no_clip","R_native_minus_S_native","native_minus_no_clip_difference_of_differences"):
        aggregate[cname]={}
        for split in ("validation","adversarial"):
            aggregate[cname][split]={}
            for metric in ("js_similarity","top1_agreement"):
                vals=[p["contrasts"][cname][split][metric] for p in pairs]; vals=[v for v in vals if v is not None]; aggregate[cname][split][metric]=None if not vals else float(np.mean(vals))
    return {"experiment":"017_clipping_susceptibility","issue":23,"protocol_version":PROTOCOL_VERSION,"terminal_battery_touched":False,"raw_recurrent_edge_indices_compared_across_topologies":False,"neurons":neurons,"seeds":list(seeds),"cyclic_pairs":[list(p) for p in CYCLIC_PAIRS],"core_fraction":CORE_FRACTION,"training_neural_steps_per_topology":training_steps,"independent_topology_training_neural_steps":training_steps*len(seeds),"permutation_control_training_neural_steps":training_steps,"total_training_neural_steps":training_steps*(len(seeds)+1),"plasticity_multiplier":16.0,"evaluation":{"settle_steps":40,"probe_steps":60},"alignment_metric":"frozen Experiment 013 cosine response-fingerprint metric with hard Dale compatibility","condition_definition":{"R_native":"exact deterministic Experiment 016 empirical-resampled R","S_native":"exact deterministic Experiment 016 RMS-matched half-normal S","R_no_clip":"R_native times phenotype-independent common alpha_pair","S_no_clip":"S_native times identical alpha_pair","Z":"zero deltas"},"permutation_isomorph_control":permutation,"pair_results":pairs,"aggregate_contrasts_mean_recovery_difference":aggregate}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--neurons",type=int,default=1024); p.add_argument("--training-neural-steps",type=int,default=24000); a=p.parse_args(); report=run(a.neurons,a.training_neural_steps); out=ROOT/"results"/"experiment_017_clipping_susceptibility"; out.mkdir(parents=True,exist_ok=True); path=out/f"report_n{a.neurons}_steps{a.training_neural_steps}.json"; path.write_text(json.dumps(report,indent=2)); print("saved",path)
if __name__=="__main__": main()
