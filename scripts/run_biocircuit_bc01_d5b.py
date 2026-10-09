#!/usr/bin/env python3
"""BC01-D5B: precommitted matched-blank presynaptic teaching control."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biocircuit.bc01 import SOURCE_PATH, load_corpus
from biocircuit.bc01_decisions import load_cards
from biocircuit.bc01_d5b import comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",type=Path,default=Path("upstream-connectome")/SOURCE_PATH)
    p.add_argument("--connectome-root",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--seeds",default="31,37,43")
    p.add_argument("--modes",default="local,global,generic")
    p.add_argument("--gains",default="1,12")
    p.add_argument("--neurons",type=int,default=256)
    p.add_argument("--background-ticks",type=int,default=8)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--card-ticks",type=int,default=32)
    p.add_argument("--settle-ticks",type=int,default=32)
    p.add_argument("--eta",type=float,default=0.03)
    p.add_argument("--checkpoint-root",type=Path)
    p.add_argument("--output",type=Path,default=Path("results/biocircuit/BC01_D5B.json"))
    a=p.parse_args()
    seeds=tuple(int(x) for x in a.seeds.split(","))
    modes=tuple(x.strip() for x in a.modes.split(","))
    gains=tuple(float(x) for x in a.gains.split(","))
    if (not seeds or len(set(seeds))!=len(seeds) or
            not modes or len(set(modes))!=len(modes) or
            set(modes)-{"local","global","generic"} or gains!=(1.,12.)):
        p.error("D5B is fixed to gains 1,12 and distinct valid seeds/modes")
    corpus=load_corpus(a.corpus)
    if corpus.source_kind!="full":
        p.error("Only the frozen complete 450-event corpus is allowed")
    cards=load_cards(corpus)
    from biocircuit.shared_features import BioCircuitSharedFeatures
    trials=[]
    for seed in seeds:
        shared=BioCircuitSharedFeatures(a.cache_root/f"seed{seed}",a.connectome_root)
        shared.verify_corpus(corpus)
        if (shared.cache.manifest["random_seed"]!=seed or
                shared.cache.manifest["schema_version"]!="pretorius.shared-features.v2" or
                shared.cache.manifest["encoder_name"]!="sklearn-tfidf-word12-rankstable-v2"):
            raise ValueError("Incorrect deterministic shared L2 v2 cache")
        for mode in modes:
            for gain in gains:
                started=perf_counter()
                result=comparison(
                    corpus,cards,shared=shared,seed=seed,mode=mode,
                    sensory_gain=gain,neurons=a.neurons,
                    background_ticks=a.background_ticks,epochs=a.epochs,
                    card_ticks=a.card_ticks,settle_ticks=a.settle_ticks,
                    eta=a.eta,checkpoint_root=a.checkpoint_root,
                )
                result["observed_elapsed_seconds"]=round(perf_counter()-started,6)
                trials.append(result)
                corrected=result["cue_contrast"]["accuracy"]
                print(
                    f"seed={seed} mode={mode} gain={gain:g} "
                    f"targeted={corrected['targeted']:.4f} "
                    f"lesion={corrected['targeted_recurrent_lesion']:.4f} "
                    f"shuffled={corrected['targeted_shuffled']:.4f} "
                    f"blank={corrected['targeted_blank_cue']:.4f} "
                    f"motor={result['motor_decoder_only']['accuracy']:.4f} "
                    f"eligible_frac={result['cue_evoked_eligibility']['eligible_fraction_mean']:.5f} "
                    f"passes={result['passes_all_gates']}",flush=True,
                )
    summary={}
    for gain in gains:
        chunk=[row for row in trials if row["sensory_gain"]==gain]
        labels=("targeted","targeted_recurrent_lesion","targeted_shuffled",
                "targeted_blank_cue","no_training","generic_hebb")
        summary[str(int(gain))]={
            "trials":len(chunk),
            "accuracy_mean":{key:sum(row["cue_contrast"]["accuracy"][key]
                for row in chunk)/len(chunk) for key in labels},
            "prior_D4_targeted_accuracy_mean":sum(row["original_D4"]["accuracy"]["targeted"]["accuracy"]
                for row in chunk)/len(chunk),
            "motor_decoder_only_accuracy_mean":sum(row["motor_decoder_only"]["accuracy"]
                for row in chunk)/len(chunk),
            "eligible_fraction_mean":sum(row["cue_evoked_eligibility"]["eligible_fraction_mean"]
                for row in chunk)/len(chunk),
            "passes_all_gates":sum(row["passes_all_gates"] for row in chunk),
        }
    report={"schema":"BC01-D5B-one-cue-minus-blank-correction-v1",
            "source_git_blob":corpus.blob_sha,"source_events":len(corpus.records),
            "source_card_count":len(cards),
            "source_card_sha256":sha256(Path("resources/biocircuit/bc01_decision_cards_v1.json").read_bytes()).hexdigest(),
            "seeds":list(seeds),"modes":list(modes),"gains":list(gains),
            "status":"development-only, no independent heldout source decision validation",
            "summary":summary,"trials":trials}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("D5B complete:",a.output,flush=True)


if __name__=="__main__":
    main()
