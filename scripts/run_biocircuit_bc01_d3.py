#!/usr/bin/env python3
"""D3: audited BC01 recurrent teaching and activity-collapse diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biocircuit.bc01 import load_corpus, SOURCE_PATH
from biocircuit.bc01_decisions import load_cards
from biocircuit.bc01_d3 import experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",type=Path,default=Path("upstream-connectome")/SOURCE_PATH)
    p.add_argument("--connectome-root",type=Path,default=None)
    p.add_argument("--cache-root",type=Path,default=None,
                   help="Directory containing independently fit seed31/seed37/... caches")
    p.add_argument("--seeds",default="31,37,43")
    p.add_argument("--modes",default="local,global,generic")
    p.add_argument("--neurons",type=int,default=256)
    p.add_argument("--eta",type=float,default=0.03)
    p.add_argument("--gain",type=float,default=1.0)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--card-ticks",type=int,default=32)
    p.add_argument("--background-ticks",type=int,default=8)
    p.add_argument("--settle-ticks",type=int,default=32)
    p.add_argument("--output",type=Path,default=Path("results/biocircuit/BC01_D3.json"))
    p.add_argument("--checkpoint-dir",type=Path,default=None)
    args=p.parse_args()
    if bool(args.cache_root)!=bool(args.connectome_root):
        p.error("Source-owned cache requires --cache-root and --connectome-root")
    seeds=tuple(int(x) for x in args.seeds.split(","))
    modes=tuple(x.strip() for x in args.modes.split(","))
    if not seeds or len(set(seeds))!=len(seeds) or not modes or len(set(modes))!=len(modes):
        p.error("Distinct nonempty seeds and modes required")
    corpus=load_corpus(args.corpus)
    if corpus.source_kind!="full":
        p.error("Full immutable 450-event source required")
    cards=load_cards(corpus)
    rows=[]
    for seed in seeds:
        shared=None
        if args.cache_root:
            from biocircuit.shared_features import BioCircuitSharedFeatures
            shared=BioCircuitSharedFeatures(args.cache_root/f"seed{seed}", args.connectome_root)
            shared.verify_corpus(corpus)
            if shared.cache.manifest["random_seed"]!=seed:
                raise ValueError("Source-cache fit seed does not match experiment seed")
        for mode in modes:
            trial=experiment(
                corpus,cards,neurons=args.neurons,seed=seed,mode=mode,
                shared=shared,eta=args.eta,gain=args.gain,epochs=args.epochs,
                background_ticks=args.background_ticks,
                card_ticks=args.card_ticks,settle_ticks=args.settle_ticks,
                checkpoint_dir=args.checkpoint_dir,
            )
            rows.append(trial)
            print(f"seed={seed} mode={mode} targeted={trial['result']['targeted']['accuracy']:.3f} "
                  f"generic={trial['result']['generic_hebb']['accuracy']:.3f} "
                  f"lesion={trial['result']['targeted_recurrent_lesion']['accuracy']:.3f} "
                  f"shuffle={trial['result']['targeted_shuffled']['accuracy']:.3f} "
                  f"cue_spread={trial['neural_cue_rms_spread']['targeted']:.8f} "
                  f"distinct={trial['result']['targeted']['distinct_choices']} "
                  f"passes={trial['passes_all_development_gates']}",flush=True)
    result={"schema":"BC01-D3-exploratory-v1",
            "status":"development-only, provisional source-action labels, not confirmatory",
            "source_git_blob":corpus.blob_sha,
            "source_events":len(corpus.records),"action_cards":len(cards),
            "input":"Connectome-owned L1/L2 lexical cache" if args.cache_root else "legacy signed lexical",
            "trials":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("Raw research evidence:",args.output)


if __name__=="__main__":
    main()
