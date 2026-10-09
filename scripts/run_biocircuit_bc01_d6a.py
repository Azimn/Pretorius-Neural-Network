#!/usr/bin/env python3
"""D6A provisional *transductive* label-card holdout, NOT independent benchmark."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biocircuit.bc01 import SOURCE_PATH, load_corpus
from biocircuit.bc01_decisions import load_cards
from biocircuit.bc01_d6a import experiment, PARAPHRASES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",type=Path,default=Path("upstream-connectome")/SOURCE_PATH)
    p.add_argument("--connectome-root",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--seeds",default="31,37,43")
    p.add_argument("--modes",default="local,global,generic")
    p.add_argument("--checkpoint-root",type=Path)
    p.add_argument("--output",type=Path,default=Path("results/biocircuit/BC01_D6A.json"))
    a=p.parse_args()
    seeds=tuple(int(x) for x in a.seeds.split(","))
    modes=tuple(x.strip() for x in a.modes.split(","))
    if (not seeds or len(set(seeds))!=len(seeds) or not modes or
        len(set(modes))!=len(modes) or set(modes)-{"local","global","generic"}):
        p.error("D6A requires unique seeds and valid neural modes")
    corpus=load_corpus(a.corpus)
    if corpus.source_kind!="full":
        p.error("D6A source is the immutable original full 450-event biography")
    cards=load_cards(corpus)
    from biocircuit.shared_features import BioCircuitSharedFeatures
    trials=[]
    for seed in seeds:
        shared=BioCircuitSharedFeatures(a.cache_root/f"seed{seed}",a.connectome_root)
        shared.verify_corpus(corpus)
        if (shared.cache.manifest["random_seed"]!=seed or
            shared.cache.manifest["schema_version"]!="pretorius.shared-features.v2" or
            shared.cache.manifest["encoder_name"]!="sklearn-tfidf-word12-rankstable-v2"):
            raise ValueError("D6A requires deterministic source-owned L2 v2")
        for mode in modes:
            start=perf_counter()
            result=experiment(corpus,cards,seed=seed,mode=mode,shared=shared,
                              checkpoint_dir=(a.checkpoint_root/f"seed{seed}_{mode}"
                                              if a.checkpoint_root else None))
            result["elapsed_seconds_observed"]=round(perf_counter()-start,6)
            trials.append(result)
            for variant in ("original_source_probe","assistant_paraphrase"):
                original=sum(f["metrics"][variant]["decoder_frozen_recurrence"]
                             ["accuracy_with_abstain_counted_wrong"] for f in result["folds"])/4
                revised=sum(f["metrics"][variant]["decoder_with_cue_contrast_W"]
                            ["accuracy_with_abstain_counted_wrong"] for f in result["folds"])/4
                lesion=sum(f["metrics"][variant]["decoder_with_W_only_lesion"]
                           ["accuracy_with_abstain_counted_wrong"] for f in result["folds"])/4
                print(f"seed={seed} mode={mode} variant={variant} "
                      f"decoder={original:.4f} recurrent={revised:.4f} "
                      f"W_lesion={lesion:.4f} elapsed={result['elapsed_seconds_observed']}",flush=True)
    summary={}
    cond=("decoder_frozen_recurrence","decoder_shuffled_labels",
          "decoder_with_cue_contrast_W","decoder_with_W_only_lesion",
          "fixed_population_readout","train_card_lexical_retrieval",
          "decoder_training_confidence_abstain")
    for variant in ("original_source_probe","assistant_paraphrase"):
        fold_evaluations=4*len(trials)
        heldout_cases=4*fold_evaluations
        summary[variant]={
            "fold_evaluations":fold_evaluations,
            "heldout_card_evaluations":heldout_cases,
            "accuracy_mean":{name:sum(
                f["metrics"][variant][name]["accuracy_with_abstain_counted_wrong"]
                for t in trials for f in t["folds"])/fold_evaluations for name in cond},
            "macro_f1_mean":{name:sum(
                f["metrics"][variant][name]["macro_f1"]
                for t in trials for f in t["folds"])/fold_evaluations for name in cond},
            "coverage_mean":{name:sum(
                f["metrics"][variant][name]["coverage"]
                for t in trials for f in t["folds"])/fold_evaluations for name in cond},
            "recurrent_accuracy_better_than_W_lesion_folds":sum(
                f["metrics"][variant]["decoder_with_cue_contrast_W"]
                 ["accuracy_with_abstain_counted_wrong"] >
                f["metrics"][variant]["decoder_with_W_only_lesion"]
                 ["accuracy_with_abstain_counted_wrong"] for t in trials for f in t["folds"]),
        }
    report={
        "schema":"BC01-D6A-transductive-cross-validation-v1",
        "scope":"PROVISIONAL: labels held out only, original 450-memory unlabeled curriculum and original owner L2 v2 may expose heldout event/episode; not reviewed or inductive",
        "source_git_blob":corpus.blob_sha,
        "source_events":len(corpus.records),
        "cards_sha256":sha256(Path("resources/biocircuit/bc01_decision_cards_v1.json").read_bytes()).hexdigest(),
        "paraphrases_sha256":sha256(PARAPHRASES.read_bytes()).hexdigest(),
        "seeds":list(seeds),"modes":list(modes),"gain":12.0,
        "summary":summary,
        "trials":trials,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("D6A transductive provisional report:",a.output,flush=True)


if __name__=="__main__":
    main()
