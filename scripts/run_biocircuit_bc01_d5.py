#!/usr/bin/env python3
"""D5: exact source-pinned per-synapse eligibility and policy counterfactuals."""
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
from biocircuit.bc01_d5 import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=Path("upstream-connectome") / SOURCE_PATH)
    p.add_argument("--connectome-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--seeds", default="31,37,43")
    p.add_argument("--modes", default="local,global,generic")
    p.add_argument("--gains", default="1,12")
    p.add_argument("--neurons", type=int, default=256)
    p.add_argument("--background-ticks", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--card-ticks", type=int, default=32)
    p.add_argument("--settle-ticks", type=int, default=32)
    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/biocircuit/BC01_D5.json"))
    args = p.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    modes = tuple(s.strip() for s in args.modes.split(","))
    gains = tuple(float(s) for s in args.gains.split(","))
    if (not seeds or len(set(seeds)) != len(seeds)
            or not modes or len(set(modes)) != len(modes)
            or set(modes) - {"local", "global", "generic"}
            or gains != (1.0, 12.0)):
        p.error("D5 requires independent seeds, valid modes and frozen gains 1,12")
    corpus = load_corpus(args.corpus)
    if corpus.source_kind != "full":
        p.error("D5 requires original complete 450-event source")
    cards = load_cards(corpus)
    from biocircuit.shared_features import BioCircuitSharedFeatures
    trials = []
    for seed in seeds:
        shared = BioCircuitSharedFeatures(args.cache_root / f"seed{seed}", args.connectome_root)
        shared.verify_corpus(corpus)
        if (shared.cache.manifest["random_seed"] != seed or
                shared.cache.manifest["schema_version"] != "pretorius.shared-features.v2" or
                shared.cache.manifest["encoder_name"] != "sklearn-tfidf-word12-rankstable-v2"):
            raise ValueError("D5 source-owned TF-IDF cache not stable L2 v2 or wrong split")
        for mode in modes:
            for gain in gains:
                start = perf_counter()
                report = audit(
                    corpus, cards, seed=seed, mode=mode, sensory_gain=gain,
                    neurons=args.neurons, shared=shared,
                    background_ticks=args.background_ticks, epochs=args.epochs,
                    card_ticks=args.card_ticks, settle_ticks=args.settle_ticks,
                    eta=args.eta, checkpoint_dir=args.checkpoint_dir,
                )
                report["elapsed_seconds_observed"] = round(perf_counter() - start, 6)
                trials.append(report)
                s = report["summary"]
                print(f"seed={seed} mode={mode} gain={gain:g} "
                      f"accuracy={report['fixed_D4_baseline_accuracy']:.4f} "
                      f"eligibility={s['mean']['eligible_edges']:.2f} "
                      f"clipped={s['mean']['clipped_edges']:.2f} "
                      f"W_l1={s['targeted_recurrent_total_delta_l1']:.5f} "
                      f"margin_W={s['mean_abs_target_margin_due_to_recurrence']:.8f} "
                      f"margin_cue={s['mean_abs_target_margin_due_to_cue']:.8f} "
                      f"lesion_flips={s['lesion_choice_flips']} "
                      f"wall_s={report['elapsed_seconds_observed']}", flush=True)
    outcome = {
        "schema": "BC01-D5-credit-localization-full-v1",
        "status": "exploratory instrumentation only, no neuronal modification",
        "source_git_blob": corpus.blob_sha,
        "source_events": len(corpus.records),
        "source_card_count": len(cards),
        "source_card_sha256": sha256(Path("resources/biocircuit/bc01_decision_cards_v1.json").read_bytes()).hexdigest(),
        "seeds": list(seeds), "modes": list(modes), "gains": list(gains),
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Saved D5 observational full per-event evidence:", args.output, flush=True)


if __name__ == "__main__":
    main()
