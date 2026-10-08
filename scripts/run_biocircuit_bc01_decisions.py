#!/usr/bin/env python3
"""Run the BC01 source-anchored decision assay, never an identity benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocircuit.bc01 import SOURCE_PATH, load_corpus  # noqa: E402
from biocircuit.bc01_decisions import benchmark, load_cards, ALLOWED_MODES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=Path("Pretorius-Connectome") / SOURCE_PATH)
    parser.add_argument("--neurons", type=int, default=256)
    parser.add_argument("--seeds", default="31,37,43")
    parser.add_argument("--modes", default="local,global,generic")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--card-ticks", type=int, default=32)
    parser.add_argument("--background-ticks", type=int, default=8)
    parser.add_argument("--settle-ticks", type=int, default=32)
    parser.add_argument("--plasticity-gain", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("results/biocircuit/BC01_D1.json"))
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    modes = tuple(x.strip() for x in args.modes.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must be distinct")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(ALLOWED_MODES):
        parser.error("Invalid topology modes")
    corpus = load_corpus(args.corpus)
    if corpus.source_kind != "full":
        parser.error("Decision-card assay requires pinned FULL 450-memory corpus")
    cards = load_cards(corpus)
    results = []
    for seed in seeds:
        for mode in modes:
            row = benchmark(
                corpus, cards, mode=mode, neurons=args.neurons, seed=seed,
                epochs=args.epochs, card_ticks=args.card_ticks,
                background_ticks=args.background_ticks,
                settle_ticks=args.settle_ticks, plasticity_gain=args.plasticity_gain,
                checkpoint_dir=args.checkpoint_dir,
            )
            results.append(row)
            print(
                f"seed={seed} mode={mode}: "
                f"intact={row['accuracy']['intact']:.3f} "
                f"lesion={row['accuracy']['recurrent_lesion']:.3f} "
                f"shuffled={row['accuracy']['shuffled_labels']:.3f} "
                f"decoder={row['accuracy']['decoder_only']:.3f} "
                f"retrieval={row['accuracy']['retrieval_only']:.3f} "
                f"lesion flips={row['recurrent_lesion_choice_flips']} "
                f"restart_exact={row['checkpoint_reproduced_all']}",
                flush=True,
            )
    report = {
        "schema": "BC01-D1-exploratory-v1",
        "status": "development set only; not independently reviewed",
        "source_blob": corpus.blob_sha,
        "events": len(corpus.records), "decision_cards": len(cards),
        "seeds": list(seeds), "modes": list(modes),
        "chance_majority_class": max(sum(c["action"] == a for c in cards) for a in set(c["action"] for c in cards)) / len(cards),
        "trials": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Saved development-only raw evidence:", args.output)
    print("Do not infer an emergent autobiographical identity from in-sample action scores.")


if __name__ == "__main__":
    main()
