#!/usr/bin/env python3
"""BioCircuit D4: fixed-source input-drive gain sweep (development-only)."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biocircuit.bc01 import load_corpus, SOURCE_PATH
from biocircuit.bc01_decisions import load_cards
from biocircuit.bc01_d4 import experiment

SUPPORTED_GAINS = (1.0, 4.0, 12.0)


def _accuracy(trials, condition):
    return sum(t["result"][condition]["accuracy"] for t in trials) / len(trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("upstream-connectome") / SOURCE_PATH)
    parser.add_argument("--connectome-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seeds", default="31,37,43")
    parser.add_argument("--modes", default="local,global,generic")
    parser.add_argument("--gains", default="1,4,12")
    parser.add_argument("--neurons", type=int, default=256)
    parser.add_argument("--background-ticks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--card-ticks", type=int, default=32)
    parser.add_argument("--settle-ticks", type=int, default=32)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/biocircuit/BC01_D4.json"))
    args = parser.parse_args()
    seeds = tuple(int(v) for v in args.seeds.split(","))
    modes = tuple(v.strip() for v in args.modes.split(","))
    gains = tuple(float(v) for v in args.gains.split(","))
    if (not seeds or len(set(seeds)) != len(seeds) or
            not modes or len(set(modes)) != len(modes) or
            set(modes) - {"local", "global", "generic"} or
            gains != SUPPORTED_GAINS):
        parser.error("D4 requires unique seeds/modes and prespecified gains 1,4,12")
    corpus = load_corpus(args.corpus)
    if corpus.source_kind != "full":
        parser.error("Only the immutable 450-record source is allowed")
    cards = load_cards(corpus)
    from biocircuit.shared_features import BioCircuitSharedFeatures
    cases = []
    # Structured cache is read once per seed. Every gain for that seed
    # reuses the same hashed source manifest and fitted 8192 coordinates.
    for seed in seeds:
        shared = BioCircuitSharedFeatures(args.cache_root / f"seed{seed}", args.connectome_root)
        shared.verify_corpus(corpus)
        if (shared.cache.manifest["random_seed"] != seed or
                shared.cache.manifest["schema_version"] != "pretorius.shared-features.v2" or
                shared.cache.manifest["encoder_name"] != "sklearn-tfidf-word12-rankstable-v2"):
            raise ValueError("Wrong episode split or nondeterministic L2 v1 cache")
        for mode in modes:
            for gain in gains:
                started = perf_counter()
                trial = experiment(
                    corpus, cards, seed=seed, mode=mode, neurons=args.neurons,
                    sensory_gain=gain, shared=shared, epochs=args.epochs,
                    card_ticks=args.card_ticks, background_ticks=args.background_ticks,
                    settle_ticks=args.settle_ticks, eta=args.eta,
                    checkpoint_dir=args.checkpoint_dir,
                )
                # Time is runtime diagnostic and intentionally excluded
                # from the strict numeric cross-process replay comparison.
                elapsed = round(perf_counter() - started, 6)
                trial["elapsed_seconds_observed"] = elapsed
                cases.append(trial)
                print(f"seed={seed} mode={mode} gain={gain:g} "
                      f"targeted={trial['result']['targeted']['accuracy']:.4f} "
                      f"lesion={trial['result']['targeted_recurrent_lesion']['accuracy']:.4f} "
                      f"shuffled={trial['result']['targeted_shuffled']['accuracy']:.4f} "
                      f"blank={trial['result']['targeted_blank_cue']['accuracy']:.4f} "
                      f"drive={trial['input_drive']['sensory_to_teacher_ratio_mean']:.4f} "
                      f"replay={trial['checkpoint_restart_exact']} "
                      f"cpu_wall_seconds={elapsed}", flush=True)
    summary = {}
    for gain in gains:
        subset = [trial for trial in cases if trial["sensory_gain"] == gain]
        summary[str(int(gain))] = {
            "trials": len(subset),
            "accuracy_mean": {c: _accuracy(subset, c) for c in (
                "targeted", "targeted_recurrent_lesion", "targeted_shuffled",
                "targeted_blank_cue", "no_training", "generic_hebb"
            )},
            "sensory_to_teacher_drive_ratio_mean": sum(
                t["input_drive"]["sensory_to_teacher_ratio_mean"] for t in subset
            ) / len(subset),
            "recurrent_lesion_choice_flips": sum(
                int(row["conditions"]["targeted"]["choice"] !=
                    row["conditions"]["targeted_recurrent_lesion"]["choice"])
                for t in subset for row in t["rows"]
            ),
            "passes_all_development_gates": sum(t["passes_all_development_gates"]
                                                for t in subset),
            "wall_seconds_total_observed": sum(t["elapsed_seconds_observed"] for t in subset),
        }
    report = {
        "schema": "BC01-D4-gain-sweep-v1",
        "status": "posthoc parameter mechanism test on reused interpreted source cards, not held-out",
        "source_git_blob": corpus.blob_sha,
        "source_event_count": len(corpus.records), "annotated_decision_cards": len(cards),
        "card_manifest_sha256": sha256(Path("resources/biocircuit/bc01_decision_cards_v1.json").read_bytes()).hexdigest(),
        "seeds": list(seeds), "modes": list(modes), "gains": list(gains),
        "summary": summary, "trials": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("Completed D4; report:", args.output, flush=True)


if __name__ == "__main__":
    main()
