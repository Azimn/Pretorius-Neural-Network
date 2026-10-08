#!/usr/bin/env python3
"""BC01 exploratory Pretorius demo. CPU only, no API and no model download."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biocircuit.bc01 import FIXTURE_PATH, demo, load_corpus  # noqa: E402

DEFAULT_QUESTIONS = [
    "What do you remember of the millstream map and turned glove?",
    "What happened with the beetle and the specimen drawer?",
    "Tell me about the purple spacecraft in 1982.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=FIXTURE_PATH,
                        help="Pinned original 450-event v12 JSONL or bundled 3-event smoke JSONL")
    parser.add_argument("--neurons", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1842)
    parser.add_argument("--mode", choices=["local", "global", "generic"], default="local")
    parser.add_argument("--exposures", type=int, default=8)
    parser.add_argument("--question", action="append", default=[],
                        help="Question without event ID. Repeat for multiple.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for another question")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/biocircuit/BC01_preview.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/biocircuit/BC01_preview.json"))
    args = parser.parse_args()
    corpus = load_corpus(args.corpus)
    questions = args.question or DEFAULT_QUESTIONS
    if args.interactive:
        print("Enter an additional question, or an empty line to finish:")
        while True:
            try:
                question = input("Pretorius question> ").strip()
            except EOFError:
                break
            if not question:
                break
            questions.append(question)
    result = demo(corpus, questions, neurons=args.neurons, seed=args.seed,
                  mode=args.mode, exposures=args.exposures, checkpoint=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("BioCircuit BC01 exploratory Pretorius")
    print(f"Pinned {result['corpus_kind']} autobiography: {result['corpus_events']} reconstructed events")
    print("Encoding: hashed lexical fallback (NOT semantic embedding)")
    print(f"Recurrent synapses changed: {result['recurrent_changed_synapses']}")
    for row in result["tests"]:
        print("\nQuestion:", row["query"])
        candidate = row["retrieval"]["candidate"]
        if candidate is None:
            print("Autobiographical evidence: UNKNOWN, no source candidate")
        else:
            print(f"Source candidate: {candidate['event_id']} ({candidate['provenance']})")
            print("Source excerpt:", candidate["source_quote"])
        print("Support/refutation: UNKNOWN (not semantically verified)")
        print("Neural policy:", row["local_or_selected"]["choice"])
        print("Learned-recurrence lesion policy:", row["recurrent_weight_lesion"]["choice"])
        print("Score difference:", round(row["recurrent_delta_score_max_abs"], 9))
        print("Policy changed after lesion:", row["recurrent_delta_changes_choice"])
        print("Checkpoint/restart exact:", row["checkpoint_restart_exact"])
    print("\nResearch JSON:", args.output)
    print("Neural checkpoint:", args.checkpoint)
    print("A neural score difference alone is not evidence of useful autobiographical decisions.")


if __name__ == "__main__":
    main()
