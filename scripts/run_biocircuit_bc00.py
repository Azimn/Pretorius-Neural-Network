#!/usr/bin/env python3
"""Generate a runnable, self-contained BC00 comparison result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biocircuit.prototype import assay  # noqa: E402


def run(seeds: tuple[int, ...] = (31, 37, 43), neurons: int = 4096,
        items: int = 128, epochs: int = 4) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    return {
        "project": "Pretorius BioCircuit",
        "prototype": "BC00",
        "status": "synthetic association kernel, NOT full developmental connectome",
        "neuron_count_per_condition": neurons,
        "seeds": list(seeds),
        "training_budget": {"events": items, "passes": epochs},
        "hypothesis": "local versus global inhibitory competition, exact matched wiring",
        "trials": [assay(seed, neurons, items, epochs) for seed in seeds],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=4096)
    parser.add_argument("--items", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seeds", default="31,37,43")
    parser.add_argument("--output", type=Path, default=Path("results/biocircuit/BC00.json"))
    options = parser.parse_args()
    report = run(
        tuple(int(x) for x in options.seeds.split(",")),
        options.neurons, options.items, options.epochs,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("BioCircuit BC00, synthetic associations (not character memory)")
    for trial in report["trials"]:
        print("seed", trial["seed"], end=" ")
        for mode in ("local", "global"):
            x = trial["conditions"][mode]
            print(f"{mode}: partial={x['partial_cue_accuracy']:.3f} lesion={x['lesioned_partial_cue_accuracy']:.3f}",end=" ")
        print()
    print("Saved", options.output)


if __name__ == "__main__":
    main()
