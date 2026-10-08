# BioCircuit BC01 preview: executable Pretorius, not yet verified memory learning

**Status:** exploratory implementation on `feature/bc01-autobiographical-demo`; see [Issue #26](https://github.com/Azimn/Pretorius-Neural-Network/issues/26). This document does not claim that the BC01 acceptance gate has been passed.

## Start without an API, language-model download or custom dataset

From the repository root, with Python 3.11+, NumPy and SciPy installed:

```sh
python -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2"
python scripts/demo_biocircuit_bc01.py
```

This command launches a source-grounded, noninteractive smoke demonstration using three **unaltered** JSONL source records (`E09-001`, `E09-002`, `E01-001`) from the upstream v12 autobiographical corpus. It shows an evidence candidate, source quote, `reconstructed` provenance, a neural policy choice, the same policy after restoring its original recurrent weights, the maximum score change, and the checkpoint/restart equality check.

It saves `results/biocircuit/BC01_preview.json`, `results/biocircuit/BC01_preview.npz` and `results/biocircuit/BC01_preview.bc01.json`. These are generated artifacts, not source canon.

Interactive single-turn questions:

```sh
python scripts/demo_biocircuit_bc01.py --interactive --question "What do you remember of the millstream map?"
```

The project intentionally rejects queries containing event IDs, so there is no event-identifier input channel to the neural model.

## Actual 450-event source, pinned and fail-closed

Upstream canonical reconstruction: `Azimn/Pretorius-Connectome` at commit `597fb23473a60eecf2e1b50f79c22bfbea816be5`. The JSONL source path is `memories/current/Pretorius_v12_450_Events_Complete.jsonl`. Expected Git blob SHA-1 (including blob header and exact raw bytes): `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`.

Bundled three-event offline fixture is an exact byte-for-byte selection of the first three original JSONL rows, with Git blob `6e9404d8568b5e3431cdecce0ebd94a73248d6c8`. The importer accepts **only** the pinned 450-event original or pinned three-event fixture. Mutation, missing events, extra newlines, corrupt ordering, duplicate event IDs, unexpected provenance or unrecognized bytes raise an error rather than substituting synthetic memories.

A local adjacent checkout of the upstream source permits the full run:

```sh
git clone https://github.com/Azimn/Pretorius-Connectome.git
git -C Pretorius-Connectome checkout 597fb23473a60eecf2e1b50f79c22bfbea816be5
python scripts/demo_biocircuit_bc01.py --corpus Pretorius-Connectome/memories/current/Pretorius_v12_450_Events_Complete.jsonl --neurons 256
```

The GitHub Actions BC01 workflow checks out that exact upstream commit and validates the full 450-event original in CI. Neither the upstream autobiography nor The Doctor Lives production state is modified by the demo.

## Implemented boundaries

The input is deterministic **signed feature hashing of tokens and bigrams**, not a semantic embedding. The same encoder is used for both stored experiences and queries. BioCircuit's BC00 projection and compartmental winner-take-all activity create a fixed secondary signal into the existing `PlasticRecurrentPersonaNet`; the generic donor condition receives the same unmodified lexical stream. No new recurrent simulator is introduced.

The external lexical index can locate a candidate source event and quote its exact text. It cannot determine if a novel claim is supported, contradicted or merely related, so the verdict always remains `unknown`, including when a plausible passage is retrieved. This is intentional in light of the Pilot 06 false-accept/rejection tradeoff. Source event IDs and titles are never a neural training label or a neural query channel. All v12 memories are **reconstructed fictional history** and must not be mistaken for lived post-awakening state.

The recurrent donor receives the narrative text, eight unsupervised exposure ticks per memory by default, and **no target action and no reward gate**. The decoder is initialized once and never trained during this run. Consequently, its ten generic action scores are an inspectable *unvalidated* policy signal, not a meaningful historical decision prediction. The specialized bridge is a fixed fold; it is not a trained FlyWire topology.

## Causal controls and interpretation

The preview reports the compartment+recurrent condition, a generic recurrent donor condition, a no-plasticity control and a targeted lesion that replaces only learned recurrent synaptic weights with initial values, preserving learned biases and the motor decoder. It checks exact policy-probability reproducibility after saving and reloading the donor checkpoint. A score difference is not automatically a policy flip; neither counts as proof of semantic memory or improved Pretorius characterization. This exploratory test does not yet match full compute/feature-statistics budgets or independently validate action targets.

The comparison is not a completed BC01 evaluation. Still outstanding under Issue #26: independent reviewed prompts; matched decoder-only and retrieval-only scoring; a verified action-target definition grounded in prior decisions/consequences; recurrent-output mediation experiments; production-proof checkpoint metadata validation at arbitrary reload; contradiction entailment verification; multiseed outcomes and adverse results reporting. Pilot 04, 05 and 06 probes are development-exposed and are **not** new sealed evaluation.

Run tests with:

```sh
python -m unittest discover -s tests_biocircuit -v
```

CI publishes JSON and checkpoint artifacts. The final BC01 results document must include actual measured neural deltas, ablation effects and failures before describing the milestone as complete.
