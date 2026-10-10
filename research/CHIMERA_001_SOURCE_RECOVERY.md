# Experiment 001 historical phenotype battery source recovery

Audit initiated 2026-10-09. This document extends, but does not change or overrule, the blocked original reproduction recorded in [RUN_REPORT.md](../results/chimera_001/RUN_REPORT.md). This is a source-forensics investigation, not neural scoring.

The exact Experiment 001 release inputs have four required original v1 SHA-256 values in [chimera_001.json](../config/chimera_001.json): 100 axis-grounded training items, 30 additional legacy training items, 40 validation items, and 20 adversarial items. The profile file is separately verified. The 20 sealed terminal inputs and scoring key remain excluded.

## Where the surviving records reside

The history of the repository has two independently pinned nonterminal preservation branches. The public v0.4 split payload is commit `7b7eb19ad852dc016ee0d370528d903bab78a5cf`, under `data/v0_4/`, with four 25-item training files, four 10-item validation files, and one 20-item adversarial file. The older v0.4 phenotype preservation snapshot is `a3af4d2433d9df95b2476c556e9e2b9b308944a4` and holds `data/phenotype_battery/pretorius_profile_v1.json` plus an adversarial file of the same name as the missing original.

An initial read-only audit found 140 distinct IDs across the v0.4 100+40 train/validation slices. Its adversarial set has exactly the same parsed records as the separate phenotype preservation branch's 20-item adversarial set. The historical profile matches the canonical original v1 SHA-256. **But** the historical adversarial file does *not* match its original v1 manifest SHA-256: historical byte hash `20f87fd29798be3540a67daec93d71175d019e5611e806c36f4263e196eb6a87`, original required `e1510589bbf468d4c4d94a9587ac5dba9cd3284370efe9dae6779d043ccbc5cf`. Parsed equality between preservation branches cannot substitute for the original source hash.

The v0.4 100-item subset cannot replace the original 130-item training split; no historical branch or available file-library search has yielded the 30 legacy training records. The original v1 presentation order is also not independently specified in the narrative log.

## Reproducible audit

The read-only [audit_chimera_001_sources.py](../scripts/audit_chimera_001_sources.py) checks pinned source commits, total item counts, cross-split ID uniqueness, 20-domain stratification, decoded original SHA-256 profile proof, and the semantic identity of the two preserved adversarial copies. It also tries a tightly scoped family of ordinary JSON envelope/indentation/line-ending renderings to see if any candidate naturally hashes to a precommitted original v1 source digest. A *positive SHA-256 match* can authenticate candidate bytes after separate review. A nonmatch is not evidence the historical data are factually wrong or that the neural hypothesis fails.

[Source audit workflow](../.github/workflows/chimera-001-source-audit.yml) checks out two exact historical commits, runs the audit, and uploads its complete machine-readable JSON. The workflow intentionally cannot access the sealed battery, cannot train any model and cannot relabel Experiment 001 as reproduced.

## Release boundary

Phase 1 reproduction remains blocked until the original v1 training (all 130), validation (40) and adversarial (20) sources are authenticated and input presentation order/parameter assumptions are resolved or explicitly labeled. Only then run six seeds with the original runner and preserve all per-seed scores. Do not tune to the historical mean. The fresh-decoder control stays gated behind Phase 1 as already specified. The journal's CC-E01 remains narrative-only until the complete six-seed results and immutable result commit exist.

## 2026-10-09: Exact v1 adversarial restoration from serialization proof

The pinned historical audit passed on [CI run 38024167448](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/38024167448). It independently verified 100/40/20 recovered split sizes, all 160 distinct train/validation/adversarial IDs, 20-domain stratification, identical parsed adversarial data across the two historical preservation commits and the original profile SHA-256. A bounded search of ordinary JSON envelope and whitespace representations found **one exact original v1 SHA-256 match** for adversarial: `{"version":"1.0","split":"adversarial","items":<the preserved 20 records>}`, serialized with Python `json.dumps(..., indent=2, ensure_ascii=False)`, followed by LF. That byte sequence hashes to the original manifest's `e1510589bbf468d4c4d94a9587ac5dba9cd3284370efe9dae6779d043ccbc5cf` and is now restored at `data/phenotype_battery/pretorius_adversarial_v1.json`.

The same declared serialization search found **zero** exact v1 hash matches for the 100 axis-grounded training records and 40 validation records. This is a bounded search result, not proof those surviving records differ semantically from their missing originals. The 30 legacy training items have not been located. Consequently 001 remains **source-blocked** and no six-seed numerical experiment has been run.

## 2026-10-09: Inverted stride partition recovers original validation

Inspection of the four v0.4 train parts found the indices originally distributed round-robin: the first element of each part, interleaved, yields `sovereignty_help_train_01` through `_05`, followed by `recognition_train_01` through `_05`; validation likewise interleaves the expected pair of records for each of 20 domains. Concatenating the four files, as the v0.4 loader does, changes the original source ordering. The audit now checks both the concatenated order and the stride-inverted order, with no record contents or target annotations modified.

The inverted validation order, combined with the ordinary v1 envelope `{"version":"1.0","split":"validation","items":...}`, two-space-indented UTF-8 JSON with a final LF, exactly matches the original v1 validation SHA-256: `54aeab2e4ab0bde7f9a4c1acdaad062731657f2a664b5e3cdf5409636b5d1246`. CI wrote and committed the matching 40-item original `data/phenotype_battery/pretorius_validation_v1.json`. Both original v1 evaluation datasets, validation and adversarial, are therefore now **byte-authenticated and available**. The 20-item terminal remains unexamined.

Neither concatenation nor the justified inverted order matched the original 100-item training file SHA-256 under the bounded JSON-format candidates. The 100 records are preserved as historical candidates, but not certified as the original v1 source file. The 30 legacy training items remain missing entirely. Experiment 001's `130 train / 40 validation / 20 adversarial` reproduction is still **blocked**, not failed; the original neural and decoder experimental scores have not been generated.
