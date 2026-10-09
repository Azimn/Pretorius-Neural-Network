# Experiment 001: identity-surgery reconstruction and fresh-decoder control

**Attempt dated:** 2026-10-08  
**Outcome:** NOT REPRODUCED. Canonical input-data preflight is blocked; no Experiment 001 six-seed numerical run has been executed or fabricated.  
**Original narrative:** [Experiment log](../../EXPERIMENT_LOG.md), [Issue 1](https://github.com/Azimn/Pretorius-Neural-Network/issues/1).  
**Runner:** [run_chimera_001.py](../../scripts/run_chimera_001.py).  
**Pinned configuration:** [chimera_001.json](../../config/chimera_001.json).  
**Machine-readable status:** [STATUS.json](STATUS.json).

## Provenance gap

The historical Phenotype Battery v1 manifest, preserved in the `v0.4-recurrent-persona` branch, identifies 20 domains, 100 axis-grounded training rows, 30 legacy training rows, 40 validation rows, 20 adversarial rows and 20 sealed terminal rows. It gives source SHA-256 hashes, which the runner verifies before using any input.

This reconstruction branch recovers only the **hash-verified original** `pretorius_profile_v1.json`. The historical branch contains a file named `pretorius_adversarial_v1.json`, but its observed SHA-256 is `20f87fd29798be3540a67daec93d71175d019e5611e806c36f4263e196eb6a87`, **not** the original manifest value `e1510589bbf468d4c4d94a9587ac5dba9cd3284370efe9dae6779d043ccbc5cf`. It has therefore been **excluded**, not silently treated as canonical. The canonical training files `pretorius_phenotype_train_battery_v1.json` (100), `pretorius_train_v1.json` (30), validation `pretorius_validation_v1.json` (40), and original adversarial file `pretorius_adversarial_v1.json` (20) are missing from this reconstruction. The later `v0.4-reproducibility-payload` branch contains a 100-row training subset and a split 40-row validation set, but no complete original 130-item training presentation sequence or canonical byte-for-byte source files. Substituting its rows would be a *different* evaluation.

The historical log pins six seeds (1842, 101, 202, 303, 404, 505), 1,024 neurons and 12,000 phenotype-training ticks, but does **not** pin all historical network hyperparameter overrides or the order of the 100 plus 30 training rows. The versioned config copies `main`'s `config/default.json` values and labels the train ordering as an assumption. These are frozen reconstruction choices, not proof of an original-authored config. Restoring the inputs without resolving this ambiguity permits an attempted reproduction but not a byte-identical historical replay.

## Reproduction protocol

Run `python scripts/run_chimera_001.py --preflight-only` before expensive work. A missing file, hash mismatch, duplicate ID, invalid distribution or unexpected split count blocks the experiment before training. The code never loads the terminal battery or key.

Place the *original hash-matching* files under `data/phenotype_battery/` before executing `python scripts/run_chimera_001.py`. The pinned config defines every current network parameter, fixed seed list, evaluation noise, settle/probe ticks, dataset hashes and exact training budget. The existing `PhenotypeCurriculum`, `PlasticRecurrentPersonaNet`, `run_items` and `score_results` implement training and Jensen-Shannon scoring.

Within each seed, an untouched founder and identically initialized trained sibling are separated. The four readouts copy full recurrent state and input projections from either the trained or virgin sibling and copy motor weight and bias arrays from either sibling. Sources and destinations have distinct memory; there is no retraining during transplant. Identical fixed evaluation noise is used across conditions.

Each completed seed writes `results/chimera_001/seed_<seed>.json`, containing the four validation and adversarial results, 40/20 item-level scores, fingerprints of the transplanted components and actual network-step counts. `RUN_REPORT.json` retains four six-element distributions, the descriptive mean of each, and deviation from the narrative numbers. Only if **all four** six-seed validation means fall within ±0.01 of the historical values (0.7791, 0.8545, 0.7791 and 0.8546 in virgin/virgin, trained/trained, trained/virgin, virgin/trained order) may Phase 2 run. If any fails, the runner writes mismatch results and exits with code 3, without hyperparameter tuning.

## The matched fresh-decoder control

Phase 2 starts a truly untrained motor decoder on the *frozen mature recurrent network*, and applies the same 12,000 training step schedule, using the original PIS target distributions. The only permitted learning calls are to `learn_motor_distribution`; recurrent and input weights, homeostatic biases and eligibility must remain byte-for-byte unchanged. Each seed also trains a fresh decoder on the same **virgin** fixed recurrent network with the same 12,000 steps. This second arm is required for a valid inference: a supervised decoder can learn a phenotype from fixed random features even if *no phenotype-specific information was acquired by recurrent plasticity*.

Report intact-vs-fresh and, critically, the six paired `trained_recurrent_fresh_decoder minus virgin_recurrent_fresh_decoder` validation differences, with adversarial outcomes. A fresh mature-core decoder reaching intact scores alone does **not** demonstrate character information specific to the mature recurrent synapses. Conversely, failure to reach those scores in this single decoder-fitting setting would not prove an information-theoretic absence of recurrent representation. This experiment isolates the practical decoder-recovery question, not a universal impossibility theorem.

## Current release rule

Do **not** revise CC-E01's journal provenance classification to "reproduced from pinned config" yet. There are no validated six-seed results or historical input files to substantiate that upgrade. After the Phase 1 gate actually passes, append a dated source-log entry, commit the six per-seed files and exact config/runner hashes, then amend the journal evidence register with the immutable commit SHA. If it fails, preserve the documented mismatch as a negative reproduction outcome and keep CC-E01 narrative-only.
