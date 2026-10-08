# BioCircuit BC00 | Executed functional prototype

**Run:** October 8, 2026. **Status:** Functional, tested, matched hardware-independent synthetic assay.  
**Evidence:** [BC00 GitHub Actions run 37832584866](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37832584866) with a downloadable `biocircuit-bc00-matched-4096` full JSON artifact. SHA-256 of the JSON: `964271f7299e40aeb6a9dd3c3eb3f8633b83c286e71a72110b36c7633fcab4b0`.

## What is delivered

An executable `biocircuit/prototype.py` kernel, runnable `scripts/run_biocircuit_bc00.py` CLI, checkpoint save/load, synapse lesion API, and deterministic unit tests. It is a synthetic **4,096-unit** sparse associative pathway with an outcome-gated trainable four-action neural readout. All 4,096 units recruit into four compartments for local inhibition, or compete globally in an otherwise exactly matched control. This is **not** yet a recurrent circuit, FlyWire anatomical simulation, autobiographical memory substrate, or live Pretorius.

Per condition the projection contains **49,152 identical fixed sensory edges**. Each model has **16,384 trainable action synapses**, exactly 128 active units per input, **512 event presentations**, and **262,144 counted synaptic parameter updates**. The two models use identical seeded stimuli, learned outcome labels, projection topology and plasticity rule. Only inhibitory competition differs. The 128 synthetic episodes each have a random 4-class outcome; a test variant corrupts roughly one third of the original cue features and introduces distractors. Three seeds are `31,37,43`.

## Actual results

The results below are unweighted mean accuracy across three runs. Higher is better except in intentionally destroyed-weight conditions. Since the 4 outcomes are random, the approximately 25% untrained/lesioned performance is a chance baseline, **not** a semantic generalization rate.

| Condition | Before training | After training on exact cues | After training, corrupted cues | All learned synapses lesioned |
| --- | ---: | ---: | ---: | ---: |
| Compartment-local inhibition | 25.52% | **96.61%** | 57.29% | 25.52% |
| Global inhibition control | 25.52% | 88.28% | **66.67%** | 25.52% |

Per-seed corrupted-cue accuracy, local: `0.460938`, `0.632812`, `0.625000`. Global: `0.695312`, `0.695312`, `0.609375`. Differences are small exploratory three-seed samples; they are not statistically powered.

The local-compartment circuit fit familiar training patterns more accurately but **lost** to the global-competition control on corrupted-cue retrieval on average. This should be recorded as a genuinely mixed finding, not a validation of a biological advantage. After ablating all learned action synapses, both revert to the original ~25.5% chance-level readout, confirming that those synapses carried the learned responses under this toy task. It does **not** establish causal necessity of a deeply recurrent character identity, which earlier experiments have not demonstrated.

The JSON runner reports `recovered_checkpoint_exact`, but this indicates exact restoration from a temporary in-memory weight copy during the assay; the separate `test_checkpoint_roundtrip_rejects_cross_mode` exercises actual `np.savez_compressed` filesystem persistence and reload. Do not interpret the assay field as a separate persisted restart experiment.

## Reproducible run

```sh
python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2'
python -m unittest discover -s tests_biocircuit -v
python scripts/run_biocircuit_bc00.py --neurons 4096 --items 128 --epochs 4 --seeds 31,37,43 --output results/biocircuit/BC00.json
```

The output file is a fully inspectable JSON artifact, not a screenshot or proposed simulation. A consumer can change `--neurons`, `--items`, `--epochs` and `--seeds` without modifying source files, subject to the compartment divisibility and sparse configuration constraints.

## Engineering and scientific interpretation

BC00 establishes end-to-end feasibility of a specialized sparse competition module with working plastic synapses, exact matched topology controls, destructive lesions, and reproducible checkpoints. The evidence does **not** establish that local compartment inhibition improves performance, that a fly-like connectome is best, or that Pretorius's personality has been embedded.

A more capable deliverable requires integrating **actual recurrent state**, training representations of Pretorius's real 450-memory frozen source, direct neural behavior readouts with memory-driven decisions, and independent semantic/truth testing. This should be BC01/BC02, reusing `persona_net` recurrence and The Doctor Lives donor mechanisms while preserving current checkpoints. Scaling to 65,536 units should be contingent on observed resource and learning behavior, not used as a substitute for a functional end-to-end demonstration.

The correct near-term objective is **a small runnable Pretorius who can demonstrate a learned choice and explain the source memory through evidence-backed retrieval**, not another standalone architecture RFC. Continue the actual FlyWire import/imprinting track independently.
