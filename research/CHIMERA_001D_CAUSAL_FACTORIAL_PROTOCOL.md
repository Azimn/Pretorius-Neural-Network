# Experiment 001D: Synaptic and decoder causal-factorial decomposition

**Protocol:** `chimera-001d-isolated-synapse-bias-decoder-components-v1`. **Evidence class:** exploratory mechanism ablation on old exposed evaluation items. Historical Experiment 001 is still source-blocked.

## Why this is necessary

001B yielded validation JS similarities of 0.854292 for intact, 0.853628 for virgin recurrence + trained decoder, 0.853672 for virgin recurrence + equally fitted fresh decoder, and 0.858388 for a no-neural training-label marginal prior. Mean *trained vs virgin recurrence with matched fresh decoder* was -0.000502. These results suggest that the output decoder and action base rates dominate the coarse JS metric, but the previous recurrent transplant grouped learned sparse recurrent **W** with homeostatic **bias**. A measurable change in W was not independently shown to be causally responsible for predictions.

001D fixes that confound with a true static-parameter **2×2 sparse recurrent W × homeostatic bias** factorial, while holding the already-trained motor decoder fixed. It also separately tests the trained motor weight matrix and the motor bias vector in a second 2×2 factorial on fully mature recurrence. All four parameters have exact independent SHA-256 fingerprints and nonaliased copied arrays. Fixed Win, sparse topology, neuronal signs, action populations, training procedure and random initialization are unchanged. Plasticity traces and fast state are not used as learned variables during evaluation; each probe resets fast state before scoring.

## Prespecified conditions

Recurrent factorial, *trained motor decoder held constant:*
- `r00`: virgin W, virgin recurrent bias
- `r10`: trained W, virgin recurrent bias
- `r01`: virgin W, trained recurrent bias
- `r11`: trained W, trained recurrent bias (intact reference)

Motor decoder factorial, *trained recurrence W+bias held constant:*
- `d00`: virgin motor weights, virgin motor bias
- `d10`: trained motor weights, virgin motor bias
- `d01`: virgin motor weights, trained motor bias
- `d11`: trained motor weights, trained motor bias (same intact reference; score equality must be exact to r11)

The primary synapse-only effect is **r10 minus r00**. Additional paired contrasts include **r11 minus r01** (W when biases are trained), **r01 minus r00** (homeostasis-only), and motor **d10 minus d01**. Per-seed component L2 displacements establish whether a purported lesion restored a genuine changed tensor rather than a no-op. No definition includes network state or output labels from another source.

For the intact model, two additional test-label-preserving **context reassignment probes** expose whether response scores track actual scenario/scalar context, above an action prior: global shuffled contexts in validation and adversarial; plus domain-pair swapped contexts in validation. Only `scenario` and `scalars` move, while item IDs, target distributions, evaluation split membership, model and noise remain fixed. Changes in JS similarity under context swap are *descriptive* on exposed fixtures, not a new independent holdout.

## Fixed source and execution

Same six seeds `1842,101,202,303,404,505`, 1,024-neuron architecture, 12,000 actual steps per seed, original v0.4 100-row train subset **at pinned commit `7b7eb19ad852dc016ee0d370528d903bab78a5cf`** with SHA-256-verified four split files, stride-inverted training order, the exactly v1-authenticated 40 validation and 20 adversarial records. Neither the missing 30 original legacy labels nor terminal data are manufactured or used. Compare the same previously exposed validation and adversarial items; neither is a fresh claim of generalization. Preserve six per-seed machine-readable `seed_<number>.json`, raw per-item scores, source/config digests and aggregate `RUN_REPORT.json` in `results/chimera_001d/`.

Use the same 001B learning code `PhenotypeCurriculum`, `PlasticRecurrentPersonaNet`, `ExperienceEncoder` and battery score unmodified; the only new code controls exact graft factors and context mapping. CI must gate a full 6-seed execution behind unit tests, independent source hashes and small-scale smoke, and commit raw outputs without changing 001B or historical Experiment 001.

## Interpretation gates

A small W-only effect relative to the trained decoder is a null **for this scoring protocol**, not a claim that no recurrent representations exist. A motor bias-only readout approaching intact scores implies output frequency is a stronger competitor than sophisticated neural dynamics. If context shuffles leave scores mostly unchanged, behavioral accuracy is likely insensitive to scenario details *on these data*. Distinguish motor biases from class priors; they are not assumed identical. Do not retrofit hyperparameters, redraw 40/20 source labels, or claim independent evaluation.

Any move to the 450-event Pretorius biography requires adjudicated behavioral labels, provenance, and genuinely new counterfactual scenarios. The 001C source audit verified 448/450 events belong to one memory-reference-connected component, so a basic episode-disjoint split is not memory-independent. A separate `001E` labeling protocol may be designed later; 001D does not invent labels.
