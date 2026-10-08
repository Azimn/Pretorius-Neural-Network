# BC01-D4: sensory-to-teaching drive ratio control

**Status:** protocol fixed before inspecting D4 trial outcomes, 2026-10-08.
**Parent:** [Issue #26](https://github.com/Azimn/Pretorius-Neural-Network/issues/26).
**Engineering target:** [D4 Issue #33](https://github.com/Azimn/Pretorius-Neural-Network/issues/33).

## Motivation and testable mechanism

BioCircuit D3's unchanged recurrent donor has 450 validated **reconstructed** Pretorius memories, 16 developer-interpreted action cards (four of each class) and no proven useful source-conditioned recurrent policy. Source-verified L2 v2 D3 target-local accuracy was 20.14%, exactly the same as no training, shuffled labels, blank-cue and restoring the pre-card recurrent W. An offline direct-drive audit found action-teaching column norm approximately 6.24 against source sensory cue 0.533. The measured drive-norm ratio is around 0.085, not a verified mechanistic cause. D4 isolates only the sensory amplitude relative to the fixed teacher.

## Frozen design

Use the original `PlasticRecurrentPersonaNet`, `biocircuit.prototype.Circuit`, fixed recurrent topology, action population layout, neural learning parameter `eta=0.03`, source cards and pre-card L2 v2 inputs. Shared feature provider is Pretorius-Connectome `7cd631f3b533203e39465ec31c4ee42945610e2f`, schema `pretorius.shared-features.v2`; frozen original source blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`, 450 events, 27 episodes. Each seed gets one L2 cache fitted on its own episode-disjoint training partition. Do not alter encoder vocabulary, L0/L1 corpus, neural topology, action teaching columns, motor decoder or labels.

Prespecified **card sensory gain values**: `1`, `4`, `12`. Multiply only the 256 sensory input components of each encoded card's training cue, and of its matched evaluation probe, by the same factor. Do **not** scale action teaching, reward, event ID (never used) or any non-sensory scalar. Leave the 450-event unlabeled background curriculum at gain `1` for every condition, so its initial recurrent W and homeostatic bias are identical for a given seed/mode. This isolates the effect of boosting source-card input during associative learning and testing, not the entire developmental exposure history. The `1` arm must reproduce D3 exactly in choice, score, physiology, recurrent W changes and input drive.

With seeds 31/37/43 and network configurations local/global/generic, this is **27 distinct conditions**, each with 16 development-only source cards. Exposures fixed: eight background ticks per 450 memories, three labelled-card epochs, 32 ticks per card per epoch, 32 probe settling ticks and 256 neurons. The targeted learner uses a matched same-fast-state no-teacher counterfactual for its local three-factor recurrent update; this uses more internal inference ticks than generic Hebbian, so report wall time and do not claim compute-matched efficiency.

For every gain, seed and mode run (a) source-anchored targeted recurrent learning, (b) original generic Hebbian update, (c) targeted with deterministically shuffled wrong teaching labels, (d) targeted with blank source sensory cues, (e) no card learning, and (f) targeted with all card-learned recurrent W reverted while keeping frozen original input and readout parameters. Record exact per-source question, fixed interpretation label, target/action confusion matrix, population rates, saturation, cue-dependent RMS state variance, absolute recurrent synapse changes, sensory/teacher drive norms and ratio, and exact checkpoint restart.

Two independent Python processes with `PYTHONHASHSEED=17` and `71337` must emit exactly the same per-condition neural results and source/encoder hashes. Wall times are measured by `time.perf_counter` for each trial, and **excluded from equality** because runtime elapsed seconds inherently vary. Raw machine-readable per-case output and failed runs must remain permanently in GitHub (not only expiring Actions artifacts).

## Failure and decision rules

A change in class histogram, better training-card accuracy, greater action-rate separation, large recurrent delta, or score-dependent lesion effect is not enough. Source-conditioned recurrent improvement would require targeted accuracy **above each matched gain's** shuffled labels, blank-cue training, no-training, generic Hebbian and recurrent W-lesion outcomes, with >1 predicted action, across seeds. A positive direction would **still be development-only** and require new independently reviewed held-out and contradiction-sensitive probes, not the already reused D1-D4 questions. Do not treat 27 correlated seed/mode/gain trials as 27 independent subject replicates. Preserve negative cases and stop any production inference if this gate fails.

The definitive Pretorius in `The-Doctor-Lives` is unaffected. The independent FlyWire neural tests are unaffected.
