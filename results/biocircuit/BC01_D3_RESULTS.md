# BioCircuit BC01-D3: stable-source recurrent credit-assignment diagnostic

**Date:** 2026-10-08. **Source:** Pretorius-Connectome v12, 450 reconstructed source events, immutable Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`. **Versioned feature provider:** [Pretorius-Connectome PR #15](https://github.com/Azimn/Pretorius-Connectome/pull/15), source code commit `7cd631f3b533203e39465ec31c4ee42945610e2f`, `pretorius.shared-features.v2` with independent-process stable vocabulary tie-breaks. **Status:** negative exploratory in-sample neural development task; no claim of Pretorius identity or semantic memory.

## Why the first D3 evidence was invalid for a multi-run comparison

The originally shared **v1** TF-IDF builder gave different vocabulary and IDF hashes between independent GitHub Actions executions despite identical source and package versions. Those two runs also yielded different D3 policy accuracies. Their complete measured per-case results were explicitly preserved in `BC01_D3_V1_UNSTABLE_FULL_PER_CASE.json` and `BC01_D3_V1_UNSTABLE_LEGACY.json`, not retroactively edited or mistaken for stable findings. The root incident and corrected source owner tests are permanently documented [here](https://github.com/Azimn/Pretorius-Connectome/blob/main/docs/SHARED_MEMORY_L2_V2_REPRODUCIBILITY.md). This report presents **only version-pinned L2 v2** outcomes. It is not a blinded experiment.

## What D3 actually tested

The prior D1 global Hebbian rule broadcast learning through every recurrent synapse and D2 at tenfold gain collapsed to one action. D3 adds a local candidate to the existing `PlasticRecurrentPersonaNet` rather than introducing another network. After first-half cue exposure, the same fast state is used to simulate a no-teacher and action-teacher second half. Their paired postsynaptic rate difference gates an activity-dependent adjustment on existing recurrent edges arriving in the selected action population. The target action originates from a literal first-person source `decisions` field with a separately recorded, provisional human interpretation. Targets never enter evaluation cues. Topology, existing input matrix, sparse synapse structure, Dale-like constraints and motor decoder remain frozen.

For each of three seeds (31/37/43) and each of three modes (compartment-local, global and generic), the entire 450-event corpus is replayed as unlabelled background and the same 16 balanced action cards receive three supervised exposure epochs. Tests include the existing generic Hebbian learner, target-local learning, pre-card synaptic lesion, shuffled teaching labels, teaching with blank sensory cues and no training. All models receive equal labelled exposure windows; the targeted rule additionally computes an untrained counterfactual branch for 16 ticks, so **total inference compute is not matched**. Read [exact methods](../../research/BIOCIRCUIT_D3_PROTOCOL.md).

The baseline and D3 intervention were designed on these reused source cues and cannot count as generalization or semantic comprehension.

## Stable v2 results and negative conclusion

Preliminary pinned-v2 CI [run 37843003731](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37843003731) completed all 25 tests and nine experiments. Its replay stage initially failed because the primary run requested persistent checkpoints and the repeated run did not, so the JSON `checkpoint_name` metadata was different. An audit of both full raw artifacts found **no different neural trial fields whatsoever**, only this optional filename field on each trial. The workflow was corrected to request matching checkpoint metadata and to fail if any actual trial value diverges.

Across nine stable-v2 conditions, aggregate development accuracy was:

| Condition | Mean correct choice |
| --- | ---: |
| Existing generic Hebbian recurrence | 15.97% |
| Targeted three-factor recurrent update | **20.14%** |
| Targeted after pre-card recurrent-weight restoration | **20.14%** |
| Targeted with shuffled action teaching labels | **20.14%** |
| Targeted after blank sensory cue teaching | **20.14%** |
| No new card learning | **20.14%** |
| Four-class balanced chance | 25.00% |

Every targeted model selected three or four different actions across the 16 cards. Thus the pathological **single-action collapse of D2 did not recur**, but the decision distribution still contains no demonstrated beneficial, source-conditioned recurrent learning. The mean targeted synaptic change L1 was approximately 4.619. Restoring all targeted learned recurrent weights changed only **one of 144 probe choices**, without a reproducible accuracy benefit. No configuration passed the preregistered developmental causal-control gate. All targeted checkpoints reloaded exactly.

## New mechanistic input diagnostic

The fixed sensory input drive from the 16 prompts had mean L2 norm approximately **0.533**, compared with **6.241** for the separately measured action-teaching input. The mean sensory-to-teacher drive ratio was **0.0854**, i.e., sensory cues produced about 8.5% of the direct input-drive magnitude of the teaching channel. The mean across-cue neural activity RMS variation was approximately 0.00252. These are direct input-matrix/activation measurements, not semantic or biologically measured neural signal strengths.

This imbalance is a plausible **testable explanation** for weak cue-conditioned action learning, but not proof of causal failure. The effect could also arise from the recurrent update rule, saturation/homeostasis, fixed action readout or the undersized 16-case development assay. It would be premature to claim that equalizing sensory and teaching input norms will create autobiographical memory.

## Next engineering gate

A D4 experiment should **first** expose and hold fixed the ratio of sensory drive to teaching drive as an experimental variable, using the version-pinned L2 v2 input, unchanged source cue cards and strict label/lesion/blank-cue controls. Measure per-class *accuracy*, action diversity, source-specific recurrent lesion dependence and internal state separation, not only weight magnitude. Hold out new, independently reviewed narrative probes before any positive extrapolation. Preserve an unchanged matched v2 baseline. Do not silently transfer any experimental network into The Doctor Lives.

The complete per-case `BC01_D3_FULL_PER_CASE.json` and matched historical-input `BC01_D3_LEGACY_PER_CASE.json` are committed by the successful workflow directly to the research branch, then merged with code. The raw files and report remain accessible on GitHub beyond CI artifact retention.
