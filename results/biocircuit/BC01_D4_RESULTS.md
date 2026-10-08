# BC01-D4 results: sensory input gain changes choices, not learned autobiography

**Date:** October 8, 2026. **Status:** complete, negative mechanistic development experiment. No neural identity or semantic comprehension claimed. [Prespecified D4 protocol](../../research/BIOCIRCUIT_D4_PROTOCOL.md) · [full source-linked per-case evidence](BC01_D4_FULL_PER_CASE.json) · [green measured CI](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37847739209).

## Scientific question

D3's cue input produced only about 8.5% of the direct synaptic drive of its teaching-channel input. Could bringing the cue drive into rough parity with the teacher enable its learned recurrent synapses to recover cue-specific source-derived policy decisions?

D4 changed **one parameter at a time**: multiply only the 256 sensory cue input components at action-card training and evaluation by gains `1`, `4` or `12`. Action teaching signal, sparse recurrent topology, static sensory Win matrix, fixed input vocabulary, network capacity (256), action-population readout, targets, developmental exposure budget and all source data were preserved. The 450 unlabeled autobiographical background exposures were intentionally held at gain 1 in all conditions, yielding a common pre-card recurrent W state within each seed/topology.

Original source is 450 reconstructed v12 Pretorius events, blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`; all 16 action annotations are development-only human interpretations of literal first-person `decisions` strings. The source-owned L2 v2 TF-IDF encoder is pinned to Pretorius-Connectome `7cd631f3b533203e39465ec31c4ee42945610e2f`, which corrected the independent-process feature drift found in v1. No semantic embeddings are claimed.

## Measured evidence

9 matched seed/topology settings (31/37/43 × local/global/generic) per gain, 27 neural learning runs total, with the same four balanced action classes in 16 cards and 25% majority chance rate:

| Gain | Sensory/teacher direct-drive norm ratio | Targeted recurrent | W-only recurrent lesion | Shuffled teaching labels | Blank-cue teaching | No new card learning | Generic Hebbian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1× | 0.08543 | 20.14% | 20.14% | 20.14% | 20.14% | 20.14% | 15.97% |
| 4× | 0.34171 | 23.61% | 23.61% | 23.61% | 23.61% | 23.61% | 22.22% |
| 12× | 1.02513 | 27.08% | 26.39% | **27.08%** | 26.39% | 26.39% | 25.69% |

The 12× condition brought sensory and teacher direct-input magnitudes into approximate parity. Targeted accuracy rose from 20.14% to 27.08%, but the **shuffled-label control also reached 27.08%**, and the no-card-learning control reached 26.39%. This is not reliable cue-to-source-action associative learning. Any apparent 12× gain over lesioned/blank/no-learning controls corresponds to only one extra correct choice among nine runs × 16 repeated development probes (144 probe evaluations) and fails the matched shuffled-label condition. Across 1×,4×,12× there were respectively 1, 1 and 2 recurrent-weight-lesion choice flips out of 144 probes each. No trial passed all development gates. Targeted predicted between 3 and 4 different action classes instead of collapsing to a single class; this is not sufficient evidence of correct learning.

The mean L1 change in recurrent weights after action-card training was approximately 4.619 at 1×, 4.653 at 4× and 4.994 at 12×, confirming a numerically effective but behaviorally unvalidated intervention. All three gains passed exact checkpoint/restart and finite physiology checks. The targeted conditions required additional cue-only counterfactual computation relative to the generic Hebbian rule, so they are **not compute-matched** comparisons.

Wall-clock time for the nine trials of each gain in this CI environment was 10.790, 10.789 and 10.803 seconds respectively, excluding dependency setup, source cache construction and the independent full sweep rerun. These times are not a general hardware performance claim, and were appropriately excluded from cross-process numeric-equality assertions.

## Verification and durable artifacts

The clean [D4 CI run 37847739209](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37847739209) passed **33 BioCircuit regression tests**, exact D3 ≡ D4 1× smoke equivalence, source-pin validation, all 27 full-corpus conditions, exact restart and 27-trial equality under two fresh Python processes with different `PYTHONHASHSEED` values (after excluding wall-clock seconds). The first run's separate compute/learning stages passed but its results publication step encountered a non-fast-forward GitHub push because documentation changed while the workflow ran. The workflow now fetches and rebases before pushing, with no force push. The second run succeeded and committed `results/biocircuit/BC01_D4_FULL_PER_CASE.json` to the feature branch. All raw event-linked score distributions, per-class confusion, physiology and lesion controls live in GitHub beyond the 30-day checkpoint Actions artifact retention.

## Interpretation and next gate

**Result:** stronger cue drive influences network predictions, not demonstrably the correct source-conditioned learned recurrent mapping. Preserving D3 1× and improving sensory-to-teacher norm parity did not solve the credit assignment problem. Do not infer benefits to autobiographical identity, learned semantic memory, independent paraphrase generalization or the definitive Pretorius. D4 used the same provisional 16 training/evaluation cards across gains, so its outcome is development-exposed and correlated across 27 conditions.

**Next:** [BC01-D5 Issue #35](https://github.com/Azimn/Pretorius-Neural-Network/issues/35) instruments where the three-factor eligibility rule fails: cue-conditioned presynaptic response, teacher-versus-counterfactual postsynaptic response, eligible synapses by action class, sign clipping, net plastic change relative to direct input, and readout decision margins. Only after a measured bottleneck should D5 try a single targeted correction with W-only lesion, no-training, shuffled-label and blank-cue controls. Larger networks, new encoders and transfer into `The-Doctor-Lives` are not justified by D4.
