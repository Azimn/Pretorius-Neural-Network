# Experiment 001F: The Autobiographical Echo, completed results

**Completed 2026-10-10.** [Protocol](../../research/CHIMERA_001F_AUTOBIOGRAPHICAL_ECHO_PROTOCOL.md), [source manifest](RUN_METADATA.json), [six-seed raw report](RUN_REPORT.json), six individual `seed_<seed>.json` records. The complete [CI run 38058087497](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/38058087497) passed source checks, smoke, six-seed compute, evaluation integrity contract and output archival.

This is a **new exploratory experiment**, not reconstruction of historical Experiment 001, and not a validated behavioral holdout. The autobiographical source consists of 450 deliberately reconstructed fictional memory records, SHA-256 `becdf4ca72b85365c35940ccd68f3b6a9bbb1093b7b3cc89c33b113bbe5608ec`, committed in Pretorius-Connectome at `5364f43dfe3c192b6c13b7bf373405ee7a41b420`. Every condition uses those same exact texts. The only scoring uses the old, **previously exposed** 40 validation and 20 adversarial cards and their historical labels.

Six paired seeds (`1842,101,202,303,404,505`), 1,024 neurons, 12,000 phenotype-foundation training steps from a separately authenticated 100-example v0.4 historical source, then **11,700 unsupervised memory replay steps per condition**, 450 memories × 26 epochs. Replays encode narrative memory/observation/decision/consequence/belief/relationship *text* with the existing fixed signed lexical hasher, neutral environmental scalars, **zero action teaching and zero reward**. The trained motor decoder remains exactly unchanged in all replay conditions. Tests assert frozen exposure cannot modify recurrent W or homeostatic bias and produces exactly the original score.

## Observed results

| Condition | Mean validation JS similarity | Mean adversarial JS similarity | Recurrent W L2 change from post-phenotype | Homeostatic bias L2 change |
| --- | ---: | ---: | ---: | ---: |
| No memory replay | 0.854291551 | 0.841098451 | 0 | 0 |
| Frozen chronological replay (`learn=False`) | 0.854291551 | 0.841098451 | **0** | **0** |
| Plastic chronological (`learn=True`) | 0.854055558 | 0.840922775 | **0.140921** | **0.010838** |
| Plastic reverse chronological | 0.854055529 | 0.840922725 | 0.140921 | 0.010840 |
| Plastic shuffled each epoch | 0.854055499 | 0.840922780 | 0.140921 | 0.010845 |

**Primary paired validation effects, plastic minus frozen:** chronological **−0.000235993**; reverse **−0.000236022**; shuffled **−0.000236053**. Chronological-minus-reverse **+0.0000000285**; chronological-minus-shuffled **+0.0000000594**. All six seeds had a small negative old validation difference for plastic chronological versus frozen replay. Sequence ordering effects were many orders smaller than the total deviation and are essentially indistinguishable at the precision of this phenotype metric. The old adversarial diagnostics show the same slight downward trend.

The source and test gates were verified at each seed. Neural W and recurrent homeostatic biases **did** update under zero-reward unsupervised text exposure, as shown by paired norms and tensor hashes. The motor decoder is byte-identical throughout. The frozen chronology and no-replay conditions agree exactly per seed on both existing diagnostic splits, so a change in presentation-time fast neural state alone is not being misread as trained memory.

## Scientific interpretation

001F demonstrates **generic activity-dependent synaptic change** from the narrative corpus but **does not demonstrate** that the content of the autobiography has been encoded into an accessible, order-sensitive identity representation, or that the intervention improves observable Pretorius-specific action selection. A 0.000236 deficit on an old exposed JS metric is small and may be a benign distributional perturbation, not damage to a character; do not describe it as such. The 0.00000003 order differences are too tiny to merit claims of a robust chronological mechanism.

The current signed lexical hashing removes much semantic structure. The experiment trains neither temporal credit assignment nor episodic indexing, and trains no readout for autobiographical features. Repeating a fixed set of 450 memories 26 times in a rate network with local Hebbian updates is **not** a mechanistic replica of remembering episodes in human cognition. Some order effects may still be hidden by the present metric. Direct content-selective probes with matched distractors and a readout gate are an appropriate next experiment, with genuinely fresh independent labels if character consistency is claimed.

**Do not tune parameters to the exposed historical validation/adversarial cards.** Freeze these negative results and preserve the source hashes, per-seed outcomes and terminal exclusion. The 001E counterfactuals remain unscored, unlabeled and blind-review-gated; no Hugging Face data were used. The historical Experiment 001 original 130-item training input is still incomplete, and no part of 001F is a reproduction of it.
