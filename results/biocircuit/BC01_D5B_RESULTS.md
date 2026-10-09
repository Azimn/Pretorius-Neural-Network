# BC01-D5B measured results: source-specific eligibility does not rescue recurrent learning

**October 8, 2026 research session (CI completed 2026-10-09 UTC).** [Frozen D5B protocol](../../research/BIOCIRCUIT_D5B_PROTOCOL.md) · [Source-verified independent replay CI](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37863157361) · [complete 18-trial per-case JSON](BC01_D5B_FULL_PER_CASE.json).

## The test

The D5A [diagnostic](BC01_D5_RESULTS.md) found that the old D4 presynaptic gating admitted 100% of incoming action edges at 1× sensory gain and ~97% at 12×. D5B made **one targeted change** to the existing recurrent learner:

```
old: max(cue_pre_rate - global_target_rate, 0)
new: max(cue_pre_rate - matched_blank_pre_rate, 0)
```

A matched blank first-half stimulus is simulated for the same tick budget from the same initial fast state; cue state and network tick are then exactly restored. Everything else is unchanged: action teacher vs same-timestep no-teacher response, source-grounded but provisional card target labels, action mask, L2 v2 frozen 8192-term TF-IDF model and BioCircuit 256-feature projection, existing 256-neuron recurrent topology, constant action readout, Dale signs, recurrent learning rate, checkpoint format and original 450 autobiographical source events. The new rule requires **16 additional blank-branch untrained ticks per learning card** (not compute matched to D4).

The old `target_rate` rule remains the default and its original results reproduce exactly in independent tests. The D5B corrected condition is compared with its own pre-card W-only recurrent lesion, deranged teaching labels, blank source cues during training, no card training and unchanged generic Hebbian. A separately supervised **motor-decoder-only** comparator trains output weights with recurrent weights frozen, to identify whether input information can be exploited through a conventional downstream classifier instead.

No targets, action labels or event IDs are allowed in neural evaluation inputs. The frozen full corpus is only reconstructed fictional autobiographical source material, not evidence of an actually lived experience.

## Measured 18-condition findings

Seeds 31,37,43 × configurations local/global/generic × gains 1× and 12×, with 16 reused development-only interpreted source decisions per condition (four actions equally represented; chance 25%). Mean percentages over nine matched seed/configuration cases per gain:

| Condition | 1× | 12× |
| --- | ---: | ---: |
| Original D4 fixed-action recurrent rule | 20.14% | 27.08% |
| D5B cue-minus-blank recurrent rule | **20.14%** | **26.39%** |
| D5B with all learned recurrent W restored to pre-card state | 20.14% | 26.39% |
| D5B with shuffled source teaching labels | 20.14% | 27.08% |
| D5B with blank-cue teaching | 20.14% | 26.39% |
| D5B motor-decoder-only comparison (supervised output weights, W frozen) | **26.39%** | **100.00%** |

The eligibility fraction dropped substantially:

| Input condition | Old D5A eligibility | New cue-vs-blank eligibility |
| --- | ---: | ---: |
| 1× sensory gain | 100% | 50.00% |
| 12× sensory gain | ~97.1% | 39.78% |

The cue-minus-blank learner successfully suppresses a large fraction of indiscriminately eligible edges. It does **not** improve source-conditioned action prediction. The D5B intact and W-only lesioned models have **identical accuracy** at both gains. In the high-gain seed31/global case, shuffled labels did *better* than the correct teaching labels (31.25% versus 25%). **Zero of eighteen** model configurations passed the strict recurrent-learning controls.

### Strong motor-decoder contrast — appropriately limited

At 12× sensory gain, the existing `PlasticRecurrentPersonaNet` using *supervised learned output decoder weights* classifies all 16 reused development cards correctly across nine seed/configuration combinations, with its recurrent W unchanged after the source background curriculum. This is a useful engineering demonstration: the encoded recall-cue features and recurrent activity are at least **in-sample decodable** by an output head. It is **not** evidence of synaptically stored autobiographical identity, biologically faithful memory, semantic understanding, independent novel paraphrase generalization or an unsupervised character. It has seen the interpreted decisions during motor learning, and its test cues are closely related to those training cues.

A decoder-only model can be useful as a future *behavioral delivery baseline*, but success in this non-sealed task cannot be attributed to source-specific recurrent plasticity. It should not be silently substituted for the original intended biological-memory hypothesis.

## Reproduction

The [source-verified Actions run 37863157361](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37863157361) passed **40 BioCircuit regression tests**, 18 full 450-memory matched source assays, original model invariance, correct-input/shuffled/blank/lesion controls, exact W and checkpoint persistence, and **independent process numerical replay** under two different `PYTHONHASHSEED` values. No score tuning on a withheld development set occurred; all 16 cards are reused and developer interpreted. Raw trained source-card scores, confusion matrices, corrected eligible edges and mean change, fixed D4 controls, held W and separately trained motor scores, source/manifest hashes, and exact time diagnostics are permanently committed in `BC01_D5B_FULL_PER_CASE.json`; temporary Actions artifacts additionally preserve per-model checkpoints.

Reproduce using `python -m unittest discover -s tests_biocircuit -v`, then the [D5B runner](../../scripts/run_biocircuit_bc01_d5b.py) with the source-owner [immutable L2 v2 provider checkout](https://github.com/Azimn/Pretorius-Connectome/commit/7cd631f3b533203e39465ec31c4ee42945610e2f) and seed31/37/43 split-specific caches, as scripted in `.github/workflows/biocircuit-bc01-d5b.yml`.

## Decision and priority

1. **Stop arbitrary gain/learning-rate or larger-network sweeps** for this BC01-D1–D5B action-learning protocol. The major measured defect is no longer just eligibility sparsity: the learned recurrent weight changes fail to produce beneficial differentiated choices even after cue-specific gating, whereas a separately trained output decoder fits all exposed development decisions.
2. Next scientifically justified work: inspect decoder–recurrent interaction and design *independently reviewed, never-trained* new source-cue paraphrases and conflict probes. Compare the decoder-only baseline against recurrence with the decoder frozen, random labels, and recurrent W-only lesion. This will distinguish useful, transferable recurrent memory from a trained static classifier, rather than re-labeling supervised decoder performance as biological memory.
3. Keep canonical Pretorius `The-Doctor-Lives` unchanged until an independently tested learned recurrent contribution is demonstrated and the production gate explicitly approves it. Retain both D5A's original negative diagnosis and D5B's failed surgical correction as immutable evidence.

**Scientific disposition:** engineered source sharing is operational; candidate rule improves *eligibility selectivity*, not source-grounded causal recurrent learning.
