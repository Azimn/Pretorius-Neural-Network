# BC01-D6A results: supervised decoder's 100% in-sample score does not transfer to withheld action labels

**Study:** 2026-10-08 local / GitHub Actions 2026-10-09 UTC. **Outcome:** negative developmental transductive source-decision transfer. [Frozen D6A protocol](../../research/BIOCIRCUIT_D6A_PROTOCOL.md) · [candidate paraphrase data](../../resources/biocircuit/bc01_d6a_paraphrase_candidates_v1.json) · [full source-linked per-case output](BC01_D6A_FULL_PER_CASE.json) · [original successful CI](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37868002509).

## Why D6A was necessary

D5B's **100% motor-decoder-only result** was measured after supervising the output decoder on all 16 source-linked action cards and probing highly related prompts from those same cards. A train/test label split did not exist. The result confirmed in-sample decodability, **not** generalized autobiographical memory.

D6A explicitly excludes each action-labelled event from the decoder's supervised training. Four stratified folds use 12 cards to train the output decoder (three/class) and **four cards to test** (one/class). All 16 cards appear exactly once in the test fold for each of three seeds (31,37,43) × three circuit configurations (local/global/generic). There are 36 seed/mode/fold evaluations and **144 held-out action-card decisions per prompt variant**, one original probe and one separately assistant-authored unreviewed candidate paraphrase for each event. The same existing 256-neuron donor, source-owned frozen L2 v2 TF-IDF + 256-feature BioCircuit projection, 450 reconstructed memories and high gain12 decoder protocol remain in place.

**Exposure limitation:** D6A's supervised action labels are held out, but the underlying first-person event texts still occur in the full 450-event background neural curriculum. The source-owner L2 v2 seed-specific fitted lexical vocabulary is **not** guaranteed to exclude held-out source episodes, and all 36 folds have a held-out episode also represented among training-labelled cards. Thus this is a **transductive, development-only action-label holdout**, not independent episode/source generalization, a blinded challenge, or externally reviewed semantic inference. The assistant-authored paraphrases have not been independently adjudicated and may be ambiguous.

## Measured results

Each evaluated card's class among {challenge, create, cooperate, approach} is balanced; uniform chance = **25%**. The previous supervised decoder-only high-gain result on exposed cards was **100%**. D6A's held-out-card mean scores are:

| Model, training/source control | Original held-out probe | Candidate paraphrase |
| --- | ---: | ---: |
| Supervised output decoder, recurrent W frozen | **18.75%** | **20.14%** |
| Same frozen supervised decoder + newly trained D5B cue-specific recurrent W | **18.75%** | **20.14%** |
| Same frozen decoder with every new card-trained recurrent W change lesioned | **18.75%** | **20.14%** |
| Supervised output decoder trained on wrong permutation of action labels | 23.61% | 23.61% |
| Fixed untrained recurrent action-population readout | 26.39% | 18.75% |
| Lexical retrieval of only the 12 label-train cards, abstaining on zero overlap | 0% | 0% |
| Decoder with confidence threshold fitted on TRAIN cues only (abstain counted incorrect) | 0% | 2.08% |

**Recurrent learning benefit:** across all **36 folds per variant**, not one obtained higher action accuracy from new recurrent W relative to its matched W-only lesion. These weights changed (mean source-card-trained recurrent delta L1 approximately 0.575) but their behavioral contribution was negligible on the held-out labels. The original 100% in-sample decoder score is not a useful forecast of unseen-card decision accuracy.

The wrong-label decoder also performs about as well as or better than the true-label decoder in this crossvalidation, indicating the originally learned target association may be specific to exposed cue strings and is not reliable even at the existing lexical-discrimination level. These are correlated seed/fold evaluations across a *tiny, human-interpreted* source set, not a formal inference of a population-wide effect.

### Abstention and unknown controls

The simple abstention heuristic uses a confidence cutoff derived from the bottom 10th percentile of confidence on the **training cards only**. It abstains on all 144 non-source modern-topic unknown-control examples evaluated across 36 folds, but also abstains on essentially **all held-out valid source decisions** (0% answer coverage on original heldout probes; 2.78% on candidate paraphrases). It therefore has **no demonstrated practical discrimination ability** for valid unseen source decisions.

Lexical nearest training-card retrieval gave **zero correct held-out labels** on both variants. On the original probes it attempted an answer for 18.75% of cases and abstained otherwise; on the newly authored paraphrases it abstained on every case. This illustrates near-total absence of transferable cue overlap in this tiny development set, not any principled semantic uncertainty.

The motor decoder mean maximum action probability on the four fixed non-source unknown topics was around **0.259**, close to a uniform four-action score but not a calibrated posterior; the train-fitted threshold abstained on all 144 unknown probe records. An unknown-only abstention rate cannot validate generalization, particularly when all legitimate held-out questions are mostly rejected.

## Reproducibility

Original [D6A CI run 37868002509](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37868002509) passed **44 BioCircuit regression tests** (including 4 D6A leak/split tests), pinned source-owned L2 v2 cache builds, nine runs × four folds, separately checkpointed supervised decoders, exact W-invariance and frozen decoder comparators, and **full independent-process numerical replay** with a different `PYTHONHASHSEED`. Only elapsed seconds are ignored when comparing replay outputs. All 144 source-card probes per variant, per-condition scores, macro-F1 and confusion, label/train lists, confidence, OOD unknown controls, model weights/checkpoint metadata and source/encoder/candidate hashes are retained in `BC01_D6A_FULL_PER_CASE.json`. The original CI originally labelled `heldout_card_evaluations` as 36 despite aggregating four cases per fold; the corrected report schema explicitly records **36 fold evaluations and 144 held-out card decisions per variant** with the identical numeric accuracies.

The original 450 fictional source events, decision-card interpretations, source-owned L2 provider, D4/D5A/D5B code and The Doctor Lives are **unchanged**.

## Decision and follow-up

The apparent supervised advantage from D5B was primarily **in-sample output-decoder fitting**, not even robust transfer to unused supervised action-card labels. New recurrent W does not rescue this gap. **Neither decoder-only nor recurrent BioCircuit is ready for a definitive Pretorius.**

Continue [D6 Issue #38](https://github.com/Azimn/Pretorius-Neural-Network/issues/38) toward a fully reviewed **episode/source-disjoint** test. Before claiming independent generalization, another reviewer must adjudicate multiple candidate events and paraphrases; freeze source/action/negative controls prior to model scoring; remove held-out episode texts from *both* the unlabeled 450-memory background and the encoder fit; then test decoder-only, recurrent plus frozen decoder and W-lesion on a separate truly unseen source task. Do not tune gains, labels, thresholds or recurrence against the current evaluated D6A set and retroactively call that confirmatory.
