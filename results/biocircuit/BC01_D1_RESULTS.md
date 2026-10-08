# BioCircuit BC01-D1: grounded decisions, first three-seed evaluation

**Date:** 2026-10-08. **Run:** [GitHub Actions 37836147694](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37836147694). **Status:** all 14 tests and 9 benchmark conditions passed, but the **behavioral learning hypothesis did not pass**. This is a development-set finding, not an independently confirmed assessment. The compact numeric archive is [BC01_D1_SUMMARY.json](BC01_D1_SUMMARY.json); the Actions run has full per-probe JSON and nine neural checkpoints under `biocircuit-bc01-grounded-decision-assay` (temporary artifact, regenerable with the pinned script and source).

## Experimental subject and task

The frozen 450-event v12 fictional autobiography from `Azimn/Pretorius-Connectome` commit `597fb23473a60eecf2e1b50f79c22bfbea816be5`, raw Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`, is replayed as unlabelled context. Sixteen selected records are then exposed as cue plus interpreted action pairings. Each interpretation is stored next to the source's *exact* `decisions` field, and is rejected at load time if source wording is changed. Interpretations remain provisional, not historical ground truth. The four action classes are balanced: challenge, create, cooperate and approach, four of each.

No event identifier or explicit target action is encoded into the probe query. The action is supplied as efference-like training input only. Neural outputs come from fixed action population rates with the old motor decoder entirely untrained. A separate decoder-only control trains motor output weights while freezing the recurrent network. The external retrieval-only control has access to card cue-to-action annotations, so its perfect score is a useful upper bound for trivial lookup but not a fair neuron-only information parity comparison.

The assay used 256 neurons per condition, seeds 31/37/43, 8 unlabelled steps per one of 450 events, three labelled epochs of 32 steps per card, and 32 settling steps. Local and global share the same fixed input projection within each seed; the generic donor receives raw signed hashed lexical inputs, which do not match the transformed input distribution exactly. No semantic embedding, reward valence, or independent phenotype assessor is used.

## Primary negative finding

| Condition | Local mean | Global mean | Generic mean |
| --- | ---: | ---: | ---: |
| Intact recurrence | **22.92%** | **25.00%** | **27.08%** |
| Restore pre-card recurrent weights | 25.00% | 22.92% | 22.92% |
| No recurrent plasticity during card phase | 25.00% | 22.92% | 22.92% |
| Shuffled teaching labels | 27.08% | 27.08% | 29.17% |
| Frozen-recurrence trained motor decoder | 25.00% | 25.00% | 25.00% |
| External nearest cue-to-card lookup | 100% | 100% | 100% |

Mean of three seeds, 16 cue probes per seed, four candidate actions. Four-class majority and balanced chance rate is 25%. Differences this small have no evidentiary weight on such a small, development-exposed set; no significance test is claimed.

**Causal interpretation:** restoring only pre-card recurrent weights changed action selections on 2 to 6 of 16 probes, depending on seed and architecture. Thus the recurrent updates have a *causal influence on predictions*, unlike the previous preview whose selected actions did not flip. But their influence did **not** improve source-anchored policy accuracy reliably. A causal effect is not a beneficial learned representation. The learned synaptic delta L1 was about 154 to 165 at 256 units, but delta magnitude cannot demonstrate semantic memory.

**Decoder warning:** the decoder-only control scored exactly 25% across seeds; its readout collapsed to a single action on the inspected seed. This is an output-learning failure to diagnose separately, not evidence of recurrent superiority. The retrieval-only control reaches 100% because probe phrases deliberately reuse the source's lexical cues. A future evaluative task must test clusters and paraphrases that cannot be solved by this direct cue match.

**Checkpoint:** all nine conditions reproduced the selected fixed-readout scores exactly after save/reload. Neural weights and data provenance were validated, but independent semantic truth checking was not.

## Decision gate and next experiment

BC01's success requirement remains **OPEN**. This run demonstrates full-corpus integration and an inspectable, source-anchored behavioral failure. It does not justify production transfer into The Doctor Lives or a claim of character identity learning.

The immediate follow-up is a diagnostic sensitivity experiment with fixed label/probe records and tenfold increased existing Hebbian/reward plasticity coefficient, using unchanged neuron counts, seeds, input/decision budgets, no-plasticity/shuffled/lesion controls, and no claim of a new holdout. A gain increase producing higher raw accuracy without outperforming both its lesioned and shuffled controls is still a failure. Any parameter selected after viewing these outcomes is developmental and must be frozen before genuinely new, independently reviewed evaluation.

Future semantic adjudication belongs to BC02. No source-unsupported autobiographical text should be fabricated as an answer, and a source match remains `unknown` until evaluated for entailment.
