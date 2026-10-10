# Experiment 001F: The Autobiographical Echo

**Protocol:** `chimera-001f-autobiographical-unsupervised-replay-v1`  
**Evidence status:** preregistered executable exploratory intervention, without independent behavioral labels.

## Motivation

Experiment 001D detected a tiny causal effect of acquired recurrent synaptic weight changes (+0.001011 validation JS similarity with the learned decoder fixed) versus a much larger decoder contribution and a simple class-prior model that outperformed the intact network. That test used just 100 existing labeled situational examples. Meanwhile our canonical `Azimn/Pretorius-Connectome` v12 archive contains 450 first-person, reconstructed episodes with decisions, observations, consequences, beliefs and relationship changes, all carrying `provenance=reconstructed`. These memories belong to a fictional character's curated biography and are **not validated human experiences, spontaneous memories or ground-truth action labels**.

001F is the first *neural developmental exposure* directly using that archived autobiography, not another synthetic trait-target update. It asks whether equal-budget replay of those 450 texts changes the network's recurrent weights and output diagnostics depending on event order and Hebbian plasticity. Success is **not** equivalent to stable identity or generalizable learned phenotype, because the scoring fixtures remain exposed, and the archive itself is a synthetic editorial construction.

## Frozen sources

- `Azimn/Pretorius-Connectome` commit `5364f43dfe3c192b6c13b7bf373405ee7a41b420`
- `memories/current/Pretorius_v12_450_Events_Complete.jsonl`, exactly **SHA-256 `becdf4ca72b85365c35940ccd68f3b6a9bbb1093b7b3cc89c33b113bbe5608ec`**
- 450 unique event IDs, exact chronological permutation 1 to 450, `provenance=reconstructed` for all
- Text-only input from `memory_text`, `observations`, `decisions`, `consequences`, `belief_changes`, and `relationship_changes`, concatenated in that order.
- Earlier 001B v0.4 100-row phenotype curriculum pinned to commit `7b7eb19ad852dc016ee0d370528d903bab78a5cf`, with four original part checksums and 12,000 foundation steps per seed.
- Exactly recovered 40-item v1 validation and 20-item adversarial inputs and old labels are used ONLY as **previously exposed, nonconfirmatory diagnostic scoring**; the source-scoring manifest remains unchanged.

The encoder is the existing 512-dimensional signed lexical feature hasher with the same environmental and action channels. No `action` is supplied during autobiographical replay, all scalars default to zero, and `reward=0`. The neural motor decoder is **never trained** on memory text. This is generic recurrent plasticity under unlabeled environmental input, *not* ground-truth imitation of the `decisions` field. No 001E drafted dilemma is passed to the model before independent human assessment.

## Prespecified matched conditions

Each of the six seeds `1842,101,202,303,404,505` starts from the identical within-seed, fully 100-row phenotype-trained neural state (1,024 neurons, 12,000 training steps). For each seed, score:

1. **No replay:** exact phenotype-trained clone, evaluate without autobiography input.
2. **Frozen chronological replay:** all 450 records in archive `chronological_order`, 26 complete passes, 11,700 actual neural steps with `learn=False`. Tests whether mere recent recurrent activation/state, which is reset before evaluation, explains outcome changes.
3. **Plastic chronological replay:** same exact input sequence, 26 passes, 11,700 steps with Hebbian/homeostatic plasticity `learn=True`, zero external reward.
4. **Plastic reverse:** reversed 450-event order, 26 complete passes, identical event exposure and budget.
5. **Plastic shuffled by epoch:** all 450 same events once per epoch under deterministic seed-indexed random permutations, 26 complete passes, same budget.

The total count is exactly `450 × 26 = 11,700` so each condition sees every autobiographical event 26 times. Chronology and reverse still differ in which event is most recent. This remains a *recency/order confound* for interpreting any differences, not a perfect abstract causal isolation of sequence. No replay arm is the same time budget and hence should be compared cautiously; frozen chronological matches the exposure budget.

All conditions reset fast neural state before evaluation. The existing scorer resets it on each diagnostic scenario. The neural implementation and motor decoder are identical; repeated text is supplied only through `ExperienceEncoder` signed-hash channels. Hash each learned `W`, recurrent bias and decoder. Source files and read-only export have digest/ID/count/provenance checks.

## Endpoint and negative-control contracts

Primary descriptive endpoints are paired per-seed differences of old validation and adversarial JS similarity between plastic chronological and matched **frozen chronological** replay, plus chronology-minus-reverse and chronology-minus-epoch-shuffled paired effects. Those are not prospective phenotype-confirmation endpoints because the evaluation cards and labels already informed earlier development. Also report exact before/after recurrent `W` and homeostatic bias L2 displacements. Confirm the decoder hash stays unchanged after all replay arms. Confirm frozen chronological produces *exactly* the no-replay evaluation output; reject the run if not. Preserve individual per-seed, per-item outputs and training step counts.

A significant difference in neural `W` establishes only learning-induced synaptic change, not memory accessibility or identity. If old phenotype JS improves, interpret it as an exploratory incidental change on previously exposed fixtures. If unchanged or degraded, do not claim the memory corpus is useless: unsupervised text, generic Hebbian updates and the old action decoder may be mismatched to narrative semantics.

## Release rules

CI must verify original source SHA, run all synthetic mechanics tests, perform reduced-budget **real 450-event** smoke and then six full-budget runs. Raw results saved to `results/chimera_001f/`, with separate metadata, six per-seed JSON and an aggregate, plus no sealed terminal contact. No retrospective parameter tuning based on results. A subsequently developed memory retrieval/readout task must distinguish episodic recall from scalar output alignment and include a fresh independent evaluation if making a character continuity claim.

No modification to Experiment 001, earlier 001B/001C/001D/001E results, or Pretorius-Connectome's source archive.
