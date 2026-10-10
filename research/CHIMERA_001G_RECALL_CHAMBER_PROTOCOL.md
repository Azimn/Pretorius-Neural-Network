# Experiment 001G: The Recall Chamber, source-specific neural discrimination

**Status:** New source-pinned, executable exploratory protocol. Not independent semantic recall, a renderer evaluation, or an Experiment 001 reproduction.

## Why this assay exists

001F exposed the same recurrent network to 450 reconstructed first-person Pretorius memories 26 times, allowing unsupervised synaptic change. The learned W matrix moved approximately 0.141 in L2 norm, but the previously exposed action battery got slightly worse and chronology-versus-shuffled differences were effectively zero. That battery could not determine whether even modest source-specific *representation* survives in recurrent activity.

001G changes the **measurement**, not the model. The core question is whether source-authored memory cues retrieve their editor-associated source in the network's isolated hidden activity more reliably *after* the 450-memory replay than in a matched frozen recurrence. The experiment compares learned recurrence with a W-only lesion that holds the learned homeostatic bias fixed, with a bias-only lesion that holds learned W fixed, and against **pure sensory-space nearest-neighbor** ranking without any neural weights.

**Crucial distinction:** This is a **cue-to-source association benchmark**, not unaided engram readout. The experiment explicitly represents both cue text **and every candidate memory text at query time**, allowing their response vectors to be compared. A non-neural lexical scorer and an existing external memory index can perform the same task. Even success would prove only improved *conditional source discrimination* under this protocol, not that Pretorius can autonomously recall or narrate the source without external content or that a memory is stored within recurrent weights.

## Source ownership and reuse

Canonical **450** reconstructive autobiography from `Azimn/Pretorius-Connectome` commit `5364f43dfe3c192b6c13b7bf373405ee7a41b420`, source SHA-256 `becdf4ca72b85365c35940ccd68f3b6a9bbb1093b7b3cc89c33b113bbe5608ec`. The original BC01 `biocircuit.bc01.load_corpus` performs byte-level Git blob and source-event/order/provenance verification, so no second source normalizer, memory schema or repository is invented. The 001F input loader remains authoritative for the 100-item phenotype foundation, original validation/adversarial input digests, narrative-feature construction and actual replay function. Shared TF-IDF caches and frozen BC lexical caches already belong to Pretorius-Connectome/BioCircuit; 001G does **not** copy or refit them.

The cue associated with each event is the concatenation of its **first two `recall_cues`**, which are editorial candidates already present in the v12 archive. Their correspondence to the record is **not independently annotated behavioral truth**. A cue may be repeated across records, may be inferred from the same prose being tested and may be an incomplete or unreliable reminder. Identifiers matching event-ID syntax are forbidden in any cue. No event ID, known motor action label, rendered answer, original Experiment 001 terminal item, 001E new situation, or source-provenance field is input to the neural cue encoding.

The 450 candidate representations come from the already implemented 001F encoder of `memory_text`, `observations`, `decisions`, `consequences`, `belief_changes`, `relationship_changes`. All vectors contain only lexical sensory dimensions, with zero action and zero scalar channels. Their input construction is identical for every control condition.

## Matched conditions

Six paired seeds `1842,101,202,303,404,505`; 1,024 neurons, existing 100-example historical v0.4 phenotype foundation, 12,000 neural steps, and 450 reconstructed events in 26 chronological passes, exactly 11,700 replay steps in **each** of the frozen and plastic arms.

- **Frozen replay**: same 11,700 input presentations, recurrence learning disabled; control for content exposure and no learned W.
- **Plastic replay**: learning enabled, zero reward, decoder completely frozen.
- **W-only lesion of plastic replay**: restore frozen W while preserving the plastic homeostatic bias.
- **Bias-only lesion of plastic replay**: restore frozen homeostatic bias while retaining learned W.
- **Signed lexical no-neural baseline**: nearest candidate cosine on the *very same* encoded cue/text sensory channels before any neural transformation.
- **Existing BC01 lexical retriever**: source-owned lexical cue candidate search using the pinned `Corpus`; its candidate result is explicitly `verdict=unknown`, not semantic support.
- **Same-episode target derangement**: evaluate the unchanged cue/candidate affinity matrix against a deterministic within-episode permutation of source labels, preserving episode distribution and forbidding target-self assignment.
- **Equal neutral key**: submit one arbitrary repeated cue ("neutral archival key") against all candidates; it cannot identify specific memories. This is a key-only null, not a recovery cue.

A neural embedding is the zero-input-blank-subtracted **mean firing-rate response** to stimulus over 20 inference-only ticks following 12 blank-settle ticks. Each probe resets fast state; `learn=False`. No decoder weights or action labels enter the readout. It uses the same parameter implementation `PlasticRecurrentPersonaNet` from 001F, not a new architecture, decoder or trained retrieval head. W-only and bias-only lesion SHA-256 invariants verify actual component isolation.

## Endpoints and interpretation thresholds

Record per-seed and source-event **exact top-1**, top-5, mean reciprocal rank, and **within-episode top-1 / MRR** across the 450-by-450 cue-to-record affinity matrix. Ranking uses stable tie-breaking. Report per-record identities, source provenance, target/source permutation scores and all motor-decoder fingerprints. The null full-archive random exact match rate is 1/450, but real cues are *not independent random text*. Equal keys and label derangement are empirical matched controls.

The primary paired differences are plastic minus frozen, plastic minus W-lesion, plastic minus bias-lesion, and plastic minus no-neural lexical, evaluated independently across all six seeds. A *strong positive neural-specific claim* would require all three of: (i) consistent substantial (at least **+0.05 absolute top-1**) improvement over frozen, (ii) at least **+0.05** above the W-lesion, and (iii) at least **+0.05** above the direct lexical baseline, with the same matched cue targets. This threshold is a prospective conservative **engineering screen**, not a p-value or sample-size-powered inferential criterion. A smaller positive effect is reportable as exploratory but not sufficient for a useful neural recall claim.

Even clearing that screen cannot imply independent semantic recall because candidate labels are source-editorial, candidate documents are supplied during test, source exposure is transductive, and the lexical hasher may directly encode the same words as the source cue. A strong BC01 lexical retrieval result is an external **information retrieval** result only, not proof that recurrence learned.

No alterations to any 001B–001F raw results, no retroactive reclassification of CC-E16, no source corpus editing, no weights migrated into The Doctor Lives. No external Hugging Face material is necessary for this targeted neural-control gate.

## Next decision

If the neural gain fails, treat 001G as a **negative mechanism assay**, not as a failure of Pretorius's underlying fictional identity. Do not fit a cue-specific neural decoder on these 450 source/answer pairs and then cite its in-sample accuracy as learned memory. Instead, test a preregistered **source-provenance-gated external retrieval → subject-access integration** with fresh reviewed cases, as governed by the existing Synthetic Ipseity and source ownership projects. Explicitly distinguish external autobiographical retrieval from any endogenously learned neural contribution.

If the neural gain passes, follow with different human-authored cue paraphrases, held-out source events, wrong-subject provenance, unrelated document matching, and reversal of cue-label assignments before making any autobiographical identity claim.
