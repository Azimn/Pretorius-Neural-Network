# Experiment 001B: Pretorius corpus reuse and external contrast controls

**Status:** candidate protocol, not yet preregistered or executed  
**Initiated:** 2026-10-09  
**Scope:** new research experiment; **not** a replacement for the historical Experiment 001 reproduction.

## Research question and boundary

The historical Experiment 001 runner lives at `scripts/run_chimera_001.py`. It must remain blocked until the original 130-item training set is authenticated. The recovered v1 validation (40) and adversarial (20) batteries must not be relabeled as independently unseen evaluation data. Do not insert replacement rows into `config/chimera_001.json`, reconstruct the missing 30 rows from unrelated examples, or match model parameters against the historical 0.854x/0.779x figures.

001B instead asks whether observed phenotype performance depends causally on *Pretorius-specific learned recurrent changes*, as opposed to a supervised motor decoder learning an action distribution from essentially generic features. New data sources, labels, comparisons and endpoints must be versioned independently of the original 001 experiment.

## Source inventory and ownership

**Canonical Pretorius autobiography:** `Azimn/Pretorius-Connectome`, `memories/current/Pretorius_v12_450_Events_Complete.jsonl` (450 reconstructed first-person memories, 27 episodes). Relevant fields include `event_id`, `episode_id`, `memory_text`, `decisions`, `consequences`, `belief_changes`, `relationship_changes`, `provenance`, `source_anchors`, and `causal_inference_status`. Source and feature cache ownership stays in Pretorius-Connectome. Do not make a second mutable copy. Pin exact source Git commit and file SHA-256 at fixture creation. These are constructed character-history records, not observed psychological facts or independently scored motor targets.

**Source-owned candidate percepts:** the v12 450-event sidecars and cue registry under `memories/annotations/` provide candidate recall cues, not independently reviewed action labels or verified perception. Do not silently promote them to behavioral ground truth.

**Existing phenotype framework:** the 20-domain Pretorius Phenotype Battery and ten abstract action outputs `explore/challenge/approach/avoid/cooperate/dominate/create/persist/conceal/comply` remain the comparison vocabulary. The survived historical v0.4 four-part 100-row axis-grounded training subset may serve as a fixed *legacy development reference* after recording its own exact file hashes and original stride interleaving. Its missing 30 original legacy rows cannot be replaced inside Experiment 001.

**Potential external controls, subject to source/license review:**
- [PersonaChat / ParlAI](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat) or a pinned [PersonaChat Hugging Face mirror](https://huggingface.co/datasets/Cynaptics/persona-chat), for artificial persona facts and multi-turn conversational consistency. Native records are *not* ten-action Pretorius phenotype scores. Prefer the primary-source license documentation over mirror metadata.
- [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog), for dialogue-act and emotion annotations. Native labels are *not* equivalent to `ACTION` categories. Original data are reported CC BY-NC-SA 4.0: observe noncommercial/share-alike requirements and separately verify permitted redistribution.

Do not co-mingle external text with Pretorius's autobiographical claims or permanently embed unrelated personas in his production state. Store external controls as source refs, pinned revisions and derived experimental cards only where licensing permits.

## Experimental fixtures

Define a **source-adjudicated phenotype card** with immutable ID, linked originating `event_id`, `episode_id`, source commit/SHA, narrative cue and historical context, nonterminal scenario, optional environment scalars, proposed ten-action probability distribution and label-rater provenance. The `decisions` field may generate candidate labels; it does not by itself supply a validated ten-action distribution. Human reviewers should independently judge action choices without seeing model predictions, document disagreement, and mark unmappable decisions as excluded rather than forcing an action label. A scientific code path must refuse cards with missing/unsupported labels.

Fit encoder statistics on training records only. Prevent any exact event ID, paraphrase family, episode or later linked event from crossing train/validation/test boundaries. Prefer episode-disjoint train and independently authored, adjudicated challenge evaluations with targeted counterfactuals and source-truth/unknown cases. The old v1 validation/adversarial battery remains a *previously exposed comparability check*, not confirmatory fresh holdout. Persist the frozen splits and hashes before model evaluation.

For external controls, choose a distinct fictional persona with independently labeled behavior on an information- and target-entropy-matched card set, not superficial token-count matching. PersonaChat continuity is an additional **separate textual task**, not a direct JS-score comparator unless ten-action labels are separately adjudicated.

## Proposed causal factorial design

Keep the existing v0.3 `PlasticRecurrentPersonaNet`, `PhenotypeCurriculum`, `ExperienceEncoder`, `battery.score_results` and training laws unchanged for the first cohort. Pin 1,024 neurons, six seeds (1842, 101, 202, 303, 404, 505), 12,000 actual `net.step` training calls, and settle/probe values; these are **new 001B conditions**, not claims of original 001 provenance. Preserve exact paired random initialization, recurrent topology/input projections and identical evaluation noise across conditions.

For each source condition (Pretorius labels, information-matched alternate-persona labels, shuffled Pretorius labels), measure intact-trained, trained recurrent + virgin decoder, virgin recurrent + trained decoder, virgin + virgin, fresh decoder fitted with recurrent plasticity *frozen* after Pretorius training, and fresh decoder fitted on an identically exposed but **never plasticity-trained** virgin recurrent substrate. The last pair is the primary disambiguation: if fresh decoders reach similar phenotype accuracy on trained and virgin recurrence, freshly supervised decoder performance alone does not imply learned character information in recurrent weights.

Report paired per-seed trained-vs-virgin recurrent differences after fresh decoder fitting, an inference-without-decoder action-population readout, and lesion/rewiring changes to learned recurrent deltas under equal budgets. No claim of absent information can be made solely because one chosen decoder fails to recover it. Separate learning effects from readout mismatch and dataset leakage. Score out-of-sample JS similarity, top-1, macro domains, source episode stratification, null controls, uncertainty and actual neural update counts. Negative outcomes and seed heterogeneity must be preserved.

A second cohort may replay autobiography as a chronological `BiographyCurriculum`, but it needs its **own** measured neural step and label exposure budget; compare with PIS only under explicit compute and supervision parity. Autobiography has no native distributional targets and must not be passed to PIS as if it did.

## Execution and release gates

At this stage only **dataset candidates and an experimental plan** have been established. No new 001B dataset has been labeled or frozen; no Hugging Face dataset has been imported; no 001B neural runs or performance claims have been made. The Hugging Face connector's dataset-search operation was unavailable, so candidate names are based on public dataset cards and require source/revision/license verification before code integration.

Gate A: pin source commit + checksums; record exact candidate/actual labels, license terms and derivation ledger.

Gate B: create reviewed, episode-disjoint Pretorius cards, independently reviewed comparison persona cards, and unambiguous abstract action mapping; freeze evaluation before looking at predictions.

Gate C: implement isolated `scripts/run_chimera_001b.py` and `config/chimera_001b.json`, reusing current modules without modifying `scripts/run_chimera_001.py`, `config/chimera_001.json` or the sealed terminal battery.

Gate D: run preregistered controls across all six seeds, commit per-seed JSON and source hashes, and append dated outcomes to the experiment log; revise CC-E01 only with **new separately numbered evidence** for 001B, not as a retroactive reproduction of 001.

**Immediate practical recommendation:** start with the internally controlled 100-row historical phenotype subset for a distinct *exploratory* training-only pilot, plus the 450-event source archive for independent card construction and provenance, before importing third-party material. Do not treat either source as the missing historical 30 rows.
