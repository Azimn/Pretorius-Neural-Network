# Experiment 001E: New autobiographical counterfactuals and independent behavioral adjudication

**Status: Source-linked drafts and release validator only. No approved answer labels. No model training.**  
**Initiated:** 2026-10-10.

## Aim

The completed 001B and 001D experiments on exposed phenotype sets show a small causal effect of recurrent W, a larger contribution from motor decoder weights, modest scenario sensitivity, and a strong no-network class-frequency prior. Further repeated work on those exposed sources will not establish behavioral generalization.

001C's pinned Pretorius archive contains 450 autobiographical events across 27 episodes. Their history is deeply entangled (one 448-event connected component), the sensory sidecars are unreviewed, and 200 causal statuses are missing. The autobiographical text supplies **fictional character history**, not a validated ten-action choice label.

001E therefore starts with **12 new, source-linked hypothetical dilemmas** grounded in twelve differently themed existing memories (`E01-001`, `E03-001`, `E05-001`, `E09-001`, `E11-001`, `E13-001`, `E15-001`, `E18-001`, `E20-001`, `E22-001`, `E25-001`, `E27-001`). These situations are drafted here by ChatGPT, so they are **not independently authored ground truth or a confirmatory unseen test** just because the wording is new. Each carries `draft_source=assistant_authored_counterfactual_not_independent_validation`, `target_actions=null`, `review_status=unreviewed_draft`, and `evaluation_role=unassigned`.

## Review and release contract

`scripts/validate_chimera_001e_reviews.py` verifies the draft rows against all 450 source pointers in `results/chimera_001c/REVIEW_QUEUE.jsonl` and the exact source commit. It rejects exact scenario duplicates with the old 40/20 v1 exposed evaluation items. Duplicate scenario content, changed source hashes, prefilled correct action labels or premature evaluation-role assignments fail before any export. The 12 draft cards are in `research/chimera_001e/SCENARIO_DRAFTS.jsonl`.

Two **different reviewers** must independently view each novel scenario, without seeing model predictions, and submit a `JSONL` row containing:

- `candidate_id`
- `scenario_sha256` (SHA-256 of the exact UTF-8 scenario text from its draft)
- `reviewer_id` (stable distinct ID)
- `attests_independent_read: true`
- `saw_model_predictions: false`
- `target_actions` (a legal ten-action soft target distribution summing to 1)
- `review_note` (a contextual explanation, at least 20 characters)

A **third adjudicator**, distinct from both reviewers, must submit a separate JSONL row with matching `candidate_id` and `scenario_sha256`, `adjudicator_id`, `decision: approve`, `target_actions`, and `adjudication_note`. A disagreement ledger and individual review rationales must be preserved. The validator never chooses a correct answer automatically, even when reviewers agree.

If either set of input review files is missing, or no card has all three sign-offs, non-preflight export fails closed. Even with administrative review, independence is only attested, not proved cryptographically. The output status remains `provisional_adjudication_not_confirmatory`, and accepted draft cards retain `evaluation_role=candidate_pending_holdout_freeze`. A **separate** frozen, source-audited independent evaluation release is needed before they are used to claim confirmatory generalization.

## Phased workflow

1. The current CI **preflights all twelve unlabeled drafts** against the 450-record immutable source queue and produces a machine-readable `results/chimera_001e/STATUS.json` saying no human-adjudicated label exists.
2. Independent reviewers record their own judgments. Do not show them results from 001B, 001D, an LLM answer key or other reviewers' rationales before submission.
3. Independent adjudication resolves ambiguous action-target mappings, records substantive disagreement and may reject an inherently underspecified dilemma. Do not force labels just to fill a test set.
4. Freeze approved scenarios and targets plus source SHA hashes before running a single neural evaluation on the resulting holdout. Test all previously declared neural controls and stronger non-neural conditional classifiers. Preserve any negative outcomes.
5. A later developmental `BiographyCurriculum` may ingest the original 450 memories, but its representational and neural-step budgets must be matched separately and it may not train on these adjudicated test labels.

The proposed twelve drafts are a **small pilot for reviewer calibration**, not a statistically adequate comprehensive personality test. Larger coverage should be balanced across source themes and counterfactual mechanisms, with provenance and category labels reviewed before introducing trained model evaluation.

Neither this process nor the 001D numerical work modifies the original historical Experiment 001 corpus, sealed terminal, reconstructed neural control laws, or Pretorius-Connectome source. All results must remain labeled as their own protocol IDs.
