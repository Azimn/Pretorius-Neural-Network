# 001E: Independent reviewer packets and safe collection

**Status:** Two research-ready packets; zero scored labels and zero validated reviewers.

Run `python scripts/build_chimera_001e_review_packets.py` from the repository root to regenerate both packets from the frozen twelve 001E scenarios and their 450-memory source index. The command never reads predictions or previous neural outcomes and does not generate an answer key. GitHub CI checks its output against the immutable source pointers and action vocabulary.

Give **Reviewer A** `review_packets/REVIEWER_A.md` and **Reviewer B** `review_packets/REVIEWER_B.md` separately. The items are presented in two independent deterministic randomized orders to reduce shared order bias. Each displays a short provenance-linked summary of Pretorius's *fictional reconstructed* antecedent and the newly authored hypothetical decision. The historical examples are supplied as context, not objectively correct actions. Explain in advance that the current experiment tests **predicted behavior in a fictional character**, not moral correctness, consciousness, or historical fact.

Reviewers each need to independently read the scenario, distribute 1.00 probability across plausible generic actions, supply a rationale and optional ambiguity note, and explicitly attest that no model prediction or other reviewer's response was seen. Accept substantive uncertainty, abstention and disagreement. The `REVIEWER_A.example.jsonl` and `REVIEWER_B.example.jsonl` files are **deliberately incomplete templates**, not submissions: both have `target_actions:null`, `attests_independent_read:false`, `saw_model_predictions:null`, and no signed reviewer ID or note. They **fail** the `validate_chimera_001e_reviews.py` release gate unchanged.

Keep Reviewer A and B's completed response files **separate and non-public until both have finalized their evaluations**; public interim answers would compromise independence. Collect them through a private channel. After both are complete, provide an independent adjudicator, who must review both separate distributions and rationales, and who must be a third person. The existing 001E release validator requires distinct reviewer identities, a separate adjudicator, and unchanged SHA-256 scenario text. It can issue only `candidate_pending_holdout_freeze` items and does **not** certify independent authorship or an unseen holdout by itself.

This twelve-situation pilot was generated inside ChatGPT and is associated with v12 memories used for background, making it **researcher-authored**, not truly independent material. It is suitable for measuring rater agreement and calibrating an action ontology. Avoid publishing strong out-of-sample phenotype claims without a separately authored, blinded, preregistered test corpus. There is no current claim of reviewed labels.

Do not allow either reviewer to see 001B, 001D, or 001F outcomes, trained model completions, inferred answer templates, or the other reviewer's responses before submission. If a reviewer reports prior exposure, keep the response as an explicitly marked non-independent sensitivity analysis instead of representing it as blind.

No source autobiography is altered; no student data, personal files, external Hugging Face datasets, production cognition or original Experiment 001 terminal data enter this review workflow.
