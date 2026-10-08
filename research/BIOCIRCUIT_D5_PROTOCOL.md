# BioCircuit BC01-D5: baseline-preserving eligibility and policy-credit audit

**Protocol frozen before examining D5 full-corpus observations: 2026-10-08.**

The prior [D4 experiment](../results/biocircuit/BC01_D4_RESULTS.md) explicitly tested scaling of sensory cues from one-twelfth of teaching-drive magnitude to approximate parity. No matched shuffled-label / W-lesion controls demonstrated source-grounded learning, despite a rise in absolute decision accuracy to 27.08%. D5 addresses the resulting *diagnostic* question without a new learning rule: at which stage does the existing model's cue-conditioned eligibility signal cease to affect correct choices?

## Fixed source, model and controls

- Original 450 source memories, 27 reconstructed episodes, source Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`. Interpretive decisions are provisional source-matched development labels, not adjudicated identity facts.
- Pretorius-Connectome commit `7cd631f3b533203e39465ec31c4ee42945610e2f` owns L2 v2 train-episode-fitted deterministic lexical encoding, never rebuilt in BioCircuit.
- Existing 256-unit recurrent donor, original sparse W/Win, action populations and motor output weights, D4 `eta=0.03`, original curriculum (450 × eight ticks), 16 action cards × three epochs × 32 ticks, 32 test-settling ticks.
- Three seeds (31/37/43), three modes (local/global/generic), sensory gains **1 and 12**, yielding 18 independent *experimental configurations*, not 18 independent subjects. These two strengths reproduce D4 baseline and near-parity sensory/teacher conditions.
- The sole modification to the existing D4 training implementation is an **optional observer** that copies preupdate W only in diagnostic mode, computes trace statistics, and never changes activity, eligibility, sparse indices, chosen labels or actual synaptic update. Unit tests compare unobserved versus observed W, voltage, rate and exact tick values, plus existing D4 policy equality.

## Per-event telemetry and counterfactuals

At each labelled training presentation (16 cards × three epochs = 48 updates per condition), collect the exact first-half cue-only presynaptic activity, matched-time teacher-minus-no-teacher postsynaptic signal and action-population size. Record candidate action-directed incoming edges, eligibility-positive edges, excitatory/inhibitory candidate count, actual modified W, candidate updates truncated by Dale sign or maximum synaptic weight, L1 delta for excitation and inhibition, mean/max eligibility surrogate, direct Win cue-to-teacher drive ratio and network rate/bias statistics.

After all updates, rerun the exact D4 targeted model and **compare every resulting action score exactly** with a separately reconstructed observational copy. Evaluate each of 16 fixed probes under (i) original trained W, (ii) original pre-card W substituted (recurrent lesion), (iii) only the single learned edge with maximum absolute delta reverted to pre-card W, and (iv) no sensory cue at test. Track target-vs-best-alternative action probability margin, intact/lesioned action flips, cue-driven policy changes and source-ID-conditioned event traces. No action-teacher channel is present in any query.

A largest-edge reversion tests *local sensitivity*, not the existence of an interpretable single-memory synapse. All intact-vs-lesioned policy results must be compared to D4's original shuffled-label, blank-cue and no-card-training outcome. No output decoder training and no new W-learning mechanism is introduced in D5.

## Engineering guardrails and outcome gate

All 18 per-case outcomes and their 48 training-event traces are generated and repeated from a fresh Python process with a different `PYTHONHASHSEED`; exclude only observed wall seconds when checking exact numerical identity. Require 450 source / 16 cards, valid L2 v2 schema/fit seed, 48 presentations per configuration, every checkpoint exact and the intact model **bit-identical to D4**. Record exact command and frozen dataset/model commitments, first-run failures and full JSON evidence on GitHub, not only the ephemeral Actions artifact.

A finding that only sign-clipped inhibitory edges changed, that all source cue activity collapsed, or that one single W edge dominates would be a hypothesis for **one separately scoped correction**. Diagnostic correlations alone do not justify a claim of causal autobiographical behavior. If no clear bottleneck is observed, publish a negative result and keep the current donor without further ad hoc tuning. Never transfer learned synapses to The Doctor Lives without independent held-out narratives and their own production gate.
