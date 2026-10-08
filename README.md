# Pretorius Neural Network


> **BioCircuit current handoff and next action:** [Start here: status, decisions, evidence ledger and chat-independent handoff](research/BIOCIRCUIT_CURRENT_STATUS_AND_HANDOFF.md). The first functional BC00 prototype has shipped; [BC01 Issue #26](https://github.com/Azimn/Pretorius-Neural-Network/issues/26) is the prioritized next deliverable: a one-command evidence-grounded Pretorius memory/decision demo with recurrent causal tests and checkpoint restart. This remains a research module of this repository, not a separate production character. For actual FlyWire imprinting use [Pretorius-Connectome](https://github.com/Azimn/Pretorius-Connectome); for the definitive character use [The Doctor Lives](https://github.com/Azimn/The-Doctor-Lives).


Experimental research repository for building, testing, and deliberately perturbing a neural representation of Doctor Pretorius.

The project is centered on a falsifiable question: can a character phenotype become an property of a plastic recurrent neural substrate rather than merely a prompt, lookup table, or trained output layer?

## Two construction routes

**Biography development** starts with a generic network and exposes it to a chronological developmental history.

**Persona Inverse Synthesis (PIS)** starts with the same generic architecture and trains from independently specified mature behavioral constraints.

The long-term comparison is whether those routes converge on similar behavior and neural organization when evaluated on unseen situations.

## Current status

The current stable experimental baseline is **v0.3.1**.

The Pretorius Phenotype Battery contains 20 phenotype dimensions, 130 training situations, 40 validation situations, 20 adversarial situations, and a sealed 20-item terminal battery. The original eight holdouts are preserved as a separate historical benchmark.

Experiments 001 through 004 found an important architecture problem and then a possible route through it:

- In the default v0.3 system, almost all measurable inverse-synthesized phenotype performance follows the trainable motor decoder rather than the recurrent substrate.
- Biography development shows the same qualitative localization.
- When the decoder is bypassed, default recurrent plasticity is too weak to expose a useful phenotype directly from neural action populations.
- Increasing recurrent Hebbian and reward-modulated plasticity together produced a reproducible recurrent-only phenotype signal, including at the full 4,096-neuron scale.
- Very high plasticity degraded discrimination, suggesting a bounded learning regime rather than a simple more-is-better effect.
- Biography and PIS runners currently use different effective neural-update budgets. v0.4 will make actual neural steps explicit and matched.

See GitHub Issue #1 and `EXPERIMENT_LOG.md` for recorded results.

## Experimental rule

Pretorius-specific personality variables are not placed inside the organism. There is no `arrogance`, `authority_resistance`, `recognition_need`, or equivalent hidden score. Character is evaluated externally from behavior and should, where possible, be represented through distributed neural dynamics.

Language models may eventually render a neural decision into dialogue, but they should not manufacture the underlying phenotype during neural evaluation.

## Repository layout

`persona_net/` contains the neural implementation.

`data/` contains experimental inputs that runners are permitted to consume.

`experiments/` contains destructive and comparative research protocols.

`results/` contains machine-readable experimental reports.

`resources/` contains reference material that informs the project but is not automatically treated as experimental input.

The sealed terminal battery is intentionally not committed to this public repository. Only its precommitted hashes should be public until the architecture and evaluation protocol are frozen.

## Next milestone: v0.4 recurrent persona

v0.4 promotes recurrent-substrate performance to the primary endpoint. The learned motor decoder becomes a control rather than the default evidence for persona acquisition. Neural-update budgets will be matched exactly, and the recurrent phenotype will be stress-tested using lesions, transplants, noise, washout, and partial identity grafts before the terminal set is opened.

This repository is experimental research software. It does not assert consciousness, sentience, biological equivalence, or human psychological fidelity.

## BioCircuit BC00 executable prototype

BioCircuit's initial executable kernel is available in `biocircuit/` with a runnable script `scripts/run_biocircuit_bc00.py`, deterministic tests in `tests_biocircuit/`, and an independent CI workflow. Its 4,096-unit matched local-inhibition and global-inhibition models learn 128 random synthetic associations, preserving equal connection, activity and synaptic-update budgets. On three seeds, compartment-local competition achieved 96.6% on original trained cues and 57.3% on damaged cues, while the simpler global control achieved 88.3% and **66.7%** respectively. Thus the biological-inspired circuit has NOT demonstrated an advantage in robustness. Full destruction of the learned output synapses dropped both to chance-level accuracy. This is an early working software prototype, not a real fly connectome, recurrent persona, or semantic autobiography imprint.

[Run instructions](research/BIOCIRCUIT_BC00_README.md) · [Measured results and limitations](results/biocircuit/BC00_RESULTS.md) · [Architecture RFC](research/BIOCIRCUIT_RFC_V0_1.md).

## BioCircuit BC01 exploratory Pretorius demo

BC01 development now has a source-locked offline CLI, real three-event smoke fixture, pinned 450-event full-corpus import path, fixed local/global circuit bridge into the existing recurrent donor, untrained generic policy readout, synaptic-delta lesion, and checkpoint restart comparison. **This is an exploratory implementation, not a successful autobiographical neural-imprint claim.** The external lexical retriever cites original reconstructed text but returns `unknown` for unverified support/refutation. Actions have not been independently validated for semantic appropriateness.

```sh
python scripts/demo_biocircuit_bc01.py
```

[BC01 run guide and limitations](research/BIOCIRCUIT_BC01_PREVIEW.md) · [Tracked completion gates](https://github.com/Azimn/Pretorius-Neural-Network/issues/26).
