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

## BC01: source-grounded decision experiment (D1 and D2)

BioCircuit's new development-only decision runner uses the 450-event frozen autobiography as its source, then checks 16 individually source-verified first-person decision records against fixed action-population readouts. Both a baseline plasticity dose (D1) and a tenfold increase (D2) ran successfully across three seeds and the compartment-local, global and generic recurrent conditions. **Neither establishes useful synaptic decision learning.** D1 approximated chance and D2 collapsed all predictions to a single class per seed. These negative results are as important as the runnable integration, and are retained verbatim in the measured reports.

Run on a checkout that also contains the pinned sibling `Pretorius-Connectome` repository:

```sh
python scripts/run_biocircuit_bc01_decisions.py --corpus Pretorius-Connectome/memories/current/Pretorius_v12_450_Events_Complete.jsonl
```

[Grounded decision cards](resources/biocircuit/bc01_decision_cards_v1.json) · [D1 measured results](results/biocircuit/BC01_D1_RESULTS.md) · [D2 dose sensitivity](results/biocircuit/BC01_D2_RESULTS.md) · [Persistent BC01 requirements](https://github.com/Azimn/Pretorius-Neural-Network/issues/26). The old one-command small smoke remains at `python scripts/demo_biocircuit_bc01.py`.

## Shared memory consumer (BioCircuit ↔ Pretorius-Connectome)

Both architectures can now consume the same source-owned, versioned **L2 TF-IDF feature cache** built from the existing immutable 450-memory L1 archive. BioCircuit's optional adapter lives in `biocircuit/shared_features.py` and projects the cache into 256 sensory channels using a separate, fixed signed hashing layer; FlyWire retains its own graph projection and the raw biological v783 synapse graph unchanged. Neither system imports the other's trained neural weights. The old BC01 lexical-hash runner remains the default for historical comparability.

The cross-repository proof passed 18 tests with exact 450-event source correspondence, reproducible query encoding and checkpoint reload. Both shared-cache and old BC01 inference still exhibited **no useful recurrent lesion-dependent action changes** on the three development questions. [Measured integration results](results/biocircuit/BC01_SHARED_MEMORY_RESULTS.md) · [Source L1/L2 implementation](https://github.com/Azimn/Pretorius-Connectome/blob/main/docs/SHARED_MEMORY_L2_IMPLEMENTATION.md) · [Tracker](https://github.com/Azimn/Pretorius-Connectome/issues/10).

## BC01 D3: diagnostic recurrent learning with shared v2 input

The next BioCircuit experiment is executable: `biocircuit/bc01_d3.py` uses the source-owned stable TF-IDF **L2 v2** memory features from Pretorius-Connectome, BioCircuit's existing recurrent donor and fixed 256-sensory projection. It compares action-gated recurrent eligibility to original Hebbian plasticity, no learning, shuffled teaching labels, teaching without source cues and restoring learned recurrent weights. Source-anchored evaluation labels never enter inference inputs. Three seeds, three neural modes, source hash checks and checkpoint/restart are tested in [green CI run 37843180975](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37843180975).

**Measured result is negative:** mean target-local accuracy 20.14%, exactly the same as its shuffled/blank/lesioned controls on 16 reused development cues, versus 25% balanced chance. Recurrent lesion changed only one decision in 144. The source feature pipeline had a real cross-process nondeterminism defect in L2 v1; raw defective-run inputs/results were retained and [Connectome PR #15](https://github.com/Azimn/Pretorius-Connectome/pull/15) fixed a separate deterministic v2 format. D3 was re-run with exact cross-process neural replay. See [protocol](research/BIOCIRCUIT_D3_PROTOCOL.md), [full report](results/biocircuit/BC01_D3_RESULTS.md) and [source-trace JSON](results/biocircuit/BC01_D3_FULL_PER_CASE.json). BC01 causal-learning acceptance remains open.


## BC01-D4 input-drive gain sweep (complete, negative learning outcome)

The executable [D4 runner](scripts/run_biocircuit_bc01_d4.py), [source-owned v2 input controlled experiment](biocircuit/bc01_d4.py), [protocol](research/BIOCIRCUIT_D4_PROTOCOL.md) and [measured report](results/biocircuit/BC01_D4_RESULTS.md) are now in the repository. D4 retained the 450 original reconstructed events, 16 provisional interpreted action cards and the same 256-unit recurrent donor, and tested the **prespecified sensory cue gains of 1×, 4×, and 12×** over three seeds and three modes, without altering the frozen source encoder or teaching channels. [CI 37847739209](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37847739209) passed 33 tests and independently repeated all 27 condition results exactly, with [full per-case JSON](results/biocircuit/BC01_D4_FULL_PER_CASE.json) committed permanently.

Sensory-to-teacher input-drive ratio rose from **0.085 to 1.025**; target-local accuracy rose from **20.14% to 27.08%**, but the label-shuffled model reached **the same 27.08%** and the recurrent-lesion/no-new-learning controls remained close at **26.39%**. No configuration passed the causal source-conditioned behavior gate. This points toward recurrent cue-to-action credit assignment, not insufficient sensory drive magnitude alone. Continue with [D5 Issue #35](https://github.com/Azimn/Pretorius-Neural-Network/issues/35), inspecting existing sparse eligibility before trying any further structural or capacity change. The Doctor Lives remains unaffected.

## BC01-D5 recurrent credit-localization audit (observational, negative)

[Source code](biocircuit/bc01_d5.py), [CLI](scripts/run_biocircuit_bc01_d5.py), [protocol](research/BIOCIRCUIT_D5_PROTOCOL.md), [measured report](results/biocircuit/BC01_D5_RESULTS.md) and [full raw per-case telemetry](results/biocircuit/BC01_D5_FULL_PER_CASE.json) now preserve the mechanism found after D4. The D4 synaptic update has an optional *observer-only* instrument; no model or neural training rule was changed. In 18 exact v2 source-pinned runs, **100% of candidate action-incoming edges were eligible at 1× sensory gain and ~97% at 12×**; the learned W effect on action-score margins was only ~1–2% of the sensory cue's own readout effect. Clipping was negligible. The [D5 workflow](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37862249986) passed 37 regressions, exact policy/weight identity to the original D4 and independent source-verified numerical replay. This localizes a plausible source-*nonselective* presynaptic eligibility failure without proving causal autobiographical learning. The follow-on D5B engineering hypothesis is cue-evoked activity relative to matched blank activity, measured against all original source-card controls before scaling or production migration.
