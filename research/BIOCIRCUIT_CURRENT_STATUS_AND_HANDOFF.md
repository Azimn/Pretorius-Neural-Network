
## BC01-D6A current result and mandatory D6B separation (2026-10-08)

D6A implemented [4-fold action-label training exclusion](../biocircuit/bc01_d6a.py), [runner](../scripts/run_biocircuit_bc01_d6a.py), [unreviewed rephrasing candidates](../resources/biocircuit/bc01_d6a_paraphrase_candidates_v1.json), [frozen protocol](BIOCIRCUIT_D6A_PROTOCOL.md), [complete source-case JSON](../results/biocircuit/BC01_D6A_FULL_PER_CASE.json) and [negative result report](../results/biocircuit/BC01_D6A_RESULTS.md). Nine seed/mode configurations × four folds × four label-card tests = **144 label-heldout probe records per variant**; the source-owned L2 v2, full 450 reconstructed memories and original 16 labelled interpretations are reused, but every supervised decoder fold trains on only 12 labels. The earlier D5B motor-only 100% was **in-sample**: the action-label withheld D6A scores dropped to **18.75% original probes, 20.14% candidate paraphrases** (chance 25%). Cue-contrast recurrent W and the exactly matched W-only lesion show *identical* accuracy and zero per-fold improvement. Wrong-label supervised decoder performs ~23.61% in both variants. Train-only confidence threshold abstains on 100% of 144 non-source controls but also 100% of valid original heldout probes and 97.22% of candidate paraphrases; this threshold is NOT calibrated generalization.

**Do not call this an independent or episode-heldout test.** Every 450-background-memory source event, including label-heldout examples, remains exposed; L2 seed fit remains its original episode partition rather than excluding D6A test episodes, and every fold shares a source episode across train and heldout labels. The candidate paraphrases were authored by an assistant without external adjudication. The D6A result nevertheless decisively blocks claims that D5B's 100% on previously exposed prompts demonstrates any source-action transfer.

Next [D6 #38](https://github.com/Azimn/Pretorius-Neural-Network/issues/38) requires a different pre-reviewed set of independent new source episode decisions; exact train-only L2 source-owner feature fitting and **background curriculum that omits the heldout sources**; frozen reviewer labels/probes, semantically grounded negatives and abstentions, plus decoder-only vs frozen decoder + source-recurrent W and matched lesion controls with repeatable CI. **Neither BioCircuit neural weights nor these label predictions should be transferred to The Doctor Lives.**


## D5B current disposition: cue-specific rule still fails, supervised decoder fits reused cards (2026-10-08)

[Measured report](../results/biocircuit/BC01_D5B_RESULTS.md), [one-correction protocol](BIOCIRCUIT_D5B_PROTOCOL.md), [complete raw 18-source-trial JSON](../results/biocircuit/BC01_D5B_FULL_PER_CASE.json), [implementation](../biocircuit/bc01_d5b.py), and [green CI 37863157361](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37863157361): 40 regression tests, frozen v12 450-memory source and deterministic split-specific L2 v2 cache, seeds31/37/43 × modes local/global/generic × gains1/12, exact model checkpoint and independent process numerical replay. The D4 `target_rate` eligibility mode is preserved bit-for-bit as default; the candidate `cue_minus_blank` mode simulates a matched blank first half and restores cue fast state before applying the original action-specific three-factor update.

**Measured failure:** the candidate reduces edge eligibility from 100% to 50% (gain1) and from ~97% to 39.8% (gain12), but corrected targeted accuracy is 20.14% and 26.39%, exactly matching W-only recurrent lesion and blank-cue/no-new-training controls; the gain12 shuffled-label control does slightly *better* at 27.08%. **Zero of 18** corrected models meets source-conditioned causal benefit. A separately trained output-only motor decoder using the *same fixed recurrent weights* achieves 100% correct over the previously exposed 16 source-linked development probes at gain12, but this is a supervised in-sample association task, not generalized/semantic memory or biologically retained identity.

**Do not repeat arbitrary eligibility-rate/capacity sweeps.** Prioritize independent never-trained cue paraphrase, contradiction and source-episode split evaluation of decoder-only versus frozen-decoder, learnable recurrent W and matched recurrent lesions. If the apparent motor-only gain collapses on independent prompts, report a negative generalization outcome. Production The Doctor Lives remains unchanged until its separate source/causal migration gate.

## BC01-D5A latest findings, 2026-10-08

**Diagnostic delivered; neural success still negative.** [Complete D5 result](../results/biocircuit/BC01_D5_RESULTS.md), [research method](BIOCIRCUIT_D5_PROTOCOL.md), [frozen per-event telemetry](../results/biocircuit/BC01_D5_FULL_PER_CASE.json), [source code](../biocircuit/bc01_d5.py), and [green 37-test + 18-trial exact replay CI 37862249986](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37862249986). Original D4 action scores, sparse W, activity and learned checkpoints match exactly with the new optional observer switched off or on. Frozen L2 v2 from Connectome `7cd631f3b533203e39465ec31c4ee42945610e2f` and 450-event reconstructed L0 Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5` are unchanged.

The D4 presynaptic eligibility criterion admitted **358.92 of 358.92 incoming target-action edges (100%) on average at sensory gain 1, and 348.66 of 358.92 (~97%) at gain 12**, irrespective of whether a source cue discriminated that neuron from a blank. Learned recurrent weights shifted the action target-vs-alternative probability margin ~0.00012 (1×) and ~0.00135 (12×), versus ~0.00692 and ~0.13087 from presenting vs masking the actual cue. Sign or maximum-weight clipping was essentially absent (67 of 150623 eligible edge presentations at high gain and zero at low gain). Exact W lesions changed 1/144 or 2/144 chosen actions; masking cues changed 79/144 or 99/144. Thus **source-nonselective eligibility is a strong engineering bottleneck hypothesis, not a validated causal success**.

**Next D5B subtask of [Issue #35](https://github.com/Azimn/Pretorius-Neural-Network/issues/35):** perform one frozen targeted change only: condition presynaptic eligibility on **source-cue-evoked activity minus the matched blank-cue state**, keeping action-specific teacher/no-teacher postsynaptic response, all source/seed/mode/label controls, sparse topology and pretrained motor weights fixed. Run 1×/12× across 31/37/43 and local/global/generic, compare source-anchored choice accuracy and recurrent W lesions vs shuffled-label, blank-cue and no-learning as before. Do not call increased W sparsity proof of identity or migrate into The Doctor Lives. If D5B fails, report negative and prioritize rethinking readout/credit assignment instead of scaling the model.


## BC01-D4 complete: gain improves choice activity, not causal action learning (2026-10-08)

D4's [frozen protocol](BIOCIRCUIT_D4_PROTOCOL.md), [implementation](../biocircuit/bc01_d4.py), [runner](../scripts/run_biocircuit_bc01_d4.py), [permanent full 27-condition evidence](../results/biocircuit/BC01_D4_FULL_PER_CASE.json) and [measured negative outcome](../results/biocircuit/BC01_D4_RESULTS.md) are now committed. Reuses canonical 450-event v12 source, pinned owner TF-IDF L2 v2 encoder, 16 developmental interpretations and preexisting `PlasticRecurrentPersonaNet` with four-class action populations. Prespecified gains 1,4,12 only multiply source-card sensory inputs during training and tests; 450 unlabeled background events and all action teaching channels are unchanged. [Green CI 37847739209](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37847739209): **33 tests**, full 3-seed/3-mode/3-gain matrix, exact checkpoint reload and fresh Python independent replay of all numeric results, plus archived raw files.

The source sensory/teacher direct-drive ratio rises 0.08543 → 0.34171 → 1.02513. Mean targeted accuracies rise 20.14% → 23.61% → 27.08%, but shuffled-label controls tie targeted in every gain, and the 12× recurrent-lesion/no-learning controls attain 26.39%. Only two of 144 D4 12× policy choices flip after the targeted learned-W lesion, with no demonstrated source-specific improvement. **Zero conditions pass all causal learning controls.** No increase in model size, pretrained semantic input, or direct integration with canonical Pretorius is warranted on this evidence. Implement [D5 Issue #35](https://github.com/Azimn/Pretorius-Neural-Network/issues/35) to instrument per-action recurrent eligibility, signal clipping and readout contribution under unchanged source cues; retain this negative source-v2 D4 result as the control baseline.

# BioCircuit | Current Status, Decisions, Provenance and Next-Turn Handoff

**Project:** Pretorius BioCircuit (fly-inspired synthetic developmental neural architecture).  
**Checked:** 2026-10-08. **This document is the current handoff and index.**  
**Track:** Experimental `biocircuit/` subpackage of [Azimn/Pretorius-Neural-Network](https://github.com/Azimn/Pretorius-Neural-Network).  
**Next tracked work:** [BC01 delivery issue #26](https://github.com/Azimn/Pretorius-Neural-Network/issues/26).  
**Canonical character:** [The-Doctor-Lives](https://github.com/Azimn/The-Doctor-Lives).  
**Independent anatomical experiment:** [Pretorius-Connectome](https://github.com/Azimn/Pretorius-Connectome).

## Latest BioCircuit D3 neural-learning disposition, 2026-10-08

**D3 is implemented and empirically negative.** [Measured D3 results](../results/biocircuit/BC01_D3_RESULTS.md), [protocol](BIOCIRCUIT_D3_PROTOCOL.md), `biocircuit/bc01_d3.py`, `scripts/run_biocircuit_bc01_d3.py` and the full `results/biocircuit/BC01_D3_FULL_PER_CASE.json` are permanent GitHub files. A matching historical lexical-input trace and neural checkpoint replay are included. [Successful CI 37843180975](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37843180975) passed **25 tests**, regenerated 450-memory caches for all three episode splits, ran nine shared-source conditions and the unchanged lexical control, then demonstrated **two independently executed nine-trial sets with exactly identical neural metrics**.

A serious precursor issue was resolved in source-owned [Connectome PR #15](https://github.com/Azimn/Pretorius-Connectome/pull/15) at `7cd631f3b533203e39465ec31c4ee42945610e2f`. The earlier L2 v1 builder produced different vocabularies/IDFs across fresh CI processes despite identical source and software versions. D3 v1 raw measurements were retained separately as `BC01_D3_V1_UNSTABLE_*.json`, not treated as replicable. Source-owned v2 now resolves feature ties deterministically and has distinct schema/manifest checks. Never promote v1-derived weights to v2.

For v2, the 16 development-only source choices averaged **20.14%** correct for the new action-gated local recurrent rule, with exactly the same **20.14%** from no learning, shuffled target labels, blank-cue learning and restoring the original recurrent weights. The original Hebbian learner averaged **15.97%** under the same v2 input. Only 1 of 144 policy choices changed after recurrent lesion, and no condition passed the scientific decision gate. All target-readout models produced three to four different actions (no D2 constant-class collapse), but the network still did not learn source-conditioned autobiographical choices. Sensory cue direct drive averaged **8.54% of the teaching-channel drive magnitude**. This is an empirically grounded bottleneck hypothesis, not proof that scaling inputs will solve memory.

**Immediate next focus, D4:** test controlled sensory/teacher input drive balance and cue-conditioned eligibility while keeping the source owner v2 encoding, frozen inputs, matched neurotopology and all recurrent lesion/shuffled/blank controls. Do not rebuild the corpus, duplicate an encoder or move these weights into The Doctor Lives. Parent acceptance Issue #26 and engineering blocker Issue #29 remain open.

## Cross-project shared memory L1/L2 is now operational (2026-10-08)

The immutable full 450-memory L1 source and SHA-pinned export are owned by `Pretorius-Connectome`, with L2 split-fitted lexical cache code at `src/pretorius_connectome/shared_memory_l2.py` ([source PR #13](https://github.com/Azimn/Pretorius-Connectome/pull/13), merged at `06bece269459a43d9e4ed09e5baabbacbd2082f7`). **Do not create another BioCircuit TF-IDF fitter.** The optional BC01 adapter `biocircuit/shared_features.py` imports that implementation from a source checkout and performs only the versioned L3 256-bucket projection. The original BC01 256-channel signed lexical input is retained as a distinct unchanged control. Neither topology nor learned synapses are shared.

[Cross-project measured result](../results/biocircuit/BC01_SHARED_MEMORY_RESULTS.md): GitHub Actions run 37840112970 passed 18 tests. Pinned full corpus, original and shared encoders, exact checkpoint reload and query/source feature equality were verified. No selected action changed under learned recurrent-weight lesion for any of three development prompts in either version. **Shared caching is an engineering integration win, not proof of neural memory.** Do not conflate this with BC01's still-open causal-behavior gate. Use exact source checkout in `upstream-connectome`; source implementation remains authoritative.

## BC01-D1/D2 update: 450-memory source-grounded behavioral evaluation

The BC01 source-backed learning assay is implemented in `biocircuit/bc01_decisions.py`, `scripts/run_biocircuit_bc01_decisions.py`, and `resources/biocircuit/bc01_decision_cards_v1.json`. [D1 measured report](../results/biocircuit/BC01_D1_RESULTS.md), [D2 increased-plasticity negative diagnostic](../results/biocircuit/BC01_D2_RESULTS.md), machine-readable summaries, and [.github/workflows/biocircuit-bc01-decisions.yml](../.github/workflows/biocircuit-bc01-decisions.yml) are checked in. The 450 events remain read-only in Pretorius-Connectome. Every label card checks that its decision quotation exactly matches the original source.

On the 16 development-only questions, D1 local/global/generic neural action accuracy averaged 22.92% / 25.00% / 27.08% across seeds 31, 37, 43; the four-way majority chance rate is 25%. Restoring the learned recurrent weight changes altered some decisions but did not consistently harm accuracy. The shuffled-label control frequently performed comparably or better. External lexical retrieval performed perfectly because the test reuses cue tokens and should not be mistaken for semantic generalization.

D2 increased the donor recurrent plasticity coefficients tenfold with the rest unchanged. **Every trained neural model then predicted the exact same class for all 16 input probes**, delivering 25% accuracy for all nine topology/seed combinations. The chosen class depended on seed. Its large recurrent changes and lesion-sensitive actions demonstrate **maladaptive recurrent effects, not improved autobiographical decision behavior**. Exact checkpoint restart passed. **BC01 success gate is still OPEN.**

The next implementation must diagnose per-class population activity, network saturation, selection diversity, state stability, action teaching and credit assignment, with controls detecting collapsed constant classifiers. Only if those mechanisms are resolved should further scaling or independently reviewed challenges be considered. Do not merge this experimental state into The Doctor Lives.

## BC01 implementation update: first runnable preview merged (2026-10-08)

**PR #27 merged to main at `1b6c8781a9f2283f4e8e7ae20375d30349740606`.** An executable [BC01 exploratory CLI](../scripts/demo_biocircuit_bc01.py), [pinned importer and recurrent bridge](../biocircuit/bc01.py), [run guide](BIOCIRCUIT_BC01_PREVIEW.md), [measurements and adverse findings](../results/biocircuit/BC01_PREVIEW_RESULTS.md), [tests](../tests_biocircuit/test_bc01.py), and [CI workflow](../.github/workflows/biocircuit-bc01.yml) now exist. Do not reimplement these. The runner uses the byte-pinned authentic three-event smoke fixture or the original complete 450-event v12 corpus at frozen upstream commit `597fb23473a60eecf2e1b50f79c22bfbea816be5`.

**Observed result:** the full 450-memory CI experiment successfully loads the corpus and reproduces policy probabilities after checkpoint reload, but its targeted recurrent weight lesion changed no chosen actions across the three demonstration questions. Measured score differences were around `6.6e-6` to `7.7e-6` at 256 units, one seed. A blank-input matched-tick control produced differences around `2.2e-6`. An unrelated spacecraft query initially returned a false lexical source candidate, then a lexical-overlap gate fixed that particular error. **There is no demonstrated autobiographical neural policy learning or verified claim entailment.** The first preview is successful *integration*, not a passed BC01 research hypothesis.

**Immediate next step under open Issue #26:** define independent source-anchored behavior/policy targets and a fair recurrent-only causal endpoint, without event-ID leakage. Add retrieval-only/decoder-only, matched generic/local/global and scrambled-content controls, multiseed outcome reporting, and support/refute/unknown evaluation. Do not treat the included development questions or Pilot 04-06 cases as confirmatory holdouts. Keep production The Doctor Lives untouched.

## Executive handoff: read this first in a new conversation

The user values working deliverables and high-speed, incremental execution, not another long chain of planning-only neural research. **Do not create a new repository, reimplement the existing engines from scratch, or resume from a stale chat summary.** Continue from committed code and actual evidence in these repositories. Current immediate priority is **BC01**, a runnable, source-grounded Pretorius neural-memory demonstration with checkpoint persistence and a concrete causal ablation test. Use the existing BioCircuit BC00 kernel and generic recurrent `persona_net` implementation; publish an executable script and test artifact before expanding to 65,536 neurons.

This is a research experiment, NOT a competing production Pretorius. Do not silently update `The-Doctor-Lives` production state, alter canonical reconstructed biography, mark its preawakening reconstructions as lived experience, or treat the renderer as evidence of learning. Work must be recorded in code, tests, results, README, and GitHub issue/PR, not only in prose in a chat.

## Repository roles, consciously decided

1. **[Pretorius-Neural-Network](https://github.com/Azimn/Pretorius-Neural-Network): active BioCircuit research and experiments.** Use `biocircuit/`, `tests_biocircuit/`, `scripts/`, `research/` and `results/biocircuit/`. It already includes reusable `persona_net/network.py`, `persona_net/development.py`, and `persona_net/encoding.py` for recurrent physiology, sparse weights, local plasticity, action-readout studies and developmental curriculum. Integrate rather than duplicate.
2. **[Pretorius-Connectome](https://github.com/Azimn/Pretorius-Connectome): separate literal FlyWire / synaptic-imprinting hypothesis**, plus the immutable reconstructed autobiographical source. Keep this experimental track independent and do not call its synthetic Pilots 01-06 a successful full-fly-connectome implementation.
3. **[The-Doctor-Lives](https://github.com/Azimn/The-Doctor-Lives): only canonical production character.** Experimental donors enter only after versioned architectural gate, causal necessity, checkpoint/evidence integrity, and explicit migration approval. The Doctor Lives has its own 4,096-unit recurrent neural convergence line and its own 70-node *persona* association graph; neither is the microscopic FlyWire wiring diagram.

Do not create another production Pretorius fork. A future dedicated BioCircuit repository is optional and currently **not** the selected path.

## Durable documentation and source-code index

| Resource | What survives there |
| --- | --- |
| [BioCircuit architecture RFC v0.1](BIOCIRCUIT_RFC_V0_1.md) | Biological principles, distinctions from previous experiments, 4k matched versus 65k eventual scaling, lesion controls and evaluation gates |
| [BC00 executable instructions](BIOCIRCUIT_BC00_README.md) | Current install, runnable CLI, expected output, sample parameters and explicit missing capabilities |
| [BC00 results](../results/biocircuit/BC00_RESULTS.md) | Exact three-seed measured scores, match controls, causal action-weight lesion findings and limitations |
| [BC00 kernel](../biocircuit/prototype.py) | Actual Python synaptic-learning and sparse-competition implementation, checkpoint support |
| [BC00 runner](../scripts/run_biocircuit_bc00.py) | Parameterized experiment and machine-readable JSON |
| [BC00 tests](../tests_biocircuit/test_bc00.py) | Five tests for deterministic replay, exact fixed wiring, training/lesion changes, persistence and invalid inputs |
| [BC00 CI](../.github/workflows/biocircuit-bc00.yml) | Fresh GitHub Actions executable experiment and artifact generation |
| [BC01 delivery issue](https://github.com/Azimn/Pretorius-Neural-Network/issues/26) | Ordered work packages with explicit definition of done; update this issue during implementation |
| [Pretorius Neural Network historical log](../EXPERIMENT_LOG.md) | Decoder amputation, phenotype surgery, recurrent learning sweep and known experimental caveats |
| [Doctor Lives completion plan](https://github.com/Azimn/The-Doctor-Lives/blob/main/PRETORIUS_COMPLETION_PLAN.md) | Authoritative canonical character construction and production transition gates |
| [Doctor Lives neural B08 disposition](https://github.com/Azimn/The-Doctor-Lives/blob/main/results/neural_characterization/B08_PRODUCTION_DISPOSITION.md) | Optional Neural Convergence; trained recurrent-weight delta not proven necessary on target endpoint |
| [Pretorius-Connectome Pilot 06 results](https://github.com/Azimn/Pretorius-Connectome/blob/main/results/imprinting/PILOT06_RESULTS.md) | Evidence-gating negative finding; rejects genuine and false paraphrases without reliable entailment |
| [Frozen v12 autobiography](https://github.com/Azimn/Pretorius-Connectome/tree/main/memories/current) | Original reconstructed 450-event biography; copy only under frozen-source manifest or use read-only importer |

These files and the linked GitHub Action artifacts are the durable project record. Artifacts may expire, so significant measurements belong in committed `results/*.md` and reproducible runners, with source and protocol hashes recorded.

## Completed BC00: evidence, not speculation

**PR:** [#25](https://github.com/Azimn/Pretorius-Neural-Network/pull/25), merged to `main` at exact commit `8569c5455b7717fea3598a1932cde3c07dd0d5d4`.

**Last post-merge CI:** [GitHub Actions run 37832877348](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37832877348), successful. Five deterministic unit tests and three-seed 4,096-unit matched synthetic-association benchmark completed. Results artifact named `biocircuit-bc00-matched-4096`.

**Deliverable:** local CPU program with sparse fixed sensory projection, four competing activity compartments, trainable four-action output synapses, checkpoint round trip, lesion test, deterministic control. Uses Python + NumPy + SciPy, no paid API, no LLM, no GPU requirement. It is **not** recurrent, not the real FlyWire topology, and does not yet learn any of Pretorius's 450 first-person memory narratives.

**Actual comparative finding:**

| Test | Compartment-local BioCircuit | Matched global-competition control |
| --- | ---: | ---: |
| Trained cue accuracy | 96.61% | 88.28% |
| Partially corrupted cue accuracy | 57.29% | 66.67% |
| Destroy all learned action output synapses | 25.52% | 25.52% |

The local circuit fits exact learned associations better, but the **global control wins on corrupted cues** at this setting. The synaptic lesion proves the *output* weights cause the synthetic classification behavior, **not** that internal recurrent identity memory is distributed or necessary. These results must be retained, including the negative comparison.

To run:
```sh
python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2'
python -m unittest discover -s tests_biocircuit -v
python scripts/run_biocircuit_bc00.py --neurons 4096 --items 128 --epochs 4 --seeds 31,37,43 --output results/biocircuit/BC00.json
```

## Other completed experimental evidence affecting our direction

- In `Pretorius-Neural-Network` experiments 001-004, most generic recurrent phenotype improvement initially resided in the motor decoder. Stronger plasticity permitted some recurrent-only signal but also demonstrated degradation under excessively high learning rates. Reusing only the earlier decoder outcome would not demonstrate synaptic identity.
- In `The-Doctor-Lives`, B05 Neural Convergence increased effective dimensionality but B06/B07 lesions did not establish the causal behavioral necessity of the final recurrent-weight delta. B08 retains the convergence profile optional and warns that 4,096-unit capacity adequacy remains unresolved.
- In `Pretorius-Connectome`, Pilots 01-03 used a synthetic masked association network and demonstrated cue familiarity/lexical effects, not full FlyWire wiring. Pilot 04 failed near-zero-overlap authored paraphrases. Pilot 05A used source narratives for retrieval/encoding, finding full-text BM25 stronger than the tested masked Hebbian overlay but prone to false support of contradictions. **Pilot 06A is now complete** and found that simple sentence-evidence/negation filters reduce false acceptance by rejecting almost all legitimate responses. The latest negative lesson is to separate retrieval, entailment/refutation and evidence-backed answer-or-abstain. These studies are not character production benchmarks.

**Shared limitation:** none of these results demonstrates a complete Pretorius with autonomous, semantically grounded, causally synaptic autobiographical decision making.

## Decisions already made; no need to re-litigate

- **Name:** Pretorius BioCircuit (developmental fly-inspired architecture), source under existing `Pretorius-Neural-Network` rather than another repo.
- **Biological inspiration:** mushroom-body-like sparse expansion / inhibitory competition / compartmental plasticity and modulation; use real FlyWire topology for an independent control track. Biology is an architectural prior, not a claim of fly physiological fidelity.
- **Scientific distinction:** specialized topology and compartment-level learning vs the old generic recurrent scaffold; actual counterfactual synaptic necessity, not performance gain from a separate decoder or pretrained embedding alone.
- **Delivery priority:** ship an executable autobiographical demonstration **before** the proposed 65,536-unit scale-up, and include human-readable output, usable CLI, checkpoint/restart and negative-result controls.
- **Research honesty:** source event identity and truth provenance stay external to neural weights; an external BM25/LLM prompt cannot be credited as neural autobiographical recall; human-unreviewed Pilot 04 cases are permanently development-only.
- **Cost:** target no paid APIs, no per-call subscription dependencies, CPU-first, existing Python/SciPy/NumPy infrastructure and GitHub Actions. Frozen external embedding model is allowed if locally usable and pinned.
- **Production:** preserve `The-Doctor-Lives` as definitive and avoid introducing a second live character brain.

## The next delivery: BC01, vertical-slice Pretorius

**First step:** implement source import and a demo path that loads reconstructed source events **without** altering their provenance. The source is the 450-event, 27-episode v12 corpus from `Pretorius-Connectome`; validate pinned source blobs and document whether BC01 includes a copy or uses `--corpus` for local full data. Ship a small bundled fixture for instant offline smoke testing and enable reproducible full-corpus execution. No fabricated event IDs or invented life history.

**Second:** shared, pinned local semantic encoder (or explicitly labeled lexical fallback) plus source-linked event retrieval. Both query and memory text must use the same representation. Report results against unchanged BM25/TF-IDF baseline. The retrieval system owns quoted source evidence, not Pretorius's learned behavioral choices.

**Third:** attach BC00 compartment activity to existing recurrent `persona_net` dynamics. Provide a versioned neural checkpoint; action bias and episodic priority must be measurable after training and change under targeted recurrent-weight ablation. Never rely only on destroying the action readout, and never pass correct answer IDs into neural inference.

**Fourth:** expose the **actual deliverable**: `python scripts/demo_biocircuit_bc01.py` with an inspectable start/train/ask/decide/save/load sequence and several repeatable scenarios; outputs should contain chosen event, source/provenance quote or explicit unknown, action/policy scores and whether synaptic learning affected the decision. No paywalled inference and no presumed truth from lexical match.

**Fifth:** matched controls and deliver: generic recurrent donor, pretrained encoder alone, non-neural retrieval alone, motor decoder alone, no-update neural control, scrambled topology, and learned recurrent-delta lesions. Publish JSON and readable failure report under `results/biocircuit/`, fresh-machine CI, restart checks, event/label leakage checks and README usage. Use at least three seeds for exploratory metrics and a separately frozen independently reviewed challenge for future confirmatory claims.

The **acceptance gate** is a reproducible demonstration that *specific learned synaptic changes in the recurrent path* matter for at least one memory-influenced policy outcome. If that fails, still release the demo with a truthful null result and identify the blocker. No arbitrary numerical accuracy target is asserted before establishing meaningful evaluation and cost baselines.

Follow the current [BC01 issue #26](https://github.com/Azimn/Pretorius-Neural-Network/issues/26) as the actionable checklist, not this prose alone.

## Priority after BC01

**BC02:** real semantic verification, meaning-preserving and contradictory claims, independently reviewed new challenge; evidence-supported response/abstain. **BC03:** multi-episode interference, replay/consolidation and longitudinal state transitions, checkpoint migrations. **BC04:** matched fixed-compartment vs learned-structural-mask topology, then 65,536-unit scaling if resource and causal tests support it. **Separate FlyWire arm:** independently continue anatomical imports and graph-preserving or randomized FlyWire comparisons. Never allow BC02-04 to delay the smaller BC01 playable/demonstrable vertical slice.

## Explicit handoff prompt for the next work session

> Continue BioCircuit development in `Azimn/Pretorius-Neural-Network`. Read `research/BIOCIRCUIT_CURRENT_STATUS_AND_HANDOFF.md`, `research/BIOCIRCUIT_RFC_V0_1.md`, `results/biocircuit/BC00_RESULTS.md` and GitHub issue #26. BC00 is merged and CI-green. Do not create a new repository or duplicate the 4,096-unit recurrent engine. Implement the first working BC01 deliverable with a pinned import of the 450-event reconstructed corpus, evidence-grounded CLI, causal synaptic controls and checkpoint/restart. Keep external FlyWire exploration in `Pretorius-Connectome` and production identity in `The-Doctor-Lives`. Record code, tests, exact runs, findings and next work in GitHub. Preserve negative results and do not assume the chat contains authoritative state.

## Documentation hygiene

Every substantive experiment should end with a committed protocol, implementation, runnable test, measured result report, exact source/code commits, checkpoint/artifact manifest where applicable, updated status/index, and a GitHub issue/PR reflecting unresolved work. Negative findings are retained. Neither a chat-only decision nor a temporary Actions artifact alone is adequate archival documentation. If an artifact expires, rerun from the pinned code and data; never reconstruct missing raw results from descriptive prose.


## Cross-project L2 consolidation and checkout integrity (2026-10-08)

The Connectome owner has resolved competing cache PRs. [PR #13](https://github.com/Azimn/Pretorius-Connectome/pull/13) merged split-fitted TF-IDF as an independent `shared_memory_l2.py`; [PR #12](https://github.com/Azimn/Pretorius-Connectome/pull/12) merged a separate stateless signed hashed BC01 sensory cache without replacing the frozen L1 source artifact. BioCircuit [PR #31](https://github.com/Azimn/Pretorius-Neural-Network/pull/31) merged the optional TF-IDF-based L3 sensory projection; the **original BioCircuit signed lexical encoder remains the default and is not interchangeable with FlyWire TF-IDF**.

A subsequent review found that a long-running Python process could reuse a module from the wrong Connectome checkout through `sys.modules`. `biocircuit/shared_features.py` now fails closed if an already imported `pretorius_connectome` module originates outside the requested root or if the loaded shared-memory L2 file differs from the requested file. `tests_biocircuit/test_shared_feature_consumer.py` simulates wrong-checkout module contamination. **Both [new shared-memory CI 37840943234](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37840943234) and [BC01 CI 37840943340](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37840943340) passed**. See the owner's [consolidation/security report](https://github.com/Azimn/Pretorius-Connectome/blob/main/docs/SHARED_MEMORY_CONSOLIDATION_20261008.md). The source TF-IDF loader also independently refits and verifies IDF against training-only texts. These are engineering provenance checks, **not** evidence of neural autobiographical learning.


## Exact BC01 legacy sensory cache integration — complete, 2026-10-08

[PR #32](https://github.com/Azimn/Pretorius-Neural-Network/pull/32) was merged at `2191156597b7d7cff694d7c0178cc154e081a149`. Canonical source owner [Pretorius-Connectome PR #12](https://github.com/Azimn/Pretorius-Connectome/pull/12) was merged at `81146c8b8b3cc0d538c5055bdabf1f767e0edd2f`. This adds an **optional third interoperable input path**, `python scripts/demo_biocircuit_bc01.py --shared-dir upstream-connectome/artifacts/shared_memory/v1`, consuming the published compressed 450-event L1 plus a source-owned, versioned **450×256 float32 exact BC01 signed-hash lexical cache**, with source/sidecar/manifest/vector checksum and all-row bitwise `ExperienceEncoder` equivalence. It uses cached features for *source learning exposures*; the **live query still uses original ExperienceEncoder**, with unchanged recurrent network and compartment machinery.

The two preexisting modes are preserved: default BC01 legacy lexical encoding (no external resource dependency), and explicit split-fit shared TF-IDF with separate L3 signed-bucket projection (`--shared-cache` with pinned `--connectome-root`). The source L1 manifest and original reconstructed 450 memories are unchanged. These three feature representations must NEVER be conflated with learned semantic embeddings, or weights from the FlyWire brain.

**Reproduced tests:** [Pre-merge full BC01 450-memory exact source-cache parity 37841321751](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841321751) and fresh merged commit post-merge BC00 [37841567887](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841567887), BC01 [37841567873](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841567873), canonical shared-memory [37841567641](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841567641): **all successful**. The full original and cached 450-event `tests` fields (including evidence retrieval and neural policy decisions), recurrent changed-synapse count and restart checks were matched exactly. See [measured permanent report](../results/biocircuit/BC01_LEGACY_CACHE_RESULTS.md) and [source benchmark report](https://github.com/Azimn/Pretorius-Connectome/blob/main/results/shared_memory/BC01_LEXICAL_CACHE_RESULTS.md).

**Cost finding:** Cache source verification with all-row feature replay consumed 0.165228s versus 0.154920s to compute the hashed lexical vectors on source runner; **we found no overall speed advantage for one run**. Its value is deterministic reuse and version/provenance checks. Causal evidence for character-level neural autobiographical memory, independently reviewed truth labels and The Doctor Lives deployment remain **open and separate experiments**. Never report the equal baseline/cached decision result as an improvement in neural cognition.
