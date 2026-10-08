# BioCircuit | Current Status, Decisions, Provenance and Next-Turn Handoff

**Project:** Pretorius BioCircuit (fly-inspired synthetic developmental neural architecture).  
**Checked:** 2026-10-08. **This document is the current handoff and index.**  
**Track:** Experimental `biocircuit/` subpackage of [Azimn/Pretorius-Neural-Network](https://github.com/Azimn/Pretorius-Neural-Network).  
**Next tracked work:** [BC01 delivery issue #26](https://github.com/Azimn/Pretorius-Neural-Network/issues/26).  
**Canonical character:** [The-Doctor-Lives](https://github.com/Azimn/The-Doctor-Lives).  
**Independent anatomical experiment:** [Pretorius-Connectome](https://github.com/Azimn/Pretorius-Connectome).

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
