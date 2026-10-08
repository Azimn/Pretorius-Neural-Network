# Pretorius BioCircuit | Architecture RFC v0.1

**Status:** Design proposal, not implemented or validated. Prepared 2026-10-08.
**Research designation:** BioCircuit: A Developmental, Fly-Inspired Neural Architecture for Persistent Artificial Characters.
**Implementation home:** An isolated experimental package in `Pretorius-Neural-Network` first. This does not introduce a second production Pretorius brain. Promote mechanisms only through the canonical `The-Doctor-Lives` integration gates.
**Parallel biological track:** `Pretorius-Connectome` retains its independent direct FlyWire topology and imprinting roadmap. Neither research path may alter the other's frozen experimental sources or results.

## The unique scientific question

Can a biologically organized, compartmental, developmentally plastic circuit acquire causally necessary, generalizable autobiographical associations and action dispositions more effectively than a generic sparse recurrent network, when semantic inputs, learning data, evaluation, neural-step budgets and computational resources are matched?

This is **not** the previously tested hypothesis that a generic 4,096-unit recurrent network can develop Pretorius-specific tendencies when trained using narrative/behavioral experiences. It is also **not** the FlyWire anatomical-imprinting hypothesis. In BioCircuit we design the types and organization of circuits using evidence from fly neurobiology but learn the connection weights, and in a later strictly controlled variant learn or prune parts of the connectivity mask. The resulting system is synthetic and need not reproduce fly anatomy.

## Existing work to preserve and reuse, not rebuild

**`Pretorius-Neural-Network` donor baseline:** `persona_net/network.py` already has a sparse recurrent rate model with Dale-like signs, local Hebbian and reward-eligibility updates, homeostatic adaptation, context/action feature channels, and a motor decoder. `persona_net/development.py` provides a chronological experience curriculum, action-efference training and replay. Its historical surgery studies found much of the phenotype followed the learned decoder; stronger recurrent plasticity showed some recurrent-only signal, and excessive plasticity harmed discrimination. Never reuse its sealed terminal battery as a model-selection aid.

**`The-Doctor-Lives` canonical implementation:** a production 4,096-unit recurrent substrate with optional Neural Convergence v0.5, including Oja plasticity, neuromodulatory tagging, excitability control, endogenous variation, checkpointing and felt-state inputs. Its B08 disposition retains the legacy production default and the convergence challenger as research-only. B06 lesions found no reproducible necessity of the final learned recurrent-weight delta for the reported expected-action advantage. That is a motivation for BioCircuit's causal necessity tests, not a proof that neural learning is impossible.

**`Pretorius-Connectome` anatomy track:** actual FlyWire import and planned full-connectome experiments remain separate. Its Pilots 01-04 used a synthetic bipartite topology, NOT the biological fly graph. Pilot 05A tests narrative input with BM25/TF-IDF against a limited masked associative overlay and finds retrieval is a stronger baseline on previously examined prompts. Do not conflate the 70-node Persona Connectome used by The Doctor Lives for psychological spreading activation with a microscopic neural connectome.

## What the new architecture actually is

**Sensory/semantic interface:** a frozen, local, open-weight sentence or passage embedding model provides a common semantic representation for event narrative, recall queries, contextual state and counterfactual statements. Model/version, tokenization, embedding dimension, checksum, inference cost and hardware profile must be recorded. This interface is **shared unchanged** across the BioCircuit and generic recurrent controls, so its pretrained knowledge cannot masquerade as a gain from fly-inspired circuitry. It is permissible to test multiple encoders in a later preregistered ablation, not select one on the final challenge. The network receives no event-ID oracle labels during inference.

**Projection and sparse expansion:** a bounded semantic projection stage sends overlapping inputs into a much larger sparse, competitively inhibited Kenyon-cell-inspired population. Activity is sparse by a measurable top-k/feedback-inhibition target, with pattern separation and reactivation measured. Population size and sparsity are experimental model parameters, not claims about fly KCs.

**Compartmental associative memory:** Kenyon-cell-inspired activity reaches multiple MBON-like readout populations through plastic synapses. Separate compartments permit different prediction errors and outcomes to modify different associations. DAN-like modulatory units gate local eligibility and delayed reinforcement, rather than broadcasting a single undifferentiated global reward. Compartment-specific depression/potentiation and learned response cancellation are experimental hypotheses, not assumed fly-accurate physiology.

**APL-like inhibitory competition:** a compact inhibitory feedback mechanism controls global sparseness and prevents unbounded population activity. It must be independently lesioned, not silently mixed with generic homeostasis.

**Recurrent context and consolidation:** reuse the existing recurrent network dynamics and delayed eligibility/tagging machinery rather than writing a second simulator from scratch. Feedback pathways preserve active context, supply prospective action bias and permit offline replay of earlier experience. A separate consolidation stage measures whether a new episode changes old association strength or merely overwrites it. Vary consolidation with matched update budgets.

**Developmental synaptic organization:** phase 1 holds each module's sparse topology fixed and learns ONLY permitted weights; phase 2, contingent on a positive causal gate, permits constrained synapse competition, pruning and addition at fixed total edge budget, preserving sign and per-module degree bounds. The wiring policy can use fly-derived motif statistics as a *prior*; it does not initialize from Pretorius memories by directly guessing a complete neuron-to-neuron configuration.

**Behavior and verifiability:** direct action-population tendencies and future choice shifts are primary neural endpoints, with decoder-only and narrative-retrieval-only controls. Any claim about what happened requires retrieved, source-provenance-carrying textual evidence and an explicit support/refute/unknown verification stage. The neural system is not credited with recollection because an external oracle codebook guesses an event ID. The language model, when attached later, is a renderer, not an identity store or experimental judge.

## Anchoring the biological inspiration

FlyWire's published whole-brain anatomical reconstruction includes 139,255 neurons and about 54.5 million synapses, but the structural snapshot alone does not identify all functional weights or physiological learning rules. The mushroom body is a more defensible model for this specific associative-learning question: sensory projection to sparse Kenyon cells, output through MBONs, DAN modulation of compartment-specific learning, and APL inhibitory feedback. Cross-compartment feedback and feedforward pathways have been implicated in memory consolidation and updating.

References:
- Dorkenwald et al. (2024), *Neuronal wiring diagram of an adult brain*, Nature, https://doi.org/10.1038/s41586-024-07558-y
- Schlegel et al. (2024), *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature, https://doi.org/10.1038/s41586-024-07686-5
- Lin et al. (2024), *Network statistics of the whole-brain connectome of Drosophila*, Nature, https://doi.org/10.1038/s41586-024-07968-y
- Davidson and Hige (2024), *Roles of feedback and feed-forward networks of dopamine subsystems: insights from Drosophila studies*, Learning & Memory, https://pubmed.ncbi.nlm.nih.gov/38862171/

These references justify modeling **circuit motifs and learning organization**, not transferring a fly's memories or inferring a human personality from fly anatomy.

## First concrete engineering milestone: BC00

Start by **extracting and wrapping** the current generic recurrent network behind a stable `NeuralSubstrate` test interface. Add an alternate compartmental topology generator and module-level instrumentation without changing the production `The-Doctor-Lives` API or its saved checkpoints. A first synthetic pass may use existing 4,096-unit scale for exact parity and smoke tests, but the first planned main developmental challenger should be a sparse 65,536-unit model, not an unexamined assumption that 4,096 units are sufficient. Report actual resident memory and neural-step runtime before increasing capacity. Design around CPU/CIs and local execution, using scipy.sparse and existing NumPy wherever possible; avoid new subscription, hosted inference or API dependence.

Before any biography imprinting, use a **non-Pretorius synthetic learning curriculum** with known latent causes and conflicting observations. It provides controlled tests of novelty detection, pattern separation, learned association, delayed reward, reversals, rejection of previously unseen contexts, false support of counterfactuals, and interference. A model that fails elementary context-to-outcome learning must not be interpreted as encoding Pretorius.

After passing those tests, use frozen v12 450-record autobiographical source data in a new memory experiment. Training/calibration/test are split by episode *and independently reviewed narrative relatedness clusters*, with no near-duplicate leakage. New editorially reviewed prompts must be sealed before model selection. Pilot 04 and Pilot 05A challenge cases have already been examined, so they can be retained only as diagnostics, not fresh confirmatory holdouts.

## Experimental comparisons

All critical comparisons share semantic-input encoder, train/validation/test split, curriculum, action readout type, target task, synaptic update count, trainable edge budget, and seed discipline. Run the following architectural controls:

| Network | Synaptic topology | Module specialization | Plasticity |
| --- | --- | --- | --- |
| Existing generic RNN | Existing matched sparse random | No anatomical module assignment | Baseline / convergence variants |
| BioCircuit compartmental | Fly-inspired expanded sparse and recurrent modules | Explicit KC/MBON/DAN/APL-like roles | Local compartment-gated |
| BioCircuit scrambled | Degree- and edge-budget-preserving rewired control | Roles/motif topology disrupted | Identical plasticity |
| BioCircuit no modulators | Same module topology | Intact | Reward/modulation pathway lesioned |
| BioCircuit no memory edges | Same topology | Intact | Target synapses frozen or zeroed |
| BioCircuit developmental rewiring | Constrained learnable mask after BC00 passes | Same roles as BioCircuit | Plasticity plus constrained edge change |
| Real FlyWire research arm | Verified actual connectome graph | Measured anatomy, not synthetic roles | Independently registered direct-imprint condition |

Different-size networks are useful for scaling but **are not matched-capacity architectural controls**. Full FlyWire and BioCircuit should eventually be compared at matched neuron/edge budgets where scientifically possible, along with degree-preserving shuffled FlyWire topology, not by comparing mismatched sizes and attributing a win to circuit organization.

Conventional BM25/TF-IDF narrative retrieval, and later local evidence-aware entailment checking, are strong separate **systems-level** comparators. They need not use the same synapse count, but must have compute, text access and storage recorded. If they win, report that.

## Causal gates and possible falsifications

The main positive gate is not a small gain in a scalar score. Disabling or permuting the learned compartmental synapses must reproducibly impair **correct, grounded future actions and retrieval** relative to matched no-lesion and randomly damaged controls, without relying on a trained motor decoder or an oracle answer codebook. The result must survive multiple seeds and restarts, across unseen contexts, with acceptable rejection of false or unknown memories. Track memory interference, rescue by replay, synaptic locality, storage/runtime, learned-state persistence, and full end-to-end latency.

A key failure outcome would be BioCircuit's behavior following a motor decoder, a pretrained embedding model or an external retriever rather than causal synaptic changes. That is the exact danger revealed by the earlier identity-surgery and B08 lesion studies.

**Do not** introduce named Pretorius trait neurons (arrogance, ambition, etc.), use source canon classes as personality rewards, or auto-promote experimental memories into lived history. Preserve reconstructed historical provenance separately from actual runtime experiences. Identity evaluation is external and behavior-based.

## Why this project begins now

Earlier work rationally answered more basic questions first: whether sparse recurrent dynamics could learn any character phenotype, where that phenotype lived, whether it persisted across checkpoints, how recurrence related causally to decisions, and whether the 450-event autobiography could be represented as memory associations. Those experiments revealed decoder dominance, weak causal evidence for the final recurrent-weight delta, and a severe semantic-representation bottleneck. This is why **functional circuit topology**, rather than simply adding more plasticity or source events, has become a specifically motivated intervention.

## Repository and delivery discipline

The first implementation should remain an experimental subpackage of `Pretorius-Neural-Network`, borrowing rather than forking existing `persona_net` components, with its own `biocircuit/` package, versioned configuration, BC00 smoke tests, and reproducible result ledger. Avoid opening another independent Pretorius production brain or leaving experimental work on stale feature branches. If BioCircuit merits dedicated storage, a separately named research repository can be created later with explicit lineage and reproducible release snapshots. Transfer into `The-Doctor-Lives` only following versioned equivalence, lesion and checkpoint-migration gates. `Pretorius-Connectome` continues its original actual FlyWire agenda unchanged.

**BC00 deliverable:** an architecture comparison contract, a topology generator with reproducible masks and degree/motif accounting, a validated shared input adapter, a suite of exact replay/lesion tests, and a preregistered multi-seed 4k matched control plus 65k sparse developmental challenger. No autobiographical superiority claim before that gate.
