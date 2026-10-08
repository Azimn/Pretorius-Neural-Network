# Shared Pretorius memory inputs: BioCircuit and FlyWire

**Status: reuse agreement; no shared embedding/vector cache has been implemented by this documentation change.**

The [canonical cross-project memory reuse contract](https://github.com/Azimn/Pretorius-Connectome/blob/main/docs/SHARED_MEMORY_REUSE_CONTRACT.md) lives in `Azimn/Pretorius-Connectome`. Read it before adding any memory encoding, normalization, source parsing or retrieval work to BioCircuit.

**Authoritative source:** `Azimn/Pretorius-Connectome/memories/current/Pretorius_v12_450_Events_Complete.jsonl`, blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`, 450 reconstructed first-person records, 27 episodes. Do not recreate these events or replace the frozen corpus with generated biographical material. `biocircuit/bc01.py` already provides a source-blob-validated read-only importer and offline 3-record smoke fixture.

## What to share and what must stay distinct

Reuse source records, stable IDs, provenance, candidate sensory annotations, cue/relationship metadata, and compatible deterministic preprocessing. If a shared versioned representation is built, pin source and encoder checksums, fitted training split, feature dimension and dtype, and reuse the same cached source/query encoding **only** for matched encoder + split conditions.

BC01 currently performs on-the-fly signed hashed **lexical** 256-sensory input via `persona_net.encoding.ExperienceEncoder` and `biocircuit.bc01.represent`; it has not emitted a shared semantic embedding cache. FlyWire associative memory currently fits word/bigram TF-IDF (up to 8,192 features) and maps those features to a real or synthetic graph with a fixed hash. These are different spaces. The bridge must be explicit and versioned; do not reinterpret a BioCircuit learned recurrent matrix as biological synapses. Keep BioCircuit recurrence, FlyWire v783 adjacency and original synapse counts, plastic overlays, and architecture-specific checkpoints separate.

Development-only source-anchored `resources/biocircuit/bc01_decision_cards_v1.json` contains 16 human-interpreted choices. They can inform comparative assay design but are not validated historical actions, trained semantic truth, or independent held-out evidence. Maintain episode-disjoint splits and never let action labels or event IDs enter query vectors. Read the actual failure reports for BioCircuit BC01 and Pretorius-Connectome Pilots 01–07 before claiming a neural memory advantage.

## Immediate implementation responsibility

**Do not build a second general-purpose memory-data preparation engine in BioCircuit.** Make the next BC01 data-interface change a compact adapter to a stable shared output owned by Pretorius-Connectome, or demonstrate why the current verified source importer and encoder cannot cover the requirement. Keep `bc01.py`'s frozen source-pin checks. Record measured CPU/memory cost and failing tests, if any. Reuse existing `persona_net` recurrence and the BC00 kernel rather than introducing another simulator.

The acceptance condition for a cross-project adapter is equal, version-pinned event coverage and deterministic feature reproduction (for an explicitly matched encoder and split), followed by a matched BioCircuit-vs-FlyWire control. It is **not** accuracy improvement from reusing a cached vector. The Doctor Lives remains the production Pretorius and must not silently ingest experimental weights or confuse reconstructed biography with lived events.

Tracking: [BC01 issue #26](https://github.com/Azimn/Pretorius-Neural-Network/issues/26); [FlyWire handoff](https://github.com/Azimn/Pretorius-Connectome/blob/main/docs/RESEARCH_HANDOFF.md). Do not leave subsequent interface decisions only in chat.


## Actual tested source integration (2026-10-08)

Portable L1 **is now implemented and running**, superseding the earlier documented-not-implemented L1 assumption. The [canonical manifest](https://github.com/Azimn/Pretorius-Connectome/blob/main/artifacts/shared_memory/v1/manifest.json) and [source archive](https://github.com/Azimn/Pretorius-Connectome/blob/main/artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz) are committed and source-pinned. BioCircuit implements its own compact [verified L1 adapter](../biocircuit/shared_memory_adapter.py); the existing demo CLI accepts `--shared-l1` plus `--shared-manifest`. [BC01 450-event CI run 37839516626](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37839516626) passed and found original versus shared-L1 evaluation outputs exactly equal at the same seed and exposures. All 16 unit tests passed. See [permanent BioCircuit results report](SHARED_MEMORY_L1_INTEGRATION_RESULTS.md) and [canonical report](https://github.com/Azimn/Pretorius-Connectome/blob/main/results/shared_memory/L1_INTERFACE_RESULTS.md). Previously written prospective statements remain as historical context. No common model-fitted L2 vector cache, verified speedup, or shared neural weight representation is claimed.
