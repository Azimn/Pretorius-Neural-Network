# BioCircuit BC01: exact legacy lexical cache interoperability

**2026-10-08 — IMPLEMENTED AND TESTED (exploratory engineering).** This is a second, optional reusable input path alongside the independently completed canonical L1 and train-split fitted TF-IDF L2 pathways. It is **not a semantic model, neural weight transfer, or validated behavioral improvement**.

## Canonical source and implementation

- The source-owner repository is [Pretorius-Connectome](https://github.com/Azimn/Pretorius-Connectome). Its original 450 reconstructed first-person memories across 27 episodes retain Git blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`.
- Existing shared L1 compressed source (`pretorius_l1_v1.jsonl.gz`) and shared TF-IDF fitted L2 (8,192 source features, fit **train episodes only**) remain supported and unchanged. The existing fitted feature bridge in `biocircuit/shared_features.py` continues working.
- Additional **stateless legacy BC01 signed BLAKE2b lexical sensory cache** is owned by source commit `81146c8b8b3cc0d538c5055bdabf1f767e0edd2f`, code `src/pretorius_connectome/shared_features_bc01.py`. The artifact has 450 rows × 256 float32 sensory channels, plus `bc01_l2_manifest.json` with source, encoder, file, ordered-ID and parent-L1 SHA256 checks. No IDF is fit. Original L1 bytes and FlyWire v783 CSR synapses are untouched.
- BioCircuit added `biocircuit/shared_memory.py` to consume that source cache, verify its exact source/text/order and compare **every cached sensory row** bitwise with the already implemented `persona_net.encoding.ExperienceEncoder(sensory_dim=256)`. Use `--shared-dir` on `scripts/demo_biocircuit_bc01.py` for opt-in precomputed source exposures; default legacy lexical encoding and the independently available `--shared-cache` TF-IDF L2 mode remain unchanged.
- BC01's sensory **query** representation is still computed by its existing encoder; retrieved event ID remains an external source lookup, recurrent weights/lesions, circuit compartments and restarts retain their existing definitions.

## Measured executed compatibility gate

[GitHub Actions BioCircuit BC01 run 37841321751](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841321751) passed the actual source checkout, L1 and L2 hash validation, BC00 and BC01 tests, original 450-event full demonstration, existing L1 input demonstration, new full 450-event cached sensory demonstration, and checkpoint checks. The workflow explicitly checks:

```python
assert original["tests"] == cached["tests"]
assert original["recurrent_changed_synapses"] == cached["recurrent_changed_synapses"]
assert cached["shared_cached_source"]
```

**This demonstrates equality rather than improvement.** It does not prove an efficient semantic representation or that synaptic updates represent memories. The previously completed cross-project fitted TF-IDF tests also remain green in [workflow 37841321705](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841321705). BC00 regression was green in [37841321754](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37841321754). The final source corpus is reconstructed fictional history, never lived autobiographical evidence.

Our source-side CPU benchmark [GitHub Actions 37840409964](https://github.com/Azimn/Pretorius-Connectome/actions/runs/37840409964) measured 710,405 bytes for compressed L1 plus 450×256 sensory L2/manifests; strict full hash-and-source replay took 0.165228 seconds, versus 0.154920 seconds to rehash the 450 source memories once on that runner. **No unconditional speedup was observed**; the benefit is a reproducible, compatible, versioned input with one canonical upstream implementation.

## Reproduce pinned 450-source comparison

```sh
# Source repository checked out at 81146c8b8b3cc0d538c5055bdabf1f767e0edd2f
python upstream-connectome/scripts/export_shared_memory.py --output upstream-connectome/artifacts/shared_memory/v1
python scripts/demo_biocircuit_bc01.py --corpus upstream-connectome/memories/current/Pretorius_v12_450_Events_Complete.jsonl --neurons 256 --seed 1842 --exposures 8 --output results/biocircuit/original.json --checkpoint results/biocircuit/original.npz
python scripts/demo_biocircuit_bc01.py --shared-dir upstream-connectome/artifacts/shared_memory/v1 --neurons 256 --seed 1842 --exposures 8 --output results/biocircuit/cached.json --checkpoint results/biocircuit/cached.npz
```

Source checkout and transitive dependencies are provided by the BC01 CI workflow. Do not use this legacy cached sensory vector as if it were the fitted TF-IDF feature space or an independently validated semantic embedding.

## Next step

A shared original L1, a frozen train-split shared TF-IDF L2 (with projection-specific BioCircuit use), and a separate exact legacy BC lexical L2 are now distinct operational resources. Match learning budgets, source event splits, query preprocessing, exact model and checkpoint state across any future architecture comparison; test memory-dependent behavioral improvements with ablations. Keep everything documented in GitHub. The definitive Pretorius implementation in The Doctor Lives is **not** automatically modified.
