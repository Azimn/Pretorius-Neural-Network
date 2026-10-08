# BioCircuit consumes Connectome's shared L1/L2 memory interface

**Date:** October 8, 2026. **Status:** cross-repository consumption demonstrated in CPU-only GitHub Actions. This is **not** evidence of autobiographical learning or neural semantic recall.

## Architecture

The canonical L1 source is the immutable 450-event gzip JSONL and manifest already published by Pretorius-Connectome. That source repository also provides an L2 vocabulary/IDF/TF-IDF document cache built with training-episode-only fit statistics. The BioCircuit consumer at `biocircuit/shared_features.py` imports `pretorius_connectome.shared_memory_l2.SharedCache` directly from a pinned Connectome checkout. No general-purpose text parser, tokenization rules or IDF estimation is duplicated in BioCircuit.

The shared L2 feature space has 8,192 TF-IDF coordinates (for this frozen source) and explicit seed, source, code and shard hashes. BioCircuit applies a *separate*, deterministic L3 map `bc01-signed-feature-bucket-v1`, folding cached TF-IDF coordinates into the existing 256 sensory channels using a fixed signed BLAKE2b bucket. That is not the same encoder as BC01's original signed lexical token hash, which remains available unchanged by default. Learned recurrent weights, topology, motor readouts and checkpoint state are never reused between FlyWire and BioCircuit.

Memory narrative vectors come from cached `docs.npz` via stable event ID selection, with byte-for-byte text verification. Live query vectors use the same frozen TF-IDF vocabulary/IDF followed by the same L3 projection, without fitting on the query or passing event IDs/decision labels into neural input.

## Reproduce

Have both repositories checked out side by side, and use exact Connectome source commit `06bece269459a43d9e4ed09e5baabbacbd2082f7`.

```sh
python -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2" "scikit-learn>=1.5,<2"
python upstream-connectome/scripts/build_shared_memory_l2.py --seed 31 --output-dir data/derived/shared-memory-l2-seed31
python scripts/demo_biocircuit_bc01.py --corpus upstream-connectome/memories/current/Pretorius_v12_450_Events_Complete.jsonl --shared-cache data/derived/shared-memory-l2-seed31 --connectome-root upstream-connectome --neurons 256 --seed 31
```

For the unmodified historical encoding control, omit both `--shared-cache` and `--connectome-root` and keep the same `--corpus` path. Without the source checkout, the original three-event offline demonstration remains available.

## Measured regression and limitations

[CI run 37840112970](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37840112970) passed **18 tests**, including all previous BC00/BC01 tests and four new consumer tests; source and cached features matched 450 original event identities exactly; original text and query TF-IDF folding reproduced the same vectors. Both the cached full-corpus demonstration and original uncached BC01 baseline passed checkpoint restart.

At seed 31, 256 units and the same three *development-only* questions, the shared-cache demonstration's maximum score differences after removing learned recurrent weights were 9.931e-6, 1.0386e-5 and 1.0268e-5. The corresponding unchanged historical lexical-input baseline differences were 9.139e-6, 7.492e-6 and 9.094e-6. **Neither condition changed the selected action for any of the three questions.** All model states reproduced exactly after checkpoint reload. Larger score differences do not establish an improvement. Only the feature input changed; learned synaptic behavior is still not usefully demonstrated.

The source code and a permanent result report stay in GitHub, while CI uploads machine-readable JSON plus the reproducible input manifest as artifacts. No production state in The Doctor Lives was modified.

**Next work:** matched architecture comparisons should first prove a causally beneficial policy signal, independently of the shared cache. Keep previously viewed probes development-only and preserve negative results.
