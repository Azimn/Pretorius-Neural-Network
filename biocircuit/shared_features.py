"""Opt-in bridge from the source-owned shared TF-IDF cache to BioCircuit.

This module does not implement a second tokenizer, IDF fitter or L1 parser.
Feature folding is a separately versioned *BioCircuit L3* projection.
"""
from __future__ import annotations

from hashlib import blake2b, sha256
from pathlib import Path
import sys

import numpy as np


PROJECTION = "bc01-signed-feature-bucket-v1"


class BioCircuitSharedFeatures:
    def __init__(self, cache_path: str | Path, connectome_root: str | Path):
        root = Path(connectome_root).resolve()
        library = root / "src/pretorius_connectome/shared_memory.py"
        if not library.is_file():
            raise ValueError("Missing canonical shared-memory provider checkout")
        sys.path.insert(0, str(root / "src"))
        from pretorius_connectome.shared_memory import SharedCache, SOURCE, SIDECARS
        self.cache = SharedCache(cache_path)
        self.cache.assert_original(SOURCE, SIDECARS)
        self.cache_file_sha = sha256((Path(cache_path) / "manifest.json").read_bytes()).hexdigest()
        self.projection_version = PROJECTION

    def verify_corpus(self, corpus):
        if corpus.source_kind != "full" or corpus.blob_sha != self.cache.manifest["source_git_blob"]:
            raise ValueError("Shared feature cache must match complete frozen v12 autobiography")
        if [r["event_id"] for r in corpus.records] != list(self.cache.ids):
            raise ValueError("Different ordered event IDs")
        for record, cached in zip(corpus.records, self.cache.records):
            if record["memory_text"] != cached["memory_text"] or record["episode_id"] != cached["episode_id"]:
                raise ValueError("Source text or episode diverged between cached and BC01 imports")

    def vectorize(self, text: str, event_id: str | None = None, sensory_dim: int = 256) -> np.ndarray:
        """Return only a source-derived 256-vector, no metadata/labels."""
        if sensory_dim < 1:
            raise ValueError("Invalid sensory dimension")
        if event_id is None:
            vector = self.cache.query(text)
        else:
            record = self.cache.records[self.cache.positions[event_id]]
            if record["memory_text"] != text:
                raise ValueError("Memory text differs from cached original")
            vector = self.cache.subset([event_id])
        vec = np.zeros(sensory_dim, dtype=np.float64)
        for column, weight in zip(vector.indices, vector.data):
            digest = blake2b(str(int(column)).encode("ascii"), digest_size=8,
                             person=b"bc01-l3-v1").digest()
            value = int.from_bytes(digest, "little")
            idx = value % sensory_dim
            vec[idx] += float(weight) * (1.0 if value & (1 << 63) else -1.0)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return vec.astype(np.float32)
