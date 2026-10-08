"""Thin BioCircuit adapter to canonical Pretorius L1/L2 artifact.

No duplicated preprocessing. The original persona_net ExperienceEncoder is the
algorithm oracle; compare every cached row bitwise before neural training.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from persona_net.encoding import ExperienceEncoder
from biocircuit.bc01 import Corpus, SOURCE_BLOB


def load_bc_shared(bundle: str | Path, corpus: Corpus) -> tuple[np.ndarray, dict]:
    if corpus.source_kind != "full" or corpus.blob_sha != SOURCE_BLOB:
        raise ValueError("shared L1 requires exact pinned full 450-event source")
    root = Path(bundle)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "pretorius-shared-memory-v1" or
        manifest.get("source_git_blob") != SOURCE_BLOB or
        manifest.get("annotation_git_blob") !=
            "ad32025166c382caf13e07c7e3b0863eb89e1adb" or
        manifest.get("encoder_name") != "persona-net-experience-lexical-bc01-v1" or
        manifest.get("normalization_version") !=
            "exact-source-record-plus-candidate-sidecar-v1" or
        manifest.get("vector_dim") != 256 or
        manifest.get("dtype") != "float32" or
        manifest.get("model_revision") is not None or
        manifest.get("fit_event_ids") != [] or
        manifest.get("fit_scope") != "stateless lexical hashing; no train data fit"):
        raise ValueError("incompatible shared representation or leaked fit")
    for file in ("l1_records.jsonl", "bc01_lexical_256.npy"):
        if sha256((root / file).read_bytes()).hexdigest() != manifest["files"][file]:
            raise ValueError("shared cache file hash mismatch")
    rows = [json.loads(line) for line in
            (root / "l1_records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    with (root / "bc01_lexical_256.npy").open("rb") as f:
        values = np.load(f, allow_pickle=False)
    if (len(rows) != 450 or values.shape != (450, 256) or
        values.dtype != np.dtype("float32") or
        not np.all(np.isfinite(values)) or
        [r["source"]["event_id"] for r in rows] !=
            manifest.get("record_ids_ordered")):
        raise ValueError("shared artifact shape/order mismatch")
    encoder = ExperienceEncoder(sensory_dim=256)
    for index, (original, row) in enumerate(zip(corpus.records, rows)):
        if (row["source"] != original or
            row["annotation"]["status"] != "unreviewed_candidate" or
            row["annotation"]["surface_forms"] != original["recall_cues"]):
            raise ValueError("shared original record or annotation mismatch")
        expected = encoder.encode(original["memory_text"]).vector[:256]
        if not np.array_equal(values[index], expected):
            raise ValueError("shared hashed lexical vector differs from BC01 encoder")
    return values, manifest
