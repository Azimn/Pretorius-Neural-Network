"""BioCircuit consumer for canonical gzip L1 and separately versioned BC L2.

Uses existing ExperienceEncoder as the bitwise reference and pins original
source; never recreates autobiography or treats lexical hashes as semantics.
"""
from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from persona_net.encoding import ExperienceEncoder
from biocircuit.bc01 import Corpus, SOURCE_BLOB


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_bc_shared(bundle: str | Path, corpus: Corpus) -> tuple[np.ndarray, dict]:
    if corpus.source_kind != "full" or corpus.blob_sha != SOURCE_BLOB:
        raise ValueError("shared L1 requires exact pinned full 450-event source")
    root = Path(bundle)
    l1_file = root / "pretorius_l1_v1.jsonl.gz"
    l1_manifest = root / "manifest.json"
    sensory_file = root / "bc01_sensory_256.npy"
    l2_manifest = root / "bc01_l2_manifest.json"
    l1 = json.loads(l1_manifest.read_text(encoding="utf-8"))
    l2 = json.loads(l2_manifest.read_text(encoding="utf-8"))
    if (l1.get("schema_version") != "pretorius-autobiography-l1/1" or
        l1.get("source_repo") != "Azimn/Pretorius-Connectome" or
        l1.get("source_git_blob") != SOURCE_BLOB or
        l1.get("records") != 450 or l1.get("fit_event_ids") != [] or
        l1.get("encoder_name") is not None or
        l1.get("normalization_version") !=
            "jsonl-field-projection-v1-no-authored-changes" or
        l1.get("gzip_sha256") != _sha(l1_file)):
        raise ValueError("unpinned, fitted or corrupted canonical L1")
    raw = gzip.decompress(l1_file.read_bytes())
    if sha256(raw).hexdigest() != l1.get("payload_sha256"):
        raise ValueError("canonical L1 payload checksum mismatch")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()
               if line.strip()]
    if (len(records) != 450 or
        [r["event_id"] for r in records] != l1.get("record_ids_ordered")):
        raise ValueError("canonical L1 order or coverage mismatch")
    if (l2.get("schema_version") != "pretorius-bc01-lexical-l2/1" or
        l2.get("source_git_blob") != SOURCE_BLOB or
        l2.get("encoder_name") != "persona-net-ExperienceEncoder-sensory256-v1" or
        l2.get("fit_event_ids") != [] or
        l2.get("fit_scope") != "stateless hash, no fit/IDF or episode exposure" or
        l2.get("vector_dim") != 256 or
        l2.get("dtype") != "float32" or
        l2.get("model_revision") is not None or
        l2.get("parent_l1_manifest_sha256") != _sha(l1_manifest) or
        l2.get("parent_l1_archive_sha256") != _sha(l1_file) or
        l2.get("artifact_sha256") != _sha(sensory_file) or
        l2.get("record_ids_ordered") != [r["event_id"] for r in records]):
        raise ValueError("incompatible BC L2 cache/fit/fingerprint/ordered IDs")
    with sensory_file.open("rb") as f:
        values = np.load(f, allow_pickle=False)
    if (values.shape != (450, 256) or values.dtype != np.dtype("float32")
        or not np.all(np.isfinite(values))):
        raise ValueError("invalid sensory array shape or dtype")
    encoder = ExperienceEncoder(sensory_dim=256)
    for i, (original, row) in enumerate(zip(corpus.records, records)):
        if (row.get("event_id") != original["event_id"] or
            set(row) != set(l1["projection_fields"]) & set(original) or
            any(original.get(key) != value for key, value in row.items())):
            raise ValueError("L1 source differs from pinned BC01 original text")
        expected = encoder.encode(original["memory_text"]).vector[:256]
        if not np.array_equal(values[i], expected):
            raise ValueError("BC01 L2 feature vector mismatch")
    return values, l2
