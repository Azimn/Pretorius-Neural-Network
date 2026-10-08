"""Read the shared, source-pinned L1 autobiography without reprocessing it.

This is an adapter to canonical Pretorius-Connectome L1. It does not import
FlyWire neural topology or transpose neural weights between architectures.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from biocircuit.bc01 import (
    Corpus, SOURCE_BLOB, SOURCE_COMMIT, EVENT_ID, git_blob_sha
)

L1_SCHEMA = "pretorius-autobiography-l1/1"
L1_SOURCE_REPO = "Azimn/Pretorius-Connectome"
L1_NORMALIZATION = "jsonl-field-projection-v1-no-authored-changes"
L1_MANIFEST_BLOB = "7a8d05b46b025b1bfe44339b3edbee040df22394"
L1_ARCHIVE_BLOB = "5381c3c22ab224bb6ebf9f037489c1b9be564888"
L1_ARCHIVE_SHA256 = "882b728f84fe8eb969ff924cecaebf6f229d8fe5063c6f7b4ff69687c2cede6c"
L1_PAYLOAD_SHA256 = "ce6a13d1608b10cebf64bd8208b5d7fef72ffb338ba18d1ad225f752fb0b7802"


def load_shared_corpus(archive_path: str | Path,
                       manifest_path: str | Path) -> Corpus:
    """Fail closed on any mismatch, return existing BC01 Corpus interface.

    Original BC01 signed-hash sensory encodings and recurrence are unchanged.
    No embedding cache or trained synaptic weights are shared at this stage.
    """
    manifest_raw = Path(manifest_path).read_bytes()
    archive = Path(archive_path).read_bytes()
    if (git_blob_sha(manifest_raw) != L1_MANIFEST_BLOB
            or git_blob_sha(archive) != L1_ARCHIVE_BLOB):
        raise ValueError("Shared L1 artifact differs from pinned published Git blobs")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if (manifest.get("schema_version") != L1_SCHEMA
            or manifest.get("normalization_version") != L1_NORMALIZATION
            or manifest.get("source_repo") != L1_SOURCE_REPO
            or manifest.get("source_git_blob") != SOURCE_BLOB
            or manifest.get("source_commit_pin") != SOURCE_COMMIT
            or manifest.get("records") != 450
            or manifest.get("encoder_name") is not None
            or manifest.get("fit_event_ids") != []
            or manifest.get("vector_dim") is not None
            or manifest.get("gzip_sha256") != L1_ARCHIVE_SHA256
            or manifest.get("payload_sha256") != L1_PAYLOAD_SHA256):
        raise ValueError("Shared L1 manifest has an incompatible source or feature space")
    if hashlib.sha256(archive).hexdigest() != L1_ARCHIVE_SHA256:
        raise ValueError("Shared L1 compressed archive digest mismatch")
    raw = gzip.decompress(archive)
    if hashlib.sha256(raw).hexdigest() != L1_PAYLOAD_SHA256:
        raise ValueError("Shared L1 decompressed archive digest mismatch")
    records = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines())
    if len(records) != 450:
        raise ValueError("Incomplete shared L1 records")
    ids = [item.get("event_id") for item in records]
    if (len(set(ids)) != 450
            or ids != manifest.get("record_ids_ordered")
            or [item.get("episode_id") for item in records] != manifest.get("episode_ids_ordered")
            or [item.get("chronological_order") for item in records] != list(range(1, 451))
            or len({item.get("episode_id") for item in records}) != 27):
        raise ValueError("Shared L1 corpus order/episode coverage mismatch")
    required = {
        "event_id", "episode_id", "chronological_order", "memory_text",
        "provenance", "recall_cues", "title",
    }
    for item in records:
        if (not required.issubset(item)
                or not isinstance(item["event_id"], str)
                or not EVENT_ID.fullmatch(item["event_id"])
                or item["provenance"] != "reconstructed"
                or not isinstance(item["memory_text"], str)
                or not item["memory_text"].strip()
                or not isinstance(item["recall_cues"], list)):
            raise ValueError("Shared L1 malformed reconstructed memory")
    return Corpus(records=records, source_kind="full", blob_sha=SOURCE_BLOB)
