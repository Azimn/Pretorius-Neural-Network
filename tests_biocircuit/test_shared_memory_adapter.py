"""Full-source shared L1 interoperability, replay, mutation and input parity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from biocircuit.bc01 import load_corpus, retrieve, represent
from biocircuit.shared_memory_adapter import load_shared_corpus
from persona_net.encoding import ExperienceEncoder

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream-connectome"
SOURCE = UPSTREAM / "memories/current/Pretorius_v12_450_Events_Complete.jsonl"
ARCHIVE = UPSTREAM / "artifacts/shared_memory/v1/pretorius_l1_v1.jsonl.gz"
MANIFEST = UPSTREAM / "artifacts/shared_memory/v1/manifest.json"


@unittest.skipUnless(SOURCE.is_file() and ARCHIVE.is_file() and MANIFEST.is_file(),
                     "Requires explicit upstream-connectome pinned checkout")
class SharedMemoryAdapterTests(unittest.TestCase):
    def test_exact_source_coverage_and_lexical_neural_parity(self):
        original = load_corpus(SOURCE)
        shared = load_shared_corpus(ARCHIVE, MANIFEST)
        self.assertEqual(len(shared.records), 450)
        self.assertEqual(shared.source_kind, "full")
        self.assertEqual(shared.blob_sha, original.blob_sha)
        self.assertEqual([x["event_id"] for x in shared.records],
                         [x["event_id"] for x in original.records])
        for a, b in zip(original.records, shared.records):
            self.assertEqual(a["memory_text"], b["memory_text"])
            self.assertEqual(a["recall_cues"], b["recall_cues"])
            self.assertEqual(a["episode_id"], b["episode_id"])
            self.assertEqual(a["title"], b["title"])
            self.assertEqual(a["approximate_date"], b["approximate_date"])
        encoder = ExperienceEncoder(sensory_dim=256)
        for a, b in zip(original.records[::13], shared.records[::13]):
            np.testing.assert_array_equal(
                represent(a["memory_text"], encoder, None),
                represent(b["memory_text"], encoder, None),
            )
        for question in (
            "millstream map and turned glove",
            "beetle specimen drawer",
            "purple spacecraft rocket 1982",
        ):
            self.assertEqual(retrieve(original, question), retrieve(shared, question))

    def test_tampering_with_shared_archive_and_source_metadata_fails(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "memory.gz"
            manifest = Path(td) / "manifest.json"
            data = ARCHIVE.read_bytes()
            archive.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            manifest.write_bytes(MANIFEST.read_bytes())
            with self.assertRaisesRegex(ValueError, "pinned"):
                load_shared_corpus(archive, manifest)
            archive.write_bytes(data)
            values = json.loads(MANIFEST.read_text(encoding="utf-8"))
            values["source_git_blob"] = "0" * 40
            manifest.write_text(json.dumps(values), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned"):
                load_shared_corpus(archive, manifest)


if __name__ == "__main__":
    unittest.main()
