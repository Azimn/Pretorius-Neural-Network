"""Cross-repository BC01 consumer tests using the same source-owned cache."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream-connectome"


@unittest.skipUnless((UPSTREAM / "src/pretorius_connectome/shared_memory_l2.py").is_file(),
                     "Optional source checkout not present; CI always provides it")
class BC01SharedCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(UPSTREAM / "src"))
        from pretorius_connectome.shared_memory_l2 import build_cache, SOURCE, SIDECARS
        from biocircuit.bc01 import load_corpus
        from biocircuit.shared_features import BioCircuitSharedFeatures
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "features"
        cls.manifest = build_cache(cls.dir, seed=31)
        cls.shared = BioCircuitSharedFeatures(cls.dir, UPSTREAM)
        cls.corpus = load_corpus(SOURCE)
        cls.shared.verify_corpus(cls.corpus)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_pinned_original_events_in_both_consumers(self):
        self.assertEqual(len(self.shared.cache.records), 450)
        self.assertEqual(list(self.shared.cache.ids),
                         [event["event_id"] for event in self.corpus.records])
        self.assertEqual(self.shared.cache.manifest["source_git_blob"],
                         self.corpus.blob_sha)

    def test_original_and_query_have_same_feature_transform(self):
        record = self.corpus.records[0]
        x = self.shared.vectorize(record["memory_text"], event_id=record["event_id"])
        query = self.shared.vectorize(record["memory_text"])
        np.testing.assert_array_equal(x, query)
        self.assertEqual(x.shape, (256,))
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertEqual(self.shared.projection_version, "bc01-signed-feature-bucket-v1")

    def test_tampered_input_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            self.shared.vectorize("fabricated memory", event_id=self.corpus.records[0]["event_id"])
        from biocircuit.bc01 import load_corpus
        fixture = load_corpus()
        with self.assertRaises(ValueError):
            self.shared.verify_corpus(fixture)

    def test_shared_neural_demonstration_preserves_restart(self):
        from biocircuit.bc01 import demo
        result = demo(
            self.corpus,
            ["millstream map and turned glove", "beetle specimen drawer",
             "purple spacecraft rocket 1982"],
            neurons=256, seed=31, exposures=8, shared=self.shared,
        )
        self.assertEqual(result["corpus_events"], 450)
        self.assertEqual(result["shared_fit_seed"], 31)
        self.assertEqual(result["shared_cache_manifest_sha256"], self.shared.cache_file_sha)
        self.assertEqual(result["shared_projection_version"], self.shared.projection_version)
        self.assertTrue(all(row["retrieval"]["verdict"] == "unknown" for row in result["tests"]))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "brain.npz"
            saved = demo(self.corpus, ["beetle specimen drawer"],
                         neurons=256, seed=31, exposures=8,
                         checkpoint=path, shared=self.shared)
            self.assertTrue(saved["tests"][0]["checkpoint_restart_exact"])
            meta = json.loads(path.with_suffix(".bc01.json").read_text())
            self.assertEqual(meta["shared_cache_manifest_sha256"],
                             self.shared.cache_file_sha)


if __name__ == "__main__":
    unittest.main()
