"""BC01 tests exact cross-repo cache replay without altered neural experiments."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from biocircuit.bc01 import load_corpus, demo, represent, make_circuit, FIXTURE_PATH
from biocircuit.shared_memory import load_bc_shared
from persona_net.encoding import ExperienceEncoder

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream-connectome"
ARTIFACT = UPSTREAM / "artifacts/shared_memory/v1"
FULL = UPSTREAM / "memories/current/Pretorius_v12_450_Events_Complete.jsonl"


@unittest.skipUnless(FULL.is_file() and (ARTIFACT / "bc01_l2_manifest.json").is_file(),
                     "upstream canonical bundle required; BC01 workflow creates it")
class BC01SharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(FULL)
        cls.values, cls.manifest = load_bc_shared(ARTIFACT, cls.corpus)

    def test_full_corpus_and_original_encoder_exact(self):
        self.assertEqual(self.values.shape, (450, 256))
        encoder = ExperienceEncoder(sensory_dim=256)
        for i in (0, 53, 200, 449):
            expected = encoder.encode(self.corpus.records[i]["memory_text"]).vector[:256]
            np.testing.assert_array_equal(self.values[i], expected)
        self.assertEqual(len(self.manifest["record_ids_ordered"]), 450)
        self.assertEqual(self.manifest["fit_event_ids"], [])

    def test_cached_compartment_and_generic_inputs_exact(self):
        encoder = ExperienceEncoder(sensory_dim=256)
        for i in (0, 55, 449):
            text = self.corpus.records[i]["memory_text"]
            for circuit in (None, make_circuit(128, 34, "local")):
                raw = represent(text, encoder, circuit)
                cached = represent(text, encoder, circuit, sensory_override=self.values[i])
                np.testing.assert_array_equal(raw, cached)
        question = "What did I recall of the beetle?"
        np.testing.assert_array_equal(represent(question, encoder, None),
                                      represent(question, encoder, None))

    def test_smoke_neural_replay_not_affected_by_cache(self):
        smoke = load_corpus(FIXTURE_PATH)
        index = {row["event_id"]: i for i, row in enumerate(self.corpus.records)}
        vectors = np.stack([self.values[index[row["event_id"]]]
                            for row in smoke.records])
        questions = ["beetle specimen drawer", "purple spacecraft rocket 1982"]
        baseline = demo(smoke, questions, neurons=128, seed=97,
                        mode="local", exposures=1)
        shared = demo(smoke, questions, neurons=128, seed=97,
                      mode="local", exposures=1, shared_sensory=vectors)
        self.assertEqual(baseline["tests"], shared["tests"])
        self.assertEqual(baseline["recurrent_changed_synapses"],
                         shared["recurrent_changed_synapses"])
        self.assertFalse(baseline["shared_cached_source"])
        self.assertTrue(shared["shared_cached_source"])

    def test_refuse_tamper_and_untrusted_source(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            for p in ARTIFACT.iterdir():
                if p.is_file():
                    shutil.copy2(p, dest / p.name)
            raw = json.loads((dest / "manifest.json").read_text())
            raw["encoder_name"] = "tfidf-unrelated"
            (dest / "manifest.json").write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_bc_shared(dest, self.corpus)
            shutil.copy2(ARTIFACT / "manifest.json", dest / "manifest.json")
            (dest / "bc01_sensory_256.npy").write_bytes(
                (dest / "bc01_sensory_256.npy").read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                load_bc_shared(dest, self.corpus)
        with self.assertRaises(ValueError):
            load_bc_shared(ARTIFACT, load_corpus(FIXTURE_PATH))


if __name__ == "__main__":
    unittest.main()
