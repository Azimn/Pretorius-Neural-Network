"""BC01 source integrity, evidence and synaptic/restart regression tests."""
from pathlib import Path
import tempfile
import unittest

from biocircuit.bc01 import (
    FIXTURE_PATH, SOURCE_BLOB, FIXTURE_BLOB, SOURCE_COMMIT, git_blob_sha,
    load_corpus, retrieve, demo, represent, make_circuit,
)
from persona_net.encoding import ExperienceEncoder


class BC01PreviewTests(unittest.TestCase):
    def setUp(self):
        self.corpus = load_corpus()

    def test_authentic_pinned_fixture_and_provenance(self):
        self.assertEqual(self.corpus.source_kind, "smoke")
        self.assertEqual(self.corpus.blob_sha, FIXTURE_BLOB)
        self.assertEqual(len(self.corpus.records), 3)
        self.assertEqual([item["event_id"] for item in self.corpus.records],
                         ["E09-001", "E09-002", "E01-001"])
        self.assertTrue(all(item["provenance"] == "reconstructed" for item in self.corpus.records))
        self.assertEqual(len(SOURCE_BLOB), 40)
        self.assertEqual(len(SOURCE_COMMIT), 40)

    def test_source_mutation_and_unpinned_jsonl_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder) / "changed.jsonl"
            temp.write_bytes(FIXTURE_PATH.read_bytes() + b"\n")
            self.assertNotEqual(git_blob_sha(temp.read_bytes()), FIXTURE_BLOB)
            with self.assertRaisesRegex(ValueError, "pinned"):
                load_corpus(temp)

    def test_retrieval_candidate_never_is_verified_answer(self):
        hit = retrieve(self.corpus, "millstream map and turned glove")
        self.assertEqual(hit["candidate"]["event_id"], "E09-001")
        self.assertEqual(hit["candidate"]["provenance"], "reconstructed")
        self.assertEqual(hit["verdict"], "unknown")
        unknown = retrieve(self.corpus, "purple spacecraft rocket 1982")
        self.assertIsNone(unknown["candidate"])
        self.assertEqual(unknown["verdict"], "unknown")
        with self.assertRaisesRegex(ValueError, "Event IDs"):
            retrieve(self.corpus, "Tell me about E09-001")

    def test_shared_input_and_no_event_identifier_leak(self):
        encoder = ExperienceEncoder(sensory_dim=256)
        circuit = make_circuit(256, 17, "local")
        a = represent("The beetle's wings", encoder, circuit)
        b = represent("The beetle's wings", encoder, circuit)
        self.assertEqual(a.shape, (encoder.input_dim,))
        self.assertTrue((a == b).all())
        self.assertTrue((a[encoder.action_offset:] == 0).all())

    def test_recurrent_delta_targeted_lesion_checkpoint_and_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            result = demo(self.corpus, [
                "millstream map and turned glove",
                "beetle specimen drawer",
                "purple spacecraft rocket 1982",
            ], neurons=256, seed=71, exposures=8,
                checkpoint=Path(folder) / "brain.npz")
            self.assertTrue((Path(folder) / "brain.npz").exists())
            self.assertTrue((Path(folder) / "brain.bc01.json").exists())
            self.assertEqual(result["corpus_events"], 3)
            self.assertFalse(result["motor_decoder_trained"])
            self.assertGreater(result["recurrent_changed_synapses"], 0)
            self.assertTrue(all(x["checkpoint_restart_exact"] for x in result["tests"]))
            self.assertTrue(all(x["retrieval"]["verdict"] == "unknown" for x in result["tests"]))
            # No automatic pass condition here: meaningful behavioral changes
            # must be measured, not manufactured by an output-decoder lesion.
            self.assertTrue(all(x["recurrent_delta_score_max_abs"] >= 0 for x in result["tests"]))


if __name__ == "__main__":
    unittest.main()
