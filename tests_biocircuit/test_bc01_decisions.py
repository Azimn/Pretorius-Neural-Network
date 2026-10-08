"""BC01-D1 grounded action assay guardrails (small offline fixture)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from biocircuit.bc01 import FIXTURE_PATH, load_corpus, represent
from biocircuit.bc01_decisions import (
    CARD_PATH, EVAL_ACTIONS, benchmark, read_population,
    train_decoder_only, validate_cards,
)
from biocircuit.bc01 import neural_config
from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet


class BC01DecisionTests(unittest.TestCase):
    def setUp(self):
        self.corpus = load_corpus(FIXTURE_PATH)
        cards = json.loads(CARD_PATH.read_text(encoding="utf-8"))["cards"]
        ids = {row["event_id"] for row in self.corpus.records}
        self.cards = tuple(row for row in cards if row["event_id"] in ids and row["action"] in ("challenge", "create"))
        self.assertEqual(len(self.cards), 2)

    def test_exact_decision_provenance_and_no_action_in_question(self):
        validate_cards(self.corpus, self.cards)
        source = self.corpus.by_id()
        for card in self.cards:
            self.assertEqual(card["source_decision"], source[card["event_id"]]["decisions"])
            self.assertNotIn(card["action"], card["probe"].lower())
            encoder = ExperienceEncoder(sensory_dim=256)
            cue = encoder.encode(card["probe"]).vector
            self.assertTrue(np.all(cue[encoder.action_offset:] == 0))

    def test_modified_decision_refuses_training(self):
        tampered = [copy.deepcopy(row) for row in self.cards]
        tampered[0]["source_decision"] = "I always chose to obey everyone."
        with self.assertRaisesRegex(ValueError, "original source decision"):
            validate_cards(self.corpus, tuple(tampered))
        tampered = [copy.deepcopy(row) for row in self.cards]
        tampered[0]["probe"] = "E09-001"
        with self.assertRaises(ValueError):
            validate_cards(self.corpus, tuple(tampered))

    def test_output_decoder_only_leaves_recurrent_matrix_unchanged(self):
        cfg = neural_config(256, 13)
        encoder = ExperienceEncoder(sensory_dim=256)
        net = PlasticRecurrentPersonaNet(cfg, encoder)
        baseline = net.W.data.copy()
        train_decoder_only(net, self.cards, encoder, None, epochs=2, settle_ticks=5)
        self.assertTrue(np.array_equal(baseline, net.W.data))
        cue = represent(self.cards[0]["probe"], encoder, None)
        result = read_population(net, cue, ticks=8)
        self.assertEqual(set(result), set(EVAL_ACTIONS))
        self.assertAlmostEqual(sum(result.values()), 1.0, places=8)

    def test_two_source_events_full_control_suite_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark(
                self.corpus, self.cards, neurons=256, seed=31,
                epochs=1, card_ticks=8, background_ticks=2,
                settle_ticks=8, checkpoint_dir=Path(tmp),
            )
            self.assertEqual(result["annotated_decision_cards"], 2)
            self.assertTrue(result["checkpoint_reproduced_all"])
            self.assertGreater(result["recurrent_delta_l1"], 0.0)
            self.assertFalse(result["recurrent_conditions_decoder_updated"])
            self.assertTrue(result["decoder_only_output_trained"])
            self.assertEqual(set(result["accuracy"]), {
                "intact", "recurrent_lesion", "no_plasticity", "shuffled_labels",
                "cue_only", "decoder_only", "retrieval_only",
            })
            for row in result["rows"]:
                self.assertEqual(row["conditions"]["intact"]["scores"],
                                 row["conditions"]["restarted"]["scores"])
                self.assertEqual(row["conditions"]["retrieval_only"]["choice"], row["target"])
            self.assertTrue((Path(tmp) / "bc01_d1_local_31.npz").exists())


if __name__ == "__main__":
    unittest.main()
