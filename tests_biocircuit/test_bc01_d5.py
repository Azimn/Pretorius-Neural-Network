"""BC01 D5: instrument-only guarantees, sensor/teacher separation and W necessity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from biocircuit.bc01 import FIXTURE_PATH, load_corpus, neural_config
from biocircuit.bc01_decisions import CARD_PATH
from biocircuit.bc01_d4 import _train_targeted, experiment as d4_experiment
from biocircuit.bc01_d5 import audit
from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet


class BC01D5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(FIXTURE_PATH)
        cards = json.loads(CARD_PATH.read_text(encoding="utf-8"))["cards"]
        ids = set(cls.corpus.by_id())
        cls.cards = tuple(c for c in cards if c["event_id"] in ids and
                          c["action"] in ("challenge", "create"))
        if len(cls.cards) != 2:
            raise AssertionError("Expected 2 anchored smoke source cards")

    def test_optional_observer_never_modifies_synapses_or_fast_state(self):
        encoder = ExperienceEncoder(sensory_dim=256)
        net = PlasticRecurrentPersonaNet(neural_config(256, 31), encoder)
        copied = PlasticRecurrentPersonaNet(neural_config(256, 31), encoder)
        snapshot = (net.W.indices.copy(), net.W.indptr.copy(), net.motor_w.copy())
        trace = []
        kwargs = dict(cards=self.cards, encoder=encoder, circuit=None,
                      shared=None, epochs=2, ticks=8,
                      labels=tuple(c["action"] for c in self.cards), eta=.03,
                      sensory_gain=12.0)
        _train_targeted(net, **kwargs)
        _train_targeted(copied, observer=trace.append, **kwargs)
        self.assertEqual(len(trace), 4)
        np.testing.assert_array_equal(net.W.data, copied.W.data)
        np.testing.assert_array_equal(net.rate, copied.rate)
        np.testing.assert_array_equal(net.v, copied.v)
        self.assertEqual(net.tick, copied.tick)
        np.testing.assert_array_equal(copied.W.indices, snapshot[0])
        np.testing.assert_array_equal(copied.W.indptr, snapshot[1])
        np.testing.assert_array_equal(copied.motor_w, snapshot[2])
        for row in trace:
            self.assertLessEqual(row["eligible_edges"], row["candidate_incoming_edges"])
            self.assertEqual(row["eligible_edges"],row["eligible_exc"]+row["eligible_inh"])
            self.assertLessEqual(row["modified_edges"], row["eligible_edges"])
            self.assertLessEqual(row["clipped_edges"], row["eligible_edges"])
            self.assertGreaterEqual(row["cue_teacher_input_ratio"], 0.0)
            self.assertGreaterEqual(row["recurrent_delta_l1"], 0.0)

    def test_existing_D4_policy_bitwise_matches_diagnostic(self):
        kwargs = dict(seed=31, mode="generic", sensory_gain=1.0,
                      neurons=256, epochs=1, background_ticks=2,
                      card_ticks=8, settle_ticks=8)
        with tempfile.TemporaryDirectory() as folder:
            result = audit(self.corpus, self.cards, **kwargs, checkpoint_dir=Path(folder))
        previous = d4_experiment(self.corpus, self.cards, **kwargs)
        self.assertTrue(result["checkpoint_restart_exact"])
        self.assertEqual(result["fixed_D4_baseline_accuracy"], previous["result"]["targeted"]["accuracy"])
        self.assertEqual(result["source_card_count"],2)
        self.assertEqual(result["summary"]["observations"],2)
        self.assertEqual(result["summary"]["distinct_training_card_ids"],2)
        self.assertEqual(set(result["summary"]["by_action"]), {"challenge","create"})
        self.assertEqual(len(result["query_counterfactual_traces"]),2)
        for i, row in enumerate(result["query_counterfactual_traces"]):
            self.assertEqual(row["intact"]["scores"], previous["rows"][i]["conditions"]["targeted"]["scores"])
            self.assertAlmostEqual(sum(row["no_cue_query"]["scores"].values()),1.0,places=8)
            self.assertLessEqual(abs(row["target_margin_change_from_lesion"]), 1.0)
        self.assertEqual(len(result["training_event_traces"]),2)

    def test_stronger_cue_can_be_audited_without_changing_teacher(self):
        kwargs = dict(seed=31,mode="generic",neurons=256,epochs=1,
                      background_ticks=2,card_ticks=8,settle_ticks=8)
        low=audit(self.corpus,self.cards,sensory_gain=1.,**kwargs)
        high=audit(self.corpus,self.cards,sensory_gain=12.,**kwargs)
        for x,y in zip(low["training_event_traces"], high["training_event_traces"]):
            self.assertAlmostEqual(y["fixed_teacher_drive_l2"],
                                   x["fixed_teacher_drive_l2"],places=7)
            self.assertAlmostEqual(y["fixed_cue_drive_l2"],
                                   x["fixed_cue_drive_l2"]*12,places=5)
            self.assertTrue(np.isfinite(y["teacher_counterfactual_target_mean"]))
        self.assertEqual(high["recurrent_csr_nonzeros"],low["recurrent_csr_nonzeros"])

    def test_d5_rejects_unregistered_gain(self):
        with self.assertRaises(ValueError):
            audit(self.corpus,self.cards,sensory_gain=4.,
                  neurons=256,seed=31,mode="generic",
                  background_ticks=2,epochs=1,card_ticks=8,settle_ticks=8)


if __name__ == "__main__":
    unittest.main()
