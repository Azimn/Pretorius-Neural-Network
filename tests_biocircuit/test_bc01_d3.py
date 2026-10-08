"""D3 regression: causal recurrent update, matched controls and collapse detection."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from biocircuit.bc01 import FIXTURE_PATH, load_corpus, neural_config, represent
from biocircuit.bc01_decisions import CARD_PATH, EVAL_ACTIONS
from biocircuit.bc01_d3 import _distribution, _train_targeted, experiment
from persona_net.encoding import ACTIONS, ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet


class BioCircuitD3Tests(unittest.TestCase):
    def setUp(self):
        self.corpus=load_corpus(FIXTURE_PATH)
        entries=json.loads(CARD_PATH.read_text(encoding="utf-8"))["cards"]
        ids=set(self.corpus.by_id())
        self.cards=tuple(c for c in entries if c["event_id"] in ids and
                         c["action"] in ("challenge","create"))
        self.assertEqual(len(self.cards),2)

    def test_constant_action_detector_rejects_collapsed_output(self):
        rows=[{"target":a,"conditions":{"targeted":{"choice":"create"}}}
              for a in EVAL_ACTIONS]
        stats=_distribution(rows,"targeted")
        self.assertTrue(stats["constant_action"])
        self.assertEqual(stats["distinct_choices"],1)
        self.assertEqual(stats["accuracy"],0.25)

    def test_teacher_updates_only_existing_target_action_edges(self):
        encoder=ExperienceEncoder(sensory_dim=256)
        net=PlasticRecurrentPersonaNet(neural_config(256,31),encoder)
        weights=net.W.data.copy()
        original_topology=(net.W.indices.copy(),net.W.indptr.copy())
        original_motor=net.motor_w.copy()
        _train_targeted(net,self.cards,encoder,None,epochs=2,ticks=8,
                        labels=tuple(c["action"] for c in self.cards),eta=.03)
        edited=np.flatnonzero(weights!=net.W.data)
        self.assertGreater(len(edited),0)
        allowed=np.isin(net.post_idx,np.concatenate(
            [net.action_populations[a] for a in ("challenge","create")]))
        self.assertTrue(np.all(allowed[edited]))
        np.testing.assert_array_equal(net.W.indices,original_topology[0])
        np.testing.assert_array_equal(net.W.indptr,original_topology[1])
        np.testing.assert_array_equal(net.motor_w,original_motor)
        positive=net.excitatory[net.pre_idx]
        self.assertTrue(np.all(net.W.data[positive]>=0))
        self.assertTrue(np.all(net.W.data[~positive]<=0))
        query=represent(self.cards[0]["probe"],encoder,None)
        self.assertTrue(np.all(query[encoder.action_offset:]==0))

    def test_source_anchored_diagnostic_contains_all_controls_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            trial=experiment(self.corpus,self.cards,seed=31,neurons=256,
                             mode="generic",epochs=1,background_ticks=2,
                             card_ticks=8,settle_ticks=8,
                             checkpoint_dir=Path(directory))
            self.assertEqual(trial["annotated_cards"],2)
            self.assertEqual(len(trial["rows"]),2)
            self.assertTrue(trial["checkpoint_restart_exact"])
            self.assertTrue((Path(directory)/"bc01_d3_generic_31.npz").is_file())
            self.assertGreater(trial["recurrent_delta_l1"]["targeted"],0)
            self.assertAlmostEqual(trial["recurrent_delta_l1"]["no_training"],0)
            self.assertEqual(set(trial["gates"]),{
                "more_than_one_action","above_balanced_chance",
                "beats_recurrent_lesion","beats_shuffled_teacher",
                "beats_blank_cue_training","beats_generic_hebb",
            })
            for value in trial["neural_cue_rms_spread"].values():
                self.assertTrue(np.isfinite(value))
            for row in trial["rows"]:
                self.assertEqual(set(row["conditions"]),{
                    "generic_hebb","targeted","targeted_shuffled",
                    "targeted_blank_cue","no_training","targeted_recurrent_lesion",
                })
                for condition in row["conditions"].values():
                    self.assertAlmostEqual(sum(condition["scores"].values()),1,places=9)
                    self.assertEqual(set(condition["trace"]["population_mean"]),set(EVAL_ACTIONS))

    def test_reject_bad_training_budget_and_action_leakage(self):
        with self.assertRaises(ValueError):
            experiment(self.corpus,self.cards,neurons=256,epochs=1,card_ticks=1,
                       background_ticks=1,settle_ticks=2)
        bad=(dict(self.cards[0],probe="The challenge was my choice"),self.cards[1])
        with self.assertRaises(ValueError):
            experiment(self.corpus,bad,neurons=256,epochs=1,card_ticks=8,
                       background_ticks=1,settle_ticks=2)


if __name__=="__main__":
    unittest.main()
