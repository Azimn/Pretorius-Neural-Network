"""D5B: one cue-minus-blank update, fixed controls and old-model invariants."""
import json
import unittest

import numpy as np

from biocircuit.bc01 import FIXTURE_PATH, load_corpus, neural_config
from biocircuit.bc01_decisions import CARD_PATH
from biocircuit.bc01_d4 import _train_targeted, experiment as d4_experiment
from biocircuit.bc01_d5b import comparison
from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet


class BC01D5BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus=load_corpus(FIXTURE_PATH)
        cards=json.loads(CARD_PATH.read_text(encoding="utf-8"))["cards"]
        ids=set(cls.corpus.by_id())
        cls.cards=tuple(x for x in cards if x["event_id"] in ids and
                        x["action"] in ("challenge","create"))
        if len(cls.cards)!=2:
            raise ValueError("Expected two anchored source smoke cards")

    def test_blank_cue_yields_no_recurrent_learning_under_new_local_gate(self):
        encoder=ExperienceEncoder(sensory_dim=256)
        net=PlasticRecurrentPersonaNet(neural_config(256,31),encoder)
        before=net.W.data.copy()
        _train_targeted(net,self.cards,encoder,None,None,epochs=2,ticks=8,
                        labels=tuple(x["action"] for x in self.cards),
                        eta=.03,blank_cues=True,sensory_gain=12.,
                        presynaptic_mode="cue_minus_blank")
        np.testing.assert_array_equal(net.W.data,before)

    def test_only_named_presynaptic_rule_changes_intact_source_behavior(self):
        settings=dict(seed=31,mode="generic",neurons=256,
                      sensory_gain=12.,background_ticks=2,epochs=1,
                      card_ticks=8,settle_ticks=8)
        corrected=comparison(self.corpus,self.cards,**settings)
        old=d4_experiment(self.corpus,self.cards,**settings)
        self.assertTrue(corrected["cue_contrast"]["checkpoint_exact"])
        self.assertEqual(corrected["original_D4"]["accuracy"],old["result"])
        self.assertTrue(corrected["motor_decoder_only"]["w_recurrent_unchanged"])
        self.assertEqual(len(corrected["per_card"]),2)
        self.assertEqual(corrected["cue_evoked_eligibility"]["presentations"],2)
        self.assertGreaterEqual(corrected["cue_evoked_eligibility"]["eligible_fraction_mean"],0)
        self.assertLessEqual(corrected["cue_evoked_eligibility"]["eligible_fraction_mean"],1)
        self.assertEqual(corrected["cue_contrast"]["accuracy"]["no_training"],
                         old["result"]["no_training"]["accuracy"])
        self.assertEqual(corrected["cue_contrast"]["accuracy"]["generic_hebb"],
                         old["result"]["generic_hebb"]["accuracy"])
        self.assertAlmostEqual(corrected["cue_contrast"]["recurrent_delta_l1"]["targeted_blank_cue"],
                               0.0,places=8)
        self.assertTrue(all("cue_contrast_shuffled" in row for row in corrected["per_card"]))

    def test_legacy_default_unchanged_and_reject_unregistered_rule(self):
        kwargs=dict(seed=31,mode="generic",neurons=256,epochs=1,
                    card_ticks=8,background_ticks=2,settle_ticks=8)
        original=d4_experiment(self.corpus,self.cards,**kwargs)
        explicit=d4_experiment(self.corpus,self.cards,
                               presynaptic_mode="target_rate",**kwargs)
        for name in ("rows","result","recurrent_delta_l1","neural_cue_rms_spread"):
            self.assertEqual(original[name], explicit[name])
        self.assertEqual(original["schema"],"BC01-D4-source-input-amplitude-v1")
        with self.assertRaises(ValueError):
            d4_experiment(self.corpus,self.cards,
                          presynaptic_mode="unsupported",**kwargs)
        with self.assertRaises(ValueError):
            comparison(self.corpus,self.cards,sensory_gain=4.,
                       neurons=256,epochs=1,card_ticks=8,background_ticks=2,settle_ticks=8)


if __name__=="__main__":
    unittest.main()
