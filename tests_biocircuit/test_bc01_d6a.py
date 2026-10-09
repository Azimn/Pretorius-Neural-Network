"""D6A: strict action-label fold isolation and explicit transductive status."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from biocircuit.bc01 import Corpus
from biocircuit.bc01_decisions import CARD_PATH, EVAL_ACTIONS
from biocircuit.bc01_d6a import load_candidate_challenge, strata, experiment


class BC01D6ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards=tuple(json.loads(CARD_PATH.read_text(encoding="utf-8"))["cards"])
        records=[]
        for i,c in enumerate(cls.cards):
            records.append({
                "event_id":c["event_id"],"decisions":c["source_decision"],
                "recall_cues":[c["training_cues"],c["probe"]],
                "memory_text":"I remember "+c["training_cues"]+". "+c["probe"],
                "episode_id":c["event_id"].split("-")[0],
                "chronological_order":i+1,
            })
        cls.synthetic_corpus=Corpus(tuple(records),"smoke","fixture-not-a-real-autobiography")

    def test_candidate_set_is_explicitly_unreviewed_and_cannot_leak_label(self):
        payload=load_candidate_challenge(self.cards)
        self.assertIn("NOT independently reviewed",payload["status"])
        self.assertEqual(len(payload["paraphrases"]),16)
        bad=json.loads(json.dumps(payload))
        bad["paraphrases"][self.cards[0]["event_id"]]="challenge on E09-001"
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"bad.json"
            file.write_text(json.dumps(bad),encoding="utf-8")
            with self.assertRaises(ValueError):
                load_candidate_challenge(self.cards,file)

    def test_holdout_balanced_complete_and_no_label_training_leakage(self):
        folds=strata(self.cards)
        self.assertEqual(len(folds),4)
        seen=[]
        for train,test in folds:
            self.assertEqual(len(train),12)
            self.assertEqual(len(test),4)
            self.assertEqual(set(c["action"] for c in test),set(EVAL_ACTIONS))
            self.assertEqual({a:sum(x["action"]==a for x in train) for a in EVAL_ACTIONS},
                             {a:3 for a in EVAL_ACTIONS})
            self.assertFalse(set(x["event_id"] for x in test)&
                             set(x["event_id"] for x in train))
            seen.extend(x["event_id"] for x in test)
        self.assertEqual(set(seen),set(x["event_id"] for x in self.cards))
        self.assertEqual(len(seen),len(set(seen)))

    def test_synthetic_fourfold_development_harness_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            report=experiment(
                self.synthetic_corpus,self.cards,seed=31,mode="generic",
                epochs=1,card_ticks=4,background_ticks=2,settle_ticks=4,
                checkpoint_dir=Path(directory),
            )
            self.assertTrue(report["never_independent_or_sealed"])
            self.assertEqual(len(report["folds"]),4)
            self.assertEqual(report["label_card_count"],16)
            for fold in report["folds"]:
                self.assertTrue(fold["decoder_recurrent_W_preserved"])
                self.assertTrue(fold["shuffled_recurrent_W_preserved"])
                self.assertTrue(fold["frozen_decoder_parameters_exact"])
                self.assertTrue(fold["source_background_contains_all_heldout_events"])
                self.assertFalse(fold["l2_encoder_train_fit_excludes_test_episodes"])
                self.assertEqual(len(fold["train_event_ids"]),12)
                self.assertEqual(len(fold["heldout_rows"]),8)
                self.assertEqual(len(fold["unknown_rows"]),4)
                self.assertTrue((Path(directory)/f"d6a_generic_31_fold{fold['fold']}.npz").is_file())
                for row in fold["heldout_rows"]:
                    self.assertNotIn(row["event_id"],fold["train_event_ids"])
                    for c in ("decoder_frozen_recurrence","decoder_with_cue_contrast_W",
                              "decoder_with_W_only_lesion","decoder_shuffled_labels"):
                        self.assertAlmostEqual(
                            sum(row["conditions"][c]["scores"].values()),1,places=8)
                    self.assertTrue(np.isfinite(row["conditions"]["decoder_frozen_recurrence"]["confidence"]))
            self.assertTrue(all(f["metrics"]["assistant_paraphrase"]["decoder_frozen_recurrence"]
                                ["coverage"]==1 for f in report["folds"]))

    def test_not_yet_independent_evaluation(self):
        with self.assertRaises(ValueError):
            strata(self.cards[:-1])


if __name__=="__main__":
    unittest.main()
