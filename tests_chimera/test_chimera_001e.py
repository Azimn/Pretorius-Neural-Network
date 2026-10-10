"""001E review-release gate tests. No actual labels or model results produced."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.validate_chimera_001e_reviews import (
    ROOT, PINNED_SOURCE, audit_reviews, check_drafts, check_targets,
    digest, load_rows,
)


class IndependentLabelGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queue = load_rows(ROOT/"results/chimera_001c/REVIEW_QUEUE.jsonl")
        cls.drafts = load_rows(ROOT/"research/chimera_001e/SCENARIO_DRAFTS.jsonl")
        cls.exposed = []
        for name in ("pretorius_validation_v1.json", "pretorius_adversarial_v1.json"):
            cls.exposed += json.loads(
                (ROOT/"data/phenotype_battery"/name).read_text())["items"]

    def test_real_source_queue_scenarios_unreviewed_and_no_old_exact_duplicates(self):
        status=check_drafts(self.drafts,self.queue,self.exposed)
        self.assertEqual(status["draft_count"],12)
        self.assertEqual(status["distinct_source_events"],12)
        self.assertEqual(len(status["draft_sha256"]),12)
        self.assertTrue(all(c["target_actions"] is None for c in self.drafts))
        self.assertTrue(all(c["evaluation_role"]=="unassigned" for c in self.drafts))
        approved,state=audit_reviews(self.drafts,[],[])
        self.assertEqual(approved,[])
        self.assertEqual(state["approved_count"],0)
        self.assertTrue(state["no_confirmatory_holdout_released"])

    def test_source_hash_or_unreviewed_target_breaches_block(self):
        one=copy.deepcopy(self.drafts[:1])
        one[0]["target_actions"]={"cooperate":1.0}
        with self.assertRaisesRegex(ValueError,"cannot contain"):
            check_drafts(one,self.queue,self.exposed,expected=1)
        one=copy.deepcopy(self.drafts[:1])
        one[0]["source_commit"]="different_commit"
        with self.assertRaisesRegex(ValueError,"Source commit"):
            check_drafts(one,self.queue,self.exposed,expected=1)

    def test_reviewer_attestations_separate_adjudicator_and_sum_one(self):
        one=self.drafts[:1]
        cid=one[0]["id"]
        sdigest=digest(one[0]["scenario"])
        reviews=[{
            "candidate_id":cid,
            "scenario_sha256":sdigest,
            "reviewer_id":r,
            "attests_independent_read":True,
            "saw_model_predictions":False,
            "target_actions":{"approach":0.5,"challenge":0.5},
            "review_note":"I separately reviewed the hypothetical event and alternatives.",
        } for r in ("reviewer-one","reviewer-two")]
        adjudications=[{
            "candidate_id":cid,
            "scenario_sha256":sdigest,
            "adjudicator_id":"third-reviewer",
            "decision":"approve",
            "target_actions":{"approach":0.4,"challenge":0.6},
            "adjudication_note":"Compared both independent textual rationales before choosing distribution.",
        }]
        approved,state=audit_reviews(one,reviews,adjudications)
        self.assertEqual(state["approved_count"],1)
        self.assertEqual(approved[0]["evaluation_role"],
                         "candidate_pending_holdout_freeze")
        self.assertEqual(approved[0]["target_actions"]["challenge"],0.6)
        bad=copy.deepcopy(reviews)
        bad[1]["reviewer_id"]="reviewer-one"
        with self.assertRaisesRegex(ValueError,"Duplicate rating"):
            audit_reviews(one,bad,adjudications)
        bad=copy.deepcopy(reviews)
        bad[0]["saw_model_predictions"]=True
        with self.assertRaisesRegex(ValueError,"model-blind"):
            audit_reviews(one,bad,adjudications)
        bad=copy.deepcopy(adjudications)
        bad[0]["adjudicator_id"]="reviewer-two"
        with self.assertRaisesRegex(ValueError,"separate"):
            audit_reviews(one,reviews,bad)
        bad=copy.deepcopy(adjudications)
        bad[0]["scenario_sha256"]="sha-of-changed-scenario"
        with self.assertRaisesRegex(ValueError,"changed scenario"):
            audit_reviews(one,reviews,bad)

    def test_invalid_action_mass_or_unknown_classes_fail(self):
        for targets in (
            {"invent":1.0},
            {"approach":0.6},
            {"approach":-0.2,"cooperate":1.2},
            {"approach":True},
        ):
            with self.assertRaises(ValueError):
                check_targets(targets)


if __name__=="__main__":
    unittest.main()
