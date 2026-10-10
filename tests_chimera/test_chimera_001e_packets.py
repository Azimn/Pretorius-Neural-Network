"""Review packet builder has no answers and does not accidentally validate templates."""
from __future__ import annotations

import json
import unittest

from scripts.build_chimera_001e_review_packets import (
    ACTIONS, render_packet, empty_template,
)
from scripts.validate_chimera_001e_reviews import (
    ROOT, DRAFTS, SOURCE_QUEUE, load_rows, audit_reviews,
)


class ReviewPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drafts=load_rows(DRAFTS)
        cls.queue=load_rows(SOURCE_QUEUE)

    def test_each_reviewer_gets_all_scenarios_once_with_matching_ids(self):
        self.assertEqual(len(ACTIONS),10)
        x=render_packet(self.drafts,self.queue,"A",62001)
        y=render_packet(self.drafts,self.queue,"B",62002)
        self.assertNotEqual(x,y)
        for packet in (x,y):
            for draft in self.drafts:
                self.assertEqual(packet.count(draft["scenario"]),1)
                self.assertEqual(packet.count("## Card "),12)
                self.assertNotIn("Intact JS similarity",packet)
                self.assertNotIn("trained decoder",packet.lower())
                self.assertNotIn('"target_actions":',packet)

    def test_empty_examples_cannot_pass_rater_validator(self):
        for label in ("A","B"):
            lines=empty_template(self.drafts,label).splitlines()
            self.assertEqual(len(lines),12)
            templates=[json.loads(x) for x in lines]
            self.assertTrue(all(x["target_actions"] is None for x in templates))
            self.assertTrue(all(x["template_not_submitted"] is True for x in templates))
            self.assertTrue(all(x["attests_independent_read"] is False for x in templates))
            self.assertTrue(all(x["saw_model_predictions"] is None for x in templates))
            with self.assertRaises(ValueError):
                audit_reviews(self.drafts,templates,[])


if __name__=="__main__":
    unittest.main()
