"""001C source-intake audit tests. No labels or neural experiment are produced."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_chimera_001c_v12 import (
    CORPUS_PATH, SIDECAR_PATH, audit, components,
)


def memory(event_id, chronological_order, links=()):
    return {
        "event_id": event_id,
        "episode_id": "E01",
        "chronological_order": chronological_order,
        "title": "Reconstructed event",
        "memory_text": "This is fictional and reconstructed.",
        "decisions": "I observed and considered options.",
        "consequences": "Another possibility emerged.",
        "belief_changes": "The evidence remained partial.",
        "relationship_changes": "No established change.",
        "provenance": "reconstructed",
        "links_to_prior_events": list(links),
    }


class IntakeTests(unittest.TestCase):
    def test_connected_components_detect_crosslinks(self):
        rows = [
            memory("E01-001", 1),
            memory("E01-002", 2, ["E01-001"]),
            memory("E01-003", 3, ["E01-002"]),
            memory("E01-004", 4),
        ]
        groups, links = components(rows)
        self.assertEqual([x["n"] for x in groups], [3, 1])
        self.assertEqual(links, 2)
        with self.assertRaisesRegex(ValueError, "Dangling"):
            components([memory("E01-001", 1, ["not-an-event"])])

    def test_no_action_labels_or_independent_split_are_invented(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / CORPUS_PATH
            sidecar = root / SIDECAR_PATH
            corpus.parent.mkdir(parents=True)
            sidecar.parent.mkdir(parents=True)
            rows = [memory("E01-001", 1), memory("E01-002", 2, ["E01-001"])]
            corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
            sidecar.write_text("".join(json.dumps({
                "event_id": r["event_id"],
                "annotation_status": "unreviewed_candidate",
                "cue_ids": ["cue:toy"],
            }) + "\n" for r in rows))
            report, queue = audit(root, strict=False)
            self.assertEqual(report["action_label_count"], 0)
            self.assertEqual(report["approved_evaluation_item_count"], 0)
            self.assertEqual(report["event_count"], 2)
            self.assertEqual(report["connected_components"][0]["n"], 2)
            self.assertTrue(all(row["target_actions"] is None for row in queue))
            self.assertTrue(all(row["evaluation_role"] == "unassigned_memory_graph_entangled"
                                for row in queue))
            self.assertFalse(report["terminal_battery_touched"])
            self.assertFalse(report["historical_experiment_001_reproduced"])

    def test_duplicate_sidecar_and_unexpected_behavioral_target_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / CORPUS_PATH
            sidecar = root / SIDECAR_PATH
            corpus.parent.mkdir(parents=True)
            sidecar.parent.mkdir(parents=True)
            row = memory("E01-001", 1)
            corpus.write_text(json.dumps(row) + "\n")
            one = json.dumps({"event_id": "E01-001", "annotation_status": "unreviewed_candidate"})
            sidecar.write_text(one + "\n" + one + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit(root, strict=False)
            sidecar.write_text(one + "\n")
            row["target_actions"] = {"create": 1.0}
            corpus.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "motor target"):
                audit(root, strict=False)


if __name__ == "__main__":
    unittest.main()
