import json
import unittest
from pathlib import Path

from persona_net.v04_data import load_v04_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v0_4"


class V04PublicDataTests(unittest.TestCase):
    def test_public_split_counts(self):
        self.assertEqual(len(load_v04_split(DATA, "train")), 100)
        self.assertEqual(len(load_v04_split(DATA, "validation")), 40)
        self.assertEqual(len(load_v04_split(DATA, "adversarial")), 20)

    def test_training_covers_twenty_axes_five_times_each(self):
        train = load_v04_split(DATA, "train")
        counts = {}
        for item in train:
            axis = item["id"].rsplit("_train_", 1)[0]
            counts[axis] = counts.get(axis, 0) + 1
        self.assertEqual(len(counts), 20)
        self.assertTrue(all(v == 5 for v in counts.values()))

    def test_terminal_material_is_not_public_payload(self):
        self.assertFalse((DATA / "terminal_inputs.json").exists())
        self.assertFalse((DATA / "terminal_key.json").exists())
        manifest = json.loads((DATA / "manifest.json").read_text())
        self.assertIn("terminal_inputs_v1.json", manifest["source_hashes"])
        self.assertIn("terminal_key_v1.json", manifest["source_hashes"])


if __name__ == "__main__":
    unittest.main()
