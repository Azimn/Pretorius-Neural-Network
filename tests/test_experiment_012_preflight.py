import sys
import unittest
from pathlib import Path

import numpy
import scipy

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from persona_net.v04_data import load_v04_split
from experiment_010_minimal_transferable_core import (
    compile_items,
    evaluate_both,
    graft,
    train_recurrent_only,
)
import experiment_012_core_convergence


class Experiment012PreflightTests(unittest.TestCase):
    def test_runtime_imports_are_available(self):
        self.assertTrue(numpy.__version__)
        self.assertTrue(scipy.__version__)
        self.assertIsNotNone(ExperienceEncoder)
        self.assertIsNotNone(PlasticRecurrentPersonaNet)
        self.assertIsNotNone(compile_items)
        self.assertIsNotNone(evaluate_both)
        self.assertIsNotNone(graft)
        self.assertIsNotNone(train_recurrent_only)
        self.assertIsNotNone(experiment_012_core_convergence.one_seed)

    def test_public_data_counts_match_protocol(self):
        data_root = ROOT / "data" / "v0_4"
        self.assertEqual(len(load_v04_split(data_root, "train")), 100)
        self.assertEqual(len(load_v04_split(data_root, "validation")), 40)
        self.assertEqual(len(load_v04_split(data_root, "adversarial")), 20)


if __name__ == "__main__":
    unittest.main()
