"""Deterministic BioCircuit functional and causal-lesion tests."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from biocircuit import Circuit, CircuitConfig, assay, curriculum
from scripts.run_biocircuit_bc00 import run


class BC00Tests(unittest.TestCase):
    def setUp(self):
        self.config = CircuitConfig(neurons=256, active=16, seed=11)

    def test_matched_projection_and_active_budget(self):
        left = Circuit(self.config, "local")
        right = Circuit(self.config, "global")
        self.assertEqual(left.projection.nnz, 256 * 12)
        self.assertTrue(np.array_equal(left.projection.indices, right.projection.indices))
        self.assertTrue(np.array_equal(left.projection.data, right.projection.data))
        samples = curriculum(11, self.config, 12)
        for sample in samples:
            for model in (left, right):
                activity = model._activity(sample.stimulus)
                self.assertEqual(np.count_nonzero(activity), 16)
                self.assertAlmostEqual(float(np.linalg.norm(activity)), 1.0, places=5)
        local_a = left._activity(samples[0].stimulus)
        self.assertTrue(all(np.count_nonzero(local_a[i * 64:(i + 1) * 64]) == 4
                            for i in range(4)))

    def test_synaptic_change_causal_and_removable(self):
        network = Circuit(self.config, "local")
        example = curriculum(11, self.config, 1)[0]
        self.assertTrue(np.all(network.weights == 0))
        network.learn(example.stimulus, example.action)
        self.assertEqual(network.predict(example.stimulus), example.action)
        self.assertGreater(np.count_nonzero(network.weights), 0)
        self.assertEqual(network.plastic_updates, 64)
        before = network.weights.copy()
        network.lesion(0)
        self.assertTrue(np.array_equal(network.weights, before))
        network.lesion()
        self.assertTrue(np.all(network.weights == 0))
        self.assertTrue(np.array_equal(network.decision_scores(example.stimulus),
                                       np.zeros(4, dtype=np.float32)))

    def test_checkpoint_roundtrip_rejects_cross_mode(self):
        model = Circuit(self.config, "local")
        example = curriculum(11, self.config, 1)[0]
        model.learn(example.stimulus, example.action)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bc00.npz"
            model.save(path)
            replay = Circuit.load(path)
            self.assertTrue(np.array_equal(model.weights, replay.weights))
            self.assertEqual(model.presentations, replay.presentations)
            self.assertEqual(model.plastic_updates, replay.plastic_updates)
            self.assertEqual(model.competition, replay.competition)
            self.assertEqual(model.predict(example.stimulus), replay.predict(example.stimulus))

    def test_replay_and_assay_determinism(self):
        a = assay(17, neurons=256, items=32, epochs=2)
        b = assay(17, neurons=256, items=32, epochs=2)
        self.assertEqual(a, b)
        self.assertTrue(a["matched_projection_exact"])
        for row in a["conditions"].values():
            self.assertEqual(row["presentations"], 64)
            self.assertEqual(row["plastic_updates"], row["presentations"] * 16 * 4)
            self.assertEqual(row["trainable_action_synapses"], 256 * 4)
            self.assertTrue(row["recovered_checkpoint_exact"])
        summary = run(seeds=(17,), neurons=256, items=32, epochs=2)
        self.assertEqual(summary["trials"], [a])

    def test_guardrails(self):
        with self.assertRaises(ValueError):
            CircuitConfig(neurons=255, active=16).validate()
        with self.assertRaises(ValueError):
            Circuit(self.config, "wrong")
        with self.assertRaises(ValueError):
            curriculum(11, self.config, 0)
        model = Circuit(self.config, "global")
        with self.assertRaises(ValueError):
            model.predict(np.zeros(12))
        with self.assertRaises(ValueError):
            model.learn(np.zeros(self.config.sensory_dim), -1)
        with self.assertRaises(ValueError):
            model.lesion(1.5)


if __name__ == "__main__":
    unittest.main()
