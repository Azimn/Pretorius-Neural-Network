import unittest

import numpy as np

from persona_net.development import BiographyCurriculum
from persona_net.encoding import ACTIONS, ExperienceEncoder
from persona_net.phenotype_training import PhenotypeCurriculum


class CountingNet:
    """Minimal deterministic test double for curriculum accounting."""

    def __init__(self):
        self.tick = 0
        self.rng = np.random.default_rng(1)

    def step(self, x, reward=0.0, learn=True):
        self.tick += 1

    def learn_motor(self, action):
        pass

    def learn_motor_distribution(self, target):
        pass


class NeuralStepAccountingTests(unittest.TestCase):
    def setUp(self):
        self.encoder = ExperienceEncoder(sensory_dim=32)

    def test_biography_reports_actual_network_updates(self):
        events = [{
            "id": "e1",
            "year": 1,
            "narrative": "A developmental event.",
            "action": "explore",
            "outcome": 0.5,
            "exposure_weight": 1.0,
        }]
        net = CountingNet()
        report = BiographyCurriculum(events, self.encoder).run(
            net,
            total_ticks=10,
            input_noise=0.0,
            consolidation_fraction=0.2,
            progress_every=0,
        )

        # 8 active curriculum ticks cost two updates each; 2 consolidation
        # ticks cost one each. The old `ticks` field therefore cannot be used
        # as the matched neural-update budget.
        self.assertEqual(report.ticks, 10)
        self.assertEqual(report.neural_steps, 18)
        self.assertEqual(report.start_neural_step, 0)
        self.assertEqual(report.end_neural_step, 18)
        self.assertEqual(net.tick, 18)

    def test_pis_reports_one_neural_update_per_training_tick(self):
        target = {a: 0.0 for a in ACTIONS}
        target["explore"] = 1.0
        items = [{"scenario": "A novel apparatus appears.", "target_actions": target}]
        net = CountingNet()
        report = PhenotypeCurriculum(items, self.encoder).run(
            net, total_ticks=9, progress_every=0
        )

        self.assertEqual(report.ticks, 9)
        self.assertEqual(report.neural_steps, 9)
        self.assertEqual(report.start_neural_step, 0)
        self.assertEqual(report.end_neural_step, 9)
        self.assertEqual(net.tick, 9)


if __name__ == "__main__":
    unittest.main()
