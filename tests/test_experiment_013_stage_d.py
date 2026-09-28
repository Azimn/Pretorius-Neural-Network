import copy
import unittest

import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from experiment_012_core_convergence import configured
from experiment_013_cross_topology import fingerprints
from experiment_013_stage_c import core_indices, sign_class
from experiment_013_stage_d import aligned_recipient_edges, graft_deltas, sign_matched_random_edges


class StageDPreflight(unittest.TestCase):
    def make_net(self, seed, neurons=64):
        cfg = configured(seed, neurons)
        enc = ExperienceEncoder(cfg["sensory_dim"])
        return enc, PlasticRecurrentPersonaNet(cfg, enc)

    def test_alignment_is_unique_and_dale_compatible(self):
        _, donor = self.make_net(101)
        _, recipient = self.make_net(202)
        rng = np.random.default_rng(13)
        donor_fp = rng.normal(size=(donor.n, 12)).astype(np.float32)
        recipient_fp = rng.normal(size=(recipient.n, 12)).astype(np.float32)
        donor_fp /= np.linalg.norm(donor_fp, axis=1, keepdims=True)
        recipient_fp /= np.linalg.norm(recipient_fp, axis=1, keepdims=True)
        core = np.arange(min(20, len(donor.W.data)), dtype=np.int64)
        mapped, similarity = aligned_recipient_edges(donor, donor_fp, core, recipient, recipient_fp)
        self.assertEqual(len(mapped), len(set(mapped.tolist())))
        np.testing.assert_array_equal(sign_class(donor, core), sign_class(recipient, mapped))
        self.assertEqual(len(similarity), len(core))

    def test_random_control_preserves_sign_per_position(self):
        _, donor = self.make_net(101)
        _, recipient = self.make_net(202)
        core = np.arange(min(40, len(donor.W.data)), dtype=np.int64)
        signs = sign_class(donor, core)
        random_edges = sign_matched_random_edges(recipient, signs, b"stage-d-test")
        self.assertEqual(len(random_edges), len(set(random_edges.tolist())))
        np.testing.assert_array_equal(signs, sign_class(recipient, random_edges))

    def test_graft_adds_deltas_and_respects_sign_constraints(self):
        _, recipient = self.make_net(202)
        edges = np.arange(min(12, len(recipient.W.data)), dtype=np.int64)
        before = recipient.W.data[edges].copy()
        deltas = np.linspace(-10.0, 10.0, len(edges))
        grafted = graft_deltas(recipient, edges, deltas)
        for edge in edges:
            if grafted.excitatory[grafted.pre_idx[edge]]:
                self.assertGreaterEqual(grafted.W.data[edge], 0.0)
            else:
                self.assertLessEqual(grafted.W.data[edge], 0.0)
            self.assertLessEqual(abs(float(grafted.W.data[edge])), grafted.max_abs_weight + 1e-7)
        self.assertTrue(np.allclose(recipient.W.data[edges], before))


if __name__ == "__main__":
    unittest.main()
