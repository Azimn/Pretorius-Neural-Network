import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from experiment_012_core_convergence import configured
from experiment_013_cross_topology import permute_network, align_neurons


def test_permutation_preserves_dynamics_and_alignment_recovers_mapping():
    cfg = configured(77, 64)
    cfg["avg_recurrent_degree"] = 8
    cfg["action_population_size"] = 8
    encoder = ExperienceEncoder(cfg["sensory_dim"])
    net = PlasticRecurrentPersonaNet(cfg, encoder)
    rng = np.random.default_rng(991)
    permutation = rng.permutation(net.n)
    other = permute_network(net, permutation)

    contexts = []
    for i in range(12):
        x = rng.normal(0, 0.2, encoder.input_dim).astype(np.float32)
        contexts.append((x, x))

    mapping, _ = align_neurons(net, other, contexts, probe_steps=6)
    assert np.mean(mapping == permutation) >= 0.90

    x = contexts[0][0]
    net.reset_fast_state(noise=0.0)
    other.reset_fast_state(noise=0.0)
    for _ in range(10):
        net.step(x, learn=False)
        other.step(x, learn=False)
    np.testing.assert_allclose(other.rate[permutation], net.rate, rtol=2e-5, atol=2e-6)
