import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from experiment_012_core_convergence import configured
from experiment_013_stage_c import SEEDS, sign_class
from experiment_014_edge_set_vs_assignment import (
    CORE_FRACTION,
    CYCLIC_PAIRS,
    RANDOMIZATION_NAMESPACE,
    dale_shuffle,
    fixed_seed,
    paired_random_edges,
    same_multiset,
)


def test_frozen_protocol_constants():
    assert SEEDS == (1842, 101, 202, 303, 404, 505)
    assert CYCLIC_PAIRS == ((1842,101),(101,202),(202,303),(303,404),(404,505),(505,1842))
    assert CORE_FRACTION == 0.01
    assert RANDOMIZATION_NAMESPACE == "experiment014-v1-preregistered"


def test_random_set_is_nonoverlapping_paired_and_dale_compatible():
    cfg = configured(101, 128)
    enc = ExperienceEncoder(cfg["sensory_dim"])
    net = PlasticRecurrentPersonaNet(cfg, enc)
    all_edges = np.arange(len(net.W.data), dtype=np.int64)
    signs = sign_class(net, all_edges)
    target = np.array([True, False, True, False], dtype=bool)
    aligned = np.array([
        all_edges[signs][0], all_edges[~signs][0], all_edges[signs][1], all_edges[~signs][1]
    ], dtype=np.int64)
    seed = fixed_seed(1842, 101, "random-edge-set")
    c_edges = paired_random_edges(net, target, aligned, seed)
    d_edges = paired_random_edges(net, target, aligned, seed)
    assert np.array_equal(c_edges, d_edges)
    assert not (set(c_edges.tolist()) & set(aligned.tolist()))
    assert np.array_equal(sign_class(net, c_edges), target)


def test_dale_shuffle_preserves_exact_delta_multiset_and_classes():
    delta = np.array([0.1, 0.2, -0.3, -0.4, 0.5, -0.6])
    signs = np.array([True, True, False, False, True, False])
    shuffled = dale_shuffle(delta, signs, fixed_seed(1842, 101, "aligned-shuffle"))
    assert same_multiset(delta, shuffled)
    assert same_multiset(delta[signs], shuffled[signs])
    assert same_multiset(delta[~signs], shuffled[~signs])
