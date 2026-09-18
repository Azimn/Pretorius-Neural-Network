import numpy as np

from persona_net import ExperienceEncoder, PlasticRecurrentPersonaNet
from experiment_012_core_convergence import configured
from experiment_013_cross_topology import fingerprints
from experiment_013_stage_c import edge_signatures, sign_class, sign_matched_random, symmetric_match_similarity


def test_stage_c_functional_edge_similarity_and_control_are_index_independent():
    cfg = configured(101, 64)
    enc = ExperienceEncoder(cfg["sensory_dim"])
    net = PlasticRecurrentPersonaNet(cfg, enc)
    # fingerprints() consumes the network's complete encoded input vector,
    # including the encoder's non-sensory channels. Using sensory_dim here
    # would construct a truncated synthetic probe that cannot match Win.
    compiled = [(np.zeros(enc.input_dim, dtype=np.float32), None) for _ in range(4)]
    fp = fingerprints(net, compiled, probe_steps=2)
    idx = np.arange(min(12, len(net.W.data)), dtype=np.int64)
    sig = edge_signatures(net, fp, idx)
    signs = sign_class(net, idx)
    assert np.isclose(symmetric_match_similarity(sig, signs, sig, signs), 1.0, atol=1e-6)
    rng = np.random.default_rng(13)
    control = sign_matched_random(net, len(idx), signs, rng)
    assert len(control) == len(idx)
    assert int(sign_class(net, control).sum()) == int(signs.sum())


def test_stage_c_unmatched_dale_class_is_penalized_not_dropped_or_fatal():
    # Independently selected tiny causal cores need not contain the same Dale classes.
    # Hard compatibility means an unmatched edge has no candidate, so it receives the
    # cosine lower bound rather than being silently omitted from the aggregate.
    sig_a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sign_a = np.array([True, False])
    sig_b = np.array([[1.0, 0.0]], dtype=np.float32)
    sign_b = np.array([True])
    score = symmetric_match_similarity(sig_a, sign_a, sig_b, sign_b)
    # A->B best scores are [1, -1], B->A is [1], hence symmetric mean = 0.5.
    assert np.isclose(score, 0.5, atol=1e-6)
