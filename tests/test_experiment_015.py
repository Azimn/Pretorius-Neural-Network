import numpy as np
from experiment_015_learned_delta_necessity import (
    RANDOMIZATION_NAMESPACE, SEEDS, CYCLIC_PAIRS, fixed_seed, matched_M, matched_R,
    descriptors, CORE_FRACTION,
)

def test_frozen_protocol_constants():
    assert SEEDS == (1842,101,202,303,404,505)
    assert CYCLIC_PAIRS == ((1842,101),(101,202),(202,303),(303,404),(404,505),(505,1842))
    assert CORE_FRACTION == 0.01
    assert RANDOMIZATION_NAMESPACE == "experiment015-v1-preregistered"

def test_fixed_seed_is_deterministic_and_purpose_separated():
    a=fixed_seed(1842,101,"M-magnitude-dale-0")
    assert a == fixed_seed(1842,101,"M-magnitude-dale-0")
    assert a != fixed_seed(1842,101,"M-sign-dale-0")
    assert a != fixed_seed(1842,101,"R-magnitude-dale-0")

def test_M_exact_per_dale_invariants_and_determinism():
    x=np.array([.1,-.2,.3,-.4,.5,-.6,.7,-.8,0.0,.9,-1.0,1.1])
    dale=np.array([0,0,0,0,0,0,1,1,1,1,1,1],dtype=bool)
    m1,p1=matched_M(x,dale,1842,101); m2,p2=matched_M(x,dale,1842,101)
    assert np.array_equal(m1,m2) and p1==p2
    for cls in (False,True):
        p=np.flatnonzero(dale==cls)
        assert np.array_equal(np.sort(np.abs(m1[p])),np.sort(np.abs(x[p])))
        assert np.array_equal(np.sort(np.sign(m1[p])),np.sort(np.sign(x[p])))
        if len(p)>1 and len(np.unique(x[p]))>1:
            assert not np.array_equal(m1[p],x[p])

def test_R_samples_only_empirical_per_dale_support_with_replacement():
    x=np.array([.1,-.2,.3,-.4,.5,-.6,.7,-.8,0.0,.9,-1.0,1.1])
    dale=np.array([0,0,0,0,0,0,1,1,1,1,1,1],dtype=bool)
    r1,p1=matched_R(x,dale,1842,101); r2,p2=matched_R(x,dale,1842,101)
    assert np.array_equal(r1,r2) and p1==p2
    for cls in (False,True):
        p=np.flatnonzero(dale==cls)
        assert set(np.abs(r1[p])).issubset(set(np.abs(x[p])))
        assert set(np.sign(r1[p])).issubset(set(np.sign(x[p])))

def test_descriptors_record_required_scale_and_counts():
    x=np.array([.1,-.2,0,.4]); dale=np.array([0,0,1,1],dtype=bool)
    d=descriptors(x,dale)
    for key in ("all","dale_0","dale_1"):
        assert set(("count","positive","negative","zero","mean_abs","median_abs","l1","l2")).issubset(d[key])
