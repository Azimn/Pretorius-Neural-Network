import hashlib
import numpy as np
from experiment_016_delta_distribution_necessity import RANDOMIZATION_NAMESPACE,SEEDS,CYCLIC_PAIRS,CORE_FRACTION,s_seed,synthetic_S,descriptors

def test_frozen_protocol_constants():
    assert SEEDS==(1842,101,202,303,404,505)
    assert CYCLIC_PAIRS==((1842,101),(101,202),(202,303),(303,404),(404,505),(505,1842))
    assert CORE_FRACTION==0.01 and RANDOMIZATION_NAMESPACE=="experiment016-v1-preregistered"

def test_seed_exact_frozen_sha256_contract():
    token=b"experiment016-v1-preregistered|S-halfnormal|1842|101|1"
    expected=int.from_bytes(hashlib.sha256(token).digest()[:8],"big",signed=False)
    assert s_seed(1842,101,1)==expected

def test_S_deterministic_exact_per_dale_rms_and_sign_orientation():
    x=np.array([.1,.2,.3,.4,.5,.6,.7,.8],dtype=float); dale=np.array([0,0,0,0,1,1,1,1],dtype=bool)
    s1,p1=synthetic_S(x,dale,1842,101); s2,p2=synthetic_S(x,dale,1842,101)
    assert np.array_equal(s1,s2) and p1==p2
    for cls in (False,True):
        p=np.flatnonzero(dale==cls)
        assert np.isclose(np.sqrt(np.mean(s1[p]**2)),np.sqrt(np.mean(x[p]**2)),rtol=1e-12,atol=1e-15)
        assert np.array_equal(np.sign(s1[p]),np.sign(x[p]))
        assert not set(np.round(np.abs(s1[p]),12)).issubset(set(np.round(np.abs(x[p]),12)))

def test_S_zero_rms_is_zero_without_redraw():
    x=np.zeros(6); dale=np.array([0,0,0,1,1,1],dtype=bool); s,_=synthetic_S(x,dale,1842,101); assert np.array_equal(s,x)

def test_descriptors_include_shape_scale_and_counts():
    d=descriptors(np.array([.1,.2,0,.4]),np.array([0,0,1,1],dtype=bool))
    required={"count","positive","negative","zero","mean_abs","median_abs","l1","l2","rms","q25_abs","q75_abs","q90_abs"}
    for key in ("all","dale_0","dale_1"): assert required.issubset(d[key])
