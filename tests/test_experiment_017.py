import numpy as np
from types import SimpleNamespace
from experiment_017_clipping_susceptibility import SEEDS,CYCLIC_PAIRS,CORE_FRACTION,PROTOCOL_VERSION,common_no_clip_alpha,clipping_audit

class Founder:
    def __init__(self):
        self.W=SimpleNamespace(data=np.array([.2,.8,-.2,-.8],dtype=float))
        self.max_abs_weight=1.0
        self.pre_idx=np.arange(4,dtype=int)
        self.excitatory=np.array([True,True,False,False])

def test_frozen_protocol_constants():
    assert SEEDS==(1842,101,202,303,404,505)
    assert CYCLIC_PAIRS==((1842,101),(101,202),(202,303),(303,404),(404,505),(505,1842))
    assert CORE_FRACTION==0.01 and PROTOCOL_VERSION=="experiment017-v1-preregistered"

def test_common_alpha_uses_both_vectors_and_is_shared():
    f=Founder(); edges=np.arange(4); R=np.array([.4,.4,-.4,-.4]); S=np.array([.9,.9,-.9,-.9])
    a=common_no_clip_alpha(f,edges,R,S)
    assert np.isclose(a["analytic_alpha"],2/9)
    assert 0.0<=a["alpha_pair"]<=a["analytic_alpha"]
    assert clipping_audit(f,edges,R*a["alpha_pair"])["count"]==0
    assert clipping_audit(f,edges,S*a["alpha_pair"])["count"]==0

def test_alpha_zero_is_retained_not_retuned():
    f=Founder(); f.W.data[0]=1.0; edges=np.array([0]); R=np.array([.2]); S=np.array([.3])
    a=common_no_clip_alpha(f,edges,R,S)
    assert a["analytic_alpha"]==0.0 and a["alpha_pair"]==0.0
    assert clipping_audit(f,edges,R*0.0)=={"count":0,"total_clipped_magnitude":0.0}

def test_clipping_audit_reports_count_and_total_loss():
    f=Founder(); edges=np.arange(4); d=np.array([1.,1.,-1.,-1.])
    audit=clipping_audit(f,edges,d)
    assert audit["count"]==2
    assert np.isclose(audit["total_clipped_magnitude"],1.6)

def test_no_clip_alpha_deterministic():
    f=Founder(); edges=np.arange(4); R=np.array([.1,.2,-.3,-.4]); S=np.array([.5,.6,-.7,-.8])
    assert common_no_clip_alpha(f,edges,R,S)==common_no_clip_alpha(f,edges,R,S)
