import numpy as np

from scripts.audit_original_support_hs_pilot import independent_shrink,route
from src.original_support_hs import shrink_node_probabilities


def test_audit_threshold_routing_and_independent_leaf_to_root_sum():
    left=np.array([1,-1,3,-1,-1]); right=np.array([2,-1,4,-1,-1])
    feature=np.array([0,-2,0,-2,-2]); threshold=np.array([1.,-2.,3.,-2.,-2.])
    counts,leaves=route(np.array([[0.],[2.],[4.],[4.]]),left,right,feature,threshold)
    np.testing.assert_array_equal(counts,[4,1,3,1,2])
    np.testing.assert_array_equal(leaves,[1,3,4,4])
    raw=np.array([[.5,.5],[1.,0.],[.3,.7],[0.,1.],[.5,.5]])
    for support in [counts,np.array([4,4,0,0,0])]:
        for lam in [0,1,100]:
            np.testing.assert_allclose(independent_shrink(left,right,raw,support,lam),
                                       shrink_node_probabilities(left,right,raw,support,lam),atol=1e-14)
