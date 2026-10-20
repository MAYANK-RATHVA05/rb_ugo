from copy import deepcopy

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.original_support_hs import (
    SupportAwareHS, original_node_counts, shrink_node_probabilities,
)


def topology():
    # root -> leaf1 / internal2 -> leaf3 / leaf4
    return np.array([1, -1, 3, -1, -1]), np.array([2, -1, 4, -1, -1])


def test_hand_formula_uses_parent_count_and_original_parent_prediction():
    left, right = topology()
    p = np.array([[.6,.4], [1.,0.], [.2,.8], [0.,1.], [.5,.5]])
    n = np.array([10,5,5,3,2])
    shrunk = shrink_node_probabilities(left, right, p, n, 10)
    # Root-to-node2 uses N(root)=10, node2-to-leaf3 uses N(node2)=5.
    expected2 = p[0] + (p[2]-p[0]) / 2
    expected3 = expected2 + (p[3]-p[2]) / 3
    np.testing.assert_allclose(shrunk[2], expected2)
    np.testing.assert_allclose(shrunk[3], expected3)
    np.testing.assert_array_equal(shrunk[0], p[0])


def test_zero_original_support_suppresses_deeper_increment_but_zero_lambda_is_identity():
    left, right = topology()
    p = np.array([[.6,.4], [1.,0.], [.2,.8], [0.,1.], [.5,.5]])
    n = [10,10,0,0,0]
    shrunk = shrink_node_probabilities(left, right, p, n, 10)
    # Transition into unsupported node2 still uses its supported parent root.
    # Only increments whose parent is node2 (support=0) disappear.
    np.testing.assert_allclose(shrunk[3], shrunk[2])
    np.testing.assert_allclose(shrunk[4], shrunk[2])
    np.testing.assert_allclose(shrink_node_probabilities(left,right,p,n,0), p)


def test_uniform_inflation_equivalent_to_lambda_rescaling():
    left, right = topology()
    p = np.array([[.6,.4], [1.,0.], [.2,.8], [0.,1.], [.5,.5]])
    original = np.array([10,5,5,3,2])
    c, lam = 7, 13
    ours = shrink_node_probabilities(left,right,p,original,lam)
    standard = shrink_node_probabilities(left,right,p,c*original,c*lam)
    np.testing.assert_allclose(ours, standard, rtol=0, atol=1e-15)
    # Same relative-to-root grid gives identical models under global inflation.
    for relative in [0,.01,.1,1,10,100]:
        np.testing.assert_allclose(
            shrink_node_probabilities(left,right,p,original,relative*original[0]),
            shrink_node_probabilities(left,right,p,c*original,relative*c*original[0]))


def test_actual_routing_counts_only_supplied_original_rows_and_preserves_tree():
    originals = np.array([[0.],[0.],[3.],[4.]])
    synthetic = np.array([[1.],[1.2],[2.],[2.2]])
    X = np.vstack((originals,synthetic))
    y = np.array([0,0,1,0,1,1,0,0])
    tree = DecisionTreeClassifier(random_state=42).fit(X,y)
    before = deepcopy(tree.tree_.__getstate__())
    counts = original_node_counts(tree, originals)
    manual = np.zeros(tree.tree_.node_count, dtype=int)
    # Explicit raw-threshold walk independent of decision_path implementation.
    for row in originals:
        node=0
        while True:
            manual[node]+=1
            if tree.tree_.children_left[node] == -1:
                break
            go_left = row[tree.tree_.feature[node]] <= tree.tree_.threshold[node]
            node = tree.tree_.children_left[node] if go_left else tree.tree_.children_right[node]
    np.testing.assert_array_equal(counts,manual)
    assert counts[0] == 4  # Duplicate ORIGINAL observations each count once.
    assert np.any(counts == 0)  # Synthetic-only nodes exist.
    standard=SupportAwareHS(tree,reg_param=3).fit()
    modified=SupportAwareHS(tree,reg_param=3,support_mode='original').fit(originals)
    np.testing.assert_array_equal(standard.raw_probabilities_,modified.raw_probabilities_)
    np.testing.assert_array_equal(before['nodes'],tree.tree_.__getstate__()['nodes'])
    np.testing.assert_array_equal(before['values'],tree.tree_.value)
    assert np.any(counts < tree.tree_.n_node_samples)
    for wrapper in (standard,modified):
        proba=wrapper.predict_proba(X)
        assert np.isfinite(proba).all() and np.all((proba>=0)&(proba<=1))
        np.testing.assert_allclose(proba.sum(axis=1),1)
    np.testing.assert_allclose(SupportAwareHS(tree).fit().predict_proba(X),tree.predict_proba(X))


def test_random_trees_probabilities_valid_over_lambda_grid():
    rng=np.random.default_rng(42)
    X=rng.normal(size=(150,3)); y=rng.integers(0,2,150)
    tree=DecisionTreeClassifier(random_state=42).fit(X,y)
    for lam in [0,.1,1,10,1e6]:
        view=SupportAwareHS(tree,reg_param=lam,support_mode='original').fit(X[:50])
        p=view.predict_proba(X)
        assert np.all((p>=0)&(p<=1))
        np.testing.assert_allclose(p.sum(axis=1),1)


@pytest.mark.parametrize('lam',[-1,np.inf,True])
def test_invalid_lambda(lam):
    l,r=topology()
    with pytest.raises(ValueError):
        shrink_node_probabilities(l,r,np.tile([.5,.5],(5,1)),[10,5,5,3,2],lam)


def test_reject_increasing_support_or_missing_originals():
    l,r=topology()
    with pytest.raises(ValueError,match='support increasing'):
        shrink_node_probabilities(l,r,np.tile([.5,.5],(5,1)),[10,5,5,6,2],1)
    tree=DecisionTreeClassifier().fit([[0],[1],[2]],[0,1,1])
    with pytest.raises(ValueError,match='X_original'):
        SupportAwareHS(tree,support_mode='original').fit()
