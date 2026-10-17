"""Exploratory mechanism checks, not evidence of predictive effectiveness."""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator

import src.exploratory_exposure as exposure
from src.exploratory_exposure import allocate, fair_counts, prepare, replay
from src.ugo import UGO
from scripts.run_exposure_pilot import posterior, scores


class AnchorModel(BaseEstimator):
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        p = np.where(X[:, 0] < 5, .02 + .001 * X[:, 0], .3 + .01 * (X[:, 0] - 10))
        return np.c_[1 - p, p]


class Guide(BaseEstimator):
    fitting = []

    def fit(self, X, y):
        type(self).fitting.append((X.copy(), y.copy()))
        self.classes_ = np.array([1, 0]); return self

    def predict_proba(self, X):
        p = .91 + .02 * (X[:, 0] - 10)
        return np.c_[p, 1 - p]


@pytest.fixture
def inputs():
    X = np.array([[0.], [.1], [.2], [.3], [.4], [.5], [10.], [11.], [12.], [13.]])
    y = np.array([0] * 6 + [1] * 4)
    U = np.array([[10.2], [11.2], [12.2]])
    parent = UGO(AnchorModel(), variant='smote', k_pre=1, lambda_step=.5,
                 max_iter=2, random_state=42, allow_provisional=True)
    return X, y, U, parent


def prepared(inputs, **kwargs):
    X, y, U, parent = inputs
    with pytest.warns(UserWarning, match='Provisional UGO'):
        return prepare(X, y, U, parent, guide=Guide(), endpoint_cap=5,
                       threshold=.9, exposure_rate=.03, **kwargs)


def test_allocation_feasibility_and_continuous_kkt_reference():
    # Equal errors, loose budget: strict concavity gives m=(2,2), analytically.
    a = allocate([.1, .1], [4, 4], 4, 1.)
    np.testing.assert_allclose(a.continuous, [2, 2], atol=1e-5)
    assert a.counts.sum() <= 4 and a.exposure <= 1.
    # One endpoint, active exposure: continuous optimum is exactly B/e=2.5.
    a = allocate([.2], [8], 8, .5)
    np.testing.assert_allclose(a.continuous, [2.5], atol=1e-6)
    assert a.counts.tolist() == [2] and a.rounding_loss == pytest.approx(.5)


def test_zero_error_zero_budget_and_empty_capacity():
    a = allocate([0., .1], [3, 8], 8, 0.)
    assert a.counts[1] == 0 and a.counts[0] >= 2 and a.exposure == 0
    assert allocate([.1], [3], 0, 0).counts.tolist() == [0]
    assert fair_counts(np.array([1, 3]), 4).tolist() == [1, 3]
    with pytest.raises(ValueError, match='infeasible'):
        fair_counts(np.array([1]), 2)


@pytest.mark.parametrize('e,lim,cap,budget', [([-.1], [2], 2, .1), ([np.nan], [2], 2, .1),
                                           ([.1], [1.5], 2, .1), ([.1], [2], -1, .1),
                                           ([.1], [2], 2, -.1)])
def test_invalid_allocation_fails(e, lim, cap, budget):
    with pytest.raises(ValueError): allocate(e, lim, cap, budget)


def test_solver_failure_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(exposure, 'minimize', lambda *a, **k: SimpleNamespace(success=False, message='forced failure'))
    with pytest.raises(RuntimeError, match='solver failed'):
        allocate([.1], [3], 3, .2)


def test_no_u_exactly_matches_parent_including_diagnostics(inputs):
    X, y, _, parent = inputs
    with pytest.warns(UserWarning):
        a, b = parent.fit_resample(X, y)
    with pytest.warns(UserWarning):
        p = prepare(X, y, np.empty((0, 1)), parent)
    result = replay(p)
    np.testing.assert_array_equal(result['X'], a)
    np.testing.assert_array_equal(result['y'], b)
    assert p['parent'].diagnostics_ == parent.diagnostics_
    assert result['ledger'] == [] and p['guide_status'] == 'unused_no_U'


def test_controls_share_trace_bank_and_exact_counts(inputs):
    p = prepared(inputs)
    constrained = replay(p, 'budget')
    assert constrained['u_informed_count'] > 0  # Exercise actual synthesis.
    for policy in ['count_matched', 'threshold_matched', 'weighted_pl']:
        result = replay(p, policy)
        assert [r['count'] for r in result['rounds']] == [r['count'] for r in constrained['rounds']]
        assert result['sample_weight'].sum() == len(p['baseline_X'])
    for bank, info in zip(p['banks'], constrained['rounds']):
        assert info['estimated_exposure'] <= bank['budget'] + 1e-10
        assert info['count'] <= bank['capacity']
    assert replay(p, 'unconstrained')['u_informed_count'] > constrained['u_informed_count']


def test_lineage_reconstructs_new_vectors_and_preserves_inputs(inputs):
    X, y, U, _ = inputs
    before = (X.copy(), y.copy(), U.copy())
    Guide.fitting.clear()
    p = prepared(inputs); result = replay(p)
    for child in result['ledger']:
        anchor, endpoint = X[child['anchor']], U[child['endpoint']]
        np.testing.assert_allclose(result['X'][child['output_index']], anchor + child['fraction'] * (endpoint - anchor))
        assert 0 < child['fraction'] < 1
        assert y[child['anchor']] == 1
    for value, original in zip((X, y, U), before): np.testing.assert_array_equal(value, original)
    np.testing.assert_array_equal(Guide.fitting[0][0], X)
    np.testing.assert_array_equal(Guide.fitting[0][1], y)
    assert not {'test', 'y_U', 'hidden_labels'}.intersection(inspect.signature(prepare).parameters)
    np.testing.assert_array_equal(result['X'][:len(X)], X)
    assert result['guide_status'] == 'custom_guide_uncertified'


def test_seed_repeatability_and_invalid_pool(inputs):
    a = replay(prepared(inputs)); b = replay(prepared(inputs))
    np.testing.assert_array_equal(a['X'], b['X']); assert a['ledger'] == b['ledger']
    X, y, _, parent = inputs
    with pytest.raises(ValueError, match='dimensions'):
        prepare(X, y, np.zeros((2, 2)), parent)
    with pytest.raises(ValueError, match='finite'):
        prepare(X, y, np.array([[np.nan]]), parent)


def test_candidate_rejection_is_shared_and_does_not_become_relocation(inputs, monkeypatch):
    original = exposure.noise_final_detect_and_relocate
    def reject(*args, **kwargs):
        corrected, info = original(*args, **kwargs)
        info['relocation_reference_indices'][:] = 0
        return corrected, info
    monkeypatch.setattr(exposure, 'noise_final_detect_and_relocate', reject)
    p = prepared(inputs)
    assert sum(b['rejected_candidates'] for b in p['banks']) > 0
    assert all(replay(p, policy)['u_informed_count'] == 0 for policy in ['budget', 'unconstrained', 'count_matched'])


def test_known_posterior_and_nontrivial_scoring():
    np.testing.assert_allclose(posterior(np.array([[0., 0.], [3., -3.]]), 2.), [.2, .2])
    good = scores(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert good['ap'] == good['mcc'] == good['gmean'] == 1.
    bad = scores(np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1]))
    assert bad['mcc'] == -1. and bad['gmean'] == 0.


def test_insufficient_calibration_labels_use_exact_parent_fallback(inputs):
    X, y, U, parent = inputs
    chosen = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    with pytest.warns(UserWarning):
        p = prepare(X[chosen], y[chosen], U, parent)
    result = replay(p)
    assert p['guide_status'] == 'parent_fallback_insufficient_or_tied_class_counts'
    np.testing.assert_array_equal(result['X'], p['baseline_X'])
    assert result['ledger'] == []
