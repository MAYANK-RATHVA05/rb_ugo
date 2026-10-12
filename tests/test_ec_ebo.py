"""Artificial-data tests for the audited EC-EBO minority-side generator."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors

from src.ec_ebo import ECEBO, _minority_step, entropy_balance_score


def test_paper_score_sign_and_declared_minority_side_boundary():
    centroid = np.array([0.])
    majority = np.array([2.])
    expected = 0.5 * (1 - np.exp(-2))
    assert entropy_balance_score(np.array([0.]), centroid, majority, .5, .5, 1) == pytest.approx(expected)
    assert entropy_balance_score(np.array([1.]), centroid, majority, .5, .5, 1) == pytest.approx(0)
    assert entropy_balance_score(np.array([2.]), centroid, majority, .5, .5, 1) == pytest.approx(-expected)


def test_step_uses_nearest_different_location_and_is_zero_for_identical_rows():
    assert _minority_step(np.array([[0.], [0.], [2.]]), .1) == pytest.approx(.2)
    assert _minority_step(np.array([[0.], [0.]]), .1) == 0


def test_generated_rows_follow_positive_decreasing_score_and_are_reproducible():
    X = np.array([[0.], [.2], [2.], [3.], [4.], [5.], [6.], [7.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    original_X, original_y = X.copy(), y.copy()
    first = ECEBO(k_majority=1, random_state=42)
    X_out, y_out = first.fit_resample(X, y)
    second = clone(first)
    X_again, y_again = second.fit_resample(X, y)

    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(X_out[:len(X)], X)
    np.testing.assert_array_equal(y_out[:len(y)], y)
    np.testing.assert_array_equal(X_out, X_again)
    np.testing.assert_array_equal(y_out, y_again)
    np.testing.assert_array_equal(X_out[len(X):], first.generated_samples_)
    np.testing.assert_array_equal(y_out[len(y):], [1] * 4)
    assert first.diagnostics_["status"] == "generated"
    assert first.diagnostics_["requested_count"] == first.diagnostics_["generated_count"] == 4
    assert first.diagnostics_["shortfall_count"] == 0
    assert first.diagnostics_["cluster_generated_counts"] == first.diagnostics_["cluster_allocations"]
    assert first.diagnostics_["duplicate_generated_count"] == 0
    minority_near = NearestNeighbors(n_neighbors=1).fit(X[y == 1]).kneighbors(
        first.generated_samples_, return_distance=True)[0][:, 0]
    majority_near = NearestNeighbors(n_neighbors=1).fit(X[y == 0]).kneighbors(
        first.generated_samples_, return_distance=True)[0][:, 0]
    proxy = first.diagnostics_["nearest_original_majority_proxy"]
    assert proxy["count"] == int(np.sum(majority_near < minority_near))
    assert proxy["rate"] == pytest.approx(proxy["count"] / 4)
    for item, point in zip(first.generated_info_, first.generated_samples_):
        assert item["accepted_steps"] >= 1
        assert 0 < item["final_score"] < item["initial_score"]
        cluster_id = item["cluster_id"]
        reference_id = item["majority_reference_index"]
        score = entropy_balance_score(
            point, first.allocation_.clustering_.centroids_[cluster_id], X[reference_id],
            first.allocation_.cluster_entropies_[cluster_id],
            first.allocation_.sample_entropies_[reference_id], first.alpha)
        assert score == pytest.approx(item["final_score"])


def test_zero_step_reports_full_shortfall_without_copying_seed():
    X = np.array([[0.], [0.], [2.], [3.], [4.]])
    y = np.array([1, 1, 0, 0, 0])
    fitted = ECEBO(k_majority=1)
    X_out, y_out = fitted.fit_resample(X, y)
    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(y_out, y)
    assert fitted.diagnostics_["status"] == "no_positive_step"
    assert fitted.diagnostics_["shortfall_count"] == 1
    assert fitted.diagnostics_["cluster_shortfall_reasons"] == [{"no_positive_step": 1}]
    assert fitted.diagnostics_["nearest_original_majority_proxy"]["rate"] is None


def test_no_positive_seed_reports_shortfall_without_quota_transfer():
    X = np.array([[0.], [1.], [0.], [1.], [2.]])
    y = np.array([1, 1, 0, 0, 0])
    fitted = ECEBO(k_majority=1)
    X_out, y_out = fitted.fit_resample(X, y)
    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(y_out, y)
    assert fitted.diagnostics_["status"] == "no_valid_generation"
    assert fitted.diagnostics_["cluster_allocations"] == [1, 0]
    assert fitted.diagnostics_["cluster_generated_counts"] == [0, 0]
    assert fitted.diagnostics_["cluster_shortfall_reasons"] == [{"no_positive_seed": 1}, {}]


def test_partial_generation_keeps_cluster_quota_shortfall():
    X = np.array([[-2.], [-1.], [3.], [4.], [-2.], [0.], [6.], [7.],
                  [8.], [9.], [10.], [11.]])
    y = np.array([1] * 4 + [0] * 8)
    fitted = ECEBO(k_majority=1)
    X_out, y_out = fitted.fit_resample(X, y)
    assert len(X_out) == len(y_out) == len(X) + 3
    assert fitted.diagnostics_["status"] == "partial_generation"
    assert fitted.diagnostics_["requested_count"] == 4
    assert fitted.diagnostics_["shortfall_count"] == 1
    assert fitted.diagnostics_["cluster_allocations"] == [1, 2, 1]
    assert fitted.diagnostics_["cluster_generated_counts"] == [0, 2, 1]
    assert fitted.diagnostics_["cluster_shortfall_reasons"][0] == {"no_positive_seed": 1}


def test_proposal_limit_can_leave_quota_unmet_without_returning_seed():
    X = np.array([[0.], [.2], [2.], [3.], [4.], [5.], [6.], [7.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    fitted = ECEBO(k_majority=1, n_iterations=1, random_state=0)
    fitted.fit_resample(X, y)
    assert fitted.diagnostics_["shortfall_count"] == 1
    assert fitted.diagnostics_["cluster_shortfall_reasons"] == [{"no_accepted_move": 1}]
    assert all(item["accepted_steps"] == 1 for item in fitted.generated_info_)


def test_early_stop_is_checked_only_after_a_successful_move():
    X = np.array([[0.], [.2], [2.], [3.], [4.], [5.], [6.], [7.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    fitted = ECEBO(k_majority=1, stop_probability=1)
    fitted.fit_resample(X, y)
    assert fitted.generated_info_
    assert all(item["accepted_steps"] == 1 and item["stop_reason"] == "probability"
               for item in fitted.generated_info_)


def test_coordinate_search_operates_on_two_dimensional_rows():
    X = np.array([[0., 0.], [.2, 0.], [2., 0.], [3., 0.], [4., 0.],
                  [5., 0.], [6., 0.], [7., 0.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    fitted = ECEBO(k_majority=1)
    X_out, _ = fitted.fit_resample(X, y)
    assert X_out.shape[1] == 2
    assert fitted.diagnostics_["generated_count"] == 4
    assert np.any(fitted.generated_samples_[:, 1] != 0)


def test_balanced_input_is_noop_and_invalid_refit_clears_fitted_state():
    X = np.array([[0.], [1.], [2.], [3.]])
    y = np.array([1, 1, 0, 0])
    fitted = ECEBO()
    X_out, y_out = fitted.fit_resample(X, y)
    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(y_out, y)
    assert fitted.diagnostics_["status"] == "already_balanced"
    assert len(fitted.generated_samples_) == 0

    fitted.fit_resample([[0.], [.2], [2.], [3.], [4.]], [1, 1, 0, 0, 0])
    assert hasattr(fitted, "allocation_")
    with pytest.raises(ValueError, match="step_fraction"):
        fitted.set_params(step_fraction=0).fit_resample(X, y)
    assert not hasattr(fitted, "allocation_")
    assert not hasattr(fitted, "diagnostics_")


@pytest.mark.parametrize("parameter,value", [
    ("k_majority", 0), ("n_iterations", 0), ("stop_probability", 1.1),
    ("alpha", 0), ("random_state", -1), ("entropy_clip", .5),
])
def test_invalid_generation_parameters_are_rejected(parameter, value):
    sampler = ECEBO(**{parameter: value})
    with pytest.raises(ValueError, match=parameter):
        sampler.fit_resample([[0.], [.2], [2.], [3.], [4.]], [1, 1, 0, 0, 0])
