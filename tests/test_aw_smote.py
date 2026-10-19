"""Hand-derived AW checks and small artificial software fixtures, not experiments."""
import math

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler

from src.aw_smote import AWSMOTE, AWUnsupportedError, _allocate_quotas, _nearest_other_rows


def clustered_fixture():
    # Majority-first input catches labels accidentally applied to reordered rows.
    # Minority clusters: {0,.25,.5}, {10,10.25,10.5}; isolated minority: {40}.
    majority = [-.125, .75, 2, 9.875, 10.125, 10.375, 10.75,
                -10, -20, -30, -40, 50, -50, 60]
    minority = [0, .25, .5, 10, 10.25, 10.5, 40]
    return np.asarray(majority + minority)[:, None], np.array([0] * 14 + [1] * 7)


def test_hand_derived_boundary_density_and_sigmoid_probabilities():
    # k=1: row .25 sees minority 0; row .5 has a tied majority/minority
    # neighbor, resolved to earlier original row .75 (majority).
    X = np.array([[-.125], [.75], [1.], [2.], [3.], [0.], [.25], [.5]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    sampler = AWSMOTE(k_neighbors=1)
    sampler.fit_resample(X, y)
    assert sampler.epsilon_ == .25
    np.testing.assert_array_equal(sampler.boundary_neighbors_, [[0], [5], [1]])
    np.testing.assert_array_equal(sampler.boundary_counts_[5:], [1, 0, 1])
    np.testing.assert_array_equal(sampler.cluster_weights_, [1])
    np.testing.assert_array_equal(sampler.cluster_quotas_, [2])
    # Mean distances are 3/8, 1/4, 3/8, so normalized inverse density
    # is (2/7,3/7,2/7), up to the declared 1e-8 stabilizer.
    a_edge = 1 / (1 + math.exp(-1.5))
    a_center = 1 / (1 + math.exp(1))
    edge_weight = a_edge / 2 + (1 - a_edge) * 2 / 7
    center_weight = (1 - a_center) * 3 / 7
    expected = np.array([edge_weight, center_weight, edge_weight]) / (2 * edge_weight + center_weight)
    np.testing.assert_allclose(sampler.sample_probabilities_[5:], expected, rtol=1e-7)
    np.testing.assert_array_equal(sampler.sample_probabilities_[:5], np.zeros(5))


def test_cleaned_neighbor_labels_cluster_maxima_and_original_total_budget():
    X, y = clustered_fixture()
    original_X, original_y = X.copy(), y.copy()
    sampler = AWSMOTE(k_neighbors=1)
    out_X, out_y = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(sampler.noise_indices_, [20])
    np.testing.assert_array_equal(sampler.excluded_indices_, [20])
    np.testing.assert_array_equal(sampler.cleaned_indices_, np.arange(20))
    np.testing.assert_array_equal(sampler.boundary_neighbors_, [[0], [14], [1], [3], [4], [5]])
    np.testing.assert_array_equal(sampler.boundary_counts_[14:20], [1, 0, 1, 1, 1, 1])
    # Maxima are equal despite different means: 2/3 versus 1.
    np.testing.assert_array_equal(sampler.cluster_weights_, [1, 1])
    np.testing.assert_array_equal(sampler.cluster_quotas_, [3, 4])
    assert 20 not in sampler.boundary_neighbors_
    assert not np.any(sampler.boundary_neighbors_ == sampler.eligible_indices_[:, None])
    np.testing.assert_array_equal(out_X[:20], X[:20])
    np.testing.assert_array_equal(out_y[:20], y[:20])
    np.testing.assert_array_equal(out_y[20:], np.ones(7))
    assert out_X.shape == (27, 1)
    assert sampler.diagnostics_["requested_count"] == 14 - 7 == 7
    assert sampler.diagnostics_["output_counts"] == {"majority": 14, "minority": 13}
    assert sampler.diagnostics_["removed_original_count"] == 1


def test_retention_control_changes_originals_only_with_identical_generation():
    X, y = clustered_fixture()
    remove = AWSMOTE(k_neighbors=1, n_samples=200)
    retain = clone(remove).set_params(noise_policy="retain_originals")
    remove_X, remove_y = remove.fit_resample(X, y)
    retain_X, retain_y = retain.fit_resample(X, y)
    for name in ("generated_samples_", "generated_seed_indices_", "generated_partner_indices_",
                 "generated_cluster_labels_", "interpolation_fractions_", "cluster_quotas_",
                 "boundary_counts_", "sample_probabilities_"):
        np.testing.assert_array_equal(getattr(remove, name), getattr(retain, name))
    np.testing.assert_array_equal(retain_X[:len(X)], X)
    np.testing.assert_array_equal(retain_y[:len(y)], y)
    np.testing.assert_array_equal(remove_X[20:], retain_X[21:])
    assert len(remove_y) + 1 == len(retain_y)
    assert retain.diagnostics_["removed_original_count"] == 0
    assert retain.diagnostics_["excluded_seed_count"] == 1
    assert 20 not in retain.generated_seed_indices_
    assert 20 not in retain.generated_partner_indices_


def test_provenance_reconstructs_samples_and_preserves_cluster_constraints():
    X, y = clustered_fixture()
    sampler = AWSMOTE(k_neighbors=1, n_samples=200)
    sampler.fit_resample(X, y)
    seeds, partners = sampler.generated_seed_indices_, sampler.generated_partner_indices_
    fractions = sampler.interpolation_fractions_
    assert np.all(seeds != partners)
    assert np.all(y[seeds] == 1) and np.all(y[partners] == 1)
    np.testing.assert_array_equal(sampler.cluster_labels_[seeds], sampler.cluster_labels_[partners])
    np.testing.assert_array_equal(sampler.cluster_labels_[seeds], sampler.generated_cluster_labels_)
    assert np.all(sampler.generated_cluster_labels_ >= 0)
    assert np.all((fractions >= 0) & (fractions < 1))
    # These fixed-seed draws detect accidental use of the author's .2--.8 interval.
    assert fractions.min() < .2 and fractions.max() > .8
    expected = X[seeds] + fractions[:, None] * (X[partners] - X[seeds])
    np.testing.assert_allclose(sampler.generated_samples_, expected, atol=1e-12)
    for label, quota in zip(sampler.diagnostics_["cluster_labels"], sampler.cluster_quotas_):
        assert np.count_nonzero(sampler.generated_cluster_labels_ == label) == quota


@pytest.mark.parametrize("min_samples", [2, 3, 5])
def test_default_fifth_other_distance_is_independent_of_minpts(min_samples):
    X = np.r_[np.arange(100., 114.), np.arange(7.)][:, None]
    y = np.array([0] * 14 + [1] * 7)
    sampler = AWSMOTE(min_samples=min_samples)
    sampler.fit_resample(X, y)
    # Fifth OTHER distances are [5,4,3,3,3,4,5], percentile 65 = 4.
    assert sampler.epsilon_ == 4
    np.testing.assert_array_equal(sampler.boundary_counts_[14:], np.zeros(7))
    assert sampler.diagnostics_["uniform_cluster_fallback"] is True
    mean_distances = np.array([7/2, 8/3, 13/6, 2, 13/6, 8/3, 7/2])
    density = 1 / (mean_distances + 1e-8)
    alpha = 1 / (1 + math.exp(-5 * (1/7 - .2)))
    expected = alpha / 7 + (1 - alpha) * density / density.sum()
    np.testing.assert_allclose(sampler.sample_probabilities_[14:], expected)
    assert np.all(np.isfinite(sampler.generated_samples_))


def test_self_exclusion_preserves_duplicate_locations_and_breaks_ties_by_row_id():
    X = np.array([[0.], [0.], [0.], [2.]])
    ids, distances = _nearest_other_rows(X, np.array([1]), np.array([3, 2, 1, 0]), 2)
    np.testing.assert_array_equal(ids, [[0, 2]])
    np.testing.assert_array_equal(distances, [[0., 0.]])


def test_euclidean_neighbors_use_all_feature_dimensions():
    X = np.array([[0., 0.], [2.5, 0.], [1.5, 1.5]])
    ids, distances = _nearest_other_rows(X, np.array([0]), np.arange(3), 2)
    np.testing.assert_array_equal(ids, [[2, 1]])
    np.testing.assert_allclose(distances, [[math.sqrt(4.5), 2.5]])


def test_two_feature_interpolation_preserves_both_coordinates():
    X, y = clustered_fixture()
    two_dimensional = np.column_stack((X[:, 0], -X[:, 0]))
    sampler = AWSMOTE(k_neighbors=1, n_samples=30)
    out_X, _ = sampler.fit_resample(two_dimensional, y)
    assert out_X.shape[1] == 2
    np.testing.assert_allclose(sampler.generated_samples_[:, 1], -sampler.generated_samples_[:, 0])
    seeds, partners = sampler.generated_seed_indices_, sampler.generated_partner_indices_
    t = sampler.interpolation_fractions_[:, None]
    expected = (1 - t) * two_dimensional[seeds] + t * two_dimensional[partners]
    np.testing.assert_allclose(sampler.generated_samples_, expected)


def test_valid_cluster_with_zero_boundary_weight_gets_no_synthetic_quota():
    X, y = clustered_fixture()
    X[3:7, 0] = [100, 110, 120, 130]
    sampler = AWSMOTE(k_neighbors=1)
    sampler.fit_resample(X, y)
    np.testing.assert_array_equal(sampler.cluster_weights_, [1, 0])
    np.testing.assert_array_equal(sampler.cluster_quotas_, [7, 0])
    assert np.all(sampler.generated_cluster_labels_ == 0)
    assert not np.any(np.isin(sampler.generated_seed_indices_, [17, 18, 19]))
    assert not np.any(np.isin(sampler.generated_partner_indices_, [17, 18, 19]))


@pytest.mark.parametrize("weights,count,expected", [
    ([1., 1.], 1, [0, 1]),  # half-up, then subtract from first maximum
    ([1., 1.], 7, [3, 4]),
    ([2., 1.], 5, [3, 2]),
    ([0., 0.], 3, [1, 2]),
    ([1., 0., 1.], 5, [2, 0, 3]),
])
def test_paper_quota_rounding_and_exact_correction(weights, count, expected):
    np.testing.assert_array_equal(_allocate_quotas(np.array(weights), count), expected)


def test_negative_paper_correction_is_reported_without_silent_reallocation():
    # Six equal clusters and N=4 round to six ones; subtracting two from
    # the first maximum makes it -1. This is a genuine rule edge case.
    with pytest.raises(AWUnsupportedError) as caught:
        _allocate_quotas(np.ones(6), 4)
    assert caught.value.reason == "invalid_quota"
    minority = np.array([v for start in range(0, 60, 10) for v in (start, start + .125)])
    X = np.r_[np.arange(100., 116.), minority][:, None]
    y = np.array([0] * 16 + [1] * 12)
    sampler = AWSMOTE(k_neighbors=1)
    with pytest.raises(AWUnsupportedError) as caught:
        sampler.fit_resample(X, y)
    assert caught.value.reason == "invalid_quota"
    assert caught.value.diagnostics["cluster_count"] == 6
    assert sampler.diagnostics_["generated_count"] == 0
    assert sampler.generated_samples_.shape == (0, 1)


def test_all_noise_has_structured_failure_without_single_cluster_fallback():
    X, y = clustered_fixture()
    sampler = AWSMOTE(k_neighbors=1, min_samples=8)
    with pytest.raises(AWUnsupportedError) as caught:
        sampler.fit_resample(X, y)
    assert caught.value.reason == "no_valid_cluster"
    assert caught.value.diagnostics["status"] == "unsupported"
    assert caught.value.diagnostics["excluded_seed_count"] == 7
    np.testing.assert_array_equal(sampler.noise_indices_, np.arange(14, 21))
    assert not sampler.clusters_
    sampler.diagnostics_["status"] = "changed_after_failure"
    assert caught.value.diagnostics["status"] == "unsupported"


def test_identical_minority_locations_report_nonpositive_epsilon():
    X = np.array([[2.], [3.], [4.], [5.], [0.], [0.], [0.]])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    sampler = AWSMOTE(k_neighbors=1)
    with pytest.raises(AWUnsupportedError) as caught:
        sampler.fit_resample(X, y)
    assert caught.value.reason == "nonpositive_epsilon"
    assert caught.value.diagnostics["epsilon"] == 0


def test_insufficient_neighbors_does_not_silently_reduce_k():
    sampler = AWSMOTE()
    with pytest.raises(AWUnsupportedError) as caught:
        sampler.fit_resample([[2.], [3.], [4.], [0.], [.25]], [0, 0, 0, 1, 1])
    assert caught.value.reason == "too_few_neighbors"
    assert caught.value.diagnostics["generated_count"] == 0


def test_overflowing_euclidean_geometry_is_diagnosed():
    with pytest.raises(AWUnsupportedError) as caught:
        _nearest_other_rows(np.array([[-1e308], [1e308]]), np.array([0]), np.array([0, 1]), 1)
    assert caught.value.reason == "nonfinite_distances"


def test_refit_and_clone_repeat_exactly_without_global_rng_changes():
    X, y = clustered_fixture()
    sampler = AWSMOTE(k_neighbors=1, n_samples=30, random_state=19)
    global_before = np.random.get_state()
    first_X, first_y = sampler.fit_resample(X, y)
    second_X, second_y = sampler.fit_resample(X, y)
    cloned_X, cloned_y = clone(sampler).fit_resample(X, y)
    for actual in (second_X, cloned_X):
        np.testing.assert_array_equal(actual, first_X)
    for actual in (second_y, cloned_y):
        np.testing.assert_array_equal(actual, first_y)
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    other_X, _ = clone(sampler).set_params(random_state=20).fit_resample(X, y)
    assert not np.array_equal(other_X, first_X)


def test_failed_refit_clears_prior_geometry_and_generated_results():
    X, y = clustered_fixture()
    sampler = AWSMOTE(k_neighbors=1)
    sampler.fit_resample(X, y)
    with pytest.raises(AWUnsupportedError):
        sampler.set_params(min_samples=8).fit_resample(X, y)
    assert sampler.generated_samples_.shape == (0, 1)
    assert sampler.generated_seed_indices_.size == 0
    assert not hasattr(sampler, "cluster_quotas_")
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sampler.fit_resample(bad, y)
    assert not hasattr(sampler, "generated_samples_")
    assert not hasattr(sampler, "diagnostics_")


def test_zero_generation_and_balanced_defaults_preserve_originals():
    X, y = clustered_fixture()
    sampler = AWSMOTE(n_samples=0)
    out_X, out_y = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(out_X, X)
    np.testing.assert_array_equal(out_y, y)
    assert sampler.diagnostics_["status"] == "zero_requested"
    assert sampler.diagnostics_["removed_original_count"] == 0
    balanced = AWSMOTE()
    balanced_X, balanced_y = balanced.fit_resample([[0.], [1.]], [0, 1])
    np.testing.assert_array_equal(balanced_X, [[0.], [1.]])
    np.testing.assert_array_equal(balanced_y, [0, 1])


@pytest.mark.parametrize("parameters", [
    {"min_samples": 1}, {"min_samples": True}, {"k_neighbors": 0},
    {"eps_percentile": -1}, {"eps_percentile": 101}, {"eps_percentile": float("nan")},
    {"theta": 1.1}, {"gamma": 0}, {"gamma": float("inf")},
    {"n_samples": -1}, {"n_samples": True}, {"n_samples": 1.5},
    {"noise_policy": "keepnoise"}, {"noise_policy": []},
    {"random_state": True}, {"random_state": -1}, {"random_state": 2**32},
])
def test_invalid_parameters_fail_explicitly(parameters):
    X, y = clustered_fixture()
    with pytest.raises(ValueError):
        AWSMOTE(**parameters).fit_resample(X, y)


@pytest.mark.parametrize("X,y", [
    ([[0.], [1.]], [0]),
    ([[0.], [1.]], [1, 2]),
    ([[0.], [1.]], ["0", "1"]),
    ([[0.], [1.]], [0, float("nan")]),
    ([[0.], [1.]], [0, float("inf")]),
    ([[0.], [1.]], [1, 1]),
    ([[0.], [1.], [2.]], [0, 1, 2]),
    ([[0.], [1.], [2.]], [0, 1, 1]),
    ([[0.], [float("inf")]], [0, 1]),
    ([["a"], ["b"]], [0, 1]),
])
def test_invalid_or_misaligned_inputs_fail(X, y):
    with pytest.raises(ValueError):
        AWSMOTE().fit_resample(X, y)


class RecordingEstimator(BaseEstimator):
    """Software spy: records arrays; learns no classifier or performance score."""

    def fit(self, X, y):
        self.received_X_ = np.array(X, copy=True)
        self.received_y_ = np.array(y, copy=True)
        return self

    def predict(self, X):
        self.prediction_X_ = np.array(X, copy=True)
        return np.zeros(len(X), dtype=int)


def test_pipeline_fits_scaling_on_original_training_only_and_does_not_resample_prediction():
    X, y = clustered_fixture()
    held_out = np.array([[1000.], [2000.]])
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("aw", AWSMOTE(k_neighbors=1, eps_percentile=100)),
        ("spy", RecordingEstimator()),
    ])
    pipeline.fit(pd.DataFrame(X), y)
    scaler, sampler, spy = (pipeline.named_steps[key] for key in ("scale", "aw", "spy"))
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
    assert scaler.n_samples_seen_ == len(X)
    np.testing.assert_allclose(spy.received_X_[:len(sampler.retained_indices_)],
                               scaler.transform(pd.DataFrame(X))[sampler.retained_indices_])
    samples_before = sampler.generated_samples_.copy()
    pipeline.predict(pd.DataFrame(held_out))
    np.testing.assert_allclose(spy.prediction_X_, scaler.transform(pd.DataFrame(held_out)))
    np.testing.assert_array_equal(sampler.generated_samples_, samples_before)
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
    assert scaler.n_samples_seen_ == len(X)


def test_hidden_groups_and_heldout_data_are_not_sampler_inputs():
    X, y = clustered_fixture()
    sampler = AWSMOTE(k_neighbors=1)
    with pytest.raises(TypeError, match="group_ids"):
        sampler.fit_resample(X, y, group_ids=np.arange(len(X)))
    with pytest.raises(TypeError, match="X_test"):
        sampler.fit_resample(X, y, X_test=X[:1])
