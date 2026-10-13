"""Hand-calculated Stage 2 SSHR cleaning, pseudo-labelling and fallback checks.

Fitted artificial partitions isolate Algorithm 2 from SSHC's objective choices.
These are software fixtures, not empirical evidence for the method.
"""

import copy
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.sshr_clustering import SSHC
from src.sshr_resampling import (
    SSHRResamplingError,
    majority_removal_weight,
    minority_confidence,
    resample_from_clusters,
)


def frozen_partition(X, y, U, centroids, labels, l_ids, u_ids, ids=None,
                     minority_label=1, majority_label=0):
    """Build a complete known nearest-centroid partition without fitting code."""
    centers = np.asarray(centroids, dtype=float).reshape(-1, np.asarray(X).shape[1])
    return SimpleNamespace(
        centroids_=centers,
        cluster_ids_=np.arange(len(centers)) if ids is None else np.asarray(ids),
        cluster_labels_=np.asarray(labels),
        labeled_cluster_ids_=np.asarray(l_ids, dtype=int),
        unlabeled_cluster_ids_=np.asarray(u_ids, dtype=int),
        classes_=np.unique(y), minority_label_=minority_label,
        majority_label_=majority_label, n_features_in_=np.asarray(X).shape[1],
    )


def ranked_geometry():
    X = np.array([[10.], [11.], [12.], [13.], [14.], [15.], [0.], [2.]])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    U = np.array([[.5], [1.], [1.5], [20.]])
    model = frozen_partition(X, y, U, [[12.5], [1.]], [0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 0])
    return X, y, U, model


def assert_conservation(result, X, y, U):
    counts, quotas = result.diagnostics["counts"], result.diagnostics["quotas"]
    assert len(result.X_resampled) == len(result.y_resampled) == counts["output_rows"]
    assert counts["output_rows"] == (
        len(X) - counts["removed_majority"] - counts["removed_minority"]
        + counts["selected_pseudo"] + counts["generated_smote"])
    assert quotas["additions_requested"] == (
        len(result.selected_unlabeled_indices) + len(result.synthetic_samples)
        + quotas["additions_unmet"])
    assert quotas["weighted_removal_requested"] == (
        quotas["weighted_removal_actual"] + quotas["weighted_removal_unmet"])
    assert len(result.row_lineage) == len(result.X_resampled)
    retained_count, selected_count = len(result.retained_labeled_indices), len(result.selected_unlabeled_indices)
    np.testing.assert_array_equal(result.X_resampled[:retained_count], X[result.retained_labeled_indices])
    np.testing.assert_array_equal(result.y_resampled[:retained_count], y[result.retained_labeled_indices])
    np.testing.assert_array_equal(result.X_resampled[retained_count:retained_count + selected_count],
                                  U[result.selected_unlabeled_indices])
    np.testing.assert_array_equal(result.X_resampled[retained_count + selected_count:],
                                  result.synthetic_samples)
    assert [item["kind"] for item in result.row_lineage] == (
        ["labeled"] * retained_count + ["pseudo_labeled"] * selected_count
        + ["smote"] * len(result.synthetic_samples))
    json.loads(json.dumps(result.diagnostics, allow_nan=False))
    json.loads(json.dumps(result.row_lineage, allow_nan=False))


def test_paper_eq5_distance_sign_and_overflow_stability():
    assert majority_removal_weight(0, 0) == pytest.approx(.5)
    assert majority_removal_weight(3, 1) == pytest.approx(1 / (1 + np.exp(-2)))
    assert majority_removal_weight(1, 3) == pytest.approx(1 / (1 + np.exp(2)))
    values = majority_removal_weight(np.array([1000., 0.]), np.array([0., 1000.]))
    np.testing.assert_array_equal(values, [1., 0.])


def test_paper_eq7_confidence_uses_observed_minority_fraction_and_reports_undefined():
    assert minority_confidence(2, 5, 0, 11.5) == pytest.approx(.4)
    assert minority_confidence(2, 5, .5, 12.5) == pytest.approx(.384)
    assert minority_confidence(2, 5, .5, 11.5) == pytest.approx(44 / 115)
    assert minority_confidence(2, 5, 0, 0) is None
    assert minority_confidence(2, 5, 1, 1) == 0


@pytest.mark.parametrize("own,minority", [
    (-1, 1), (1, -1), (np.nan, 0), (0, np.inf), (True, 0),
    ("1", 0), ([1, 2], [1, 2, 3]),
])
def test_weight_helper_rejects_nonphysical_or_incompatible_distances(own, minority):
    with pytest.raises(ValueError):
        majority_removal_weight(own, minority)


@pytest.mark.parametrize("n_plus,n,own,total", [
    (-1, 2, 0, 1), (3, 2, 0, 1), (1.5, 2, 0, 1),
    (True, 2, 0, 1), (1, 0, 0, 1), (1, 2, -1, 1),
    (1, 2, 2, 1), (1, 2, 0, np.inf), (1, 2, np.nan, 1),
])
def test_confidence_helper_rejects_invalid_counts_and_distances(n_plus, n, own, total):
    with pytest.raises(ValueError):
        minority_confidence(n_plus, n, own, total)


def test_weighted_removal_and_global_confidence_ranking_are_hand_calculated():
    X, y, U, model = ranked_geometry()
    result = resample_from_clusters(X, y, U, model, delta=0)
    # G=6-2=4: floor(.3*4)=1 removal, 3 additions. All additions are existing U.
    quotas = result.diagnostics["quotas"]
    assert quotas["weighted_removal_requested"] == quotas["weighted_removal_actual"] == 1
    assert quotas["additions_requested"] == 3
    np.testing.assert_array_equal(result.retained_labeled_indices, np.arange(1, 8))
    np.testing.assert_array_equal(result.selected_unlabeled_indices, [1, 0, 2])
    assert result.synthetic_samples.shape == (0, 1)
    weights = {item["labeled_index"]: item for item in result.diagnostics["weighted_candidates"]}
    assert weights[0]["own_distance"] == pytest.approx(2.5)
    assert weights[0]["minority_distance"] == pytest.approx(9.)
    assert weights[0]["weight"] == pytest.approx(1 / (1 + np.exp(6.5)))
    assert weights[1]["weight"] == pytest.approx(1 / (1 + np.exp(8.5)))
    scores = {item["unlabeled_index"]: item for item in result.diagnostics["pseudo_candidates"]}
    assert set(scores) == {0, 1, 2}
    assert scores[1]["confidence"] == pytest.approx(.4)
    assert scores[0]["confidence"] == pytest.approx(.384)
    assert scores[2]["confidence"] == pytest.approx(44 / 115)
    assert all(item["observed_minority_count"] == 2 and item["cluster_size"] == 5
               for item in scores.values())
    np.testing.assert_array_equal(result.y_resampled, [0] * 5 + [1] * 5)
    assert_conservation(result, X, y, U)


def test_small_cluster_cleaning_uses_l_plus_u_and_includes_observed_minority_rows():
    X = np.array([[0.], [0.], [0.], [9.], [10.], [11.], [10.], [10.],
                  [19.], [21.], [20.]])
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0])
    U = np.array([[0.], [10.], [10.], [10.], [10.], [20.5]])
    model = frozen_partition(X, y, U, [[0.], [10.], [20.]], [0, 0, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], [0, 1, 1, 1, 1, 2])
    result = resample_from_clusters(X, y, U, model, delta=.75)
    # Majority cluster sizes (L+U) are 4 and 9: .75 * mean = 4.875.
    # With L-only sizes the threshold is 3 and the 3-row cluster would survive.
    removals = result.diagnostics["removals"]
    assert removals["small_majority_clusters"]["indices"] == [0, 1, 2]
    assert removals["small_majority_clusters"]["minority_count"] == 1
    assert removals["small_majority_clusters"]["majority_count"] == 2
    assert result.diagnostics["small_cluster_threshold"]["threshold"] == pytest.approx(4.875)
    assert result.diagnostics["small_cluster_threshold"]["deleted_cluster_ids"] == [0]
    assert removals["minority_cluster_majority"]["indices"] == [10]
    assert result.diagnostics["counts"]["removed_majority"] == 3
    assert result.diagnostics["counts"]["removed_minority"] == 1
    # Post-clean 5 majority, 2 minority: quota uses gap 3, not the original 5.
    assert result.diagnostics["quotas"]["weighted_removal_requested"] == 0
    assert result.diagnostics["quotas"]["additions_requested"] == 3
    np.testing.assert_array_equal(result.retained_labeled_indices, [3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(result.selected_unlabeled_indices, [5])
    # Deleted centroid zero is excluded from Eq7's denominator: distances are
    # .5 to own centroid 20 and 10.5 to the frozen majority centroid 10.
    assert result.diagnostics["selected_pseudo_rows"][0]["confidence"] == pytest.approx(7 / 11)
    assert len(result.synthetic_samples) == 2
    assert not any(item["unlabeled_index"] == 0 for item in result.diagnostics["pseudo_candidates"])
    assert_conservation(result, X, y, U)


def test_small_cluster_threshold_is_computed_once_before_simultaneous_deletion():
    X = np.array([0.] + [10.] * 5 + [20.] * 10 + [30.] * 2)[:, None]
    y = np.array([0] * 16 + [1] * 2)
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[0.], [10.], [20.], [30.]], [0, 0, 0, 1],
                             [0] + [1] * 5 + [2] * 10 + [3] * 2, [])
    result = resample_from_clusters(X, y, U, model, delta=.9, removal_fraction=0)
    # .9 * mean(1,5,10)=4.8 removes only the one-row cluster. Recomputing the
    # threshold after that deletion would incorrectly remove the five-row one.
    assert result.diagnostics["removals"]["small_majority_clusters"]["indices"] == [0]
    assert result.diagnostics["counts"]["removed_majority"] == 1
    assert result.diagnostics["quotas"]["additions_requested"] == 13
    assert_conservation(result, X, y, U)


def test_cluster_exactly_at_minimum_size_is_retained():
    X = np.array([[10.], [10.], [20.], [20.], [-1.], [1.]])
    y = np.array([0, 0, 0, 0, 1, 1])
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[10.], [20.], [0.]], [0, 0, 1],
                             [0, 0, 1, 1, 2, 2], [])
    result = resample_from_clusters(X, y, U, model, delta=1, removal_fraction=0)
    assert result.diagnostics["small_cluster_threshold"]["threshold"] == 2
    assert result.diagnostics["small_cluster_threshold"]["deleted_cluster_ids"] == []
    assert result.diagnostics["removals"]["small_majority_clusters"]["indices"] == []
    assert_conservation(result, X, y, U)


def test_weight_ties_are_resolved_by_original_labeled_position():
    X = np.array([[10.], [10.], [10.], [10.], [0.]])
    y = np.array([0, 0, 0, 0, 1])
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[10.], [0.]], [0, 1], [0, 0, 0, 0, 1], [])
    result = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=1)
    np.testing.assert_array_equal(result.retained_labeled_indices, [3, 4])
    assert result.diagnostics["removals"]["weighted_majority"]["indices"] == [0, 1, 2]
    assert len(result.selected_unlabeled_indices) == len(result.synthetic_samples) == 0
    assert_conservation(result, X, y, U)


def test_global_pseudo_ties_preserve_original_unlabeled_order_without_updating_scores():
    X = np.array([[10.], [10.], [10.], [0.]])
    y = np.array([0, 0, 0, 1])
    U = np.zeros((4, 1))
    model = frozen_partition(X, y, U, [[10.], [0.]], [0, 1], [0, 0, 0, 1], [1] * 4)
    result = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0)
    np.testing.assert_array_equal(result.selected_unlabeled_indices, [0, 1])
    assert [item["confidence"] for item in result.diagnostics["pseudo_candidates"]] == pytest.approx([.2] * 4)
    assert [item["observed_minority_count"] for item in result.diagnostics["pseudo_candidates"]] == [1] * 4
    assert_conservation(result, X, y, U)


def test_pseudo_confidence_is_ranked_globally_across_different_minority_clusters():
    X = np.array([100.] * 5 + [0., 0., 20.])[:, None]
    y = np.array([0] * 5 + [1] * 3)
    U = np.array([[20.], [0.], [20.], [0.]])
    model = frozen_partition(X, y, U, [[100.], [0.], [20.]], [0, 1, 1],
                             [0] * 5 + [1, 1, 2], [2, 1, 2, 1])
    result = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0)
    # Cluster 1 has 2/4 observed minority support versus cluster 2's 1/3.
    # The higher-confidence cluster wins both slots even though its first U
    # observation occurs later, and no cluster-wise equal quota is invented.
    np.testing.assert_array_equal(result.selected_unlabeled_indices, [1, 3])
    candidates = result.diagnostics["pseudo_candidates"]
    assert [item["unlabeled_index"] for item in candidates] == [1, 3, 0, 2]
    assert [item["confidence"] for item in candidates] == pytest.approx([.5, .5, 1 / 3, 1 / 3])
    assert_conservation(result, X, y, U)


def test_smote_pool_policy_distinguishes_selected_observations_from_true_synthesis():
    X = np.array([[0.], [10.], [11.], [12.], [13.]])
    y = np.array([1, 0, 0, 0, 0])
    U = np.array([[2.], [100.]])
    model = frozen_partition(X, y, U, [[11.5], [0.]], [0, 1], [1, 0, 0, 0, 0], [1, 0])
    generated = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0,
                                       random_state=17)
    observed_only = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0,
                                           random_state=17, smote_pool_policy="observed_only")
    np.testing.assert_array_equal(generated.selected_unlabeled_indices, [0])
    assert len(generated.synthetic_samples) == 2
    assert np.all((generated.synthetic_samples >= 0) & (generated.synthetic_samples <= 2))
    assert observed_only.synthetic_samples.shape == (0, 1)
    assert observed_only.diagnostics["quotas"]["additions_unmet"] == 2
    assert generated.diagnostics["quotas"]["additions_unmet"] == 0
    for item in generated.row_lineage:
        if item["kind"] == "smote":
            for source in (item["anchor"], item["neighbor"]):
                assert source in [{"kind": "labeled", "index": 0},
                                  {"kind": "unlabeled", "index": 0}]
            anchor = X[item["anchor"]["index"]] if item["anchor"]["kind"] == "labeled" else U[item["anchor"]["index"]]
            neighbor = X[item["neighbor"]["index"]] if item["neighbor"]["kind"] == "labeled" else U[item["neighbor"]["index"]]
            assert 0 <= item["coefficient"] <= 1
            row = generated.X_resampled[generated.row_lineage.index(item)]
            np.testing.assert_allclose(row, anchor + item["coefficient"] * (neighbor - anchor))
    allocation = generated.diagnostics["smote_allocations"][0]
    assert allocation["pool_size"] == 2
    assert allocation["effective_k"] == 1
    assert allocation["k_clamped"] is True
    assert_conservation(generated, X, y, U)
    assert_conservation(observed_only, X, y, U)


def test_duplicate_coordinates_exclude_self_by_row_identity_and_preserve_lineage():
    X = np.array([[0.], [0.], [10.], [10.], [10.], [10.], [10.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0])
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[10.], [0.]], [0, 1], [1, 1, 0, 0, 0, 0, 0], [])
    result = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0)
    np.testing.assert_array_equal(result.synthetic_samples, np.zeros((3, 1)))
    for item in result.row_lineage:
        if item["kind"] == "smote":
            assert item["anchor"]["kind"] == item["neighbor"]["kind"] == "labeled"
            assert {item["anchor"]["index"], item["neighbor"]["index"]} == {0, 1}
    assert result.diagnostics["coordinate_duplicates"]["smote_duplicate_with_retained_or_pseudo_or_prior_smote"] == 3
    assert_conservation(result, X, y, U)


def test_multifeature_smote_vectors_follow_recorded_segment_and_reproduce_for_same_seed():
    X = np.array([[0., 0.], [2., 4.], [10., 10.], [11., 11.],
                  [12., 12.], [13., 13.], [14., 14.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0])
    U = np.empty((0, 2))
    model = frozen_partition(X, y, U, [[12., 12.], [1., 2.]], [0, 1],
                             [1, 1, 0, 0, 0, 0, 0], [])
    first = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0,
                                   random_state=73, k_neighbors=1)
    again = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0,
                                   random_state=73, k_neighbors=1)
    different = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0,
                                       random_state=74, k_neighbors=1)
    np.testing.assert_array_equal(first.synthetic_samples, again.synthetic_samples)
    assert first.row_lineage == again.row_lineage
    assert first.diagnostics == again.diagnostics
    assert not np.array_equal(first.synthetic_samples, different.synthetic_samples)
    np.testing.assert_allclose(first.synthetic_samples[:, 1], 2 * first.synthetic_samples[:, 0])
    assert np.all((first.synthetic_samples[:, 0] >= 0) & (first.synthetic_samples[:, 0] <= 2))
    for row, info in zip(first.X_resampled, first.row_lineage):
        if info["kind"] == "smote":
            anchor, neighbor = X[info["anchor"]["index"]], X[info["neighbor"]["index"]]
            np.testing.assert_allclose(row, anchor + info["coefficient"] * (neighbor - anchor))
    assert first.diagnostics["smote_allocations"][0]["k_clamped"] is False
    assert_conservation(first, X, y, U)


def test_fallback_allocation_uses_largest_remainder_and_cluster_id_ties():
    X = np.array([100.] * 7 + [0., 2., 20., 22.])[:, None]
    y = np.array([0] * 7 + [1] * 4)
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[100.], [1.], [21.]], [0, 1, 1],
                             [0] * 7 + [2, 2, 5, 5], [], ids=[0, 2, 5])
    result = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0)
    allocations = {item["cluster_id"]: item for item in result.diagnostics["smote_allocations"]}
    assert allocations[2]["requested"] == allocations[2]["generated"] == 2
    assert allocations[5]["requested"] == allocations[5]["generated"] == 1
    for row, lineage in zip(result.X_resampled, result.row_lineage):
        if lineage["kind"] == "smote":
            if lineage["cluster_id"] == 2:
                assert 0 <= row[0] <= 2
            else:
                assert 20 <= row[0] <= 22
    assert_conservation(result, X, y, U)


def test_unanchored_minority_centroid_influences_eq5_but_cannot_supply_pseudo_labels():
    X = np.array([[10.], [11.], [12.], [13.], [0.], [1.]])
    y = np.array([0, 0, 0, 0, 1, 1])
    U = np.array([[9.5]])
    model = frozen_partition(X, y, U, [[10.], [0.], [9.5]], [0, 1, 1],
                             [0, 0, 0, 0, 1, 1], [2])
    result = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=.5)
    assert all(item["minority_distance"] == pytest.approx(item["own_distance"] + .5)
               for item in result.diagnostics["weighted_candidates"])
    assert all(item["weight"] == pytest.approx(1 / (1 + np.exp(.5)))
               for item in result.diagnostics["weighted_candidates"])
    assert len(result.selected_unlabeled_indices) == 0
    assert len(result.synthetic_samples) == 1
    assert result.diagnostics["pseudo_candidates"] == []
    assert_conservation(result, X, y, U)


def test_no_minority_centroid_reports_separate_removal_and_addition_shortfalls():
    X = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
    y = np.array([0, 0, 0, 0, 1, 1])
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[2.5]], [0], [0] * 6, [])
    result = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=.5)
    assert result.diagnostics["quotas"]["weighted_removal_requested"] == 1
    assert result.diagnostics["quotas"]["weighted_removal_unmet"] == 1
    assert result.diagnostics["quotas"]["additions_requested"] == 1
    assert result.diagnostics["quotas"]["additions_unmet"] == 1
    np.testing.assert_array_equal(result.X_resampled, X)
    np.testing.assert_array_equal(result.y_resampled, y)
    assert_conservation(result, X, y, U)


def test_singleton_without_eligible_u_reports_shortfall_without_ros_or_cross_cluster_smote():
    X = np.array([[0.], [10.], [11.], [12.]])
    y = np.array([1, 0, 0, 0])
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[11.], [0.]], [0, 1], [1, 0, 0, 0], [])
    result = resample_from_clusters(X, y, U, model, delta=0, removal_fraction=0)
    assert len(result.synthetic_samples) == 0
    assert result.diagnostics["quotas"]["additions_requested"] == 2
    assert result.diagnostics["quotas"]["additions_unmet"] == 2
    np.testing.assert_array_equal(result.X_resampled, X)
    assert_conservation(result, X, y, U)


def test_class_erasure_from_small_impure_majority_cluster_is_a_diagnosed_failure():
    X = np.array([0., 0., 0.] + [10.] * 10)[:, None]
    y = np.array([0, 0, 1] + [0] * 10)
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[0.], [10.]], [0, 0], [0] * 3 + [1] * 10, [])
    with pytest.raises(SSHRResamplingError) as caught:
        resample_from_clusters(X, y, U, model, delta=1)
    assert caught.value.diagnostics["counts"]["removed_minority"] == 1
    json.dumps(caught.value.diagnostics, allow_nan=False)


def test_class_erasure_from_minority_cluster_cleaning_is_a_diagnosed_failure():
    X = np.array([[0.], [0.], [0.]])
    y = np.array([1, 1, 0])
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[0.]], [1], [0] * 3, [])
    with pytest.raises(SSHRResamplingError) as caught:
        resample_from_clusters(X, y, U, model, delta=0)
    assert caught.value.diagnostics["counts"]["removed_majority"] == 1
    json.dumps(caught.value.diagnostics, allow_nan=False)


def test_reproducibility_input_nonmutation_and_frozen_model_nonmutation():
    X, y, U, model = ranked_geometry()
    original_arrays = [array.copy() for array in (X, y, U)]
    original_model = copy.deepcopy(vars(model))
    first = resample_from_clusters(X, y, U, model, delta=0, random_state=19)
    second = resample_from_clusters(X, y, U, model, delta=0, random_state=19)
    for array, original in zip((X, y, U), original_arrays):
        np.testing.assert_array_equal(array, original)
    assert vars(model).keys() == original_model.keys()
    for key, original in original_model.items():
        np.testing.assert_array_equal(vars(model)[key], original)
    np.testing.assert_array_equal(first.X_resampled, second.X_resampled)
    np.testing.assert_array_equal(first.y_resampled, second.y_resampled)
    assert first.row_lineage == second.row_lineage
    assert first.diagnostics == second.diagnostics
    assert not np.shares_memory(first.X_resampled, X)
    assert not np.shares_memory(first.X_resampled, U)
    assert "y_unlabeled" not in inspect.signature(resample_from_clusters).parameters


def test_explicit_none_pool_and_empty_pool_are_equivalent_and_cluster_order_is_not_semantic():
    X = np.array([[0.], [2.], [10.], [11.], [12.], [13.], [14.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0])
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[12.], [1.]], [0, 1],
                             [1, 1, 0, 0, 0, 0, 0], [])
    empty = resample_from_clusters(X, y, U, model, delta=0)
    absent = resample_from_clusters(X, y, None, model, delta=0)
    np.testing.assert_array_equal(empty.X_resampled, absent.X_resampled)
    assert empty.diagnostics == absent.diagnostics
    model.centroids_ = model.centroids_[::-1].copy()
    model.cluster_ids_ = model.cluster_ids_[::-1].copy()
    model.cluster_labels_ = model.cluster_labels_[::-1].copy()
    reordered = resample_from_clusters(X, y, U, model, delta=0)
    np.testing.assert_array_equal(empty.X_resampled, reordered.X_resampled)
    assert empty.row_lineage == reordered.row_lineage
    assert empty.diagnostics == reordered.diagnostics


def test_success_outputs_remain_independent_after_a_subsequent_diagnosed_failure():
    X, y, U, model = ranked_geometry()
    good = resample_from_clusters(X, y, U, model, delta=0)
    saved = copy.deepcopy(good)
    bad_X = np.zeros((3, 1))
    bad_y = np.array([1, 1, 0])
    bad_U = np.empty((0, 1))
    bad_model = frozen_partition(bad_X, bad_y, bad_U, [[0.]], [1], [0, 0, 0], [])
    with pytest.raises(SSHRResamplingError) as caught:
        resample_from_clusters(bad_X, bad_y, bad_U, bad_model, delta=0)
    assert caught.value.diagnostics["status"] == "failure_class_erased_by_preliminary_cleaning"
    assert "quotas" not in caught.value.diagnostics
    assert "selected_pseudo_rows" not in caught.value.diagnostics
    np.testing.assert_array_equal(good.X_resampled, saved.X_resampled)
    assert good.diagnostics == saved.diagnostics
    assert good.row_lineage == saved.row_lineage


def test_actual_sshc_to_resampling_has_verified_confidence_and_only_existing_additions():
    X = np.array([[0.], [2.], [8.], [10.], [12.], [14.]])
    y = np.array([1, 1, 0, 0, 0, 0])
    U = np.array([[1.], [3.]])
    clustering = SSHC().fit(X, y, U)
    result = resample_from_clusters(X, y, U, clustering)
    np.testing.assert_array_equal(result.selected_unlabeled_indices, [0, 1])
    selected = result.diagnostics["selected_pseudo_rows"]
    assert selected[0]["confidence"] == pytest.approx(10 / 21)
    assert selected[1]["confidence"] == pytest.approx(8 / 19)
    assert len(result.synthetic_samples) == 0
    assert result.diagnostics["counts"]["generated_smote"] == 0
    assert result.diagnostics["counts"]["selected_pseudo"] == 2
    assert_conservation(result, X, y, U)


@pytest.mark.parametrize("minority_count,majority_count", [(2, 2), (3, 2)])
def test_fixed_class_identity_with_tied_or_inverted_counts_requests_no_balancing(minority_count, majority_count):
    X = np.array([0.] * minority_count + [10.] * majority_count)[:, None]
    y = np.array([1] * minority_count + [0] * majority_count)
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[10.], [0.]], [0, 1],
                             [1] * minority_count + [0] * majority_count, [])
    result = resample_from_clusters(X, y, U, model, delta=0)
    assert result.diagnostics["quotas"]["weighted_removal_requested"] == 0
    assert result.diagnostics["quotas"]["additions_requested"] == 0
    np.testing.assert_array_equal(result.X_resampled, X)
    np.testing.assert_array_equal(result.y_resampled, y)
    assert_conservation(result, X, y, U)


def test_initial_tied_counts_still_run_preliminary_cleaning_before_gap_is_computed():
    X = np.array([[0.], [0.], [0.], [10.]])
    y = np.array([1, 1, 0, 0])
    U = np.empty((0, 1))
    model = frozen_partition(X, y, U, [[10.], [0.]], [0, 1], [1, 1, 1, 0], [])
    result = resample_from_clusters(X, y, U, model, delta=0)
    assert result.diagnostics["removals"]["minority_cluster_majority"]["indices"] == [2]
    assert result.diagnostics["quotas"]["additions_requested"] == 0
    assert result.diagnostics["quotas"]["weighted_removal_requested"] == 0
    assert result.diagnostics["parameters"]["minority_label"] == 1
    assert result.diagnostics["parameters"]["majority_label"] == 0
    np.testing.assert_array_equal(result.retained_labeled_indices, [0, 1, 3])
    assert_conservation(result, X, y, U)


def test_original_string_labels_are_preserved_for_existing_and_added_rows():
    X, y, U, model = ranked_geometry()
    y = np.where(y == 1, "rare", "common")
    model.classes_ = np.unique(y)
    model.cluster_labels_ = np.array(["common", "rare"])
    model.minority_label_, model.majority_label_ = "rare", "common"
    result = resample_from_clusters(X, y, U, model, delta=0)
    assert set(result.y_resampled) == {"rare", "common"}
    assert result.y_resampled[-3:].tolist() == ["rare"] * 3
    assert all(item["assigned_label"] == "rare" for item in result.row_lineage[-3:])
    assert_conservation(result, X, y, U)


def test_zero_dimensional_identity_arrays_are_canonicalized_to_json_scalars():
    X, y, U, model = ranked_geometry()
    y = y.astype(np.int32)
    model.classes_ = np.unique(y)
    model.minority_label_ = np.array(1)
    model.majority_label_ = np.array(0)
    result = resample_from_clusters(X, y, U, model, delta=0)
    assert result.y_resampled.dtype == np.dtype(np.int32)
    assert type(result.diagnostics["parameters"]["minority_label"]) is int
    assert type(result.diagnostics["parameters"]["majority_label"]) is int
    assert all(type(item["assigned_label"]) is int for item in result.row_lineage)
    np.testing.assert_array_equal(result.y_resampled, [0] * 5 + [1] * 5)
    assert_conservation(result, X, y, U)


def test_object_dtype_infinite_binary_label_is_rejected_before_partition_validation():
    X, _, U, model = ranked_geometry()
    y = np.array([0] * 6 + [np.inf] * 2, dtype=object)
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        resample_from_clusters(X, y, U, model)


def test_incomparable_fitted_class_domain_raises_value_error():
    X, y, U, model = ranked_geometry()
    model.classes_ = np.array([0, "bad"], dtype=object)
    with pytest.raises(ValueError, match="comparable binary label domain"):
        resample_from_clusters(X, y, U, model)


@pytest.mark.parametrize("unsupported_labels", [np.array([0j, 1j]), np.array([b"0", b"1"])])
def test_unsupported_fitted_cluster_label_scalars_are_rejected(unsupported_labels):
    X, y, U, model = ranked_geometry()
    model.cluster_labels_ = unsupported_labels
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        resample_from_clusters(X, y, U, model)


def test_unsigned_cluster_id_cannot_overflow_signed_index_storage():
    X, y, U, model = ranked_geometry()
    model.cluster_ids_ = np.array([0, 2**63], dtype=np.uint64)
    with pytest.raises(ValueError, match="out-of-range"):
        resample_from_clusters(X, y, U, model)


@pytest.mark.parametrize("parameter,value", [
    ("delta", -.1), ("delta", 1.1), ("delta", True), ("delta", np.nan),
    ("removal_fraction", -.1), ("removal_fraction", 1.1),
    ("removal_fraction", True), ("removal_fraction", np.inf),
    ("k_neighbors", 0), ("k_neighbors", 1.5), ("k_neighbors", True),
    ("random_state", -1), ("random_state", True), ("random_state", 1.5),
    ("smote_pool_policy", "all_unlabeled"), ("smote_pool_policy", []),
])
def test_invalid_parameters_are_rejected(parameter, value):
    X, y, U, model = ranked_geometry()
    with pytest.raises(ValueError):
        resample_from_clusters(X, y, U, model, **{parameter: value})


@pytest.mark.parametrize("field,value", [
    ("centroids_", np.array([[np.nan], [1.]])),
    ("centroids_", np.array([[12.5, 0.], [1., 0.]])),
    ("cluster_ids_", np.array([0, 0])),
    ("cluster_ids_", np.array([0])),
    ("cluster_ids_", np.array([-1, 1])),
    ("cluster_ids_", np.array([0., 1.])),
    ("cluster_labels_", np.array([0, 2])),
    ("cluster_labels_", np.array([1, 0])),
    ("classes_", np.array([0, 2])),
    ("labeled_cluster_ids_", np.array([0] * 8)),
    ("unlabeled_cluster_ids_", np.array([1, 1, 1, 7])),
    ("unlabeled_cluster_ids_", np.array([1, 1, 1])),
    ("n_features_in_", 2),
    ("n_features_in_", True),
    ("minority_label_", 2),
    ("majority_label_", 1),
    ("labeled_predictions_", np.ones(8, dtype=int)),
    ("clusters_", (np.array([0, 1, 2, 3, 4, 5, 6]), np.array([7, 8, 9, 10, 11]))),
    ("diagnostics_", {"not_json": np.array([1, 2])}),
])
def test_inconsistent_fitted_partitions_are_rejected(field, value):
    X, y, U, model = ranked_geometry()
    setattr(model, field, value)
    with pytest.raises(ValueError):
        resample_from_clusters(X, y, U, model)


def test_invalid_input_alignment_nonfinite_and_categorical_features_are_rejected():
    X, y, U, model = ranked_geometry()
    for bad_X, bad_y, bad_U in (
            (X, y[:-1], U), (X, y, np.zeros((2, 2))),
            (X, y, np.array([[np.inf]])),
            (X.astype(str), y, U), (X.astype(bool), y, U),
            (X, y, U.astype(bool)),
            (pd.DataFrame({"category": pd.Categorical(X[:, 0])}), y, U)):
        with pytest.raises(ValueError):
            resample_from_clusters(bad_X, bad_y, bad_U, model)
