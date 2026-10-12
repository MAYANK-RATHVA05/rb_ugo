import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import src.bfch_smote as bfch
from src.bfch_smote import BFCHSMOTE, _allocate_areas, _contains_majority
from src.evaluation import EvaluationConfig, evaluate_ten_fold


def two_class_geometry():
    minority = np.array([[0., 0.], [0., 2.]])
    majority = np.array([[4., 0.], [4., 2.], [5., -1.], [5., 1.], [5., 3.], [6., 1.]])
    return np.vstack((minority, majority)), np.array([1, 1, 0, 0, 0, 0, 0, 0])


def hand_geometry(with_occupied_point=False):
    # Indices 0,1 are retained minority anchors; 2 is filtered. Indices 3,4
    # are boundary majority anchors; 5,6 are not. Index 7, when present, is
    # filtered by RF but remains an observed majority point inside the hull.
    X = np.array([[0., 0.], [0., 2.], [10., 10.],
                  [4., 0.], [4., 3.], [5., 1.], [10., 11.]])
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    positive_probabilities = [0.3, 0.9, 0.29, 0.2, 0.05, 0.21, 0.1]
    if with_occupied_point:
        X = np.vstack((X, [[1., 0.5]]))
        y = np.r_[y, 0]
        positive_probabilities.append(0.6)
    return X, y, np.asarray(positive_probabilities)


def stub_forest(monkeypatch, positive_probabilities):
    calls = []

    class FixedProbabilitiesForest:
        def __init__(self, **parameters):
            calls.append({"parameters": parameters})

        def fit(self, X, y):
            calls[-1]["fitted_rows"] = len(X)
            calls[-1]["fitted_labels"] = y.copy()
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            assert len(X) == len(positive_probabilities)
            return np.column_stack((1 - positive_probabilities, positive_probabilities))

    monkeypatch.setattr(bfch, "RandomForestClassifier", FixedProbabilitiesForest)
    return calls


def test_hand_calculated_filter_votes_fusion_area_and_allocation(monkeypatch):
    X, y, probabilities = hand_geometry()
    calls = stub_forest(monkeypatch, probabilities)
    sampler = BFCHSMOTE(k_neighbors=2, n_samples=3, fusion=0.5)
    augmented, labels = sampler.fit_resample(X, y)
    assert calls[0]["fitted_rows"] == len(X)
    np.testing.assert_array_equal(calls[0]["fitted_labels"], y)
    np.testing.assert_array_equal(sampler.retained_indices_["minority"], [0, 1])
    np.testing.assert_array_equal(sampler.retained_indices_["majority"], [3, 4, 6])
    np.testing.assert_array_equal(sampler.boundary_indices_["majority"], [3, 4])
    np.testing.assert_array_equal(sampler.boundary_indices_["minority"], [0, 1])
    assert len(sampler.hulls_) == 1
    hull = sampler.hulls_[0]
    assert hull["minority_pair_indices"] == (0, 1)
    assert hull["majority_index"] == 3
    np.testing.assert_array_equal(hull["midpoint"], [0., 1.])
    np.testing.assert_array_equal(hull["fusion_point"], [2., 0.5])
    assert hull["area"] == pytest.approx(2.)
    assert hull["allocated_count"] == 3
    assert sampler.diagnostics_["candidate_hulls"] == 1
    assert sampler.diagnostics_["valid_hulls"] == 1
    assert len(augmented) == len(labels) == len(X) + 3


def test_filtered_majority_point_inside_hull_still_rejects_it(monkeypatch):
    X, y, probabilities = hand_geometry(with_occupied_point=True)
    stub_forest(monkeypatch, probabilities)
    sampler = BFCHSMOTE(k_neighbors=2, n_samples=3, fusion=0.5)
    augmented, labels = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(sampler.retained_indices_["majority"], [3, 4, 6])
    assert sampler.diagnostics_["candidate_hulls"] == 1
    assert sampler.diagnostics_["rejected_occupied"] == 1
    assert sampler.diagnostics_["status"] == "no_valid_hulls"
    assert sampler.hulls_ == []
    np.testing.assert_array_equal(augmented, X)
    np.testing.assert_array_equal(labels, y)


def test_one_neighbor_per_minority_cannot_pass_two_vote_boundary_rule(monkeypatch):
    X, y, probabilities = hand_geometry()
    stub_forest(monkeypatch, probabilities)
    sampler = BFCHSMOTE(k_neighbors=1)
    augmented, labels = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(sampler.boundary_indices_["majority"], [])
    assert sampler.diagnostics_["status"] == "insufficient_majority_boundary"
    np.testing.assert_array_equal(augmented, X)
    np.testing.assert_array_equal(labels, y)


def test_failed_refit_clears_previous_geometry(monkeypatch):
    X, y, probabilities = hand_geometry()
    stub_forest(monkeypatch, probabilities)
    sampler = BFCHSMOTE(k_neighbors=2)
    sampler.fit_resample(X, y)
    assert sampler.hulls_
    with pytest.raises(ValueError, match="two numerical features"):
        sampler.fit_resample(X[:, :1], y)
    assert not hasattr(sampler, "hulls_")
    assert not hasattr(sampler, "boundary_indices_")


def test_generated_count_labels_geometry_and_reproducibility():
    X, y = two_class_geometry()
    sampler = BFCHSMOTE(k_neighbors=2, theta_min=0, theta_maj=0,
                        n_samples=11, random_state=19)
    augmented, labels = sampler.fit_resample(X, y)
    assert sampler.diagnostics_["status"] == "generated"
    assert sampler.diagnostics_["generated_count"] == 11
    assert sampler.diagnostics_["valid_hulls"] == 1
    assert augmented.shape == (19, 2)
    np.testing.assert_array_equal(augmented[:len(X)], X)
    np.testing.assert_array_equal(labels[:len(y)], y)
    np.testing.assert_array_equal(labels[len(y):], np.ones(11))
    # All synthetic rows lie inside the one fused triangle.
    assert augmented[len(X):, 0].min() >= 0
    assert np.all(augmented[len(X):, 0] < 2.01)
    synthetic = augmented[len(X):]
    assert np.all(synthetic[:, 1] >= 0.25 * synthetic[:, 0])
    assert np.all(synthetic[:, 1] <= 2 - 0.25 * synthetic[:, 0])
    again, _ = clone(sampler).fit_resample(X, y)
    np.testing.assert_allclose(again, augmented)


def test_barycentric_membership_rejects_inside_and_edge_but_not_outside_plane():
    triangle = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    assert _contains_majority(triangle, np.array([[0.5, 0.5, 0.]]))
    assert _contains_majority(triangle, np.array([[1., 1., 0.]]))
    assert not _contains_majority(triangle, np.array([[2., 2., 0.]]))
    assert not _contains_majority(triangle, np.array([[0.5, 0.5, 1.]]))


def test_zero_fusion_has_no_area_and_never_falls_back_to_smote():
    X, y = two_class_geometry()
    sampler = BFCHSMOTE(k_neighbors=2, theta_min=0, theta_maj=0, fusion=0)
    augmented, labels = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(augmented, X)
    np.testing.assert_array_equal(labels, y)
    assert sampler.diagnostics_["status"] == "no_valid_hulls"
    assert sampler.diagnostics_["rejected_degenerate"] >= 1


def test_largest_remainder_allocation_is_exact_and_stable():
    np.testing.assert_array_equal(_allocate_areas(np.array([1., 1., 1.]), 5), [2, 2, 1])
    np.testing.assert_array_equal(_allocate_areas(np.array([1., 3.]), 8), [2, 6])


@pytest.mark.parametrize("kwargs", [
    {"k_neighbors": 0}, {"n_samples": -1}, {"n_samples": True},
    {"fusion": 1.1}, {"theta_min": float("nan")}, {"random_state": -1},
])
def test_invalid_parameters_fail(kwargs):
    X, y = two_class_geometry()
    with pytest.raises(ValueError):
        BFCHSMOTE(**kwargs).fit_resample(X, y)


def test_paired_runner_accepts_bfch_without_changing_default_methods(tmp_path):
    rng = np.random.default_rng(9)
    minority = rng.normal(loc=[0, 0], scale=[0.3, 0.7], size=(20, 2))
    majority = rng.normal(loc=[4, 0], scale=[0.5, 1.], size=(40, 2))
    X = pd.DataFrame(np.vstack((minority, majority)), columns=["a", "b"])
    y = np.r_[np.ones(20, dtype=int), np.zeros(40, dtype=int)]
    result = evaluate_ten_fold(
        X, y, dataset_id="bfch_artificial_geometry", positive_label=1,
        provenance={"kind": "artificial software fixture"}, output_dir=tmp_path,
        config=EvaluationConfig(classifiers=("GNB",), methods=("nores", "bfch_smote"),
                                method_parameters={"bfch_smote": {
                                    "k_neighbors": 5, "theta_min": 0, "theta_maj": 0}}),
    )
    assert len(result["records"]) == 20
    assert all(record["status"] == "success" for record in result["records"])
    for baseline, bfch in zip(result["records"][::2], result["records"][1::2]):
        assert baseline["test_indices"] == bfch["test_indices"]
        assert baseline["train_indices"] == bfch["train_indices"]
        assert bfch["resampler_diagnostics"]["requested_count"] == 18
        assert bfch["sample_counts"]["added_rows"] == 18
