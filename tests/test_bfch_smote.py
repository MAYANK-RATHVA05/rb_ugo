import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.bfch_smote import BFCHSMOTE, _allocate_areas, _contains_majority
from src.evaluation import EvaluationConfig, evaluate_ten_fold


def two_class_geometry():
    minority = np.array([[0., 0.], [0., 2.]])
    majority = np.array([[4., 0.], [4., 2.], [5., -1.], [5., 1.], [5., 3.], [6., 1.]])
    return np.vstack((minority, majority)), np.array([1, 1, 0, 0, 0, 0, 0, 0])


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
