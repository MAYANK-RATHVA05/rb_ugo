"""Public-estimator checks using only artificial labelled/unlabelled pools."""

from dataclasses import replace
import inspect
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone

import src.sshr as sshr_module
from src.sshr import SSHR
from src.sshr_clustering import SSHC
from src.sshr_resampling import SSHRResamplingError, resample_from_clusters


def separated_pools():
    # The true held-out data is absent. U's existing rows have no labels here.
    X = np.array([[-1.], [1.], [10.], [11.], [12.], [13.], [14.], [15.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    U = np.array([[0.], [.5], [13.]])
    return X, y, U


def fitted_names(estimator):
    return {name for name in vars(estimator) if name.endswith("_")}


def test_estimator_fits_actual_sshc_then_core_with_hand_calculated_results():
    X, y, U = separated_pools()
    sampler = SSHR()
    X_out, y_out = sampler.fit_resample(X, y, U)
    independent_clustering = SSHC().fit(X, y, U)
    independent_result = resample_from_clusters(X, y, U, independent_clustering)
    assert isinstance(sampler, BaseEstimator)
    assert isinstance(sampler.clustering_, SSHC)
    np.testing.assert_allclose(sampler.clustering_.centroids_, [[88 / 7], [1 / 8]])
    np.testing.assert_array_equal(sampler.retained_labeled_indices_, [0, 1, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(sampler.selected_unlabeled_indices_, [0, 1])
    assert sampler.synthetic_samples_.shape == (1, 1)
    assert len(X_out) == len(y_out) == 10
    assert np.count_nonzero(y_out == 0) == np.count_nonzero(y_out == 1) == 5
    assert sampler.diagnostics_["counts"]["removed_majority"] == 1
    assert sampler.diagnostics_["counts"]["selected_pseudo"] == 2
    assert sampler.diagnostics_["counts"]["generated_smote"] == 1
    selected = sampler.diagnostics_["selected_pseudo_rows"]
    assert selected[0]["confidence"] == pytest.approx(352 / 711)
    assert selected[1]["confidence"] == pytest.approx(338 / 697)
    np.testing.assert_array_equal(X_out, independent_result.X_resampled)
    np.testing.assert_array_equal(y_out, independent_result.y_resampled)
    assert sampler.row_lineage_ == independent_result.row_lineage
    assert sampler.diagnostics_ == independent_result.diagnostics
    assert sampler.n_features_in_ == 1
    assert sampler.minority_label_ == 1
    assert sampler.majority_label_ == 0
    np.testing.assert_array_equal(sampler.classes_, [0, 1])
    assert sampler.diagnostics_["author_code_verified"] is False
    assert sampler.diagnostics_["clustering_diagnostics"]["parameters"]["objective_mode"] == "paper_printed"
    assert sampler.diagnostics_["clustering_diagnostics"]["objective"]["anomaly"] is True
    json.dumps(sampler.diagnostics_, allow_nan=False)
    json.dumps(sampler.row_lineage_, allow_nan=False)


def test_each_call_fits_a_fresh_clustering_on_exact_ordered_supplied_pools(monkeypatch):
    X, y, U = separated_pools()
    calls = []

    class RecordingSSHC(SSHC):
        def fit(self, X_labeled, y_labeled, X_unlabeled=None):
            calls.append((self, np.asarray(X_labeled).copy(), np.asarray(y_labeled).copy(),
                          None if X_unlabeled is None else np.asarray(X_unlabeled).copy()))
            return super().fit(X_labeled, y_labeled, X_unlabeled)

    monkeypatch.setattr(sshr_module, "SSHC", RecordingSSHC)
    sampler = SSHR()
    sampler.fit_resample(X, y, U)
    first_model = sampler.clustering_
    modified_U = np.array([[-.5], [.25], [14.]])
    sampler.fit_resample(X, y, modified_U)
    assert sampler.clustering_ is not first_model
    assert len(calls) == 2
    for _, fitted_X, fitted_y, fitted_U in calls:
        np.testing.assert_array_equal(fitted_X, X)
        np.testing.assert_array_equal(fitted_y, y)
    np.testing.assert_array_equal(calls[0][3], U)
    np.testing.assert_array_equal(calls[1][3], modified_U)
    np.testing.assert_allclose(sampler.clustering_.centroids_, [[89 / 7], [-1 / 16]])


def test_estimator_forwards_all_explicit_algorithm_policies(monkeypatch):
    X, y, U = separated_pools()
    clustering_parameters, resampling_parameters = [], []
    real_core = sshr_module.resample_from_clusters

    class RecordingSSHC(SSHC):
        def __init__(self, **parameters):
            clustering_parameters.append(parameters.copy())
            super().__init__(**parameters)

    def recording_core(*args, **kwargs):
        resampling_parameters.append(kwargs.copy())
        return real_core(*args, **kwargs)

    monkeypatch.setattr(sshr_module, "SSHC", RecordingSSHC)
    monkeypatch.setattr(sshr_module, "resample_from_clusters", recording_core)
    sampler = SSHR(minority_label=1, mu=.25, objective_mode="standard_minority_f1",
                   prune_scope="all", tol=1e-6, max_split_rounds=3, max_refine_iter=4,
                   delta=.2, removal_fraction=.4, k_neighbors=3, random_state=9,
                   smote_pool_policy="observed_only")
    sampler.fit_resample(X, y, U)
    assert clustering_parameters == [{
        "minority_label": 1, "mu": .25, "objective_mode": "standard_minority_f1",
        "prune_scope": "all", "tol": 1e-6, "max_split_rounds": 3, "max_refine_iter": 4,
    }]
    assert resampling_parameters == [{
        "delta": .2, "removal_fraction": .4, "k_neighbors": 3, "random_state": 9,
        "smote_pool_policy": "observed_only",
    }]
    assert sampler.diagnostics_["parameters"]["smote_pool_policy"] == "observed_only"
    assert sampler.clustering_.diagnostics_["parameters"]["objective_mode"] == "standard_minority_f1"


def test_inputs_remain_intact_and_returned_arrays_do_not_share_fitted_results():
    X, y, U = separated_pools()
    original = [array.copy() for array in (X, y, U)]
    sampler = SSHR()
    X_out, y_out = sampler.fit_resample(X, y, U)
    for array, before in zip((X, y, U), original):
        np.testing.assert_array_equal(array, before)
    saved_X, saved_y = sampler.result_.X_resampled.copy(), sampler.result_.y_resampled.copy()
    X_out[:] = 999
    y_out[:] = 999
    np.testing.assert_array_equal(sampler.result_.X_resampled, saved_X)
    np.testing.assert_array_equal(sampler.result_.y_resampled, saved_y)
    assert not np.shares_memory(sampler.retained_labeled_indices_, sampler.result_.retained_labeled_indices)
    assert not np.shares_memory(sampler.selected_unlabeled_indices_, sampler.result_.selected_unlabeled_indices)
    assert not np.shares_memory(sampler.synthetic_samples_, sampler.result_.synthetic_samples)
    sampler.row_lineage_[0]["original_index"] = 999
    sampler.diagnostics_["counts"]["output_rows"] = 999
    assert sampler.result_.row_lineage[0]["original_index"] == 0
    assert sampler.result_.diagnostics["counts"]["output_rows"] == 10


def test_default_absent_and_explicit_empty_unlabeled_pools_are_equivalent():
    X, y, _ = separated_pools()
    absent, empty = SSHR(), SSHR()
    absent_X, absent_y = absent.fit_resample(X, y)
    empty_X, empty_y = empty.fit_resample(X, y, np.empty((0, 1)))
    np.testing.assert_array_equal(absent_X, empty_X)
    np.testing.assert_array_equal(absent_y, empty_y)
    assert absent.diagnostics_ == empty.diagnostics_
    assert len(absent.selected_unlabeled_indices_) == 0
    assert len(absent.synthetic_samples_) == 3


def test_clone_get_params_same_seed_and_pickle_preserve_full_estimator_results():
    X, y, U = separated_pools()
    sampler = SSHR(minority_label=1, mu=.7, objective_mode="standard_minority_f1",
                   tol=1e-6, random_state=29, k_neighbors=2)
    parameters = sampler.get_params()
    assert set(parameters) == {
        "minority_label", "mu", "objective_mode", "prune_scope", "tol",
        "max_split_rounds", "max_refine_iter", "delta", "removal_fraction",
        "k_neighbors", "random_state", "smote_pool_policy",
    }
    X_first, y_first = sampler.fit_resample(X, y, U)
    duplicate = clone(sampler)
    assert duplicate.get_params() == parameters
    assert fitted_names(duplicate) == set()
    X_duplicate, y_duplicate = duplicate.fit_resample(X, y, U)
    np.testing.assert_array_equal(X_first, X_duplicate)
    np.testing.assert_array_equal(y_first, y_duplicate)
    assert sampler.row_lineage_ == duplicate.row_lineage_
    assert sampler.diagnostics_ == duplicate.diagnostics_
    restored = pickle.loads(pickle.dumps(sampler))
    assert restored.get_params() == parameters
    np.testing.assert_array_equal(restored.result_.X_resampled, X_first)
    assert restored.row_lineage_ == sampler.row_lineage_
    assert restored.diagnostics_ == sampler.diagnostics_


def test_feature_pool_changes_geometry_without_any_hidden_label_interface():
    X, y, U = separated_pools()
    sampler = SSHR()
    sampler.fit_resample(X, y, U)
    assert list(inspect.signature(SSHR.fit_resample).parameters) == [
        "self", "X_labeled", "y_labeled", "X_unlabeled"]
    assert not any("hidden" in name or "y_unlabeled" in name for name in vars(sampler))
    with pytest.raises(TypeError, match="unexpected keyword"):
        sampler.fit_resample(X, y, U, y_unlabeled=[1, 1, 0])
    with pytest.raises(TypeError, match="unexpected keyword"):
        sampler.fit_resample(X, y, U, X_test=np.array([[100.]]))


def test_string_class_domain_is_preserved_for_selected_and_generated_rows():
    X, y, U = separated_pools()
    names = np.where(y == 1, "rare", "common")
    sampler = SSHR(minority_label="rare")
    X_out, y_out = sampler.fit_resample(X, names, U)
    assert sampler.minority_label_ == "rare"
    assert sampler.majority_label_ == "common"
    np.testing.assert_array_equal(sampler.classes_, ["common", "rare"])
    assert set(y_out) == {"common", "rare"}
    assert y_out[-3:].tolist() == ["rare"] * 3
    assert all(item["assigned_label"] == "rare" for item in sampler.row_lineage_[-3:])
    assert len(X_out) == 10
    json.dumps(sampler.diagnostics_, allow_nan=False)


@pytest.mark.parametrize("minority_count,majority_count", [(2, 2), (3, 2)])
def test_explicit_source_identity_is_fixed_with_tied_or_inverted_observed_counts(minority_count, majority_count):
    X = np.array([0.] * minority_count + [10.] * majority_count)[:, None]
    y = np.array([1] * minority_count + [0] * majority_count)
    sampler = SSHR(minority_label=1, delta=0)
    X_out, y_out = sampler.fit_resample(X, y)
    assert sampler.minority_label_ == 1
    assert sampler.majority_label_ == 0
    np.testing.assert_array_equal(X_out, X)
    np.testing.assert_array_equal(y_out, y)
    assert sampler.diagnostics_["quotas"]["additions_requested"] == 0


@pytest.mark.parametrize("parameter,value", [
    ("delta", -.1), ("delta", True), ("removal_fraction", 1.1),
    ("k_neighbors", 0), ("k_neighbors", True), ("random_state", -1),
    ("random_state", True), ("smote_pool_policy", []),
])
def test_invalid_resampling_parameters_fail_before_clustering_is_constructed(monkeypatch, parameter, value):
    X, y, U = separated_pools()

    class ShouldNotBeConstructed:
        def __init__(self, **kwargs):
            raise AssertionError("Invalid core settings must be checked before fitting SSHC")

    monkeypatch.setattr(sshr_module, "SSHC", ShouldNotBeConstructed)
    sampler = SSHR(**{parameter: value})
    with pytest.raises(ValueError):
        sampler.fit_resample(X, y, U)
    assert fitted_names(sampler) == set()


@pytest.mark.parametrize("parameter,value", [
    ("mu", 1.1), ("objective_mode", []), ("prune_scope", "unsupported"),
    ("tol", -1), ("max_split_rounds", 0), ("max_refine_iter", True),
    ("minority_label", "missing"),
])
def test_invalid_clustering_parameters_are_forwarded_and_leave_no_fitted_state(parameter, value):
    X, y, U = separated_pools()
    sampler = SSHR(**{parameter: value})
    with pytest.raises(ValueError):
        sampler.fit_resample(X, y, U)
    assert fitted_names(sampler) == set()


@pytest.mark.parametrize("bad_pool", ["category", "boolean", "complex_labeled", "complex_unlabeled"])
def test_categorical_boolean_and_complex_feature_inputs_are_rejected(bad_pool):
    X, y, U = separated_pools()
    if bad_pool == "category":
        X = pd.DataFrame({"category": pd.Categorical(X[:, 0])})
    elif bad_pool == "boolean":
        X = X.astype(bool)
    elif bad_pool == "complex_labeled":
        X = X.astype(complex) + 1j
    else:
        U = U.astype(complex) + 1j
    sampler = SSHR()
    with pytest.raises(ValueError):
        sampler.fit_resample(X, y, U)
    assert fitted_names(sampler) == set()


def test_clustering_failure_on_refit_clears_all_previous_geometry():
    X, y, U = separated_pools()
    sampler = SSHR()
    sampler.fit_resample(X, y, U)
    assert hasattr(sampler, "result_")
    with pytest.raises(ValueError, match="Tied class counts"):
        sampler.fit_resample([[0.], [1.], [10.], [11.]], [1, 1, 0, 0])
    assert fitted_names(sampler) == set()


def test_resampling_validation_failure_on_refit_clears_all_previous_geometry():
    X, y, U = separated_pools()
    sampler = SSHR()
    sampler.fit_resample(X, y, U)
    with pytest.raises(ValueError, match="k_neighbors"):
        sampler.set_params(k_neighbors=0).fit_resample(X, y, U)
    assert fitted_names(sampler) == set()


def test_diagnosed_class_erasure_clears_stale_geometry_and_preserves_only_fresh_failure_diagnostics():
    X, y, U = separated_pools()
    sampler = SSHR(minority_label=1, delta=0)
    sampler.fit_resample(X, y, U)
    with pytest.raises(SSHRResamplingError) as caught:
        sampler.fit_resample(np.zeros((3, 1)), [1, 1, 0])
    assert fitted_names(sampler) == {"diagnostics_"}
    assert sampler.diagnostics_["status"] == "failure_class_erased_by_preliminary_cleaning"
    assert sampler.diagnostics_["counts"]["removed_majority"] == 1
    assert sampler.diagnostics_["estimator_failure"]["exception_type"] == "SSHRResamplingError"
    assert "selected_pseudo_rows" not in sampler.diagnostics_
    assert "quotas" not in sampler.diagnostics_
    assert not hasattr(sampler, "clustering_")
    assert not hasattr(sampler, "result_")
    json.dumps(sampler.diagnostics_, allow_nan=False)
    caught.value.diagnostics["counts"]["removed_majority"] = 999
    assert sampler.diagnostics_["counts"]["removed_majority"] == 1
    with pytest.raises(ValueError):
        sampler.set_params(k_neighbors=0).fit_resample(X, y, U)
    assert fitted_names(sampler) == set()
    sampler.set_params(k_neighbors=5)
    sampler.fit_resample(X, y, U)
    assert sampler.diagnostics_["status"] == "quotas_met"
    assert "estimator_failure" not in sampler.diagnostics_


def test_unexpected_clustering_exception_also_clears_stale_fitted_state(monkeypatch):
    X, y, U = separated_pools()
    sampler = SSHR()
    sampler.fit_resample(X, y, U)

    class FailedSSHC(SSHC):
        def fit(self, *args, **kwargs):
            raise RuntimeError("controlled clustering failure")

    monkeypatch.setattr(sshr_module, "SSHC", FailedSSHC)
    with pytest.raises(RuntimeError, match="controlled clustering failure"):
        sampler.fit_resample(X, y, U)
    assert fitted_names(sampler) == set()


@pytest.mark.parametrize("payload", [{"not_finite": np.nan}, ["not_a_dictionary"]])
def test_malformed_success_diagnostics_cannot_publish_partially_fitted_state(monkeypatch, payload):
    X, y, U = separated_pools()
    sampler = SSHR()
    sampler.fit_resample(X, y, U)
    real_core = sshr_module.resample_from_clusters

    def corrupted_core(*args, **kwargs):
        return replace(real_core(*args, **kwargs), diagnostics=payload)

    monkeypatch.setattr(sshr_module, "resample_from_clusters", corrupted_core)
    with pytest.raises(ValueError):
        sampler.fit_resample(X, y, U)
    assert fitted_names(sampler) == set()


@pytest.mark.parametrize("payload", [{"not_finite": np.nan}, ["not_a_dictionary"]])
def test_malformed_failure_diagnostics_are_rejected_without_publishing_state(monkeypatch, payload):
    X, y, U = separated_pools()
    sampler = SSHR()
    sampler.fit_resample(X, y, U)

    def failed_core(*args, **kwargs):
        raise SSHRResamplingError("controlled malformed diagnostics", payload)

    monkeypatch.setattr(sshr_module, "resample_from_clusters", failed_core)
    with pytest.raises(ValueError, match="failure diagnostics"):
        sampler.fit_resample(X, y, U)
    assert fitted_names(sampler) == set()


def test_malformed_success_lineage_cannot_publish_fitted_state(monkeypatch):
    X, y, U = separated_pools()
    sampler = SSHR()
    sampler.fit_resample(X, y, U)
    real_core = sshr_module.resample_from_clusters

    def corrupted_core(*args, **kwargs):
        return replace(real_core(*args, **kwargs), row_lineage=[{"source": np.array([1])}])

    monkeypatch.setattr(sshr_module, "resample_from_clusters", corrupted_core)
    with pytest.raises(TypeError):
        sampler.fit_resample(X, y, U)
    assert fitted_names(sampler) == set()
