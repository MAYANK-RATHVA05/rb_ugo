"""Deterministic UGO software checks, not evidence of research performance."""
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import src.ugo as ugo_module
from src.ugo import UGO, informative_seed_indices


class ToyEstimator(BaseEstimator):
    """Controlled probabilities, including reversed class columns."""

    fits = []

    def __init__(self, uncertain_region="low", random_state=None):
        self.uncertain_region = uncertain_region
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)[::-1]
        type(self).fits.append((X.copy(), y.copy()))
        return self

    def predict_proba(self, X):
        high = X[:, 0] >= 5
        p1 = np.where(high, 0.9 + 0.01 * (X[:, 0] - 10), 0.8 - 0.01 * X[:, 0])
        if self.uncertain_region == "high":
            p1 = np.where(high, 0.2 + 0.01 * (X[:, 0] - 10), 0.1 + 0.01 * X[:, 0])
        if self.uncertain_region == "equal":
            p1 = np.where(high, 0.8 - 0.1 * (X[:, 0] - 10), 0.2 + 0.1 * X[:, 0])
        if self.uncertain_region == "constant":
            p1 = np.full(len(X), 0.5)
        return np.column_stack((p1, 1 - p1))  # classes_=[1,0]


@pytest.fixture
def toy():
    return np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0]]), np.array([0, 0, 0, 0, 1, 1])


def run(X, y, **kwargs):
    estimator = kwargs.pop("estimator", ToyEstimator())
    params = {"k_pre": 1, "max_iter": 2, "allow_provisional": True, **kwargs}
    sampler = UGO(estimator, **params)
    with pytest.warns(UserWarning, match="Provisional UGO"):
        X_new, y_new = sampler.fit_resample(X, y)
    return sampler, X_new, y_new


def test_explicit_provisional_opt_in_required(toy) -> None:
    with pytest.raises(ValueError, match="allow_provisional"):
        UGO(ToyEstimator()).fit_resample(*toy)


def test_informative_seed_order_and_noise_exclusion() -> None:
    seeds = informative_seed_indices([0.2, 0.9, 0.7, 0.7, 0.99],
                                      [0, 0, 0, 0, 1], 0,
                                      [False, True, False, False, False], 0.5)
    assert seeds.tolist() == [2, 3]  # ceil(0.5 * 3), stable equal-score order.


@pytest.mark.parametrize("variant", ["ros", "smote"])
def test_majority_class_can_be_selected_and_outputs_preserved(toy, variant) -> None:
    X, y = toy
    X_before, y_before = X.copy(), y.copy()
    sampler, X_new, y_new = run(X, y, variant=variant)
    history = sampler.diagnostics_["history"]
    assert all(record["selected_class"] == 0 for record in history)
    assert history[0]["seed_indices"] == [0, 1]  # Largest MMA scores, not random seeds.
    assert len(X_new) > len(X)
    assert X_new.shape[1] == X.shape[1] and y_new.shape == (len(X_new),)
    assert (y_new[len(y):] == 0).all()
    np.testing.assert_array_equal(X_new[:len(X)], X)
    np.testing.assert_array_equal(y_new[:len(y)], y)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    assert sampler.diagnostics_["stopping_reason"] == "max_iterations_safeguard"
    assert sampler.diagnostics_["reproduction_status"] == "provisional_prompt_based"
    assert sampler.diagnostics_["final_class_counts"][0]["count"] > 4


@pytest.mark.parametrize("variant", ["ros", "smote"])
def test_seed_reproducibility(toy, variant) -> None:
    first, X1, y1 = run(*toy, variant=variant)
    second, X2, y2 = run(*toy, variant=variant)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert first.diagnostics_ == second.diagnostics_


def test_smote_interpolation_really_uses_selected_seeds(toy) -> None:
    sampler, X_new, _ = run(*toy, variant="smote", max_iter=1, lambda_step=0.5)
    record = sampler.diagnostics_["history"][0]
    generation = record["generation"]
    X, _ = toy
    expected = []
    for seed, neighbor, step in zip(generation["drawn_seed_indices"],
                                    generation["neighbor_indices"], generation["interpolation_steps"]):
        assert seed in record["seed_indices"]
        assert seed != neighbor
        expected.append((1 - step) * X[seed] + step * X[neighbor])
    np.testing.assert_allclose(X_new[len(X):], expected)
    assert not np.isin(X_new[len(X):, 0], X[:, 0]).all()


def test_ros_only_copies_selected_seeds(toy) -> None:
    sampler, X_new, _ = run(*toy, max_iter=1, lambda_step=1.0)
    selected = toy[0][sampler.diagnostics_["history"][0]["seed_indices"], 0]
    assert np.isin(X_new[len(toy[0]):, 0], selected).all()


def test_cap_truncates_last_batch(toy) -> None:
    sampler, X_new, _ = run(*toy, phi=1.0, lambda_step=1.0, max_iter=20)
    # Original counts [4,2]: floor(1*2*4-6)=2 synthetic rows.
    assert sampler.diagnostics_["generation_cap"] == 2
    assert sampler.diagnostics_["number_of_synthetic_samples"] == 2
    assert len(X_new) == 8
    assert sampler.diagnostics_["stopping_reason"] == "provisional_generation_cap"


def test_zero_cap_no_op(toy) -> None:
    sampler, X_new, y_new = run(*toy, phi=0.1)
    assert sampler.diagnostics_["number_of_iterations"] == 0
    assert sampler.diagnostics_["stopping_reason"] == "provisional_generation_cap"
    np.testing.assert_array_equal(X_new, toy[0])
    np.testing.assert_array_equal(y_new, toy[1])


def test_all_pre_noise_no_op() -> None:
    sampler, X_new, _ = run([[0.0], [10.0]], [0, 1])
    assert sampler.diagnostics_["pre_detected_noisy_samples"] == 2
    assert sampler.diagnostics_["stopping_reason"] == "no_eligible_non_noisy_seeds"
    assert len(X_new) == 2


def test_undefined_welch_safeguard(toy) -> None:
    sampler, X_new, _ = run(*toy, estimator=ToyEstimator("constant"))
    assert sampler.diagnostics_["stopping_reason"] == "undefined_welch_safeguard:both_groups_constant"
    assert len(X_new) == len(toy[0])
    assert sampler.diagnostics_["history"][0]["welch"]["p_value"] is None


def test_welch_stop_and_balanced_data() -> None:
    X, y = [[0.0], [1.0], [10.0], [11.0]], [0, 0, 1, 1]
    sampler, X_new, _ = run(X, y, estimator=ToyEstimator("equal"))
    assert sampler.diagnostics_["stopping_reason"] == "welch_fail_to_reject"
    assert len(X_new) == 4
    # Balance alone is not an added stopping rule.
    sampler, X_new, _ = run(X, y, estimator=ToyEstimator())
    assert len(X_new) > 4


def test_cloning_training_sizes_and_no_test_inputs(toy, monkeypatch) -> None:
    assert list(inspect.signature(UGO.fit_resample).parameters) == ["self", "X", "y"]
    ToyEstimator.fits = []
    original_estimator = ToyEstimator()
    references = []
    relocate = ugo_module.noise_final_detect_and_relocate

    def spy(generated, label, reference_X, reference_y, *args, **kwargs):
        references.append((reference_X.copy(), reference_y.copy()))
        return relocate(generated, label, reference_X, reference_y, *args, **kwargs)

    monkeypatch.setattr(ugo_module, "noise_final_detect_and_relocate", spy)
    sampler, _, _ = run(*toy, estimator=original_estimator, max_iter=3)
    assert not hasattr(original_estimator, "classes_")
    assert [len(X) for X, _ in ToyEstimator.fits] == [6, 7, 8]
    assert sampler.estimator_.random_state == 42
    for X, y in references:
        np.testing.assert_array_equal(X, toy[0])
        np.testing.assert_array_equal(y, toy[1])
    assert all(not (X == 9999).any() for X, _ in ToyEstimator.fits)


def test_insufficient_smote_neighbors_is_diagnosed(toy, monkeypatch) -> None:
    # Only one clean selected-class sample: keep a defined controlled Welch
    # decision to exercise the separate generator guard.
    from src.stopping import WelchDecision
    monkeypatch.setattr(ugo_module, "welch_stopping",
                        lambda *a: WelchDecision(5.0, 0.001, 2.0, False, True, "reject_equal_means"))
    monkeypatch.setattr(ugo_module, "noise_pre_detect",
                        lambda X, y, k: np.array([False, True, True, True, False, False]))
    sampler, X_new, _ = run(*toy, variant="smote")
    assert sampler.diagnostics_["stopping_reason"] == "insufficient_same_class_neighbors"
    assert len(X_new) == 6


def test_real_estimator_both_variants_and_string_labels() -> None:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=80, n_features=4, n_informative=3,
                                n_redundant=0, weights=[0.75, 0.25], random_state=42)
    labels = np.where(y == 0, "normal", "rare")
    for variant in ["ros", "smote"]:
        sampler, X_new, y_new = run(X, labels, variant=variant, k_pre=3,
                                    estimator=LogisticRegression(max_iter=1000))
        assert X_new.shape[0] == len(y_new)
        assert set(y_new) == {"normal", "rare"}
        assert sampler.diagnostics_["stopping_reason"] is not None


def test_demo_executes_generation_for_both_variants() -> None:
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "main_stage3.py")],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PROVISIONAL" in result.stdout
    assert "UGO-ROS (provisional)" in result.stdout
    assert "UGO-SMOTE (provisional)" in result.stdout
    assert result.stdout.count("stopping_reason: welch_fail_to_reject") == 2
    assert "number_of_synthetic_samples: 3" in result.stdout
    assert "number_of_synthetic_samples: 6" in result.stdout


@pytest.mark.parametrize("features", [np.array([["red"], ["blue"]]),
                                       pd.DataFrame({"x": pd.Categorical([1, 2])}),
                                       np.array([[np.nan], [1.0]]),
                                       np.array([[np.inf], [1.0]])])
def test_unsupported_features_rejected(features) -> None:
    with pytest.raises(ValueError, match="numerical|finite"):
        UGO(ToyEstimator(), allow_provisional=True).fit_resample(features, [0, 1])


def test_estimator_without_probabilities_rejected(toy) -> None:
    with pytest.raises(ValueError, match="predict_proba"):
        UGO(LinearSVC(), allow_provisional=True).fit_resample(*toy)


def test_identical_vectors_terminate_without_algorithm_switch() -> None:
    sampler, X_new, _ = run(np.ones((6, 1)), [0, 0, 0, 0, 1, 1], variant="smote")
    assert sampler.diagnostics_["stopping_reason"] == "class_without_eligible_non_noisy_samples"
    assert len(X_new) == 6


def test_too_small_uncertainty_group_is_diagnosed() -> None:
    # All samples are clean, but class 1 has only one original sample.
    X = [[0.0], [1.0], [2.0], [10.0]]
    sampler, _, _ = run(X, [0, 0, 0, 1], k_pre=2)
    assert sampler.diagnostics_["stopping_reason"] in {
        "class_without_eligible_non_noisy_samples", "undefined_welch_safeguard:too_small_groups",
    }


def test_explicit_estimator_seed_is_preserved(toy) -> None:
    sampler, _, _ = run(*toy, estimator=ToyEstimator(random_state=7))
    assert sampler.estimator_.random_state == 7
    assert sampler.diagnostics_["effective_parameters"]["estimator_random_states"] == {"random_state": 7}


def test_failed_second_run_clears_stale_diagnostics(toy) -> None:
    sampler, _, _ = run(*toy)
    sampler.set_params(alpha=1)
    with pytest.raises(ValueError):
        sampler.fit_resample(*toy)
    assert not hasattr(sampler, "diagnostics_")


@pytest.mark.parametrize("kwargs", [{"lambda_select": 0}, {"lambda_step": np.nan},
                                      {"k_pre": 0}, {"max_iter": 0}, {"phi": -1},
                                      {"alpha": 1}, {"random_state": None}, {"variant": "ordinary_smote"}])
def test_invalid_parameters(toy, kwargs) -> None:
    with pytest.raises(ValueError):
        UGO(ToyEstimator(), allow_provisional=True, **kwargs).fit_resample(*toy)
