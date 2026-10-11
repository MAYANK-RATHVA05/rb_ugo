"""Holdout isolation, final-model fitting and numerical metric checks."""
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_loader import LoadedDataset
import src.training as training
from src.training import TrainingPreprocessor, binary_metrics, imbalanced_accuracy, run_experiment


def dataset(n_majority=60, n_minority=20, *, categorical=False):
    rng = np.random.default_rng(19)
    y = np.r_[np.zeros(n_majority, dtype=int), np.ones(n_minority, dtype=int)]
    X = pd.DataFrame({"x": rng.normal(size=len(y)) + 0.8 * y,
                      "z": rng.normal(size=len(y))})
    categories = ()
    if categorical:
        X["kind"] = pd.Categorical(np.where(np.arange(len(y)) % 2, "a", "b"),
                                    categories=["a", "b", "unused_metadata_category"])
        categories = ("kind",)
    original = pd.Series(np.where(y == 0, "negative", "positive"))
    return LoadedDataset(
        X=X, y=pd.Series(y), original_y=original,
        label_mapping={"negative": 0, "positive": 1},
        feature_names=tuple(X.columns), numeric_feature_names=("x", "z"),
        categorical_feature_names=categories, majority_label="negative", minority_label="positive",
        majority_count=n_majority, minority_count=n_minority, target_name="class",
        source_filename="software-test-fixture.csv", source_format="csv",
    )


def labels_from_matrix(matrix):
    truth, predicted = [], []
    for true in (0, 1):
        for pred in (0, 1):
            truth.extend([true] * matrix[true][pred])
            predicted.extend([pred] * matrix[true][pred])
    return np.asarray(truth), np.asarray(predicted)


@pytest.mark.parametrize(("matrix", "expected"), [
    ([[90, 0], [0, 10]], 1.0),
    ([[0, 90], [10, 0]], -1.0),
    ([[90, 0], [10, 0]], -0.1),
    ([[0, 90], [0, 10]], -0.9),
    ([[8, 2], [1, 3]], 0.4),
])
def test_iam_known_confusion_matrices(matrix, expected):
    truth, predicted = labels_from_matrix(matrix)
    assert imbalanced_accuracy(truth, predicted) == pytest.approx(expected)
    # Swapping which class is called positive cannot alter IAM.
    assert imbalanced_accuracy(1 - truth, 1 - predicted) == pytest.approx(expected)


def test_metrics_report_standard_orientation_and_zero_division():
    truth, predicted = labels_from_matrix([[90, 0], [10, 0]])
    metrics = binary_metrics(truth, predicted, np.full(100, 0.1))
    assert metrics["confusion_matrix"] == [[90, 0], [10, 0]]
    assert metrics["F1"] == metrics["minority_precision"] == metrics["MCC"] == 0
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["AUPRC"] == pytest.approx(0.1)


@pytest.mark.parametrize(("truth", "predicted"), [([0, 0], [0, 0]), ([0, 1], [0]), ([0, 1], [0, 2])])
def test_iam_rejects_invalid_or_single_true_class(truth, predicted):
    with pytest.raises(ValueError):
        imbalanced_accuracy(truth, predicted)


def test_preprocessing_uses_training_values_labels_and_observed_categories_only():
    X = pd.DataFrame({
        "x": [0, np.nan, 2, 3, 10, 11, np.nan, 13],
        "within_class_constant": [5, 5, 5, 5, 10, 11, 12, 13],
        "all_missing": [np.nan] * 8,
        "kind": pd.Categorical(["left", "right"] * 4, categories=["left", "right", "held_out"]),
    })
    preprocessor = TrainingPreprocessor(("x", "within_class_constant", "all_missing"), ("kind",))
    transformed = preprocessor.fit_transform(X, [0] * 4 + [1] * 4)
    heldout = pd.DataFrame({"x": [1e9], "within_class_constant": [999],
                           "all_missing": [999], "kind": ["held_out"]})
    numeric = preprocessor.transformer_.named_transformers_["numeric"]
    assert numeric.named_steps["imputer"].statistics_.tolist() == [6.5]
    assert numeric.named_steps["scaler"].mean_.tolist() == [6.5]
    encoder = preprocessor.transformer_.named_transformers_["categorical"].named_steps["encoder"]
    assert encoder.categories_[0].tolist() == ["left", "right"]
    assert set(preprocessor.dropped_features_) == {"within_class_constant", "all_missing"}
    assert transformed.shape == (8, 3)
    np.testing.assert_array_equal(preprocessor.transform(heldout)[0, 1:], [0, 0])
    assert numeric.named_steps["scaler"].mean_.tolist() == [6.5]


def test_preprocessing_rejects_no_usable_features_and_metadata_mismatch():
    with pytest.raises(ValueError, match="No usable features"):
        TrainingPreprocessor(("x",), ()).fit(pd.DataFrame({"x": [1, 1, 2, 2]}), [0, 0, 1, 1])
    with pytest.raises(ValueError, match="metadata"):
        TrainingPreprocessor(("unknown",), ()).fit(pd.DataFrame({"x": [1, 2, 3, 4]}), [0, 0, 1, 1])


class RecordingUGO:
    """Controlled augmentation makes isolation/final-fit checks independent of UGO stopping."""

    instances = []

    def __init__(self, estimator, variant, **kwargs):
        self.estimator = estimator
        self.variant = variant
        self.kwargs = kwargs
        self.instances.append(self)

    def fit_resample(self, X, y):
        self.input_X, self.input_y = X.copy(), y.copy()
        self.estimator_ = clone(self.estimator).fit(X, y)
        # Deliberately add a final minority batch that changes the fitted model.
        self.output_X = np.vstack([X, np.repeat(X[y == 1][:2], 5, axis=0)])
        self.output_y = np.r_[y, np.ones(10, dtype=int)]
        self.diagnostics_ = {"number_of_synthetic_samples": 10,
                             "stopping_reason": "controlled_test_batch"}
        return self.output_X.copy(), self.output_y.copy()


def test_shared_split_resampler_isolation_and_fresh_final_model_on_last_batch(monkeypatch):
    fixture = dataset()
    RecordingUGO.instances = []
    monkeypatch.setattr(training, "UGO", RecordingUGO)
    result = run_experiment(fixture)
    assert set(result.train_indices).isdisjoint(result.test_indices)
    assert set(result.train_indices) | set(result.test_indices) == set(range(len(fixture.y)))
    expected_train = result.preprocessor.transform(fixture.X.iloc[result.train_indices])
    expected_labels = fixture.y.iloc[result.train_indices].to_numpy()
    for sampler, name in zip(RecordingUGO.instances, ("UGO-ROS", "UGO-SMOTE")):
        np.testing.assert_array_equal(sampler.input_X, expected_train)
        np.testing.assert_array_equal(sampler.input_y, expected_labels)
        model = result.models[name].named_steps["classifier"]
        assert model is not sampler.estimator_
        expected = LogisticRegression(max_iter=1000, random_state=42).fit(sampler.output_X, sampler.output_y)
        np.testing.assert_allclose(model.coef_, expected.coef_)
        assert not np.allclose(model.coef_, sampler.estimator_.coef_)
        assert result.report["methods"][name]["training_rows"] == len(result.train_indices) + 10
        assert result.models[name].named_steps["preprocessor"] is result.preprocessor
        np.testing.assert_array_equal(result.models[name].predict(fixture.X.iloc[result.test_indices]), result.predictions[name])
    assert result.report["methods"]["NORES"]["synthetic_rows"] == 0


def test_changing_holdout_values_cannot_change_training_or_preprocessing(monkeypatch):
    fixture = dataset()
    _, test_indices = train_test_split(np.arange(len(fixture.y)), test_size=0.25,
                                      stratify=fixture.y, random_state=42)
    mutated = fixture.X.copy()
    mutated.iloc[test_indices, :] = 1e12
    monkeypatch.setattr(training, "UGO", RecordingUGO)
    first = run_experiment(fixture)
    second = run_experiment(replace(fixture, X=mutated))
    assert first.preprocessor.metadata() == second.preprocessor.metadata()
    for name in first.models:
        np.testing.assert_array_equal(first.models[name].named_steps["classifier"].coef_,
                                      second.models[name].named_steps["classifier"].coef_)
    np.testing.assert_array_equal(first.preprocessor.transform(fixture.X.iloc[first.train_indices]),
                                  second.preprocessor.transform(fixture.X.iloc[second.train_indices]))


def test_real_pipeline_repeats_deterministically_and_persists_for_raw_inference(tmp_path):
    fixture = dataset(categorical=True)
    with pytest.warns(UserWarning):
        first = run_experiment(fixture, max_iter=2)
    with pytest.warns(UserWarning):
        second = run_experiment(fixture, max_iter=2)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    for name in first.models:
        np.testing.assert_array_equal(first.predictions[name], second.predictions[name])
        np.testing.assert_array_equal(first.probabilities[name], second.probabilities[name])
        assert first.report["methods"][name]["metrics"] == second.report["methods"][name]["metrics"]
        destination = tmp_path / f"{name}.joblib"
        joblib.dump(first.models[name], destination)
        restored = joblib.load(destination)
        heldout = fixture.X.iloc[first.test_indices].copy()
        np.testing.assert_array_equal(restored.predict(heldout), first.predictions[name])
        heldout["kind"] = "a_new_category"
        assert np.isfinite(restored.predict_proba(heldout)).all()


@pytest.mark.parametrize("test_size", [0, 1, -0.1, np.nan, True, "0.25"])
def test_bad_test_size_rejected(test_size):
    with pytest.raises(ValueError, match="test_size"):
        run_experiment(dataset(), test_size=test_size)


@pytest.mark.parametrize("random_state", [-1, 2**32, True, 1.5])
def test_bad_seed_rejected(random_state):
    with pytest.raises(ValueError, match="random_state"):
        run_experiment(dataset(), random_state=random_state)


def test_small_minority_and_unusable_split_are_clear_errors():
    with pytest.raises(ValueError, match="at least three"):
        run_experiment(dataset(n_majority=20, n_minority=2))
    with pytest.raises(ValueError, match="at least two.*training"):
        run_experiment(dataset(n_majority=20, n_minority=3), test_size=0.9)
    with pytest.raises(ValueError, match="stratified"):
        run_experiment(dataset(), test_size=0.001)


def test_bad_cap_policy_rejected():
    with pytest.raises(ValueError, match="cap_policy"):
        run_experiment(dataset(), cap_policy="guessed")
