"""Train and evaluate an original-UGO pilot on one untouched holdout.

This is a working experiment, not a reproduction of the paper's full benchmark.
All preprocessing and generation use the training partition only. No choices
are made from the holdout metrics. Paper-specific ambiguities remain explicit in
UGO diagnostics; preprocessing choices below are declared engineering choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src._validation import finite_real, positive_integer
from src.data_loader import LoadedDataset
from src.ugo import UGO


class TrainingPreprocessor(TransformerMixin, BaseEstimator):
    """Drop degenerate raw columns, then impute/encode/scale using training only.

    The paper removes features with zero variance within any class. We interpret
    this as at most one observed distinct value in either training class, before
    imputation, including nominal columns. Columns with no observed values in a
    class are also removed and reported. Numerical median imputation/scaling and
    nominal most-frequent imputation/one-hot encoding are pipeline choices,
    rather than verified author preprocessing. Unknown categories are ignored.
    """

    def __init__(self, numeric_features: tuple[str, ...], categorical_features: tuple[str, ...]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    @staticmethod
    def _frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
            raise ValueError("Raw features must be a nonempty DataFrame with unique columns.")
        return X

    def fit(self, X: pd.DataFrame, y: Any) -> TrainingPreprocessor:
        X = self._frame(X)
        labels = _labels(y, len(X), require_both=True)
        names = tuple(self.numeric_features) + tuple(self.categorical_features)
        if len(set(names)) != len(names) or set(names) != set(X.columns):
            raise ValueError("Numeric/categorical feature metadata must partition the raw columns.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.dropped_features_: dict[str, str] = {}
        retained: list[str] = []
        for name in X.columns:
            if X[name].notna().sum() == 0:
                self.dropped_features_[name] = "all_missing_in_training"
                continue
            degenerate = [int(label) for label in (0, 1)
                          if X.iloc[np.flatnonzero(labels == label)][name].nunique(dropna=True) <= 1]
            if degenerate:
                self.dropped_features_[name] = "at_most_one_observed_value_in_training_class:" + ",".join(map(str, degenerate))
            else:
                retained.append(name)
        if not retained:
            raise ValueError("No usable features remain after training-only within-class feature filtering.")
        self.retained_features_ = tuple(retained)
        numeric = [name for name in self.numeric_features if name in retained]
        categorical = [name for name in self.categorical_features if name in retained]
        transformers = []
        if numeric:
            transformers.append(("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric))
        if categorical:
            transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical))
        self.transformer_ = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
        transformed = self.transformer_.fit_transform(self._prepared_frame(X), labels)
        if transformed.shape[1] == 0 or not np.isfinite(transformed).all():
            raise ValueError("Preprocessing must produce at least one finite numerical feature.")
        self.output_feature_names_ = tuple(self.transformer_.get_feature_names_out())
        self.n_output_features_ = transformed.shape[1]
        return self

    def _prepared_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        prepared = X.loc[:, list(self.feature_names_in_)].copy()
        for name in self.numeric_features:
            prepared[name] = pd.to_numeric(prepared[name], errors="raise").astype(float)
            if np.isinf(prepared[name].to_numpy()).any():
                raise ValueError(f"Numerical feature {name!r} contains infinity.")
        for name in self.categorical_features:
            # pandas category metadata can include categories absent from training;
            # converting observed values prevents that metadata defining the encoder.
            prepared[name] = prepared[name].astype(object).where(prepared[name].notna(), np.nan)
        return prepared

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "transformer_")
        X = self._frame(X)
        if set(X.columns) != set(self.feature_names_in_):
            raise ValueError("Prediction feature columns must match the training feature columns.")
        transformed = np.asarray(self.transformer_.transform(self._prepared_frame(X)), dtype=float)
        if not np.isfinite(transformed).all():
            raise ValueError("Preprocessed features must be finite.")
        return transformed

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "transformer_")
        return np.asarray(self.output_feature_names_, dtype=object)

    def metadata(self) -> dict[str, Any]:
        check_is_fitted(self, "transformer_")
        return {
            "retained_raw_features": list(self.retained_features_),
            "dropped_raw_features": dict(self.dropped_features_),
            "transformed_feature_names": list(self.output_feature_names_),
            "transformed_feature_count": int(self.n_output_features_),
            "fitted_on": "training_partition_only",
            "numeric_imputation": "median", "numeric_scaling": "StandardScaler",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot_unknown_categories_ignored",
            "within_class_filter": "at_most_one_observed_value_in_either_training_class_before_imputation",
            "categorical_smote_caveat": (
                "UGO-SMOTE interpolates encoded nominal features into fractional values; "
                "this is a numerical pipeline convention, not a verified categorical UGO rule."
                if self.categorical_features else None
            ),
        }


def _labels(y: Any, n: int, *, require_both: bool) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != n or pd.isna(labels).any():
        raise ValueError("Labels must be a nonmissing vector aligned with features.")
    if not np.isin(labels, [0, 1]).all() or (require_both and set(np.unique(labels)) != {0, 1}):
        raise ValueError("Labels must use majority=0 and minority=1, with both classes present.")
    return labels.astype(np.int64, copy=True)


def imbalanced_accuracy(y_true: Any, y_pred: Any) -> float:
    """IAM from the UGO paper, Section 4.3 (p. 9), for labels 0/1.

    Rows are true labels and columns predictions. For each class i, the term is
    (c_ii - max(row errors, column errors)) / max(row total, column total),
    averaged over the two classes. Both classes must occur in y_true, so all
    denominators are positive. This is not ordinary balanced accuracy.
    """
    truth = np.asarray(y_true)
    truth = _labels(truth, len(truth), require_both=True)
    predicted = _labels(y_pred, len(truth), require_both=False)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(float)
    diagonal = np.diag(matrix)
    rows, columns = matrix.sum(axis=1), matrix.sum(axis=0)
    return float(np.mean((diagonal - np.maximum(rows - diagonal, columns - diagonal)) / np.maximum(rows, columns)))


def binary_metrics(y_true: Any, y_pred: Any, probabilities: Any) -> dict[str, Any]:
    """Report metrics at the classifier's unchanged default decision threshold.

    Precision/F1 with no predicted positives use zero_division=0; sklearn MCC
    returns zero for a constant prediction. IAM requires both true classes.
    """
    truth = np.asarray(y_true)
    truth = _labels(truth, len(truth), require_both=True)
    predicted = _labels(y_pred, len(truth), require_both=False)
    scores = np.asarray(probabilities, dtype=float)
    if scores.shape != truth.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Minority probabilities must be finite, aligned, and lie in [0, 1].")
    return {
        "IAM": imbalanced_accuracy(truth, predicted),
        "MCC": float(matthews_corrcoef(truth, predicted)),
        "F1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "minority_recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
        "minority_precision": float(precision_score(truth, predicted, pos_label=1, zero_division=0)),
        "AUPRC": float(average_precision_score(truth, scores)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
    }


@dataclass
class ExperimentResult:
    """Artifacts for a single pilot; split indices are positional source rows."""

    report: dict[str, Any]
    models: dict[str, Pipeline]
    preprocessor: TrainingPreprocessor
    train_indices: np.ndarray
    test_indices: np.ndarray
    y_test: np.ndarray
    predictions: dict[str, np.ndarray]
    probabilities: dict[str, np.ndarray]


def run_experiment(
    dataset: LoadedDataset, *, test_size: float = 0.25, random_state: int = 42,
    cap_policy: str = "section_3_3", max_iter: int = 100,
) -> ExperimentResult:
    """Compare NORES, UGO-ROS and UGO-SMOTE on the same stratified holdout.

    LogisticRegression(max_iter=1000) supplies UGO probabilities and a fresh
    clone is trained on every final augmented dataset. Its internal UGO scoring
    model is never used as the final model. All methods use the same fitted
    preprocessing. No threshold tuning, test-guided selection or extension is
    performed. A single holdout is a pilot, not the paper's ten-fold protocol.
    """
    test_size = finite_real(test_size, "test_size")
    if not 0 < test_size < 1:
        raise ValueError("test_size must lie in (0, 1).")
    if isinstance(random_state, bool) or not isinstance(random_state, Integral) or not 0 <= random_state < 2 ** 32:
        raise ValueError("random_state must be an integer in [0, 2**32).")
    random_state = int(random_state)
    positive_integer(max_iter, "max_iter")
    if not isinstance(cap_policy, str) or cap_policy not in {"section_3_3", "table_4"}:
        raise ValueError("cap_policy must be 'section_3_3' or 'table_4'.")
    X = TrainingPreprocessor._frame(dataset.X)
    labels = _labels(dataset.y, len(X), require_both=True)
    if min(np.bincount(labels, minlength=2)) < 3:
        raise ValueError("Each class requires at least three observations for training (at least two) and an untouched test observation.")
    try:
        train_indices, test_indices = train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=random_state, stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(f"Cannot construct a stratified train/test split: {exc}") from exc
    y_train, y_test = labels[train_indices], labels[test_indices]
    if min(np.bincount(y_train, minlength=2)) < 2 or min(np.bincount(y_test, minlength=2)) < 1:
        raise ValueError("Requested split must leave at least two observations of each class in training and one in test; change test_size or use more data.")
    X_train, X_test = X.iloc[train_indices].copy(), X.iloc[test_indices].copy()
    preprocessor = TrainingPreprocessor(dataset.numeric_feature_names, dataset.categorical_feature_names)
    transformed_train = preprocessor.fit_transform(X_train, y_train)
    # Evaluation data are transformed only after the complete preprocessor has
    # been fitted using training values and training labels.
    transformed_test = preprocessor.transform(X_test)
    template = LogisticRegression(max_iter=1000, random_state=random_state)
    models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    method_reports: dict[str, Any] = {}
    for name, variant in (("NORES", None), ("UGO-ROS", "ros"), ("UGO-SMOTE", "smote")):
        started = perf_counter()
        diagnostics = None
        if variant is None:
            fitting_X, fitting_y = transformed_train.copy(), y_train.copy()
        else:
            sampler = UGO(
                estimator=clone(template), variant=variant, random_state=random_state,
                cap_policy=cap_policy, max_iter=max_iter, allow_provisional=True,
            )
            fitting_X, fitting_y = sampler.fit_resample(transformed_train.copy(), y_train.copy())
            diagnostics = sampler.diagnostics_
        final_classifier = clone(template)
        final_classifier.fit(fitting_X, fitting_y)
        fit_seconds = perf_counter() - started
        # Both components are already fitted. Persist this raw-feature pipeline
        # so local inference uses exactly the preprocessing learned in training.
        models[name] = Pipeline([("preprocessor", preprocessor), ("classifier", final_classifier)])
        predictions[name] = final_classifier.predict(transformed_test)
        positive_column = int(np.flatnonzero(final_classifier.classes_ == 1)[0])
        probabilities[name] = final_classifier.predict_proba(transformed_test)[:, positive_column]
        method_reports[name] = {
            "metrics": binary_metrics(y_test, predictions[name], probabilities[name]),
            "training_rows": int(len(fitting_y)),
            "synthetic_rows": int(len(fitting_y) - len(y_train)),
            "augmented_class_counts": {str(label): int((fitting_y == label).sum()) for label in (0, 1)},
            "fit_seconds": float(fit_seconds),
            "ugo_diagnostics": diagnostics,
        }
    report = {
        "dataset": {
            "source_filename": dataset.source_filename, "source_format": dataset.source_format,
            "rows": len(X), "raw_features": X.shape[1],
            "label_mapping": dict(dataset.label_mapping),
            "majority_label": dataset.majority_label, "minority_label": dataset.minority_label,
        },
        "protocol": {
            "evaluation": "single_stratified_holdout_pilot",
            "test_size": test_size, "random_state": random_state,
            "classifier": "LogisticRegression", "classifier_parameters": template.get_params(),
            "threshold": "default_classifier_predict; no threshold tuning",
            "cap_policy": cap_policy, "max_iter": int(max_iter),
            "reproduction_status": "paper_checked_assumptions_unresolved",
            "test_used_for_selection": False,
            "warnings": [
                "This single holdout pilot does not reproduce the published ten-fold, 74-dataset benchmark.",
                "The paper's conflicting generation caps and undocumented pool/rounding conventions remain explicit assumptions.",
                "Imputation, scaling and nominal encoding are declared pipeline choices, not verified author preprocessing.",
                "Holdout results are descriptive; a single split does not establish general superiority.",
            ],
            "metrics_conventions": {
                "confusion_matrix": "rows=true, columns=predicted, label_order=[0,1]",
                "positive_class": 1, "precision_f1_zero_division": 0,
                "MCC_constant_prediction": 0,
                "IAM": "UGO paper Section 4.3, p.9; not balanced accuracy",
                "AUPRC": "sklearn average_precision_score (non-interpolated average precision)",
            },
        },
        "split": {
            "train_indices": train_indices.tolist(), "test_indices": test_indices.tolist(),
            "index_convention": "zero_based_positional_source_rows",
            "train_rows": len(train_indices), "test_rows": len(test_indices),
            "train_class_counts": {str(label): int((y_train == label).sum()) for label in (0, 1)},
            "test_class_counts": {str(label): int((y_test == label).sum()) for label in (0, 1)},
        },
        "preprocessing": preprocessor.metadata(), "methods": method_reports,
    }
    return ExperimentResult(report, models, preprocessor, train_indices, test_indices,
                            y_test.copy(), predictions, probabilities)
