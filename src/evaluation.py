"""Paired ten-fold evaluation; every learned operation sees fitting rows only."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src._validation import binary_labels, numeric_matrix
from src.classifier_registry import get_classifier_spec, make_classifier, UnavailableClassifierError
from src.comparators import make_comparator
from src.metrics import binary_metrics, METRIC_POLICY
from src.preprocessing import feature_frame, FoldPreprocessor, PREPROCESSING_POLICY
from src.results import content_hash, json_value, ResultStore
from src.ugo import UGO


METHODS = ("nores", "rus", "ros", "smote", "rwo", "mcsmote", "leao",
           "ugo_ros", "ugo_smote", "ugo_rwo", "ugo_mcsmote")
PROTOCOL = "declared stratified ten-fold approximation; not author folds"


@dataclass(frozen=True)
class EvaluationConfig:
    classifiers: tuple[str, ...] = ("LogReg",)
    methods: tuple[str, ...] = METHODS
    seed: int = 42
    scale: bool = True
    method_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


def _validate_config(config: EvaluationConfig) -> None:
    if isinstance(config.seed, bool) or not isinstance(config.seed, Integral) or not 0 <= config.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if not isinstance(config.scale, bool):
        raise ValueError("scale must be boolean.")
    if (not config.methods or len(set(config.methods)) != len(config.methods)
            or "nores" not in config.methods or any(m not in METHODS for m in config.methods)):
        raise ValueError("Methods must be unique known names and include nores.")
    if not config.classifiers or len(set(config.classifiers)) != len(config.classifiers):
        raise ValueError("At least one unique classifier is required.")
    for name in config.classifiers:
        get_classifier_spec(name)
    for name, parameters in config.method_parameters.items():
        if name not in config.methods or not isinstance(parameters, dict):
            raise ValueError("Method parameters must map selected methods to dictionaries.")
        if set(parameters) & {"estimator", "variant", "random_state", "allow_provisional"}:
            raise ValueError("Estimator, variant, random_state and provisional status are controlled by the runner.")
        if name == "nores" and parameters:
            raise ValueError("nores takes no method parameters.")
    # Reject nonfinite configuration values rather than silently normalizing them.
    json.dumps(asdict(config), allow_nan=False)


def make_evaluation_resampler(method: str, estimator: Any, *, seed: int,
                             parameters: dict[str, Any]):
    if method.startswith("ugo_"):
        return UGO(clone(estimator), variant=method[4:], random_state=seed,
                   allow_provisional=True, **parameters)
    return make_comparator(method, random_state=seed, **parameters)


def ten_fold_assignments(y: Any, *, seed: int = 42,
                         supplied_folds: Any = None) -> tuple[list[dict[str, Any]], str]:
    """Record exact row indices. Supplied folds need provenance from the caller."""
    labels, classes = binary_labels(y, len(y))
    if min(int(np.sum(labels == c)) for c in classes) < 10:
        raise ValueError("Ten-fold evaluation requires at least ten rows in each class; folds will not be reduced.")
    if supplied_folds is None:
        pairs = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
                     .split(np.zeros((len(labels), 1)), labels))
        protocol = PROTOCOL
    else:
        pairs = list(supplied_folds)
        protocol = "caller-supplied ten-fold assignments; authority recorded in provenance"
    if len(pairs) != 10:
        raise ValueError("Exactly ten outer folds are required.")
    records, test_seen = [], np.zeros(len(labels), dtype=int)
    universe = np.arange(len(labels))
    for fold, (training, testing) in enumerate(pairs):
        train, test = np.asarray(training), np.asarray(testing)
        for indices in (train, test):
            if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                    or (indices < 0).any() or (indices >= len(labels)).any()
                    or len(np.unique(indices)) != len(indices)):
                raise ValueError("Fold indices must be nonempty, unique, in-range integer vectors.")
        if (np.intersect1d(train, test).size or not np.array_equal(np.sort(np.r_[train, test]), universe)):
            raise ValueError("Each training partition must be the exact complement of its test partition.")
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
            raise ValueError("Every training/test fold must contain both fixed classes.")
        test_seen[test] += 1
        records.append({"fold": fold, "train_indices": train.tolist(), "test_indices": test.tolist()})
    if not np.all(test_seen == 1):
        raise ValueError("Every input row must appear in exactly one test fold.")
    return records, protocol


def _input_hash(X: pd.DataFrame, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(content_hash({"columns": X.columns.tolist(), "dtypes": [str(d) for d in X.dtypes],
                                "y_dtype": str(y.dtype)}).encode())
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(pd.Series(y), index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _software_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    inputs = (sorted((root / "src").glob("*.py"))
              + sorted((root / "native").glob("*"))
              + [root / "requirements.txt", root / "requirements-classifiers.txt",
                 root / "scripts" / "setup_c50.py"])
    for path in inputs:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "stochtree"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    c50_manifest = root / ".runtime" / "c50" / "backend.json"
    return {"python": platform.python_version(), "source_sha256": digest.hexdigest(),
            "packages": packages,
            "c50_backend": (json.loads(c50_manifest.read_text())
                              if c50_manifest.is_file() else "unavailable")}


def _counts(y: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    return [{"label": label, "count": int(np.sum(y == label))} for label in labels]


def _evaluate_job(X: pd.DataFrame, y: np.ndarray, labels: np.ndarray, positive_label: Any,
                  fold: dict[str, Any], job: dict[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    started = perf_counter()
    seed = (int(config.seed) + fold["fold"]) % 2**32
    train, test = np.asarray(fold["train_indices"]), np.asarray(fold["test_indices"])
    record = {"job": job, "seed": seed, "train_indices": train, "test_indices": test,
              "positive_label": positive_label, "train_counts": _counts(y[train], labels),
              "test_counts": _counts(y[test], labels), "metrics": None, "predictions": None,
              "probabilities": None, "error": None, "timings_seconds": {},
              "preprocessing": None, "resampler_diagnostics": None}
    stage = "classifier_constructor"
    sampler = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            template = make_classifier(job["classifier"], seed=seed)
            record["classifier_parameters"] = template.get_params(deep=False)
            stage = "preprocessing"
            stage_start = perf_counter()
            preprocessing = FoldPreprocessor(scale=config.scale)
            fitting = preprocessing.fit_transform(X.iloc[train].copy(), y[train].copy())
            # Test features are not passed to fit, resampling, calibration or selection.
            testing = preprocessing.transform(X.iloc[test].copy())
            record["preprocessing"] = preprocessing.diagnostics_
            record["timings_seconds"][stage] = perf_counter() - stage_start
            stage = "resampling"
            stage_start = perf_counter()
            sampler = make_evaluation_resampler(job["method"], template, seed=seed,
                                                parameters=config.method_parameters.get(job["method"], {}))
            parameters = sampler.get_params(deep=False)
            record["resampler_parameters"] = {k: v for k, v in parameters.items() if k != "estimator"}
            augmented, augmented_y = sampler.fit_resample(fitting.copy(), y[train].copy())
            augmented = numeric_matrix(augmented)
            augmented_y, augmented_classes = binary_labels(augmented_y, len(augmented))
            if augmented.shape[1] != fitting.shape[1] or not np.array_equal(labels, augmented_classes):
                raise ValueError("Resampler changed the feature width or label identities.")
            record["resampler_diagnostics"] = getattr(sampler, "diagnostics_", None)
            record["timings_seconds"][stage] = perf_counter() - stage_start
            record["resampled_counts"] = _counts(augmented_y, labels)
            record["sample_counts"] = {"training": len(train), "test": len(test),
                                       "resampled_training": len(augmented),
                                       "added_rows": max(0, len(augmented) - len(train)),
                                       "removed_rows": max(0, len(train) - len(augmented))}
            stage = "final_fit"
            stage_start = perf_counter()
            # UGO.estimator_ can predate the last batch. Always fit a fresh clone.
            final_model = clone(template)
            final_model.fit(augmented.copy(), augmented_y.copy())
            record["timings_seconds"][stage] = perf_counter() - stage_start
            stage = "prediction"
            stage_start = perf_counter()
            predicted = np.asarray(final_model.predict(testing.copy()))
            probabilities = np.asarray(final_model.predict_proba(testing.copy()), dtype=float)
            if (probabilities.shape != (len(test), 2) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)
                    or not np.array_equal(final_model.classes_, labels)):
                raise ValueError("Final classifier violated the probability contract.")
            negative_label = labels[labels != positive_label][0]
            metrics = binary_metrics(y[test], predicted, negative_label=negative_label,
                                     positive_label=positive_label)
            record.update(status="success", metrics=metrics, predictions=predicted,
                          probabilities=probabilities, probability_labels=labels, true_labels=y[test])
            record["timings_seconds"][stage] = perf_counter() - stage_start
        except Exception as exc:
            record["status"] = "unavailable" if isinstance(exc, UnavailableClassifierError) else "failed"
            record["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            if sampler is not None:
                record["resampler_diagnostics"] = getattr(sampler, "diagnostics_", None)
        record["warnings"] = [{"category": item.category.__name__, "message": str(item.message)}
                              for item in caught]
    record["timings_seconds"]["total"] = perf_counter() - started
    return json_value(record)


def evaluate_ten_fold(X: Any, y: Any, *, dataset_id: str, positive_label: Any,
                      provenance: dict[str, Any], output_dir: str | Path,
                      config: EvaluationConfig | None = None, supplied_folds: Any = None,
                      retry_failures: bool = False,
                      on_record: Callable[[dict[str, Any], bool], None] | None = None) -> dict[str, Any]:
    """Execute fixed paired folds or resume them, without tuning on test results.

    Terminal failures are retained by default; retry_failures retries failed jobs,
    never successful or unavailable ones. A callback gets (record, was_cached).
    Exceptions before job execution (e.g. infeasible folds) fail explicitly.
    """
    config = config or EvaluationConfig()
    _validate_config(config)
    if (not isinstance(dataset_id, str) or not dataset_id.strip()
            or not isinstance(provenance, dict) or not provenance):
        raise ValueError("An explicit dataset identity and nonempty provenance are required.")
    if not isinstance(retry_failures, bool):
        raise ValueError("retry_failures must be boolean.")
    frame = feature_frame(X)
    labels_y, labels = binary_labels(y, len(frame))
    if positive_label not in labels:
        raise ValueError("positive_label must be an explicit source/domain class label.")
    if supplied_folds is not None and not provenance.get("fold_source"):
        raise ValueError("Supplied folds require fold_source provenance; do not assume they are official.")
    if supplied_folds is None and provenance.get("requires_group_or_time_split"):
        raise ValueError("Provider group/time constraints require supplied compliant folds.")
    folds, protocol = ten_fold_assignments(labels_y, seed=config.seed, supplied_folds=supplied_folds)
    manifest = {"schema_version": 1, "stage": "4R-5", "dataset_id": dataset_id,
                "input_sha256": _input_hash(frame, labels_y), "provenance": provenance,
                "rows": len(frame), "input_features": frame.columns.tolist(), "labels": labels,
                "positive_label": positive_label, "config": asdict(config), "folds": folds,
                "folds_sha256": content_hash(folds), "fold_protocol": protocol,
                "preprocessing_policy": PREPROCESSING_POLICY, "metric_policy": METRIC_POLICY,
                "classifier_specs": [asdict(get_classifier_spec(n)) for n in config.classifiers],
                "software": _software_identity(),
                "fidelity": "source-audited algorithms with declared ambiguities and classifier approximations"}
    store = ResultStore(output_dir, manifest)
    records, reused = [], 0
    for fold in folds:
        for classifier in config.classifiers:
            for method in config.methods:
                job = {"fold": fold["fold"], "classifier": classifier, "method": method}
                record = store.read(job)
                cached = record is not None and not (retry_failures and record["status"] == "failed")
                if cached:
                    reused += 1
                else:
                    record = store.write(_evaluate_job(frame, labels_y, labels, positive_label,
                                                        fold, job, config))
                records.append(record)
                if on_record:
                    on_record(record, cached)
    # Coverage is explicit; no silently dropping failures from paired comparisons.
    coverage = []
    for classifier in config.classifiers:
        baseline = {r["job"]["fold"] for r in records if r["job"]["classifier"] == classifier
                    and r["job"]["method"] == "nores" and r["status"] == "success"}
        for method in config.methods:
            subset = [r for r in records if r["job"]["classifier"] == classifier and r["job"]["method"] == method]
            successes = {r["job"]["fold"] for r in subset if r["status"] == "success"}
            coverage.append({"classifier": classifier, "method": method,
                             "success": len(successes), "failed": sum(r["status"] == "failed" for r in subset),
                             "unavailable": sum(r["status"] == "unavailable" for r in subset),
                             "paired_successful_folds_with_nores": sorted(successes & baseline)})
    return {"run_id": store.run_id, "output_dir": str(store.directory.resolve()),
            "records": records, "reused_jobs": reused, "coverage": coverage}
