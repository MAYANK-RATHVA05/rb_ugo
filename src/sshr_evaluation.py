"""Opt-in controlled SSHR evaluation with training-only hidden-label pools.

This path deliberately leaves the existing evaluation registry and defaults
unchanged. Stratified label hiding is an experimental-design choice, not a
reconstruction of the SSHR authors' undisclosed unlabelled-pool protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import hashlib
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import json
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src._validation import binary_labels, numeric_matrix
from src.classifier_registry import get_classifier_spec, make_classifier, UnavailableClassifierError
from src.evaluation import _counts, _input_hash, _software_identity, make_evaluation_resampler
from src.metrics import binary_metrics, METRIC_POLICY
from src.preprocessing import feature_frame, FoldPreprocessor, PREPROCESSING_POLICY
from src.results import content_hash, json_value, ResultStore
from src.sshr import SSHR
from src.sshr_resampling import _check_label_scalars


CONTROLLED_METHODS = ("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr")
MASK_POLICY = "stratified training-only label hiding; not a verified author protocol"
FOLD_PROTOCOL = "declared repeated stratified outer CV; exact author folds unavailable"


@dataclass(frozen=True)
class ControlledEvaluationConfig:
    classifiers: tuple[str, ...] = ("LogReg",)
    methods: tuple[str, ...] = ("nores", "sshr")
    seed: int = 42
    scale: bool = True
    n_splits: int = 5
    n_repeats: int = 5
    labeled_fraction: float = 0.5
    min_labeled_per_class: int = 2
    use_unlabeled: bool = True
    method_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _seed(value: Any) -> int:
    result = _integer(value, "seed", minimum=0)
    if result >= 2**32:
        raise ValueError("seed must be in [0, 2**32).")
    return result


def _fraction(value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 < value <= 1):
        raise ValueError("labeled_fraction must be finite and in (0, 1].")
    return float(value)


def _derived_seed(seed: int, repeat: int, fold: int, purpose: int) -> int:
    return int(np.random.SeedSequence([seed, repeat, fold, purpose]).generate_state(1)[0])


def _validate_config(config: ControlledEvaluationConfig) -> None:
    if not isinstance(config, ControlledEvaluationConfig):
        raise ValueError("config must be a ControlledEvaluationConfig.")
    _seed(config.seed)
    _integer(config.n_splits, "n_splits", 2)
    _integer(config.n_repeats, "n_repeats")
    _integer(config.min_labeled_per_class, "min_labeled_per_class", 2)
    _fraction(config.labeled_fraction)
    if not isinstance(config.scale, bool) or not isinstance(config.use_unlabeled, bool):
        raise ValueError("scale and use_unlabeled must be boolean.")
    if (not isinstance(config.methods, (tuple, list)) or not config.methods
            or any(not isinstance(name, str) for name in config.methods)
            or len(set(config.methods)) != len(config.methods)
            or "nores" not in config.methods
            or any(name not in CONTROLLED_METHODS for name in config.methods)):
        raise ValueError("Methods must be unique supported names and include nores.")
    if (not isinstance(config.classifiers, (tuple, list)) or not config.classifiers
            or any(not isinstance(name, str) for name in config.classifiers)
            or len(set(config.classifiers)) != len(config.classifiers)):
        raise ValueError("At least one unique classifier is required.")
    if "ec_ebo" in config.methods and not config.scale:
        raise ValueError("ec_ebo requires fold-local scaling.")
    for name in config.classifiers:
        get_classifier_spec(name)
    if not isinstance(config.method_parameters, dict):
        raise ValueError("method_parameters must be a dictionary.")
    reserved = {"estimator", "variant", "random_state", "allow_provisional", "minority_label"}
    for name, parameters in config.method_parameters.items():
        if name not in config.methods or not isinstance(parameters, dict):
            raise ValueError("Method parameters must map selected methods to dictionaries.")
        if set(parameters) & reserved:
            raise ValueError("Estimator, variant, random_state, provisional status and minority_label are runner controlled.")
        if name == "nores" and parameters:
            raise ValueError("nores takes no method parameters.")
    try:
        json.dumps(asdict(config), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Configuration must contain finite JSON-compatible values.") from exc


def _indices(value: Any, *, name: str, rows: int, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value)
    # An explicit empty Python list has float dtype; normalize only that case.
    if array.ndim == 1 and not len(array) and allow_empty:
        return np.empty(0, dtype=np.int64)
    if (array.ndim != 1 or array.dtype.kind not in "iu" or not len(array)
            or (array < 0).any() or (array >= rows).any()
            or len(np.unique(array)) != len(array)):
        raise ValueError(f"{name} must contain unique, in-range integer row indices.")
    return np.array(array, dtype=np.int64, copy=True)


def _key(record: dict[str, Any], n_repeats: int, n_splits: int) -> tuple[int, int]:
    if not isinstance(record, dict):
        raise ValueError("Fold and mask records must be dictionaries.")
    repeat = _integer(record.get("repeat"), "repeat", 0)
    fold = _integer(record.get("fold"), "fold", 0)
    if repeat >= n_repeats or fold >= n_splits:
        raise ValueError("Repeat/fold identifiers are outside the configured range.")
    return repeat, fold


def controlled_fold_assignments(y: Any, *, seed: int = 42, n_splits: int = 5,
                                n_repeats: int = 5, supplied_folds: Any = None
                                ) -> tuple[list[dict[str, Any]], str]:
    """Freeze exact global row indices; every row is tested once per repeat."""
    seed = _seed(seed)
    n_splits = _integer(n_splits, "n_splits", 2)
    n_repeats = _integer(n_repeats, "n_repeats")
    labels, classes = binary_labels(y, len(y))
    _check_label_scalars(classes)
    if min(int(np.sum(labels == label)) for label in classes) < n_splits:
        raise ValueError("Each class needs at least n_splits rows; folds are not silently reduced.")
    if supplied_folds is None:
        pairs = []
        for repeat in range(n_repeats):
            repeat_seed = _derived_seed(seed, repeat, 0, 1)
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            pairs.extend({"repeat": repeat, "fold": fold, "train_indices": train,
                          "test_indices": test, "split_seed": repeat_seed}
                         for fold, (train, test) in enumerate(splitter.split(np.zeros((len(labels), 1)), labels)))
        protocol = FOLD_PROTOCOL
    else:
        pairs = list(supplied_folds)
        protocol = "caller-supplied repeated outer folds; authority recorded in provenance"
    if len(pairs) != n_splits * n_repeats:
        raise ValueError("Exactly n_splits * n_repeats fold records are required.")
    records, keys = [], set()
    seen = np.zeros((n_repeats, len(labels)), dtype=int)
    universe = np.arange(len(labels))
    for record in pairs:
        repeat, fold = _key(record, n_repeats, n_splits)
        if (repeat, fold) in keys:
            raise ValueError("Duplicate repeat/fold identifier.")
        keys.add((repeat, fold))
        train = _indices(record.get("train_indices"), name="train_indices", rows=len(labels))
        test = _indices(record.get("test_indices"), name="test_indices", rows=len(labels))
        if np.intersect1d(train, test).size or not np.array_equal(np.sort(np.r_[train, test]), universe):
            raise ValueError("Outer train/test indices must be disjoint exact complements.")
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
            raise ValueError("Every outer training and test partition must contain both fixed classes.")
        seen[repeat, test] += 1
        item = {"repeat": repeat, "fold": fold,
                "train_indices": train.tolist(), "test_indices": test.tolist()}
        if supplied_folds is None:
            item["split_seed"] = record["split_seed"]
        records.append(item)
    if not np.all(seen == 1):
        raise ValueError("Every input row must be tested exactly once per repeat.")
    return sorted(records, key=lambda item: (item["repeat"], item["fold"])), protocol


def controlled_label_masks(y: Any, folds: list[dict[str, Any]], *, seed: int = 42,
                           labeled_fraction: float = 0.5, min_labeled_per_class: int = 2,
                           supplied_masks: Any = None) -> tuple[list[dict[str, Any]], str]:
    """Partition each outer training fold into L and U before learned operations.

    True labels are used exclusively to create the stratified design here.
    They are never an unlabelled-pool argument to a learning method.
    """
    seed, fraction = _seed(seed), _fraction(labeled_fraction)
    minimum = _integer(min_labeled_per_class, "min_labeled_per_class", 2)
    labels, classes = binary_labels(y, len(y))
    _check_label_scalars(classes)
    if not isinstance(folds, list) or not folds:
        raise ValueError("Nonempty validated fold records are required.")
    keys = {(record["repeat"], record["fold"]) for record in folds}
    if len(keys) != len(folds):
        raise ValueError("Fold identifiers must be unique.")
    if supplied_masks is not None:
        supplied = list(supplied_masks)
        if len(supplied) != len(folds) or any(not isinstance(item, dict) for item in supplied):
            raise ValueError("One supplied mask record is required for each outer fold.")
        mapped = {}
        for item in supplied:
            key = (_integer(item.get("repeat"), "repeat", 0), _integer(item.get("fold"), "fold", 0))
            if key not in keys or key in mapped:
                raise ValueError("Supplied mask identifiers must match the outer folds exactly.")
            mapped[key] = item
        protocol = "caller-supplied training-only label masks; authority recorded in provenance"
    else:
        mapped, protocol = {}, MASK_POLICY
    records = []
    for fold in folds:
        repeat, number = fold["repeat"], fold["fold"]
        train = _indices(fold["train_indices"], name="train_indices", rows=len(labels))
        mask_seed = _derived_seed(seed, repeat, number, 2)
        clamps = []
        if supplied_masks is None:
            rng = np.random.default_rng(mask_seed)
            selected = []
            for label in classes:
                candidates = train[labels[train] == label]
                if len(candidates) < minimum:
                    raise ValueError("Outer training class cannot meet min_labeled_per_class.")
                requested = int(np.floor(fraction * len(candidates)))
                count = min(len(candidates), max(minimum, requested))
                selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
                clamps.append({"label": label, "training_count": len(candidates),
                               "fraction_requested_count": requested, "labeled_count": count,
                               "minimum_clamp_applied": count != requested})
            chosen = set(selected)
            labeled = np.asarray([index for index in train if int(index) in chosen], dtype=np.int64)
            unlabeled = np.asarray([index for index in train if int(index) not in chosen], dtype=np.int64)
        else:
            source = mapped[(repeat, number)]
            labeled = _indices(source.get("labeled_indices"), name="labeled_indices", rows=len(labels))
            unlabeled = _indices(source.get("unlabeled_indices"), name="unlabeled_indices",
                                 rows=len(labels), allow_empty=True)
        if (np.intersect1d(labeled, unlabeled).size
                or not np.array_equal(np.sort(np.r_[labeled, unlabeled]), np.sort(train))):
            raise ValueError("Labeled and unlabelled indices must partition only the outer training rows.")
        if any(int(np.sum(labels[labeled] == label)) < minimum for label in classes):
            raise ValueError("Every observed labeled class must meet min_labeled_per_class.")
        records.append({"repeat": repeat, "fold": number,
                        "mask_seed": mask_seed if supplied_masks is None else None,
                        "labeled_indices": labeled.tolist(), "unlabeled_indices": unlabeled.tolist(),
                        "requested_labeled_fraction": fraction,
                        "actual_labeled_fraction": len(labeled) / len(train),
                        "per_class_design": json_value(clamps) if supplied_masks is None else None})
    return records, protocol


def make_controlled_resampler(method: str, estimator: Any, *, seed: int,
                              positive_label: Any, parameters: dict[str, Any]) -> Any:
    """Separate opt-in factory; existing registries and default methods are unchanged."""
    if method not in CONTROLLED_METHODS:
        raise ValueError("Unknown controlled-study method.")
    if method == "sshr":
        return SSHR(minority_label=positive_label, random_state=seed, **parameters)
    if method == "ec_ebo":
        return make_evaluation_resampler(method, estimator, seed=seed,
                                        parameters={"minority_label": positive_label, **parameters})
    return make_evaluation_resampler(method, estimator, seed=seed, parameters=parameters)


def _validate_source(frame: pd.DataFrame, provenance: dict[str, Any]) -> np.ndarray | None:
    metadata = provenance.get("feature_types")
    if (not isinstance(metadata, dict) or set(metadata) != set(frame.columns)
            or any(not isinstance(value, str) or value not in {"numeric", "continuous"}
                   for value in metadata.values())):
        raise ValueError("feature_types must identify every original source column as numeric/continuous; categorical and boolean sources are unsupported.")
    invalid = [name for name in frame if not pd.api.types.is_numeric_dtype(frame[name])
               or pd.api.types.is_bool_dtype(frame[name]) or pd.api.types.is_complex_dtype(frame[name])
               or isinstance(frame[name].dtype, pd.CategoricalDtype)]
    if invalid:
        raise ValueError(f"Original numerical features are required; unsupported columns: {invalid}.")
    if np.isinf(frame.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Infinite numerical features are unsupported.")
    if "coordinate_groups" in provenance:
        supplied = provenance["coordinate_groups"]
        if (not isinstance(supplied, list) or len(supplied) != len(frame)
                or any(isinstance(value, bool) or not isinstance(value, Integral) for value in supplied)):
            raise ValueError("coordinate_groups must provide one integer group ID per original row.")
        groups = np.asarray(supplied)
        if (groups.ndim != 1 or groups.dtype.kind not in "iu"
                or np.any(groups > np.iinfo(np.int64).max)):
            raise ValueError("coordinate_groups must contain integer IDs representable as int64.")
        groups = np.array(groups, dtype=np.int64, copy=True)
        # Compare coordinate identities only. Hidden class labels play no role
        # in determining or validating these source row groups.
        coordinate_codes = frame.groupby(list(frame.columns), dropna=False, sort=False).ngroup().to_numpy()
        coordinate_memberships = pd.DataFrame({"coordinate": coordinate_codes, "declared_group": groups})
        if (coordinate_memberships.groupby("coordinate")["declared_group"].nunique() > 1).any():
            raise ValueError("Every identical feature-coordinate row must belong to the same declared coordinate group.")
        return groups
    if frame.duplicated().any():
        raise ValueError("Duplicate feature-coordinate rows are unsupported in this controlled path; resolve group-aware provenance before evaluation.")
    return None


def _validate_group_partitions(groups: np.ndarray, folds: list[dict[str, Any]],
                               masks: list[dict[str, Any]]) -> None:
    """Keep every declared group together across outer folds and L/U pools."""
    for fold in folds:
        training = groups[np.asarray(fold["train_indices"], dtype=np.int64)]
        testing = groups[np.asarray(fold["test_indices"], dtype=np.int64)]
        if np.intersect1d(training, testing).size:
            raise ValueError("Coordinate groups must remain disjoint across outer training and test partitions.")
    for mask in masks:
        labeled = groups[np.asarray(mask["labeled_indices"], dtype=np.int64)]
        unlabeled = groups[np.asarray(mask["unlabeled_indices"], dtype=np.int64)]
        if np.intersect1d(labeled, unlabeled).size:
            raise ValueError("Coordinate groups must remain together in either the labeled or unlabelled training pool.")


def _overlap_proxy(points: np.ndarray, fitting: np.ndarray, y_labeled: np.ndarray,
                   positive_label: Any) -> dict[str, Any]:
    """Training-only nearest-observed-class proxy; never a true overlap estimate."""
    if not len(points):
        return {"rows": 0, "nearest_majority_count": 0, "nearest_majority_rate": None, "distance_tie_count": 0}
    minority = fitting[y_labeled == positive_label]
    majority = fitting[y_labeled != positive_label]
    minority_distance = cdist(points, minority).min(axis=1)
    majority_distance = cdist(points, majority).min(axis=1)
    if not np.isfinite(np.r_[minority_distance, majority_distance]).all():
        raise ValueError("Overlap-proxy distance overflow; scale numerical inputs.")
    count = int(np.sum(majority_distance < minority_distance))
    return {"rows": len(points), "nearest_majority_count": count,
            "nearest_majority_rate": count / len(points),
            "distance_tie_count": int(np.sum(majority_distance == minority_distance))}


def _global_lineage(lineage: list[dict[str, Any]], labeled: np.ndarray,
                    unlabeled: np.ndarray) -> list[dict[str, Any]]:
    """Preserve algorithm-relative indices while adding evaluator row identity."""
    if not isinstance(lineage, list) or any(not isinstance(item, dict) for item in lineage):
        raise ValueError("SSHR row lineage must be a list of row records.")
    result = deepcopy(lineage)

    def global_index(value: Any, source: np.ndarray) -> int:
        index = _integer(value, "lineage source index", 0)
        if index >= len(source):
            raise ValueError("SSHR lineage source index is outside its fitting pool.")
        return int(source[index])

    for record in result:
        kind = record.get("kind")
        if not isinstance(kind, str) or kind not in {"labeled", "pseudo_labeled", "smote"}:
            raise ValueError("SSHR lineage kinds must be labeled, pseudo_labeled or smote.")
        if kind in {"labeled", "pseudo_labeled"}:
            source = labeled if kind == "labeled" else unlabeled
            record["global_index"] = global_index(record.get("original_index"), source)
        else:
            for name in ("anchor", "neighbor"):
                point = record.get(name)
                if (not isinstance(point, dict) or not isinstance(point.get("kind"), str)
                        or point["kind"] not in {"labeled", "unlabeled"}):
                    raise ValueError("SSHR SMOTE lineage endpoints must identify labeled/unlabeled fitting rows.")
                source = labeled if point["kind"] == "labeled" else unlabeled
                point["global_index"] = global_index(point.get("index"), source)
    return result


def _evaluate_job(X: pd.DataFrame, y: np.ndarray, labels: np.ndarray, positive_label: Any,
                  fold: dict[str, Any], mask: dict[str, Any], job: dict[str, Any],
                  config: ControlledEvaluationConfig) -> dict[str, Any]:
    started = perf_counter()
    seed = _derived_seed(config.seed, fold["repeat"], fold["fold"], 3)
    train, test = np.asarray(fold["train_indices"]), np.asarray(fold["test_indices"])
    labeled = np.asarray(mask["labeled_indices"], dtype=np.int64)
    hidden = np.asarray(mask["unlabeled_indices"], dtype=np.int64)
    actual_u = hidden if config.use_unlabeled else np.empty(0, dtype=np.int64)
    record = {"job": job, "seed": seed, "train_indices": train, "test_indices": test,
              "labeled_indices": labeled, "unlabeled_indices": hidden,
              "algorithm_unlabeled_indices": actual_u if job["method"] == "sshr" else [],
              "positive_label": positive_label, "labeled_counts": _counts(y[labeled], labels),
              "test_counts": _counts(y[test], labels), "metrics": None, "predictions": None,
              "probabilities": None, "error": None, "timings_seconds": {},
              "preprocessing": None, "resampler_diagnostics": None,
              "pseudo_label_audit": None, "row_lineage": None, "training_overlap_proxy": None}
    stage, sampler = "classifier_constructor", None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            template = make_classifier(job["classifier"], seed=seed)
            record["classifier_parameters"] = template.get_params(deep=False)
            stage, stage_start = "preprocessing", perf_counter()
            preprocessing = FoldPreprocessor(scale=config.scale)
            fitting = preprocessing.fit_transform(X.iloc[labeled].copy(), y[labeled].copy())
            fitting = numeric_matrix(fitting)
            # Baselines receive only the same L; no method sees hidden labels.
            unlabelled_features = (preprocessing.transform(X.iloc[actual_u].copy())
                                  if job["method"] == "sshr" and len(actual_u)
                                  else np.empty((0, fitting.shape[1]), dtype=float))
            record["preprocessing"] = preprocessing.diagnostics_
            record["timings_seconds"][stage] = perf_counter() - stage_start
            stage, stage_start = "resampling", perf_counter()
            sampler = make_controlled_resampler(job["method"], template, seed=seed,
                                                positive_label=positive_label,
                                                parameters=config.method_parameters.get(job["method"], {}))
            record["resampler_parameters"] = {key: value for key, value in sampler.get_params(deep=False).items()
                                               if key != "estimator"}
            if job["method"] == "sshr":
                augmented, augmented_y = sampler.fit_resample(fitting.copy(), y[labeled].copy(),
                                                              unlabelled_features.copy())
            else:
                augmented, augmented_y = sampler.fit_resample(fitting.copy(), y[labeled].copy())
            augmented = numeric_matrix(augmented)
            augmented_y, augmented_classes = binary_labels(augmented_y, len(augmented))
            if augmented.shape[1] != fitting.shape[1] or not np.array_equal(labels, augmented_classes):
                raise ValueError("Resampler changed feature width or fixed label identities.")
            record["resampler_diagnostics"] = getattr(sampler, "diagnostics_", None)
            record["resampled_counts"] = _counts(augmented_y, labels)
            record["sample_counts"] = {"outer_training": len(train), "labeled_training": len(labeled),
                                       "hidden_training": len(hidden), "algorithm_unlabeled": len(unlabelled_features),
                                       "test": len(test), "resampled_training": len(augmented),
                                       "net_added_rows": max(0, len(augmented) - len(labeled)),
                                       "net_removed_rows": max(0, len(labeled) - len(augmented))}
            if job["method"] == "sshr":
                selected = _indices(sampler.selected_unlabeled_indices_, name="selected_unlabeled_indices_",
                                     rows=len(actual_u), allow_empty=True)
                source_rows = actual_u[selected]
                # Retrospective audit only AFTER the method has completed.
                correct = int(np.sum(y[source_rows] == positive_label))
                record["pseudo_label_audit"] = {"selected_count": len(selected), "correct_count": correct,
                                                "incorrect_count": len(selected) - correct,
                                                "accuracy": correct / len(selected) if len(selected) else None,
                                                "selected_global_indices": source_rows,
                                                "purpose": "retrospective only; hidden gold labels never supplied to the algorithm"}
                record["row_lineage"] = _global_lineage(sampler.row_lineage_, labeled, actual_u)
                lineage_counts = {kind: sum(item["kind"] == kind for item in record["row_lineage"])
                                  for kind in ("labeled", "pseudo_labeled", "smote")}
                retained = _indices(sampler.retained_labeled_indices_, name="retained_labeled_indices_",
                                    rows=len(labeled), allow_empty=True)
                if (len(record["row_lineage"]) != len(augmented)
                        or lineage_counts["labeled"] != len(retained)
                        or lineage_counts["pseudo_labeled"] != len(selected)
                        or lineage_counts["smote"] != len(sampler.synthetic_samples_)):
                    raise ValueError("SSHR row lineage does not conserve retained, pseudo-labelled and synthetic output counts.")
                record["sample_counts"].update({"selected_pseudo_labeled": len(selected),
                                                 "generated_smote": len(sampler.synthetic_samples_),
                                                 "generated_synthetic": len(sampler.synthetic_samples_),
                                                 "generated_minority": len(sampler.synthetic_samples_),
                                                 "generated_majority": 0,
                                                 "retained_labeled": len(retained),
                                                 "removed_majority": sampler.diagnostics_["counts"]["removed_majority"],
                                                 "removed_minority": sampler.diagnostics_["counts"]["removed_minority"]})
                record["training_overlap_proxy"] = {
                    "definition": "nearest original observed majority distance < nearest original observed minority distance; strict ties separate; not true class overlap",
                    "pseudo_labeled": _overlap_proxy(unlabelled_features[selected], fitting, y[labeled], positive_label),
                    "smote": _overlap_proxy(sampler.synthetic_samples_, fitting, y[labeled], positive_label)}
            else:
                # These named supervised comparators append generated rows.
                # Verify that contract rather than infer removals from net size.
                if (len(augmented) < len(fitting)
                        or not np.array_equal(augmented[:len(fitting)], fitting)
                        or not np.array_equal(augmented_y[:len(fitting)], y[labeled])):
                    raise ValueError("Controlled supervised baseline violated its append-only original labeled-row prefix contract.")
                generated = augmented[len(fitting):]
                generated_y = augmented_y[len(fitting):]
                minority_generated = generated_y == positive_label
                record["sample_counts"].update({"retained_labeled": len(labeled),
                                                 "selected_pseudo_labeled": 0,
                                                 "generated_synthetic": len(generated),
                                                 "generated_minority": int(np.sum(minority_generated)),
                                                 "generated_majority": int(np.sum(~minority_generated)),
                                                 "removed_majority": 0, "removed_minority": 0})
                record["training_overlap_proxy"] = {
                    "definition": "nearest original observed majority distance < nearest original observed minority distance; strict ties separate; not true class overlap",
                    "synthetic_all": _overlap_proxy(generated, fitting, y[labeled], positive_label),
                    "synthetic_minority": _overlap_proxy(generated[minority_generated], fitting, y[labeled], positive_label),
                    "synthetic_majority": _overlap_proxy(generated[~minority_generated], fitting, y[labeled], positive_label)}
            record["timings_seconds"][stage] = perf_counter() - stage_start
            stage, stage_start = "final_fit", perf_counter()
            final_model = clone(template)
            final_model.fit(augmented.copy(), augmented_y.copy())
            record["timings_seconds"][stage] = perf_counter() - stage_start
            stage, stage_start = "prediction", perf_counter()
            # Held-out features are transformed only after resampling and fitting.
            testing = preprocessing.transform(X.iloc[test].copy())
            predicted = np.asarray(final_model.predict(testing.copy()))
            probabilities = np.asarray(final_model.predict_proba(testing.copy()), dtype=float)
            if (probabilities.shape != (len(test), 2) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)
                    or not np.array_equal(final_model.classes_, labels)):
                raise ValueError("Final classifier violated the probability contract.")
            negative_label = labels[labels != positive_label][0]
            positive_column = int(np.flatnonzero(labels == positive_label)[0])
            metrics = binary_metrics(y[test], predicted, negative_label=negative_label,
                                     positive_label=positive_label,
                                     positive_probabilities=probabilities[:, positive_column])
            counts = metrics["counts"]
            metrics["probability_auc"] = metrics["auc"]
            metrics["fpr"] = counts["fp"] / (counts["tn"] + counts["fp"])
            record.update(status="success", metrics=metrics, predictions=predicted,
                          probabilities=probabilities, probability_labels=labels, true_labels=y[test])
            record["timings_seconds"][stage] = perf_counter() - stage_start
        except Exception as exc:
            record["status"] = "unavailable" if isinstance(exc, UnavailableClassifierError) else "failed"
            record["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            if sampler is not None:
                record["resampler_diagnostics"] = getattr(sampler, "diagnostics_", None)
        record["warnings"] = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
    record["timings_seconds"]["total"] = perf_counter() - started
    return json_value(record)


def evaluate_controlled(X: Any, y: Any, *, dataset_id: str, positive_label: Any,
                        provenance: dict[str, Any], output_dir: str | Path,
                        config: ControlledEvaluationConfig | None = None,
                        supplied_folds: Any = None, supplied_masks: Any = None,
                        retry_failures: bool = False,
                        on_record: Callable[[dict[str, Any], bool], None] | None = None) -> dict[str, Any]:
    """Execute or strictly resume a fixed, paired reduced-label comparison.

    Successful and unavailable jobs are never refitted. Failed jobs remain in
    coverage and are retried only when explicitly requested. No test score is
    used for parameter selection, pseudo-label selection, or label-mask design.
    """
    config = ControlledEvaluationConfig() if config is None else config
    _validate_config(config)
    # Freeze mutable parameter dictionaries/lists before manifest creation or
    # user callbacks, so every job obeys the saved run configuration.
    config = deepcopy(config)
    if (not isinstance(dataset_id, str) or not dataset_id.strip()
            or not isinstance(provenance, dict) or not provenance):
        raise ValueError("Explicit dataset_id and nonempty provenance are required.")
    if not isinstance(retry_failures, bool):
        raise ValueError("retry_failures must be boolean.")
    provenance = deepcopy(provenance)
    # Provenance is part of exact run identity. Diagnostic normalization must
    # not collapse an invalid NaN/infinite source claim to a legitimate null.
    try:
        json.dumps(provenance, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Provenance must contain finite JSON-compatible values.") from exc
    frame = feature_frame(X)
    coordinate_groups = _validate_source(frame, provenance)
    labels_y, labels = binary_labels(y, len(frame))
    _check_label_scalars(labels)
    candidate = np.asarray(positive_label)
    if candidate.ndim != 0 or not np.isin(candidate, labels):
        raise ValueError("positive_label must be an explicit source/domain class scalar.")
    positive_label = labels[int(np.flatnonzero(labels == candidate.item())[0])]
    if supplied_folds is not None and not provenance.get("fold_source"):
        raise ValueError("Supplied folds require fold_source provenance.")
    if supplied_masks is not None and not provenance.get("mask_source"):
        raise ValueError("Supplied masks require mask_source provenance.")
    if coordinate_groups is not None and (supplied_folds is None or supplied_masks is None):
        raise ValueError("Coordinate-group inputs require supplied outer folds and supplied label masks with recorded provenance.")
    if supplied_folds is None and provenance.get("requires_group_or_time_split"):
        raise ValueError("Group/time constraints require caller-supplied compliant outer folds.")
    folds, fold_protocol = controlled_fold_assignments(labels_y, seed=config.seed,
                                                       n_splits=config.n_splits, n_repeats=config.n_repeats,
                                                       supplied_folds=supplied_folds)
    masks, mask_protocol = controlled_label_masks(labels_y, folds, seed=config.seed,
                                                  labeled_fraction=config.labeled_fraction,
                                                  min_labeled_per_class=config.min_labeled_per_class,
                                                  supplied_masks=supplied_masks)
    if coordinate_groups is not None:
        _validate_group_partitions(coordinate_groups, folds, masks)
    software = _software_identity()
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sshr_study.py"
    software["sshr_runner_script_sha256"] = hashlib.sha256(script_path.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "stage": "SSHR-3", "dataset_id": dataset_id,
                "input_sha256": _input_hash(frame, labels_y), "provenance": provenance,
                "rows": len(frame), "input_features": frame.columns.tolist(), "labels": labels,
                "positive_label": positive_label, "config": asdict(config),
                "folds": folds, "masks": masks, "folds_sha256": content_hash(folds),
                "masks_sha256": content_hash(masks), "fold_protocol": fold_protocol,
                "mask_protocol": mask_protocol,
                "coordinate_group_policy": ("preserve all original rows; every declared group remains together across outer train/test and labeled/unlabelled training pools"
                                            if coordinate_groups is not None else "duplicate feature-coordinate rows rejected"),
                "coordinate_groups_sha256": content_hash(coordinate_groups) if coordinate_groups is not None else None,
                "preprocessing_policy": {**PREPROCESSING_POLICY, "fit_pool": "observed L only; original numeric features only"},
                "metric_policy": {**METRIC_POLICY, "probability_auc": "explicit alias of probability ROC AUC; not SSHR printed AUC",
                                  "fpr": "held-out FP/(TN+FP)"},
                "seed_policy": "NumPy SeedSequence([root seed, repeat, fold, purpose]); purpose 1 folds, 2 mask, 3 method/classifier",
                "classifier_specs": [asdict(get_classifier_spec(name)) for name in config.classifiers],
                "software": software,
                "fidelity": "source-audited algorithms with declared interpretations; author SSHR code unverified"}
    store = ResultStore(output_dir, manifest)
    records, reused = [], 0
    mask_by_key = {(mask["repeat"], mask["fold"]): mask for mask in masks}
    for fold in folds:
        mask = mask_by_key[(fold["repeat"], fold["fold"])]
        for classifier in config.classifiers:
            for method in config.methods:
                job = {"repeat": fold["repeat"], "fold": fold["fold"], "classifier": classifier, "method": method}
                record = store.read(job)
                cached = record is not None and not (retry_failures and record["status"] == "failed")
                if cached:
                    reused += 1
                else:
                    record = store.write(_evaluate_job(frame, labels_y, labels, positive_label, fold, mask, job, config))
                records.append(record)
                if on_record:
                    on_record(record, cached)
    coverage = []
    for classifier in config.classifiers:
        baseline = {(record["job"]["repeat"], record["job"]["fold"]) for record in records
                    if record["job"]["classifier"] == classifier and record["job"]["method"] == "nores"
                    and record["status"] == "success"}
        for method in config.methods:
            subset = [record for record in records if record["job"]["classifier"] == classifier
                      and record["job"]["method"] == method]
            successes = {(record["job"]["repeat"], record["job"]["fold"]) for record in subset
                         if record["status"] == "success"}
            coverage.append({"classifier": classifier, "method": method, "success": len(successes),
                             "failed": sum(record["status"] == "failed" for record in subset),
                             "unavailable": sum(record["status"] == "unavailable" for record in subset),
                             "paired_successful_folds_with_nores": [list(key) for key in sorted(successes & baseline)]})
    return {"run_id": store.run_id, "output_dir": str(store.directory.resolve()),
            "records": records, "reused_jobs": reused, "coverage": coverage}


__all__ = ["ControlledEvaluationConfig", "CONTROLLED_METHODS", "controlled_fold_assignments",
           "controlled_label_masks", "make_controlled_resampler", "evaluate_controlled"]
