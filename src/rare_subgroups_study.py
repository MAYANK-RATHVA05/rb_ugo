"""Train/select on X/y only, then evaluate the frozen diagnostic cases.

No research runs happen on import. The caller controls execution. The training
API intentionally cannot receive calibration, test labels, or subgroup IDs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.aw_smote import AWSMOTE, AWUnsupportedError
from src.rare_subgroups_protocol import (CaseData, array_hash, calibrate_threshold, case_id,
                                        evaluate_scores, positive_scores)


@dataclass
class Sampled:
    X: np.ndarray | None
    y: np.ndarray | None
    retained: np.ndarray | None
    metadata: dict
    trace: dict


@dataclass
class Fitted:
    model: Any
    sampled: Sampled
    record: dict


@dataclass
class Training:
    methods: dict[str, Fitted]
    candidates: list[dict]
    classifier_fit_attempts: int


def _error(exc: Exception, stage: str) -> dict:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc),
            "reason": exc.reason if isinstance(exc, AWUnsupportedError) else "exception"}


def _aw_trace(sampler: AWSMOTE) -> dict:
    names = ("cluster_labels_", "eligible_indices_", "excluded_indices_", "boundary_counts_",
             "boundary_neighbors_", "cluster_weights_", "cluster_quotas_", "sample_probabilities_",
             "generated_seed_indices_", "generated_partner_indices_", "generated_cluster_labels_",
             "interpolation_fractions_")
    return {name.removesuffix("_"): getattr(sampler, name).copy()
            for name in names if hasattr(sampler, name)}


def _sample(X: np.ndarray, y: np.ndarray, config: dict, seed: int,
            method: str, parameters: tuple[int, float] | None = None) -> Sampled:
    sampler = None
    started = perf_counter()
    try:
        if method in {"NORES", "WEIGHTED"}:
            out_X, out_y, retained = X.copy(), y.copy(), np.arange(len(X))
        elif method in {"ROS", "SMOTE"}:
            sampler = (RandomOverSampler(random_state=seed) if method == "ROS" else
                       SMOTE(k_neighbors=config["smote_k_neighbors"], random_state=seed))
            out_X, out_y = sampler.fit_resample(X, y)
            retained = np.arange(len(X))
        else:
            aw = config["aw_conventions"]
            sampler = AWSMOTE(min_samples=parameters[0], eps_percentile=parameters[1],
                              k_neighbors=aw["boundary_k_other_rows"], theta=aw["theta"], gamma=aw["gamma"],
                              n_samples=aw["generated_count"], random_state=seed)
            out_X, out_y = sampler.fit_resample(X, y)
            retained = sampler.retained_indices_.copy()
        generated = len(out_X) - len(retained)
        if method not in {"NORES", "WEIGHTED"} and generated != config["aw_conventions"]["generated_count"]:
            raise ValueError("A generating treatment did not meet the fixed original-count budget.")
        metadata = {"status": "success", "generated_count": generated,
                    "output_counts": {"majority": int(np.sum(out_y == 0)), "minority": int(np.sum(out_y == 1))},
                    "output_X_sha256": array_hash(out_X), "output_y_sha256": array_hash(out_y)}
        if isinstance(sampler, AWSMOTE):
            metadata["resampler_diagnostics"] = deepcopy(sampler.diagnostics_)
        trace = _aw_trace(sampler) if isinstance(sampler, AWSMOTE) else {}
        return Sampled(out_X, out_y, retained, {**metadata, "resample_seconds": perf_counter() - started}, trace)
    except Exception as exc:
        metadata = {"status": "unsupported" if isinstance(exc, AWUnsupportedError) else "failed",
                    "error": _error(exc, "resampling"), "resample_seconds": perf_counter() - started}
        if isinstance(sampler, AWSMOTE) and hasattr(sampler, "diagnostics_"):
            metadata["resampler_diagnostics"] = deepcopy(sampler.diagnostics_)
        return Sampled(None, None, None, metadata,
                       _aw_trace(sampler) if isinstance(sampler, AWSMOTE) else {})


def _retention_control(default: Sampled, X: np.ndarray, y: np.ndarray) -> Sampled:
    """Restore originals using the default AW's actual synthetic rows, not a redraw."""
    if default.X is None:
        return Sampled(None, None, None, deepcopy(default.metadata), default.trace)
    synthetic = default.X[len(default.retained):]
    out_X = np.vstack((X, synthetic))
    out_y = np.r_[y, np.ones(len(synthetic), dtype=y.dtype)]
    metadata = deepcopy(default.metadata)
    metadata.update(resample_seconds=0.0, resampling_reused=True,
                    output_counts={"majority": int(np.sum(out_y == 0)), "minority": int(np.sum(out_y == 1))},
                    output_X_sha256=array_hash(out_X), output_y_sha256=array_hash(out_y))
    metadata["resampler_diagnostics"].update(noise_policy="retain_originals", removed_original_count=0,
                                             output_counts=metadata["output_counts"])
    return Sampled(out_X, out_y, np.arange(len(X)), metadata, default.trace)


def _new_classifier(config: dict, seed: int, weighted: bool = False):
    params = {k: v for k, v in config["classifier"].items() if k not in {"class", "version_source"}}
    params["random_state"] = seed
    params["class_weight"] = config["weighted_control_class_weight"] if weighted else None
    return RandomForestClassifier(**params)


def _fit(sampled: Sampled, config: dict, seed: int, weighted: bool = False) -> Fitted:
    record = {**deepcopy(sampled.metadata), "classifier_fit_attempted": False, "fit_seconds": 0.0}
    if sampled.X is None:
        return Fitted(None, sampled, record)
    started = perf_counter()
    try:
        model = _new_classifier(config, seed, weighted)
        record["classifier_fit_attempted"] = True
        model.fit(sampled.X, sampled.y)
        record.update(classifier_parameters=model.get_params(deep=False), fit_seconds=perf_counter() - started)
        return Fitted(model, sampled, record)
    except Exception as exc:
        record.update(status="failed", error=_error(exc, "classifier_fit"), fit_seconds=perf_counter() - started)
        return Fitted(None, sampled, record)


def train_methods(X_fit: np.ndarray, y_fit: np.ndarray, X_selection: np.ndarray,
                  y_selection: np.ndarray, config: dict, seed: int) -> Training:
    """Fit fixed methods and choose tuned AW by aggregate validation AP only."""
    if (X_fit.ndim != 2 or X_selection.ndim != 2 or X_fit.shape[1] != 2
            or X_selection.shape[1] != 2 or len(X_fit) != len(y_fit)
            or len(X_selection) != len(y_selection)):
        raise ValueError("Training/selection must contain exactly two aligned feature columns.")
    gap = int(np.sum(y_fit == 0) - np.sum(y_fit == 1))
    if gap != config["aw_conventions"]["generated_count"]:
        raise ValueError("Original class-count difference does not match the declared generation budget.")
    methods = {}
    fitted_once = []
    for method in ("NORES", "WEIGHTED", "ROS", "SMOTE"):
        fitted = _fit(_sample(X_fit, y_fit, config, seed, method), config, seed, method == "WEIGHTED")
        methods[method] = fitted
        fitted_once.append(fitted)
    aw = config["aw_conventions"]
    default_parameters = (aw["dbscan_min_samples"], aw["epsilon_percentile"])
    default_sample = _sample(X_fit, y_fit, config, seed, "AW_REMOVE", default_parameters)
    methods["AW_REMOVE"] = _fit(default_sample, config, seed)
    methods["AW_SEED_FILTER_RETAIN"] = _fit(_retention_control(default_sample, X_fit, y_fit), config, seed)
    fitted_once.extend((methods["AW_REMOVE"], methods["AW_SEED_FILTER_RETAIN"]))
    candidates, eligible = [], []
    for min_samples in config["aw_tuning"]["min_samples_grid"]:
        for percentile in config["aw_tuning"]["epsilon_percentile_grid"]:
            parameters = (min_samples, percentile)
            reused = parameters == default_parameters
            fitted = (methods["AW_REMOVE"] if reused else
                      _fit(_sample(X_fit, y_fit, config, seed, "AW_REMOVE", parameters), config, seed))
            if not reused:
                fitted_once.append(fitted)
            candidate = {"min_samples": min_samples, "eps_percentile": percentile,
                         "reused_fixed_default": reused, "status": fitted.record["status"],
                         "validation_ap": None, "fit": deepcopy(fitted.record), "trace": fitted.sampled.trace}
            if fitted.model is not None:
                try:
                    ap = float(average_precision_score(y_selection, positive_scores(fitted.model, X_selection)))
                    if not np.isfinite(ap):
                        raise ValueError("Selection AP is nonfinite.")
                    candidate["validation_ap"] = ap
                    eligible.append((-ap, 0 if reused else 1, min_samples, percentile, fitted))
                except Exception as exc:
                    candidate.update(status="failed", error=_error(exc, "validation_selection"))
            candidates.append(candidate)
    if eligible:
        best = min(eligible, key=lambda item: item[:4])
        chosen = best[-1]
        record = deepcopy(chosen.record)
        record.update(fit_reused=True, selected_parameters={"min_samples": best[2], "eps_percentile": best[3]},
                      selected_validation_ap=-best[0])
        methods["AW_REMOVE_TUNED"] = Fitted(chosen.model, chosen.sampled, record)
    else:
        status = "unsupported" if all(c["status"] == "unsupported" for c in candidates) else "failed"
        empty = Sampled(None, None, None, {}, {})
        methods["AW_REMOVE_TUNED"] = Fitted(None, empty, {
            "status": status, "error": {"stage": "validation_selection", "reason": "no_eligible_candidate"},
            "classifier_fit_attempted": False})
    attempts = sum(int(f.record["classifier_fit_attempted"]) for f in fitted_once)
    return Training(methods, candidates, attempts)


def _retention(groups: np.ndarray, retained: np.ndarray | None) -> dict | None:
    if retained is None:
        return None
    return {name: {"original": int(np.sum(groups == group)),
                   "retained": int(np.sum(groups[retained] == group)),
                   "removed": int(np.sum(groups == group) - np.sum(groups[retained] == group)),
                   "fraction": float(np.sum(groups[retained] == group) / np.sum(groups == group))}
            for group, name in ((1, "common"), (2, "rare"))}


def evaluate_case(data: CaseData, config: dict, job: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Select everything before access to calibration/test predictions or hidden groups."""
    started = perf_counter()
    if not np.all(data.calibration.y == 0):
        raise ValueError("Threshold calibration must contain known negatives only.")
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(data.fitting.X)
    X_selection = scaler.transform(data.selection.X)
    training = train_methods(X_fit, data.fitting.y, X_selection, data.selection.y, config, job["seed"])
    # Tuning is now locked; calibration/test features never enter the training API.
    X_calibration, X_test = scaler.transform(data.calibration.X), scaler.transform(data.test.X)
    rows, predictions, prediction_cache = [], {"y": data.test.y, "groups": data.test.groups}, {}
    for method in config["methods"]:
        fitted = training.methods[method]
        row = {**deepcopy(fitted.record), "method": method, "metrics": None,
               "group_retention": _retention(data.fitting.groups, fitted.sampled.retained),
               "retained_indices": fitted.sampled.retained, "trace": fitted.sampled.trace}
        if fitted.model is not None:
            try:
                key = id(fitted.model)
                if key not in prediction_cache:
                    calibration = calibrate_threshold(positive_scores(fitted.model, X_calibration),
                                                      config["threshold"]["target_empirical_calibration_fpr"])
                    prediction_cache[key] = (calibration, positive_scores(fitted.model, X_test))
                calibration, scores = prediction_cache[key]
                row["metrics"] = evaluate_scores(data.test.y, data.test.groups, scores, calibration)
                row["test_scores_sha256"] = array_hash(scores)
                predictions[method] = scores
            except Exception as exc:
                row.update(status="failed", error=_error(exc, "calibration_or_test_evaluation"))
        rows.append(row)
    fingerprints = {name: {"X": array_hash(part.X), "y": array_hash(part.y), "groups": array_hash(part.groups)}
                    for name, part in vars(data).items()}
    return {"status": "success", "record_type": "rare_subgroup_case", "job": job, "case_id": case_id(job),
            "case_status": "complete" if all(r["status"] == "success" for r in rows) else "partial",
            "classifier_fit_attempts": training.classifier_fit_attempts,
            "elapsed_seconds": perf_counter() - started, "partitions_sha256": fingerprints,
            "scaled_fitting_X_sha256": array_hash(X_fit),
            "scaler": {"mean": scaler.mean_, "scale": scaler.scale_, "fitted_rows": int(scaler.n_samples_seen_)},
            "rows": rows, "selection_candidates": training.candidates}, predictions
