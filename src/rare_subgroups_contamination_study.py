"""Contamination-study fitting/selection on observed X/y, then held-out evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src._validation import binary_labels, numeric_matrix
import src.rare_subgroups_study as baseline
from src.rare_subgroups_protocol import array_hash, positive_scores
from src.rare_subgroups_contamination import (
    ContaminationCase, calibrate_threshold, case_id, evaluate_scores, tie_uniforms,
)


class TracedSMOTE(SMOTE):
    """Observe the library's already-drawn interpolation inputs; consume no RNG.

    This instrumentation targets the pinned imbalanced-learn API. Equivalence to
    its reference SMOTE, including repeated calls, is checked in software tests.
    """

    def fit_resample(self, X, y):
        self._positive_ids = np.flatnonzero(np.asarray(y) == 1)
        self.generated_seed_indices_ = np.empty(0, dtype=int)
        self.generated_partner_indices_ = np.empty(0, dtype=int)
        self.interpolation_fractions_ = np.empty(0)
        return super().fit_resample(X, y)

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps, y_type=None, y=None):
        if y_type != 1 or y is not None:
            raise ValueError("Trace supports standard binary minority-label-1 SMOTE only.")
        self.generated_seed_indices_ = self._positive_ids[rows].copy()
        self.generated_partner_indices_ = self._positive_ids[nn_num[rows, cols]].copy()
        self.interpolation_fractions_ = steps[:, 0].copy()
        return super()._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)


def _sample(X, y, config, seed, method, parameters=None) -> baseline.Sampled:
    if method not in {"ROS", "SMOTE"}:
        return baseline._sample(X, y, config, seed, method, parameters)
    started = perf_counter()
    try:
        sampler = (RandomOverSampler(random_state=seed) if method == "ROS" else
                   TracedSMOTE(random_state=seed, k_neighbors=config["smote_k_neighbors"]))
        out_X, out_y = sampler.fit_resample(X, y)
        generated = len(out_X) - len(X)
        if generated != config["aw_conventions"]["generated_count"]:
            raise ValueError("Library sampler did not meet the observed original-count budget.")
        trace = ({"duplicated_indices": sampler.sample_indices_[len(X):].copy()} if method == "ROS" else
                 {key: getattr(sampler, key + "_").copy() for key in
                  ("generated_seed_indices", "generated_partner_indices", "interpolation_fractions")})
        return baseline.Sampled(out_X, out_y, np.arange(len(X)), {
            "status": "success", "generated_count": generated,
            "output_counts": {"majority": int(np.sum(out_y == 0)), "minority": int(np.sum(out_y == 1))},
            "output_X_sha256": array_hash(out_X), "output_y_sha256": array_hash(out_y),
            "resample_seconds": perf_counter() - started,
        }, trace)
    except Exception as exc:
        return baseline.Sampled(None, None, None, {
            "status": "failed", "error": baseline._error(exc, "resampling"),
            "resample_seconds": perf_counter() - started,
        }, {})


def candidate_id(min_samples: int, percentile: float) -> str:
    return f"aw-min{min_samples}-eps{format(percentile, '.8g')}"


@dataclass
class Training:
    methods: dict
    candidates: list[dict]
    selection_vectors: dict[str, np.ndarray]
    classifier_fit_attempts: int


def train_methods(X_fit, y_fit, X_selection, y_selection, design: dict, seed: int) -> Training:
    """Cannot accept hidden truth, calibration, test arrays or tie uniforms."""
    X_fit, X_selection = numeric_matrix(X_fit), numeric_matrix(X_selection)
    y_fit, labels = binary_labels(y_fit, len(X_fit))
    y_selection, selection_labels = binary_labels(y_selection, len(X_selection))
    if X_fit.shape[1] != 2 or X_selection.shape[1] != 2 or set(labels) != {0, 1} or set(selection_labels) != {0, 1}:
        raise ValueError("Training requires two feature columns and aligned binary labels 0/1.")
    gap = int(np.sum(y_fit == 0) - np.sum(y_fit == 1))
    if gap <= 0:
        raise ValueError("Observed label 1 must be strictly minority in fitting data.")
    config = deepcopy(design)
    config["aw_conventions"]["generated_count"] = gap
    methods, all_fits = {}, []
    for method in ("NORES", "WEIGHTED", "ROS", "SMOTE"):
        fitted = baseline._fit(_sample(X_fit, y_fit, config, seed, method), config, seed, method == "WEIGHTED")
        methods[method] = fitted
        all_fits.append(fitted)
    aw = config["aw_conventions"]
    default_parameters = (aw["dbscan_min_samples"], aw["epsilon_percentile"])
    default = _sample(X_fit, y_fit, config, seed, "AW_REMOVE", default_parameters)
    methods["AW_REMOVE"] = baseline._fit(default, config, seed)
    methods["AW_SEED_FILTER_RETAIN"] = baseline._fit(baseline._retention_control(default, X_fit, y_fit), config, seed)
    all_fits.extend((methods["AW_REMOVE"], methods["AW_SEED_FILTER_RETAIN"]))
    candidates, eligible, vectors = [], [], {"selection_y": np.asarray(y_selection).copy()}
    for minimum in config["aw_tuning"]["min_samples_grid"]:
        for percentile in config["aw_tuning"]["epsilon_percentile_grid"]:
            identifier = candidate_id(minimum, percentile)
            reused = (minimum, percentile) == default_parameters
            fitted = (methods["AW_REMOVE"] if reused else baseline._fit(
                _sample(X_fit, y_fit, config, seed, "AW_REMOVE", (minimum, percentile)), config, seed))
            if not reused:
                all_fits.append(fitted)
            candidate = {"candidate_id": identifier, "min_samples": minimum, "eps_percentile": percentile,
                         "reused_fixed_default": reused, "status": fitted.record["status"],
                         "validation_ap": None, "fit": deepcopy(fitted.record), "trace": fitted.sampled.trace,
                         "selection_vector": None}
            if fitted.model is not None:
                try:
                    probabilities = positive_scores(fitted.model, X_selection)
                    ap = float(average_precision_score(y_selection, probabilities))
                    if not np.isfinite(ap):
                        raise ValueError("Selection AP is nonfinite.")
                    key = "selection__" + identifier
                    vectors[key] = probabilities
                    candidate.update(validation_ap=ap, selection_vector=key)
                    eligible.append((-ap, 0 if reused else 1, minimum, percentile, fitted, identifier))
                except Exception as exc:
                    candidate.update(status="failed", error=baseline._error(exc, "validation_selection"))
            candidates.append(candidate)
    if eligible:
        best = min(eligible, key=lambda item: item[:4])
        chosen = best[4]
        record = deepcopy(chosen.record)
        record.update(fit_reused=True, selected_parameters={"min_samples": best[2], "eps_percentile": best[3]},
                      selected_validation_ap=-best[0], selected_candidate_id=best[5])
        methods["AW_REMOVE_TUNED"] = baseline.Fitted(chosen.model, chosen.sampled, record)
    else:
        status = "unsupported" if all(c["status"] == "unsupported" for c in candidates) else "failed"
        methods["AW_REMOVE_TUNED"] = baseline.Fitted(None, baseline.Sampled(None, None, None, {}, {}), {
            "status": status, "error": {"stage": "validation_selection", "reason": "no_eligible_candidate"},
            "classifier_fit_attempted": False,
        })
    return Training(methods, candidates, vectors, sum(int(f.record["classifier_fit_attempted"]) for f in all_fits))


def original_retention(fitting, retained) -> dict | None:
    if retained is None:
        return None
    masks = {"common": (fitting.groups == 1) & ~fitting.corrupted,
             "rare": (fitting.groups == 2) & ~fitting.corrupted, "noise": fitting.corrupted}
    result = {}
    for name, mask in masks.items():
        original, kept = int(mask.sum()), int(mask[retained].sum())
        result[name] = {"original": original, "retained": kept, "removed": original - kept,
                        "fraction": kept / original if original else None}
    return result


def parent_diagnostics(corrupted: np.ndarray, trace: dict) -> dict:
    """Called only after selection; parent labels do not certify synthetic truth."""
    if "duplicated_indices" in trace:
        flags = corrupted[trace["duplicated_indices"]]
        return {"kind": "duplication", "generated": len(flags), "corrupted_parents": int(flags.sum()),
                "corrupted_parent_fraction": float(flags.mean()) if len(flags) else None}
    if "generated_seed_indices" in trace:
        seeds = corrupted[trace["generated_seed_indices"]]
        partners = corrupted[trace["generated_partner_indices"]]
        return {"kind": "interpolation", "generated": len(seeds),
                "clean_clean": int(np.sum(~seeds & ~partners)),
                "clean_corrupt": int(np.sum(seeds != partners)), "corrupt_corrupt": int(np.sum(seeds & partners))}
    return {"kind": "none"}


def evaluate_case(data: ContaminationCase, design: dict, job: dict) -> tuple[dict, dict]:
    started = perf_counter()
    if not np.all(data.calibration.y == 0):
        raise ValueError("Threshold calibration must contain known negatives only.")
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(data.fitting.X)
    training = train_methods(X_fit, data.fitting.y, scaler.transform(data.selection.X), data.selection.y, design, job["seed"])
    # The training API has returned; model selection is locked before held-out
    # prediction, tie generation or any hidden fitting metadata is read.
    X_cal, X_test = scaler.transform(data.calibration.X), scaler.transform(data.test.X)
    partition_ids = design["distribution"]["partition_ids"]
    cal_ties = tie_uniforms(job["seed"], partition_ids["calibration"], len(X_cal))
    test_ties = tie_uniforms(job["seed"], partition_ids["test"], len(X_test))
    vectors = {**training.selection_vectors, "calibration_y": data.calibration.y,
               "test_y": data.test.y, "test_groups": data.test.groups,
               "calibration_ties": cal_ties, "test_ties": test_ties,
               **{"fitting_" + name: getattr(data.fitting, name) for name in
                  ("y", "true_y", "groups", "corrupted", "stable_ids")}}
    rows, cache = [], {}
    for method in design["methods"]:
        fitted = training.methods[method]
        row = {**deepcopy(fitted.record), "method": method, "metrics": None, "vectors": {},
               "group_retention": original_retention(data.fitting, fitted.sampled.retained),
               "parent_diagnostics": parent_diagnostics(data.fitting.corrupted, fitted.sampled.trace),
               "retained_indices": fitted.sampled.retained, "trace": fitted.sampled.trace}
        if fitted.model is not None:
            try:
                identity = id(fitted.model)
                if identity not in cache:
                    cal_scores = positive_scores(fitted.model, X_cal)
                    calibration = calibrate_threshold(cal_scores, cal_ties, design["threshold"]["target_false_positive_count"])
                    cache[identity] = (cal_scores, positive_scores(fitted.model, X_test), calibration)
                cal_scores, test_scores, calibration = cache[identity]
                row["metrics"] = evaluate_scores(data.test.y, data.test.groups, test_scores, test_ties, calibration)
                row["vectors"] = {"calibration": "calibration__" + method, "test": "test__" + method}
                vectors[row["vectors"]["calibration"]] = cal_scores
                vectors[row["vectors"]["test"]] = test_scores
            except Exception as exc:
                row.update(status="failed", error=baseline._error(exc, "calibration_or_test_evaluation"))
        rows.append(row)
    fingerprints = {name: {key: array_hash(value) for key, value in vars(partition).items()}
                    for name, partition in vars(data).items()}
    return {"status": "success", "record_type": "rare_subgroups_contamination_case", "job": job,
            "case_id": case_id(job), "case_status": "complete" if all(r["status"] == "success" for r in rows) else "partial",
            "classifier_fit_attempts": training.classifier_fit_attempts, "elapsed_seconds": perf_counter() - started,
            "partitions_sha256": fingerprints, "scaled_fitting_X_sha256": array_hash(X_fit),
            "scaler": {"mean": scaler.mean_, "scale": scaler.scale_, "fitted_rows": int(scaler.n_samples_seen_)},
            "input_observed_counts": {"majority": int(np.sum(data.fitting.y == 0)), "minority": int(np.sum(data.fitting.y == 1))},
            "rows": rows, "selection_candidates": training.candidates,
            "vector_sha256": {name: array_hash(value) for name, value in vectors.items()}}, vectors
