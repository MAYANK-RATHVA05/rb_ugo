"""Frozen NADOS-only diagnostic layered on the completed contamination cases.

The sampler and classifier receive only scaled fitting features and observed
labels.  True labels, latent groups, corruption flags and stable row IDs are
read only after training has returned and are used solely for diagnostics.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.nados import NADOS, NADOSUnsupportedError
from src.rare_subgroups_contamination import (
    ContaminationCase,
    ROOT,
    calibrate_threshold,
    case_id,
    evaluate_scores,
    generate_case as generate_contamination_case,
    load_protocol as load_contamination_protocol,
    planned_jobs as contamination_jobs,
    tie_uniforms,
)
from src.rare_subgroups_protocol import array_hash, positive_scores
import src.rare_subgroups_study as baseline


DEFAULT_PROTOCOL = ROOT / "configs/rare_subgroups_nados_v1.json"
FROZEN_DESIGN_SHA256 = "c86301b697708983c3884256a0e055bde57f60ee3f7d8f5df5e42b6da9693b9f"
PARENT_DESIGN_SHA256 = "adfdb2961963ff83b9552e84b06963689f938d853859d9bafc224106a08849db"
PARENT_RUN_COMMIT = "975020ab3cc3ba5e15cfd29499c7ee5b776a0116"
SHARED_SECTIONS = (
    "seeds", "distribution", "preprocessing", "classifier", "label_corruption", "threshold",
)
TRACE_ATTRIBUTES = (
    "minority_indices_", "neighbor_indices_", "neighbor_distances_",
    "minority_neighbor_counts_", "minority_support_", "mean_minority_distance_",
    "mean_majority_distance_", "relative_distance_evidence_", "reliability_scores_",
    "boundary_scores_", "difficulty_scores_", "eligible_mask_", "eligible_indices_",
    "excluded_indices_", "synthesis_priorities_", "seed_probabilities_",
    "generated_seed_indices_", "generated_partner_indices_", "interpolation_fractions_",
    "retained_indices_",
)


def design_hash(design: dict) -> str:
    serialized = json.dumps(design, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_protocol(config: dict) -> dict:
    if not isinstance(config, dict) or not isinstance(config.get("scientific_design"), dict):
        raise ValueError("A NADOS diagnostic protocol must contain scientific_design.")
    design = config["scientific_design"]
    digest = design_hash(design)
    if digest != FROZEN_DESIGN_SHA256 or config.get("design_sha256") != digest:
        raise ValueError("Configuration differs from the frozen NADOS diagnostic design.")
    if not isinstance(config.get("execution_enabled"), bool):
        raise ValueError("execution_enabled must be a boolean.")
    source = design.get("case_source", {})
    if (source.get("config") != "configs/rare_subgroups_contamination_v1.json"
            or source.get("scientific_design_sha256") != PARENT_DESIGN_SHA256
            or source.get("run_commit") != PARENT_RUN_COMMIT
            or tuple(source.get("shared_sections", ())) != SHARED_SECTIONS):
        raise ValueError("NADOS case-source identity or shared-section contract changed.")
    parent = load_contamination_protocol(ROOT / source["config"])
    if parent["design_sha256"] != PARENT_DESIGN_SHA256:
        raise ValueError("Parent contamination design identity changed.")
    if parent.get("run_commit") != PARENT_RUN_COMMIT:
        raise ValueError("Parent contamination run identity changed.")
    if design.get("methods") != ["NADOS"]:
        raise ValueError("The diagnostic must contain only the fixed NADOS method.")
    return config


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def parent_design(config: dict) -> dict:
    validate_protocol(config)
    return load_contamination_protocol(ROOT / config["scientific_design"]["case_source"]["config"])[
        "scientific_design"
    ]


def runtime_design(config: dict) -> dict:
    """Materialize the referenced frozen sections for execution and fixtures."""
    design = deepcopy(config["scientific_design"])
    parent = parent_design(config)
    for name in SHARED_SECTIONS:
        design[name] = deepcopy(parent[name])
    return design


def planned_jobs(config: dict) -> list[dict]:
    return contamination_jobs(parent_design(config))


def generate_case(config: dict, job: dict) -> ContaminationCase:
    return generate_contamination_case(parent_design(config), job)


def _nados_trace(sampler: NADOS) -> dict[str, np.ndarray]:
    return {name.removesuffix("_"): np.asarray(getattr(sampler, name)).copy()
            for name in TRACE_ATTRIBUTES if hasattr(sampler, name)}


def sample_nados(X: np.ndarray, y: np.ndarray, design: dict, seed: int) -> baseline.Sampled:
    """Apply only fixed NADOS; unsupported geometry remains unsupported."""
    convention = design["nados_conventions"]
    sampler = NADOS(
        k_neighbors=convention["k_neighbors"],
        reliability_weight=convention["reliability_weight"],
        difficulty_weight=convention["difficulty_weight"],
        reliability_threshold=convention["reliability_threshold"],
        target_ratio=convention["target_ratio"],
        minority_label=convention["minority_label"],
        random_state=seed,
    )
    started = perf_counter()
    try:
        out_X, out_y = sampler.fit_resample(X, y)
        generated = len(out_X) - len(X)
        expected = int(np.sum(y == 0) - np.sum(y == 1))
        if generated != expected:
            raise ValueError("NADOS did not meet the fixed observed class-gap generation budget.")
        if not np.array_equal(out_X[:len(X)], X) or not np.array_equal(out_y[:len(y)], y):
            raise ValueError("NADOS did not retain every original fitting row in order.")
        metadata = {
            "status": "success",
            "generated_count": generated,
            "output_counts": {"majority": int(np.sum(out_y == 0)),
                              "minority": int(np.sum(out_y == 1))},
            "output_X_sha256": array_hash(out_X),
            "output_y_sha256": array_hash(out_y),
            "resampler_diagnostics": deepcopy(sampler.diagnostics_),
            "resample_seconds": perf_counter() - started,
        }
        return baseline.Sampled(out_X, out_y, sampler.retained_indices_.copy(), metadata,
                                _nados_trace(sampler))
    except Exception as exc:
        status = "unsupported" if isinstance(exc, NADOSUnsupportedError) else "failed"
        return baseline.Sampled(None, None, None, {
            "status": status,
            "error": {"stage": "resampling", "type": type(exc).__name__, "message": str(exc),
                      "reason": exc.reason if isinstance(exc, NADOSUnsupportedError) else "exception"},
            "resample_seconds": perf_counter() - started,
        }, _nados_trace(sampler))


def train_nados(X_fit: np.ndarray, y_fit: np.ndarray, design: dict, seed: int) -> baseline.Fitted:
    """Training boundary: hidden truth and held-out partitions cannot be passed."""
    sampled = sample_nados(np.asarray(X_fit), np.asarray(y_fit), design, seed)
    classifier_config = {"classifier": deepcopy(design["classifier"]),
                         "weighted_control_class_weight": "balanced"}
    return baseline._fit(sampled, classifier_config, seed)


def eligibility_diagnostics(fitting, trace: dict) -> dict | None:
    if "minority_indices" not in trace:
        return None
    minority = trace["minority_indices"].astype(int)
    reliability = trace["reliability_scores"]
    difficulty = trace["difficulty_scores"]
    eligible = trace["eligible_mask"].astype(bool)
    masks = {
        "common": (fitting.groups[minority] == 1) & ~fitting.corrupted[minority],
        "rare": (fitting.groups[minority] == 2) & ~fitting.corrupted[minority],
        "noise": fitting.corrupted[minority],
    }
    result = {}
    for name, mask in masks.items():
        original = int(mask.sum())
        accepted = int(np.sum(mask & eligible))
        result[name] = {
            "original": original,
            "eligible": accepted,
            "excluded": original - accepted,
            "fraction": accepted / original if original else None,
            "mean_reliability": float(np.mean(reliability[mask])) if original else None,
            "mean_difficulty": float(np.mean(difficulty[mask])) if original else None,
        }
    return result


def parent_diagnostics(corrupted: np.ndarray, trace: dict) -> dict | None:
    if "generated_seed_indices" not in trace:
        return None
    seeds = corrupted[trace["generated_seed_indices"].astype(int)]
    partners = corrupted[trace["generated_partner_indices"].astype(int)]
    generated = len(seeds)
    any_corrupted = int(np.sum(seeds | partners))
    return {
        "kind": "interpolation",
        "generated": generated,
        "clean_clean": int(np.sum(~seeds & ~partners)),
        "clean_corrupt": int(np.sum(seeds != partners)),
        "corrupt_corrupt": int(np.sum(seeds & partners)),
        "any_corrupted_parent": any_corrupted,
        "corrupted_parent_fraction": any_corrupted / generated if generated else None,
    }


def evaluate_case(data: ContaminationCase, design: dict, job: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit NADOS before reading any evaluator-only metadata or held-out features."""
    started = perf_counter()
    if not np.all(data.calibration.y == 0):
        raise ValueError("Threshold calibration must contain known negatives only.")
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(data.fitting.X)
    fitted = train_nados(X_fit, data.fitting.y, design, job["seed"])

    # Training is locked. Hidden fitting metadata and held-out partitions are
    # accessed only below this point for evaluation and audit diagnostics.
    X_cal = scaler.transform(data.calibration.X)
    X_test = scaler.transform(data.test.X)
    partition_ids = design["distribution"]["partition_ids"]
    cal_ties = tie_uniforms(job["seed"], partition_ids["calibration"], len(X_cal))
    test_ties = tie_uniforms(job["seed"], partition_ids["test"], len(X_test))
    vectors: dict[str, np.ndarray] = {
        "calibration_y": data.calibration.y,
        "test_y": data.test.y,
        "test_groups": data.test.groups,
        "calibration_ties": cal_ties,
        "test_ties": test_ties,
        **{"fitting_" + name: getattr(data.fitting, name)
           for name in ("y", "true_y", "groups", "corrupted", "stable_ids")},
    }
    trace_vector_names = {}
    for name, value in fitted.sampled.trace.items():
        key = "trace__" + name
        vectors[key] = np.asarray(value)
        trace_vector_names[name] = key

    row = {
        **deepcopy(fitted.record),
        "method": "NADOS",
        "metrics": None,
        "vectors": {},
        "trace_vectors": trace_vector_names,
        "eligibility": eligibility_diagnostics(data.fitting, fitted.sampled.trace),
        "parent_diagnostics": parent_diagnostics(data.fitting.corrupted, fitted.sampled.trace),
    }
    if fitted.model is not None:
        try:
            cal_scores = positive_scores(fitted.model, X_cal)
            calibration = calibrate_threshold(
                cal_scores, cal_ties, design["threshold"]["target_false_positive_count"]
            )
            test_scores = positive_scores(fitted.model, X_test)
            row["metrics"] = evaluate_scores(
                data.test.y, data.test.groups, test_scores, test_ties, calibration
            )
            row["vectors"] = {"calibration": "calibration__NADOS", "test": "test__NADOS"}
            vectors["calibration__NADOS"] = cal_scores
            vectors["test__NADOS"] = test_scores
        except Exception as exc:
            row.update(status="failed", error=baseline._error(exc, "calibration_or_test_evaluation"))

    fingerprints = {name: {key: array_hash(value) for key, value in vars(partition).items()}
                    for name, partition in vars(data).items()}
    record = {
        "status": "success",
        "record_type": "rare_subgroups_nados_case",
        "job": job,
        "case_id": case_id(job),
        "case_status": "complete" if row["status"] == "success" else "partial",
        "classifier_fit_attempts": int(fitted.record.get("classifier_fit_attempted", False)),
        "elapsed_seconds": perf_counter() - started,
        "partitions_sha256": fingerprints,
        "scaled_fitting_X_sha256": array_hash(X_fit),
        "scaler": {"mean": scaler.mean_, "scale": scaler.scale_,
                   "fitted_rows": int(scaler.n_samples_seen_)},
        "input_observed_counts": {"majority": int(np.sum(data.fitting.y == 0)),
                                  "minority": int(np.sum(data.fitting.y == 1))},
        "rows": [row],
        "vector_sha256": {name: array_hash(value) for name, value in vectors.items()},
    }
    return record, vectors
