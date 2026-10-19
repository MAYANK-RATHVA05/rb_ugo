"""Frozen diagnostic design, independent partitions, and evaluation arithmetic.

No classifier or sampler is fitted in this module. Hidden component IDs are
evaluation metadata, never appended to the two numerical features.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from src._validation import finite_real
from src.results import canonical_json, content_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/rare_subgroups_pilot_v1.json"
DESIGN_KEYS = (
    "protocol_id", "seeds", "distribution", "preprocessing", "classifier", "methods",
    "weighted_control_class_weight", "smote_k_neighbors", "sampler_random_state",
    "aw_conventions", "aw_tuning", "threshold", "metrics", "budget", "screening_rules", "deferred",
)
FROZEN_DESIGN_SHA256 = "3855c803f1226f9ceefa26a2b26473a63d33c47f8c6afd89ab27c5fe4d3b9bb5"
METRICS = ("rare_recall", "common_recall", "positive_recall", "test_fpr", "f1", "ap", "worst_group_recall")


def design_hash(config: dict) -> str:
    return content_hash({key: config[key] for key in DESIGN_KEYS})


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    """Accept only the reviewed v1 numerical design; metadata may progress.

    A changed seed/grid/metric/convention must get a reviewed protocol version,
    rather than be silently ignored by this purpose-built runner.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or any(key not in config for key in DESIGN_KEYS):
        raise ValueError("Protocol must be an object containing all frozen design fields.")
    if design_hash(config) != FROZEN_DESIGN_SHA256:
        raise ValueError("Configuration differs from the frozen v1 design; use a reviewed protocol version.")
    if not isinstance(config.get("execution_enabled"), bool):
        raise ValueError("execution_enabled must be a boolean.")
    return config


def planned_jobs(config: dict) -> list[dict]:
    distribution = config["distribution"]
    return [{"seed": seed, "rare_count": count, "rare_sigma": sigma}
            for count in distribution["fit_counts"]["rare_grid"]
            for sigma in distribution["rare_minority"]["isotropic_sigma_grid"]
            for seed in config["seeds"]]


def case_id(job: dict) -> str:
    sigma = format(job["rare_sigma"], ".8g").replace(".", "p")
    return f"r{job['rare_count']}-s{sigma}-seed{job['seed']}"


def array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(canonical_json({"dtype": values.dtype.str, "shape": values.shape}).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Partition:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray  # 0=majority, 1=common minority, 2=rare minority; evaluator only


@dataclass(frozen=True)
class CaseData:
    fitting: Partition
    selection: Partition
    calibration: Partition
    test: Partition


def generate_case(config: dict, job: dict) -> CaseData:
    """Generate protocol partitions from independent keyed component streams.

    Call only for an actual run, or with reduced artificial software fixtures.
    The check-only command never calls this function.
    """
    d = config["distribution"]
    n_rare, sigma, seed = job["rare_count"], job["rare_sigma"], job["seed"]
    if (seed not in config["seeds"] or n_rare not in d["fit_counts"]["rare_grid"]
            or sigma not in d["rare_minority"]["isotropic_sigma_grid"]):
        raise ValueError("Case is not part of the configured design.")
    fit = d["fit_counts"]
    count_sets = {
        "fitting": {"majority": fit["majority"], "common": fit["total_minority"] - n_rare, "rare": n_rare},
        "selection": d["selection_counts"], "calibration": d["calibration_counts"], "test": d["test_counts"],
    }
    partitions = {}
    for name, counts in count_sets.items():
        partition_id = d["partition_ids"][name]
        features, labels, groups = [], [], []
        for group, component in enumerate(("majority", "common", "rare")):
            count = counts[component]
            pool_count = (d["fitting_pool_sizes"].get(component, count) if name == "fitting" else count)
            if count < 0 or count > pool_count:
                raise ValueError("Partition counts exceed the declared component pool.")
            rng = np.random.default_rng(np.random.SeedSequence([seed, partition_id, group]))
            draws = rng.standard_normal((pool_count, 2))[:count]
            params = d[("majority", "common_minority", "rare_minority")[group]]
            scale = sigma if group == 2 else params["isotropic_sigma"]
            features.append(np.asarray(params["mean"]) + scale * draws)
            labels.append(np.full(count, int(group > 0), dtype=np.int8))
            groups.append(np.full(count, group, dtype=np.int8))
        X, y, component_ids = np.vstack(features), np.concatenate(labels), np.concatenate(groups)
        order = np.random.default_rng(np.random.SeedSequence([seed, partition_id, 99])).permutation(len(y))
        partitions[name] = Partition(X[order], y[order], component_ids[order])
    return CaseData(**partitions)


def _probability_vector(values: Any) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Scores must be a nonempty finite probability vector in [0, 1].")
    return scores


def positive_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Find class 1 explicitly; never assume its predict_proba column order."""
    classes = np.asarray(model.classes_)
    probabilities = np.asarray(model.predict_proba(X))
    if (classes.shape != (2,) or set(classes.tolist()) != {0, 1}
            or probabilities.shape != (len(X), 2)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Classifier must return aligned binary class probabilities.")
    return _probability_vector(probabilities[:, int(np.flatnonzero(classes == 1)[0])])


def calibrate_threshold(negative_scores: np.ndarray, target_fpr: float) -> dict:
    """Lowest unique-score threshold meeting the negative-only empirical target.

    Infinity is represented by an explicit flag and null for strict JSON.
    """
    scores = _probability_vector(negative_scores)
    target = finite_real(target_fpr, "target_fpr")
    if not 0 <= target <= 1:
        raise ValueError("target_fpr must lie in [0, 1].")
    ordered = np.sort(scores)
    candidates, first = np.unique(ordered, return_index=True)
    counts = len(scores) - first
    eligible = np.flatnonzero(counts / len(scores) <= target)
    index = int(eligible[0]) if len(eligible) else None
    false_positives = int(counts[index]) if index is not None else 0
    return {"threshold": float(candidates[index]) if index is not None else None,
            "threshold_is_infinite": index is None, "calibration_fp": false_positives,
            "calibration_negatives": len(scores), "calibration_fpr": false_positives / len(scores)}


def evaluate_scores(y: np.ndarray, groups: np.ndarray, scores: np.ndarray, calibration: dict) -> dict:
    """Only the evaluator receives held-out true labels and component identities."""
    scores = _probability_vector(scores)
    y, groups = np.asarray(y), np.asarray(groups)
    if (y.shape != scores.shape or groups.shape != scores.shape
            or set(groups.tolist()) != {0, 1, 2}
            or not np.array_equal(y, (groups != 0).astype(int))):
        raise ValueError("Test labels and evaluator component IDs are inconsistent or incomplete.")
    threshold = np.inf if calibration["threshold_is_infinite"] else calibration["threshold"]
    predicted = scores >= threshold
    denominators = [int(np.sum(groups == group)) for group in range(3)]
    true_positives = [int(np.sum(predicted & (groups == group))) for group in range(3)]
    fp, tp = true_positives[0], sum(true_positives[1:])
    positives = sum(denominators[1:])
    common, rare = (true_positives[g] / denominators[g] for g in (1, 2))
    return {**calibration, "rare_recall": rare, "common_recall": common,
            "positive_recall": tp / positives, "test_fpr": fp / denominators[0],
            "f1": 2 * tp / (2 * tp + fp + positives - tp),
            "ap": float(average_precision_score(y, scores)), "worst_group_recall": min(common, rare),
            "test_fp": fp, "test_negatives": denominators[0], "test_tp": tp, "test_positives": positives,
            "rare_tp": true_positives[2], "rare_count": denominators[2],
            "common_tp": true_positives[1], "common_count": denominators[1]}
