"""Frozen contamination design, label-only pairing and negative-only tie calibration.

No research computation happens on import. Hidden fitting truth is evaluation
metadata; fitting and selection APIs receive only features and observed labels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.rare_subgroups_protocol import (
    ROOT, Partition, _probability_vector, generate_case as generate_clean_case,
)


DEFAULT_PROTOCOL = ROOT / "configs/rare_subgroups_contamination_v1.json"
FROZEN_DESIGN_SHA256 = "adfdb2961963ff83b9552e84b06963689f938d853859d9bafc224106a08849db"


def design_hash(design: dict) -> str:
    serialized = json.dumps(design, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_protocol(config: dict) -> dict:
    if not isinstance(config, dict) or not isinstance(config.get("scientific_design"), dict):
        raise ValueError("A contamination protocol must contain scientific_design.")
    digest = design_hash(config["scientific_design"])
    if digest != FROZEN_DESIGN_SHA256 or config.get("design_sha256") != digest:
        raise ValueError("Configuration differs from the frozen contamination design.")
    if not isinstance(config.get("execution_enabled"), bool):
        raise ValueError("execution_enabled must be a boolean.")
    return config


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def planned_jobs(design: dict) -> list[dict]:
    d = design["distribution"]
    return [{"rare_count": count, "rare_sigma": sigma, "flipped_count": noise, "seed": seed}
            for count in d["fit_counts"]["rare_grid"]
            for sigma in d["rare_minority"]["isotropic_sigma_grid"]
            for noise in design["label_corruption"]["flipped_majority_count_grid"]
            for seed in design["seeds"]]


def case_id(job: dict) -> str:
    sigma = format(job["rare_sigma"], ".8g").replace(".", "p")
    return f"r{job['rare_count']}-s{sigma}-m{job['flipped_count']}-seed{job['seed']}"


@dataclass(frozen=True)
class FittingPartition:
    X: np.ndarray
    y: np.ndarray  # observed labels; the ONLY fitting labels exposed to training
    true_y: np.ndarray
    groups: np.ndarray
    corrupted: np.ndarray
    stable_ids: np.ndarray  # positions before the original partition shuffle


@dataclass(frozen=True)
class ContaminationCase:
    fitting: FittingPartition
    selection: Partition
    calibration: Partition
    test: Partition


def generate_case(design: dict, job: dict) -> ContaminationCase:
    """Generate only on explicit execution or reduced artificial test fixtures."""
    if job not in planned_jobs(design):
        raise ValueError("Case is not part of the contamination design.")
    clean_config = deepcopy(design)
    counts = clean_config["distribution"]["fit_counts"]
    majority, minority = counts["true_majority"], counts["true_total_minority"]
    counts.update(majority=majority, total_minority=minority)
    clean_job = {key: job[key] for key in ("seed", "rare_count", "rare_sigma")}
    data = generate_clean_case(clean_config, clean_job)
    seed, noise = job["seed"], job["flipped_count"]
    if not 0 <= noise <= majority:
        raise ValueError("Corruption count exceeds true-majority fitting rows.")
    fitting_id = design["distribution"]["partition_ids"]["fitting"]
    order = np.random.default_rng(np.random.SeedSequence([seed, fitting_id, 99])).permutation(len(data.fitting.y))
    chosen = np.random.default_rng(np.random.SeedSequence([seed, 5, 100])).permutation(majority)[:noise]
    corrupted = np.isin(order, chosen)
    observed = data.fitting.y.copy()
    observed[corrupted] = 1
    return ContaminationCase(
        FittingPartition(data.fitting.X, observed, data.fitting.y, data.fitting.groups, corrupted, order),
        data.selection, data.calibration, data.test,
    )


def tie_uniforms(seed: int, partition_id: int, rows: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, partition_id, 101])).random(rows)


def _aligned_uniforms(uniforms: np.ndarray, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(uniforms, dtype=float)
    if (values.shape != scores.shape or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values >= 1)):
        raise ValueError("Tie uniforms must align with scores and lie in [0, 1).")
    return values


def predict_with_ties(scores: np.ndarray, uniforms: np.ndarray, calibration: dict) -> np.ndarray:
    scores = _probability_vector(scores)
    uniforms = _aligned_uniforms(uniforms, scores)
    threshold, cutoff = calibration["threshold"], calibration["tie_cutoff"]
    if not (np.isfinite(threshold) and 0 <= threshold <= 1 and np.isfinite(cutoff) and 0 <= cutoff < 1):
        raise ValueError("Invalid calibrated score/tie cutoff.")
    return (scores > threshold) | ((scores == threshold) & (uniforms >= cutoff))


def calibrate_threshold(negative_scores: np.ndarray, uniforms: np.ndarray, target_fp: int) -> dict:
    scores = _probability_vector(negative_scores)
    uniforms = _aligned_uniforms(uniforms, scores)
    if isinstance(target_fp, bool) or not isinstance(target_fp, Integral) or not 1 <= target_fp <= len(scores):
        raise ValueError("target_fp must be an integer in [1, calibration rows].")
    if len(np.unique(np.column_stack((scores, uniforms)), axis=0)) != len(scores):
        raise ValueError("Duplicate calibration keys; no redraw is permitted.")
    index = np.lexsort((uniforms, scores))[-target_fp]
    result = {"threshold": float(scores[index]), "tie_cutoff": float(uniforms[index]),
              "calibration_fp": int(target_fp), "calibration_negatives": len(scores),
              "calibration_fpr": target_fp / len(scores)}
    if int(predict_with_ties(scores, uniforms, result).sum()) != target_fp:
        raise ValueError("Tie calibration did not meet the declared count.")
    return result


def evaluate_scores(y: np.ndarray, groups: np.ndarray, scores: np.ndarray,
                    uniforms: np.ndarray, calibration: dict) -> dict:
    scores = _probability_vector(scores)
    y, groups = np.asarray(y), np.asarray(groups)
    if (y.shape != scores.shape or groups.shape != scores.shape
            or set(groups.tolist()) != {0, 1, 2}
            or not np.array_equal(y, (groups != 0).astype(int))):
        raise ValueError("Test labels and evaluator groups are inconsistent or incomplete.")
    predicted = predict_with_ties(scores, uniforms, calibration)
    counts = [int(np.sum(groups == group)) for group in range(3)]
    hits = [int(np.sum(predicted & (groups == group))) for group in range(3)]
    fp, tp, positives = hits[0], sum(hits[1:]), sum(counts[1:])
    common, rare = hits[1] / counts[1], hits[2] / counts[2]
    return {**calibration, "rare_recall": rare, "common_recall": common,
            "positive_recall": tp / positives, "test_fpr": fp / counts[0],
            "f1": 2 * tp / (tp + fp + positives), "ap": float(average_precision_score(y, scores)),
            "worst_group_recall": min(common, rare), "test_fp": fp, "test_negatives": counts[0],
            "test_tp": tp, "test_positives": positives, "rare_tp": hits[2], "rare_count": counts[2],
            "common_tp": hits[1], "common_count": counts[1]}
