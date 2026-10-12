"""Descriptive, same-fold comparison of saved binary evaluation jobs."""
from __future__ import annotations

from collections import Counter
import csv
import json
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from src.results import ResultStore, _atomic_json


METRICS = ("iam", "mcc", "f1", "gmean", "auc")


def _score(record: dict[str, Any], metric: str) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or metric not in metrics or metric not in metrics.get("undefined", {}):
        raise ValueError(f"Saved success record lacks {metric}; run evaluation again in a new directory.")
    value = metrics[metric]
    if value is None:
        if not metrics["undefined"][metric]:
            raise ValueError(f"Null {metric} score is not marked undefined.")
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"Saved {metric} score is not finite and numerical.")
    return float(value)


def summarize_saved_run(directory: str | Path) -> dict[str, Any]:
    """Read validated job files; keep failures and missing folds visible."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Run directory has no manifest.json.")
    envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = envelope.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest is invalid.")
    config = manifest.get("config", {})
    classifiers, methods = config.get("classifiers", []), config.get("methods", [])
    folds = [entry["fold"] for entry in manifest.get("folds", [])]
    if (not classifiers or not methods or "nores" not in methods
            or len(set(classifiers)) != len(classifiers) or len(set(methods)) != len(methods)
            or folds != list(range(10))):
        raise ValueError("Run manifest must have unique methods/classifiers and ten numbered folds with nores.")
    store = ResultStore(root, manifest)
    expected_jobs = [{"fold": fold, "classifier": classifier, "method": method}
                     for fold in folds for classifier in classifiers for method in methods]
    expected_files = {store._path(job).name for job in expected_jobs}
    extra_files = {path.name for path in root.glob("job-*.json")} - expected_files
    if extra_files:
        raise ValueError("Run directory contains unexpected job files.")
    indexed = {(job["classifier"], job["method"], job["fold"]): store.read(job)
               for job in expected_jobs}
    for record in indexed.values():
        if record is not None and record["status"] == "success":
            for metric in METRICS:
                _score(record, metric)

    coverage, rows = [], []
    for classifier in classifiers:
        method_coverage = {}
        for method in methods:
            subset = [indexed[classifier, method, fold] for fold in folds]
            counts = Counter("missing" if record is None else record["status"] for record in subset)
            successes = [record for record in subset if record is not None and record["status"] == "success"]
            additions = [record["sample_counts"]["added_rows"] for record in successes]
            sampler_statuses = Counter(
                record["resampler_diagnostics"]["status"] for record in successes
                if isinstance(record.get("resampler_diagnostics"), dict)
                and "status" in record["resampler_diagnostics"])
            condition = {
                "classifier": classifier, "method": method,
                **{status: counts[status] for status in ("success", "failed", "unavailable", "missing")},
                "zero_generation_successes": sum(value == 0 for value in additions),
                "mean_added_rows": float(np.mean(additions)) if additions else None,
                "sampler_status_counts": dict(sorted(sampler_statuses.items())),
            }
            method_coverage[method] = condition
            coverage.append(condition)
        for metric in METRICS:
            shared = [fold for fold in folds if all(
                indexed[classifier, method, fold] is not None
                and indexed[classifier, method, fold]["status"] == "success"
                and _score(indexed[classifier, method, fold], metric) is not None
                for method in methods)]
            baseline = np.array([_score(indexed[classifier, "nores", fold], metric)
                                 for fold in shared], dtype=float)
            for method in methods:
                values = np.array([_score(indexed[classifier, method, fold], metric)
                                   for fold in shared], dtype=float)
                difference = values - baseline
                rows.append({
                    "dataset_id": manifest["dataset_id"], "classifier": classifier,
                    "method": method, "metric": metric, "shared_fold_ids": shared,
                    "shared_fold_count": len(shared),
                    "mean": float(values.mean()) if len(shared) else None,
                    "sample_sd": float(values.std(ddof=1)) if len(shared) > 1 else None,
                    "nores_mean": float(baseline.mean()) if len(shared) else None,
                    "mean_difference_from_nores": float(difference.mean()) if len(shared) else None,
                    "difference_sample_sd": float(difference.std(ddof=1)) if len(shared) > 1 else None,
                    "undefined_zeros_included": sum(
                        bool(indexed[classifier, method, fold]["metrics"]["undefined"][metric])
                        for fold in shared),
                    **{key: method_coverage[method][key] for key in
                       ("success", "failed", "unavailable", "missing",
                        "zero_generation_successes", "mean_added_rows")},
                })
    complete = (all(item["success"] == 10 for item in coverage)
                and all(row["shared_fold_count"] == 10 for row in rows))
    return {"run_id": store.run_id, "dataset_id": manifest["dataset_id"],
            "positive_label": manifest["positive_label"], "fold_protocol": manifest["fold_protocol"],
            "methods": methods, "classifiers": classifiers,
            "metric_policy": manifest.get("metric_policy"),
            "pairing_policy": "Each metric uses only folds with a finite score for every selected method; "
                              "declared MCC/F1 zero-denominator zeros remain included.",
            "status": "complete" if complete else "partial", "coverage": coverage, "rows": rows}


def export_comparison_report(report: dict[str, Any], output_csv: str | Path) -> tuple[Path, Path]:
    """Write readable CSV plus complete JSON metadata; outputs are regenerable."""
    csv_path = Path(output_csv)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Report output must be a .csv path.")
    rows = report["rows"]
    if not rows:
        raise ValueError("Report has no comparison rows.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "shared_fold_ids": json.dumps(row["shared_fold_ids"])})
    json_path = csv_path.with_suffix(".json")
    _atomic_json(json_path, report)
    return csv_path, json_path
