"""Coverage-first paired summaries of saved rare-subgroup case records."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from src.rare_subgroups_protocol import METRICS, case_id, planned_jobs
from src.results import _atomic_json


def _at_most(value: float, bound: float) -> bool:
    # Inclusive protocol boundaries should not flip on floating subtraction.
    return bool(value <= bound or np.isclose(value, bound, rtol=0, atol=1e-12))


def _screen(cells: list[dict], config: dict, complete: bool) -> dict:
    rules, assessments = config["screening_rules"], []
    for cell in cells:
        indexed = cell["indexed"]
        seeds = config["seeds"]
        paired = all(indexed[seed, method] is not None and indexed[seed, method]["status"] == "success"
                     for seed in seeds for method in ("AW_REMOVE", "AW_SEED_FILTER_RETAIN"))
        assessment = {"rare_count": cell["rare_count"], "rare_sigma": cell["rare_sigma"],
                      "paired_complete": paired, "qualifies": None, "simple_controls": {}}
        if paired:
            remove = [indexed[seed, "AW_REMOVE"] for seed in seeds]
            retain = [indexed[seed, "AW_SEED_FILTER_RETAIN"] for seed in seeds]
            differences = np.array([b["metrics"]["rare_recall"] - a["metrics"]["rare_recall"]
                                    for a, b in zip(remove, retain)])
            fpr = np.mean([abs(b["metrics"]["test_fpr"] - a["metrics"]["test_fpr"])
                           for a, b in zip(remove, retain)])
            removed = {group: sum(row["group_retention"][group]["removed"] for row in remove)
                       for group in ("common", "rare")}
            qualifies = (sum(removed.values()) > 0
                         and _at_most(rules["minimum_mean_rare_recall_gain_retention_over_remove"], differences.mean())
                         and int(np.sum(differences > 0)) >= rules["minimum_positive_seed_differences_per_setting"]
                         and _at_most(fpr, rules["maximum_mean_absolute_paired_test_fpr_difference"]))
            assessment.update(qualifies=bool(qualifies), mean_rare_recall_difference=float(differences.mean()),
                              positive_seed_differences=int(np.sum(differences > 0)),
                              mean_absolute_test_fpr_difference=float(fpr), removed_originals=removed)
            if qualifies:
                for method in rules["simple_controls"]:
                    controls = [indexed[seed, method] for seed in seeds]
                    if any(row is None or row["status"] != "success" for row in controls):
                        assessment["simple_controls"][method] = None
                        continue
                    rare_gap = np.mean([r["metrics"]["rare_recall"] - c["metrics"]["rare_recall"]
                                        for r, c in zip(retain, controls)])
                    common_gap = np.mean([r["metrics"]["common_recall"] - c["metrics"]["common_recall"]
                                          for r, c in zip(retain, controls)])
                    control_fpr = np.mean([abs(r["metrics"]["test_fpr"] - c["metrics"]["test_fpr"])
                                           for r, c in zip(retain, controls)])
                    assessment["simple_controls"][method] = bool(
                        _at_most(rare_gap, rules["simple_control_rare_recall_tolerance"])
                        and _at_most(common_gap, rules["simple_control_common_recall_loss_tolerance"])
                        and _at_most(control_fpr, rules["maximum_mean_absolute_paired_test_fpr_difference"]))
        assessments.append(assessment)
    qualifying = [a for a in assessments if a["qualifies"]]
    if not complete:
        status = "incomplete_coverage"
    elif len(qualifying) < rules["minimum_qualifying_settings"]:
        status = "trigger_not_met"
    elif all(any(value is True for value in a["simple_controls"].values()) for a in qualifying):
        status = "simple_controls_suffice"
    else:
        status = "follow_up_required"
    return {"status": status, "qualifying_settings": len(qualifying), "settings": assessments,
            "interpretation": "Descriptive predeclared screening only; no significance, noise robustness, or novelty claim.",
            "group_loss_rule": "At least one minority group loses originals; common and rare losses shown separately.",
            "further_requirements": "Any residual loss requires prospective contamination and real subgroup studies."}


def summarize_cases(records: dict[str, dict], config: dict) -> dict:
    """Include every planned row; aggregate only shared successful seeds per cell."""
    jobs, methods = planned_jobs(config), config["methods"]
    if set(records) - {case_id(job) for job in jobs}:
        raise ValueError("Unexpected case records are not part of the design.")
    scores, lookup = [], {}
    for job in jobs:
        identifier = case_id(job)
        record = records.get(identifier)
        rows = record["rows"] if record is not None else []
        if record is not None and (record.get("job") != job or len(rows) != len(methods)
                                   or {r["method"] for r in rows} != set(methods)):
            raise ValueError("Saved case has mismatched jobs, methods, or duplicate rows.")
        by_method = {row["method"]: row for row in rows}
        for method in methods:
            row = by_method.get(method)
            status = row["status"] if row is not None else "missing"
            if status not in {"success", "unsupported", "failed", "missing"}:
                raise ValueError("Unknown method completion status.")
            metrics = row.get("metrics") if row is not None and status == "success" else None
            if status == "success" and (not isinstance(metrics, dict) or any(
                    metrics.get(metric) is None or not np.isfinite(metrics[metric]) for metric in METRICS)):
                raise ValueError("A successful row lacks finite evaluation metrics.")
            lookup[job["rare_count"], job["rare_sigma"], job["seed"], method] = row
            counts = (row or {}).get("output_counts", {})
            retention = (row or {}).get("group_retention") or {}
            error = (row or {}).get("error", {})
            scores.append({"case_id": identifier, "seed": job["seed"], "fit_rare_count": job["rare_count"],
                           "rare_sigma": job["rare_sigma"], "method": method, "status": status,
                           "error_stage": error.get("stage"), "reason": error.get("reason"),
                           "generated_count": (row or {}).get("generated_count"),
                           "output_majority": counts.get("majority"), "output_minority": counts.get("minority"),
                           "rare_retained_fraction": retention.get("rare", {}).get("fraction"),
                           "common_retained_fraction": retention.get("common", {}).get("fraction"),
                           "selected_parameters": json.dumps((row or {}).get("selected_parameters")),
                           **{key: (metrics or {}).get(key) for key in (*METRICS, "threshold", "threshold_is_infinite",
                               "calibration_fpr", "calibration_fp", "calibration_negatives", "test_fp", "test_negatives",
                               "test_tp", "test_positives", "rare_tp", "rare_count", "common_tp", "common_count")}})
    coverage, summary, cells = [], [], []
    d = config["distribution"]
    for count in d["fit_counts"]["rare_grid"]:
        for sigma in d["rare_minority"]["isotropic_sigma_grid"]:
            indexed = {(seed, method): lookup[count, sigma, seed, method]
                       for seed in config["seeds"] for method in methods}
            shared = [seed for seed in config["seeds"] if all(
                indexed[seed, method] is not None and indexed[seed, method]["status"] == "success"
                for method in methods)]
            cells.append({"rare_count": count, "rare_sigma": sigma, "indexed": indexed})
            for method in methods:
                statuses = [indexed[seed, method]["status"] if indexed[seed, method] is not None else "missing"
                            for seed in config["seeds"]]
                coverage.append({"fit_rare_count": count, "rare_sigma": sigma, "method": method,
                                 "planned": len(config["seeds"]), "shared_successes": len(shared),
                                 **{s: statuses.count(s) for s in ("success", "unsupported", "failed", "missing")}})
                for metric in METRICS:
                    values = [indexed[seed, method]["metrics"][metric] for seed in shared]
                    summary.append({"fit_rare_count": count, "rare_sigma": sigma, "method": method, "metric": metric,
                                    "n_shared_seeds": len(shared), "shared_seeds": json.dumps(shared),
                                    "mean": float(np.mean(values)) if values else None,
                                    "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    complete = all(row["status"] == "success" for row in scores)
    return {"status": "complete" if complete else "partial", "scores": scores, "coverage": coverage,
            "summary": summary, "screening": _screen(cells, config, complete)}


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_reports(directory: Path, records: dict[str, dict], config: dict) -> dict:
    report = summarize_cases(records, config)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("scores", "coverage", "summary"):
        _atomic_csv(directory / f"{name}.csv", report[name])
    _atomic_json(directory / "screening.json", report["screening"])
    _atomic_json(directory / "status.json", {
        "status": report["status"], "planned_cases": len(planned_jobs(config)), "saved_cases": len(records),
        "planned_rows": len(report["scores"]),
        "success_rows": sum(r["status"] == "success" for r in report["scores"]),
        "classifier_fit_attempts": sum(r.get("classifier_fit_attempts", 0) for r in records.values()),
        "summary_scope": "Within each setting, identical successful seeds for all seven methods; sample SD across seeds.",
        "screening_status": report["screening"]["status"], "not_paper_reproduction_or_novelty_evidence": True})
    return report
