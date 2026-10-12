"""Fixed simulation pilot, descriptive paired comparisons and measured resources."""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Callable

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from src.classifier_registry import get_classifier_spec
from src.evaluation import EvaluationConfig, _validate_config, evaluate_ten_fold
from src.results import _atomic_json, content_hash, json_value
from src.simulations import make_linear_simulation, make_nonlinear_simulation


PAIRED_POLICY = (
    "For each dataset/classifier/method/metric use only common successful folds "
    "with finite method and NORES scores. Report both means, mean paired difference "
    "and sample SD of differences on that same subset. Declared zero-denominator "
    "MCC/F1 zeros are retained with undefined counts. No significance claims."
)


def pilot_config(spec: dict[str, Any]) -> EvaluationConfig:
    if (spec.get("schema_version") != 1 or spec.get("native_threads") != 1
            or set(spec.get("simulations", {})) != {"linear", "nonlinear"}):
        raise ValueError("Pilot requires schema 1, one native thread and both named simulations.")
    config = EvaluationConfig(classifiers=tuple(spec["classifiers"]), methods=tuple(spec["methods"]),
                              seed=spec["seed"], scale=spec["scale"],
                              method_parameters=spec["method_parameters"])
    _validate_config(config)
    return config


def process_peak_memory() -> dict[str, Any]:
    """Native + Python resident high-water mark, not a per-job allocation measure."""
    if platform.system() not in {"Linux", "Darwin"}:
        return {"bytes": None, "status": "unsupported platform", "scope": "process lifetime"}
    import resource
    native = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"bytes": int(native * (1024 if platform.system() == "Linux" else 1)),
            "status": "measured", "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "scope": "process lifetime including imports, inputs, native libraries and retained results"}


def summarize_records(records: list[dict[str, Any]], config: EvaluationConfig) -> list[dict[str, Any]]:
    """Descriptive fold comparisons; never treat folds as independent datasets."""
    indexed = {(r["job"]["classifier"], r["job"]["method"], r["job"]["fold"]): r for r in records}
    expected = {(c, m, f) for c in config.classifiers for m in config.methods for f in range(10)}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("Summary requires exactly one terminal record for every configured fold job.")
    summaries = []
    for classifier in config.classifiers:
        for method in config.methods:
            subset = [indexed[(classifier, method, f)] for f in range(10)]
            statuses = Counter(r["status"] for r in subset)
            if set(statuses) - {"success", "failed", "unavailable"}:
                raise ValueError("Nonterminal or unknown result status.")
            metrics = {}
            for metric in ("iam", "mcc", "f1"):
                paired, baseline_values, method_values = [], [], []
                undefined_method = undefined_baseline = 0
                for fold, record in enumerate(subset):
                    baseline = indexed[(classifier, "nores", fold)]
                    if record["status"] != "success" or baseline["status"] != "success":
                        continue
                    value, reference = record["metrics"][metric], baseline["metrics"][metric]
                    if value is None or reference is None or not np.isfinite([value, reference]).all():
                        continue
                    paired.append(fold)
                    method_values.append(value)
                    baseline_values.append(reference)
                    undefined_method += bool(record["metrics"]["undefined"][metric])
                    undefined_baseline += bool(baseline["metrics"]["undefined"][metric])
                differences = np.asarray(method_values) - np.asarray(baseline_values)
                metrics[metric] = {
                    "paired_fold_ids": paired, "paired_count": len(paired),
                    "method_mean": float(np.mean(method_values)) if paired else None,
                    "nores_mean": float(np.mean(baseline_values)) if paired else None,
                    "mean_difference": float(differences.mean()) if paired else None,
                    "difference_sample_sd": float(differences.std(ddof=1)) if len(paired) > 1 else None,
                    "undefined_method_scores_included": undefined_method,
                    "undefined_nores_scores_included": undefined_baseline,
                }
            diagnostics = [r["resampler_diagnostics"] for r in subset
                           if r["status"] == "success" and r.get("resampler_diagnostics")]
            additions = [r["sample_counts"]["added_rows"] for r in subset if r["status"] == "success"]
            durations = [r["timings_seconds"]["total"] for r in subset]
            summaries.append({
                "classifier": classifier, "method": method,
                "classifier_fidelity": get_classifier_spec(classifier).implementation_status,
                "success": statuses["success"], "failed": statuses["failed"], "unavailable": statuses["unavailable"],
                "metrics": metrics, "job_seconds_total": float(sum(durations)),
                "job_seconds_mean": float(np.mean(durations)), "job_seconds_max": float(max(durations)),
                "added_rows_mean": float(np.mean(additions)) if additions else None,
                "added_rows_min": min(additions) if additions else None,
                "added_rows_max": max(additions) if additions else None,
                "ugo_stopping_reasons": dict(Counter(d.get("stopping_reason") for d in diagnostics)),
                "ugo_iterations_max": max((d.get("number_of_iterations", 0) for d in diagnostics), default=0),
                "warning_categories": dict(Counter(w["category"] for r in subset for w in r["warnings"])),
                "failures": [{"fold": r["job"]["fold"], "status": r["status"], "error": r["error"]}
                             for r in subset if r["status"] != "success"],
            })
    return json_value(summaries)


def run_simulation_pilot(spec: dict[str, Any], output_dir: str | Path,
                         *, on_record: Callable | None = None) -> dict[str, Any]:
    """Run the frozen matrix once or resume its exact saved configuration."""
    config = pilot_config(spec)
    started = perf_counter()
    memory_before = process_peak_memory()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root / "pilot-config.json"
    normalized = json_value(spec)
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != normalized:
            raise ValueError("Pilot output contains a different configuration; use a new directory.")
    else:
        if any(root.iterdir()):
            raise ValueError("Pilot directory is nonempty without its configuration.")
        _atomic_json(spec_path, normalized)
    dataset_reports = []
    reused_total = 0
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for name, factory in (("linear", make_linear_simulation), ("nonlinear", make_nonlinear_simulation)):
            simulation = factory(seed=config.seed, **spec["simulations"][name])
            start = perf_counter()
            def progress(record, cached):
                if on_record:
                    on_record(name, record, cached)
            result = evaluate_ten_fold(
                simulation.X, simulation.y, dataset_id=f"ugo_simulation_{name}",
                positive_label=simulation.minority_label,
                provenance={"kind": "declared paper simulation reconstruction", "stage": "4R-6",
                            "author_exact_rows": False, "simulation": simulation.metadata,
                            "native_threads": 1},
                output_dir=root / name, config=config, on_record=progress)
            manifest = json.loads((root / name / "manifest.json").read_text(encoding="utf-8"))
            store_records = result["records"]
            hashes = [{"job": r["job"], "record_sha256": content_hash(r)} for r in store_records]
            dataset_reports.append({
                "dataset": name, "run_id": result["run_id"], "manifest": manifest["manifest"],
                "record_hashes": hashes, "rows": len(simulation.y), "positive_label": simulation.minority_label,
                "observed_class_counts": [{"label": int(c), "count": int(np.sum(simulation.y == c))}
                                          for c in np.unique(simulation.y)],
                "noise_relabeled_rows": len(simulation.relabeled_indices),
                "summaries": summarize_records(store_records, config),
                "invocation_wall_seconds": perf_counter() - start, "reused_jobs": result["reused_jobs"],
            })
            reused_total += result["reused_jobs"]
    expected_jobs = 2 * len(config.classifiers) * len(config.methods) * 10
    invocation = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": perf_counter() - started, "reused_jobs": reused_total,
        "executed_jobs": expected_jobs - reused_total, "expected_jobs": expected_jobs,
        "memory_before": memory_before, "memory_after": process_peak_memory(),
        "platform": platform.platform(), "machine": platform.machine(),
        "native_thread_pools": pools,
    }
    measurement_path = root / "first-complete-invocation.json"
    if not measurement_path.exists():
        _atomic_json(measurement_path, invocation)
    first_measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    report = json_value({
        "schema_version": 1, "stage": "4R-6", "config": asdict(config), "pilot_spec": normalized,
        "paired_policy": PAIRED_POLICY, "datasets": dataset_reports,
        "first_complete_invocation": first_measurement, "current_invocation": invocation,
        "scope": "simulation pilot only; no real benchmarks, tuning, significance tests or full reproduction claim",
    })
    _atomic_json(root / "latest-report.json", report)
    return report


def export_pilot_report(report: dict[str, Any], directory: str | Path) -> None:
    """Export compact, auditable measured results without raw observations."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "summary.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if ([d["run_id"] for d in old["datasets"]] != [d["run_id"] for d in report["datasets"]]):
            raise ValueError("Export directory contains different run identities; use a new directory.")
    # Keep originally measured timings, rather than claiming cached replay is computation time.
    exported = {k: v for k, v in report.items() if k != "current_invocation"}
    exported["datasets"] = [{k: v for k, v in dataset.items()
                             if k not in {"invocation_wall_seconds", "reused_jobs"}} for dataset in report["datasets"]]
    _atomic_json(path, exported)
    with (destination / "paired_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "classifier", "method", "metric", "success", "failed", "unavailable",
                  "paired_count", "method_mean", "nores_mean", "mean_difference", "difference_sample_sd",
                  "undefined_method_scores_included", "undefined_nores_scores_included"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in report["datasets"]:
            for summary in dataset["summaries"]:
                for metric, values in summary["metrics"].items():
                    writer.writerow({"dataset": dataset["dataset"], "metric": metric,
                                     **{k: summary[k] for k in ("classifier", "method", "success", "failed", "unavailable")},
                                     **{k: values[k] for k in fields if k in values}})
