"""Manifest-validated, resumable orchestration for the 74-dataset benchmark."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from src.data_loader import LoadedDataset, load_dataset
from src.evaluation import EvaluationConfig, _validate_config, evaluate_ten_fold
from src.pilot import process_peak_memory, summarize_records
from src.results import _atomic_json, content_hash, json_value


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "ugo_74_approximate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_benchmark_manifest(path: str | Path = DEFAULT_MANIFEST) -> tuple[dict[str, Any], Path]:
    """Load the accepted manifest and verify all immutable source facts."""
    manifest_path = Path(path).resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = document.get("datasets")
    if (document.get("schema_version") != 1 or document.get("dataset_count") != 74
            or not isinstance(datasets, list) or len(datasets) != 74):
        raise ValueError("Stage 4R-7 requires the schema-1 74-dataset manifest.")
    identifiers = [entry.get("dataset_id") for entry in datasets]
    if any(not isinstance(item, str) or not item for item in identifiers) or len(set(identifiers)) != 74:
        raise ValueError("Benchmark dataset identifiers must be 74 unique nonempty strings.")
    for entry in datasets:
        required = {"relative_path", "sha256", "target_column", "positive_label", "minority_label",
                    "rows", "input_features", "majority_count", "minority_count", "fidelity"}
        if not required <= entry.keys():
            raise ValueError(f"Incomplete manifest entry for {entry['dataset_id']}.")
        source = (ROOT / entry["relative_path"]).resolve()
        if ROOT not in source.parents or not source.is_file():
            raise ValueError(f"Dataset source is absent or outside the repository: {entry['dataset_id']}.")
        if _sha256(source) != entry["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {entry['dataset_id']}.")
    return document, manifest_path


def _load_entry(entry: dict[str, Any]) -> tuple[LoadedDataset, Path]:
    source = (ROOT / entry["relative_path"]).resolve()
    loaded = load_dataset(source, target_column=entry["target_column"],
                          minority_label=entry["minority_label"])
    counts = np.bincount(loaded.y, minlength=2)
    expected = np.array([entry["majority_count"], entry["minority_count"]])
    if (len(loaded.y) != entry["rows"] or loaded.X.shape[1] != entry["input_features"]
            or not np.array_equal(counts, expected)
            or entry["positive_label"] not in loaded.label_mapping):
        raise ValueError(f"Loaded data differ from manifest facts: {entry['dataset_id']}.")
    return loaded, source


def benchmark_identity(document: dict[str, Any], manifest_path: Path,
                       config: EvaluationConfig) -> dict[str, Any]:
    return json_value({
        "schema_version": 1, "stage": "4R-7", "dataset_manifest": str(manifest_path.relative_to(ROOT)),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "datasets": [entry["dataset_id"] for entry in document["datasets"]],
        "config": asdict(config), "expected_jobs": 74 * len(config.classifiers) * len(config.methods) * 10,
        "source_scope": document["scope"], "accepted_approximate_variants": 18,
        "execution_policy": "dataset-level processes; one native thread per worker; atomic per-fold records",
    })


def _run_dataset(entry: dict[str, Any], output_root: str, config: EvaluationConfig,
                 retry_failures: bool) -> dict[str, Any]:
    loaded, source = _load_entry(entry)
    positive = loaded.label_mapping[entry["positive_label"]]
    started = perf_counter()
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        result = evaluate_ten_fold(
            loaded.X, loaded.y, dataset_id=entry["dataset_id"], positive_label=positive,
            provenance={
                "kind": "manifest-validated accepted UGO benchmark input",
                "stage": "4R-7", "manifest_entry": entry,
                "file_sha256": entry["sha256"], "source_positive_label": entry["positive_label"],
                "label_mapping": loaded.label_mapping, "author_folds": False,
                "native_threads": 1,
            },
            output_dir=Path(output_root) / "datasets" / entry["dataset_id"],
            config=config, retry_failures=retry_failures)
    records = result["records"]
    statuses = Counter(record["status"] for record in records)
    report = json_value({
        "dataset": entry["dataset_id"], "run_id": result["run_id"], "source": str(source.relative_to(ROOT)),
        "rows": len(loaded.y), "input_features": loaded.X.shape[1], "positive_label": positive,
        "fidelity": entry["fidelity"], "success": statuses["success"], "failed": statuses["failed"],
        "unavailable": statuses["unavailable"], "expected_jobs": len(records),
        "reused_jobs": result["reused_jobs"], "invocation_wall_seconds": perf_counter() - started,
        "summaries": summarize_records(records, config),
        "record_set_sha256": content_hash([content_hash(record) for record in records]),
        "native_thread_pools": pools,
    })
    report_path = Path(output_root) / "dataset-reports" / f"{entry['dataset_id']}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    return report


def benchmark_status(output_root: str | Path, identity: dict[str, Any]) -> dict[str, Any]:
    root = Path(output_root)
    reports = {}
    for path in (root / "dataset-reports").glob("*.json") if (root / "dataset-reports").is_dir() else []:
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[report["dataset"]] = report
    selected = [reports[name] for name in identity["datasets"] if name in reports]
    return json_value({
        "dataset_count": len(identity["datasets"]), "completed_datasets": len(selected),
        "remaining_datasets": [name for name in identity["datasets"] if name not in reports],
        "expected_jobs": identity["expected_jobs"],
        "terminal_jobs_in_completed_datasets": sum(item["expected_jobs"] for item in selected),
        "success": sum(item["success"] for item in selected),
        "failed": sum(item["failed"] for item in selected),
        "unavailable": sum(item["unavailable"] for item in selected),
    })


def run_benchmark(output_root: str | Path, config: EvaluationConfig, *,
                  manifest_path: str | Path = DEFAULT_MANIFEST,
                  selected_datasets: tuple[str, ...] | None = None, workers: int = 1,
                  retry_failures: bool = False,
                  on_dataset: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run or resume selected dataset shards within one fixed full-matrix identity."""
    _validate_config(config)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 5:
        raise ValueError("workers must be an integer from 1 through 5.")
    document, path = load_benchmark_manifest(manifest_path)
    identity = benchmark_identity(document, path, config)
    entries = {entry["dataset_id"]: entry for entry in document["datasets"]}
    selected = tuple(entries) if selected_datasets is None else selected_datasets
    if not selected or len(set(selected)) != len(selected) or any(name not in entries for name in selected):
        raise ValueError("Selected datasets must be unique names from the 74-dataset manifest.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "benchmark-config.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Benchmark output belongs to a different configuration; use a new directory.")
    else:
        if any(root.iterdir()):
            raise ValueError("Benchmark output is nonempty without its configuration.")
        _atomic_json(identity_path, identity)
    started, memory_before = perf_counter(), process_peak_memory()
    completed = []
    if workers == 1:
        for name in selected:
            report = _run_dataset(entries[name], str(root), config, retry_failures)
            completed.append(report)
            if on_dataset:
                on_dataset(report)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_dataset, entries[name], str(root), config, retry_failures): name
                       for name in selected}
            for future in as_completed(futures):
                report = future.result()
                completed.append(report)
                if on_dataset:
                    on_dataset(report)
    invocation = json_value({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "selected_datasets": list(selected),
        "workers": workers, "wall_seconds": perf_counter() - started,
        "memory_before": memory_before, "memory_after": process_peak_memory(),
        "datasets_completed_this_invocation": len(completed),
        "jobs_executed_this_invocation": sum(r["expected_jobs"] - r["reused_jobs"] for r in completed),
        "jobs_reused_this_invocation": sum(r["reused_jobs"] for r in completed),
    })
    status = benchmark_status(root, identity)
    _atomic_json(root / "latest-invocation.json", invocation)
    _atomic_json(root / "status.json", status)
    return {"identity": identity, "invocation": invocation, "status": status,
            "completed": sorted(completed, key=lambda item: identity["datasets"].index(item["dataset"]))}
