"""Run the fixed 30-variant, six-method controlled comparison locally.

This is a descriptive extension of the Ecoli4/Yeast4 pilot. It preserves the
same five-repeat/five-fold label-hiding protocol, methods and classifiers. Raw
inputs are local and ignored by Git; --check-only never fits a classifier.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_sshr_small_study import group_folds_and_masks
from src.data_loader import load_dataset
from src.results import _atomic_json, content_hash
from src.sshr_evaluation import ControlledEvaluationConfig, evaluate_controlled
from src.sshr_study_report import (
    export_study_reports, summarize_controlled_run, summarize_unlabeled_ablation,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("data/manifests/all_methods_30.json")
CLASSIFIERS = ("LogReg", "GNB")
METHODS = ("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr")
CONFIG = ControlledEvaluationConfig(
    classifiers=CLASSIFIERS, methods=METHODS, seed=42, scale=True,
    n_splits=5, n_repeats=5, labeled_fraction=0.5, min_labeled_per_class=2,
    use_unlabeled=True, method_parameters={},
)
ABLATION_CONFIG = replace(CONFIG, methods=("nores", "sshr"), use_unlabeled=False)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_registry(root: Path = ROOT) -> tuple[list[str], dict, str, str]:
    """Check the fixed selection against the larger source registry."""
    selection_path = root / SELECTION
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    names = selection.get("datasets")
    if (selection.get("schema_version") != 1 or not isinstance(names, list)
            or len(names) != 30 or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != 30):
        raise ValueError("The fixed study selection must contain exactly 30 distinct dataset IDs.")
    source_path = root / selection["source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    entries = {item["dataset_id"]: item for item in source["datasets"]}
    if len(entries) != len(source["datasets"]) or any(n not in entries for n in names):
        raise ValueError("Selected dataset ID missing or ambiguous in source registry.")
    return names, entries, _hash(selection_path), _hash(source_path)


def prepare_dataset(name: str, entry: dict, *, root: Path = ROOT,
                    selection_hash: str, source_manifest_hash: str):
    """Verify input bytes, numerical contract, counts and all frozen row splits."""
    path = root / entry["relative_path"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}. Copy the raw data before training.")
    digest = _hash(path)
    if digest != entry["sha256"]:
        raise ValueError(f"Source SHA-256 differs for {name}.")
    if entry["dataset_id"] != name or entry["categorical_features"] or entry["missing_feature_cells"]:
        raise ValueError(f"{name} is not a registered numeric, no-missing input.")
    loaded = load_dataset(path, target_column=entry["target_column"],
                          minority_label=entry["positive_label"])
    if _hash(path) != digest:
        raise ValueError(f"Source changed while loading {name}.")
    X, y = loaded.X, loaded.y.to_numpy()
    try:
        matrix = X.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has non-numerical features.") from exc
    if (not np.isfinite(matrix).all() or tuple(X.columns) != tuple(entry["feature_names"])
            or X.shape != (entry["rows"], entry["input_features"])
            or int(np.sum(y == 0)) != entry["majority_count"]
            or int(np.sum(y == 1)) != entry["minority_count"]
            or loaded.label_mapping.get(entry["positive_label"]) != 1
            or loaded.categorical_feature_names):
        raise ValueError(f"{name} dimensions, labels or feature types differ from registry.")
    groups, folds, masks = group_folds_and_masks(
        X, y, seed=CONFIG.seed, n_splits=CONFIG.n_splits,
        n_repeats=CONFIG.n_repeats, labeled_fraction=CONFIG.labeled_fraction,
        min_labeled_per_class=CONFIG.min_labeled_per_class,
    )
    provenance = {
        "kind": "hash-verified local binary benchmark variant; fixed descriptive comparison",
        "source_dataset_id": name, "source_relative_path": entry["relative_path"],
        "source_filename": path.name, "source_sha256": digest,
        "source_fidelity": entry["fidelity"],
        "reported_paper_name": entry["paper_dataset_name"],
        "source_positive_label": entry["positive_label"],
        "selection_manifest_sha256": selection_hash,
        "source_manifest_sha256": source_manifest_hash,
        "runner_script_sha256": _hash(Path(__file__)),
        "feature_types": {column: "numeric" for column in X.columns},
        "feature_selection": list(X.columns), "excluded_features": [],
        "coordinate_groups": groups.tolist(),
        "requires_group_or_time_split": True,
        "fold_source": "Five repeated stratified-group folds on identical raw feature coordinates; seed 42; no redraw",
        "mask_source": "Within each training fold, retain about half of each class in whole coordinate groups; seed 42; no redraw",
        "frozen_design": {"folds_sha256": content_hash(folds),
                          "masks_sha256": content_hash(masks)},
        "author_rows_unverified": True,
        "author_folds_unverified": True,
        "author_unlabeled_pool_protocol_unverified": True,
    }
    summary = {"dataset_id": name, "rows": len(y), "features": X.shape[1],
               "majority": entry["majority_count"], "minority": entry["minority_count"],
               "source_sha256": digest, "source_fidelity": entry["fidelity"],
               "folds": len(folds), "min_test_minority": min(
                   int(np.sum(y[fold["test_indices"]] == 1)) for fold in folds)}
    return X, y, provenance, folds, masks, summary


def export_combined_scores(report_root: Path, dataset_names: list[str]) -> Path:
    """Combine canonical same-fold scores from completed dataset reports."""
    rows = []
    for name in dataset_names:
        path = report_root / name / "reports" / "scores.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(row for row in csv.DictReader(handle)
                        if row["scope"] == "all_methods_shared_success")
    path = report_root / "all_methods_30_scores.csv"
    report_root.mkdir(parents=True, exist_ok=True)
    columns = ("dataset_id", "run_id", "classifier", "method", "scope", "metric",
               "mean", "sample_sd", "n_folds", "fold_ids", "n_paired_with_nores",
               "nores_mean", "paired_mean_difference")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    names, entries, selection_hash, source_manifest_hash = read_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/all-methods-30"))
    parser.add_argument("--datasets", nargs="+", choices=names,
                        help="Run a selected subset first; omitted means all 30.")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate source hashes, labels, group folds and masks; no fitting.")
    args = parser.parse_args(argv)
    selected = args.datasets if args.datasets is not None else names
    if len(selected) != len(set(selected)):
        parser.error("--datasets cannot contain duplicates.")
    prepared = []
    for name in selected:
        result = prepare_dataset(name, entries[name], selection_hash=selection_hash,
                                 source_manifest_hash=source_manifest_hash)
        prepared.append((name, *result))
        print(f"{name}: input OK, {result[-1]['folds']} group-disjoint fold events, "
              f"min test minority={result[-1]['min_test_minority']}", flush=True)
    if args.check_only:
        print(f"Checked {len(prepared)}/{len(names)} selected datasets; no model was fitted.")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    overview = {"schema_version": 1, "purpose": "fixed 30-variant controlled comparison",
                "selection_manifest_sha256": selection_hash,
                "source_manifest_sha256": source_manifest_hash,
                "configured_dataset_count": len(names), "selected_datasets": selected,
                "config": asdict(CONFIG), "ablation_config": asdict(ABLATION_CONFIG),
                "datasets": []}
    failed = False
    for name, X, y, provenance, folds, masks, summary in prepared:
        # A source edited between preflight and its own run cannot be trained.
        if _hash(ROOT / provenance["source_relative_path"]) != provenance["source_sha256"]:
            raise ValueError(f"Source changed after preparation: {name}.")
        item = {**summary, "runs": {}}
        overview["datasets"].append(item)
        for condition, config in (("with-u", CONFIG), ("without-u", ABLATION_CONFIG)):
            count = 0

            def progress(record: dict, cached: bool) -> None:
                nonlocal count
                count += 1
                if count % 25 == 0 or record["status"] != "success":
                    print(f"{name}/{condition}: {count} jobs, "
                          f"latest={record['status']}, cached={cached}", flush=True)

            run = evaluate_controlled(
                X, y, dataset_id=name, positive_label=1, provenance=provenance,
                output_dir=args.output / name / condition, config=config,
                supplied_folds=folds, supplied_masks=masks, on_record=progress,
            )
            item["runs"][condition] = {"run_id": run["run_id"],
                                       "coverage": run["coverage"],
                                       "reused_jobs": run["reused_jobs"]}
            failed |= any(record["status"] != "success" for record in run["records"])
            _atomic_json(args.output / "overview.json", overview)
        report = summarize_controlled_run(args.output / name / "with-u")
        ablation = summarize_unlabeled_ablation(
            args.output / name / "with-u", args.output / name / "without-u")
        export_study_reports(report, ablation, args.output / name / "reports")
        export_combined_scores(args.output, selected)
        print(f"{name}: reports written to {args.output / name / 'reports'}", flush=True)
    print(f"Combined table: {args.output / 'all_methods_30_scores.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
