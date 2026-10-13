"""Run a fixed paired SSHR pilot on two preselected, hash-verified local files.

All raw rows are retained. Identical original feature vectors stay together in
outer folds and in observed/hidden training-label pools. These folds and masks
are declared experimental design, not recovered author folds or an author
unlabelled-data protocol. This script never downloads data or tunes parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import Any
import zipfile

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src._validation import binary_labels, numeric_matrix
from src.data_loader import load_dataset
from src.results import _atomic_json, content_hash
from src.sshr_evaluation import ControlledEvaluationConfig, evaluate_controlled


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = Path("data/manifests/sshr_pilot_sources.json")
DATASETS = ("ecoli4", "yeast4")
CLASSIFIERS = ("LogReg", "GNB")
METHODS = ("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr")
CONFIG = ControlledEvaluationConfig(
    classifiers=CLASSIFIERS, methods=METHODS, seed=42, scale=True,
    n_splits=5, n_repeats=5, labeled_fraction=0.5, min_labeled_per_class=2,
    use_unlabeled=True, method_parameters={})
ABLATION_CONFIG = replace(CONFIG, methods=("nores", "sshr"), use_unlabeled=False)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _derived_seed(seed: int, repeat: int, fold: int, purpose: int) -> int:
    return int(np.random.SeedSequence([seed, repeat, fold, purpose]).generate_state(1)[0])


def group_folds_and_masks(X: Any, y: Any, *, seed: int = 42, n_splits: int = 5,
                          n_repeats: int = 5, labeled_fraction: float = 0.5,
                          min_labeled_per_class: int = 2
                          ) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Freeze group-disjoint folds and label masks before any learned operation.

    Group identity depends on raw X only. Gold labels are used solely for this
    experimental-design stratification and conflict check, never as SSHR input
    for hidden rows. Prefixes of randomly permuted whole class groups are kept
    until their observed-row target is met; any group-size overshoot is recorded.
    """
    matrix = numeric_matrix(X)
    labels, classes = binary_labels(y, len(matrix))
    seed = _integer(seed, "seed", 0)
    if seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32).")
    n_splits = _integer(n_splits, "n_splits", 2)
    n_repeats = _integer(n_repeats, "n_repeats", 1)
    minimum = _integer(min_labeled_per_class, "min_labeled_per_class", 2)
    if (isinstance(labeled_fraction, bool) or not isinstance(labeled_fraction, Real)
            or not np.isfinite(labeled_fraction) or not 0 < labeled_fraction <= 1):
        raise ValueError("labeled_fraction must be finite and in (0, 1].")
    fraction = float(labeled_fraction)
    identities: dict[tuple[float, ...], int] = {}
    groups = np.empty(len(matrix), dtype=np.int64)
    for index, row in enumerate(matrix):
        key = tuple(row)
        groups[index] = identities.setdefault(key, len(identities))
    group_labels = {}
    group_sizes = {}
    for group in range(len(identities)):
        members = np.flatnonzero(groups == group)
        observed = np.unique(labels[members])
        if len(observed) != 1:
            raise ValueError("Identical-coordinate groups contain conflicting gold labels; this pilot design is unsupported.")
        group_labels[group] = observed[0]
        group_sizes[group] = len(members)
    if any(sum(value == label for value in group_labels.values()) < n_splits
           for label in classes):
        raise ValueError("Each class needs at least n_splits coordinate groups; folds are not silently reduced.")
    folds, masks = [], []
    for repeat in range(n_repeats):
        split_seed = _derived_seed(seed, repeat, 0, 1)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(splitter.split(matrix, labels, groups)):
            if (len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2
                    or np.intersect1d(groups[train], groups[test]).size):
                raise ValueError("Every frozen group fold must have disjoint groups and both classes; no redraw is performed.")
            folds.append({"repeat": repeat, "fold": fold, "split_seed": split_seed,
                          "train_indices": train.tolist(), "test_indices": test.tolist()})
            mask_seed = _derived_seed(seed, repeat, fold, 2)
            rng = np.random.default_rng(mask_seed)
            selected_groups: set[int] = set()
            design = []
            for label in classes:
                candidates = train[labels[train] == label]
                if len(candidates) < minimum:
                    raise ValueError("Outer training class cannot meet min_labeled_per_class; no redraw is performed.")
                requested = int(np.floor(fraction * len(candidates)))
                target = min(len(candidates), max(minimum, requested))
                # Group IDs follow first occurrence in the original file, not
                # class labels or later learned coordinates.
                candidate_groups = np.unique(groups[candidates])
                permutation = rng.permutation(candidate_groups)
                chosen, count = [], 0
                for group in permutation:
                    chosen.append(int(group))
                    count += group_sizes[int(group)]
                    if count >= target:
                        break
                selected_groups.update(chosen)
                design.append({"label": label.item() if isinstance(label, np.generic) else label,
                               "training_count": len(candidates),
                               "training_group_count": len(candidate_groups),
                               "fraction_requested_count": requested,
                               "target_labeled_count": target, "labeled_count": count,
                               "minimum_clamp_applied": target != requested,
                               "whole_group_overshoot": count - target,
                               "selected_coordinate_groups": chosen})
            labeled = train[np.isin(groups[train], list(selected_groups))]
            unlabeled = train[~np.isin(groups[train], list(selected_groups))]
            if (np.intersect1d(groups[labeled], groups[unlabeled]).size
                    or any(np.sum(labels[labeled] == label) < minimum for label in classes)):
                raise ValueError("Frozen group mask violates disjointness or minimum observed counts.")
            masks.append({"repeat": repeat, "fold": fold, "mask_seed": mask_seed,
                          "labeled_indices": labeled.tolist(), "unlabeled_indices": unlabeled.tolist(),
                          "requested_labeled_fraction": fraction,
                          "actual_labeled_fraction": len(labeled) / len(train),
                          "per_class_design": design})
    return groups, folds, masks


def _duplicates(groups: np.ndarray) -> dict[str, int]:
    counts = np.bincount(groups)
    repeated = counts[counts > 1]
    return {"coordinate_group_count": len(counts),
            "duplicate_coordinate_groups": len(repeated),
            "duplicate_coordinate_pairs": int(np.sum(repeated * (repeated - 1) // 2)),
            "duplicate_excess_rows": int(np.sum(repeated - 1)),
            "rows_in_duplicate_groups": int(np.sum(repeated))}


def prepare_dataset(name: str, *, root: Path = ROOT):
    """Verify all registered source properties, then freeze exact row designs."""
    root = Path(root)
    manifest_path = root / SOURCE_MANIFEST
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [entry for entry in document["datasets"] if entry["dataset_id"] == name]
    if name not in DATASETS or len(entries) != 1:
        raise ValueError(f"Unknown or ambiguous pilot dataset {name!r}.")
    entry = entries[0]
    source = root / entry["relative_path"]
    if not source.is_file():
        raise FileNotFoundError(f"Missing local input {source}; this script never downloads data.")
    digest = _file_hash(source)
    if digest != entry["sha256"] or source.stat().st_size != entry["bytes"]:
        raise ValueError(f"Registered {name} source SHA-256 or byte size mismatch.")
    loaded = load_dataset(source, minority_label="positive")
    if _file_hash(source) != digest:
        raise ValueError("Source file changed during loading; no evaluation may run.")
    X, y = loaded.X, loaded.y.to_numpy()
    if (loaded.source_format != "dat" or loaded.categorical_feature_names
            or tuple(loaded.numeric_feature_names) != tuple(entry["feature_names"])
            or tuple(X.columns) != tuple(entry["feature_names"])):
        raise ValueError("Pilot requires the registered original numeric KEEL features; categorical sources are unsupported.")
    matrix = X.to_numpy(dtype=float)
    if (not np.isfinite(matrix).all() or len(y) != entry["rows"]
            or X.shape[1] != entry["input_features"] or loaded.target_name != entry["target_name"]
            or loaded.label_mapping != entry["label_mapping"]
            or int(np.sum(y == 0)) != entry["majority_count"]
            or int(np.sum(y == 1)) != entry["minority_count"]):
        raise ValueError("Loaded source dimensions, labels, missingness or finite numerical features differ from registration.")
    groups, folds, masks = group_folds_and_masks(X, y)
    duplicate_summary = _duplicates(groups)
    if any(duplicate_summary[key] != entry[key] for key in (
            "duplicate_coordinate_groups", "duplicate_coordinate_pairs", "duplicate_excess_rows")):
        raise ValueError("Duplicate-coordinate audit differs from source registration.")
    archive_entry = entry["archive"]
    archive_path = root / archive_entry["relative_path"]
    if archive_path.is_file():
        archive_hash = _file_hash(archive_path)
        if archive_hash != archive_entry["sha256"]:
            raise ValueError("Optional local archive exists but differs from its registered SHA-256.")
        with zipfile.ZipFile(archive_path) as original:
            members = [info for info in original.infolist()
                       if Path(info.filename).name == archive_entry["member"] and not info.is_dir()]
            if len(members) != 1 or hashlib.sha256(original.read(members[0])).hexdigest() != digest:
                raise ValueError("Optional archive raw member differs from the registered extracted source.")
        archive = {**archive_entry, "verification": "local archive SHA-256 and extracted raw-member SHA-256 verified"}
    else:
        archive = {**archive_entry, "verification": "UNVERIFIED: optional original archive absent locally; extracted source independently hash-verified"}
    provenance = {
        "kind": "hash-verified local numeric KEEL binary variant; descriptive controlled pilot",
        "source_dataset_id": name, "source_relative_path": entry["relative_path"],
        "source_filename": source.name, "source_sha256": digest, "source_bytes": entry["bytes"],
        "source_format": loaded.source_format, "target_name": loaded.target_name,
        "original_label_mapping": loaded.label_mapping, "source_positive_label": loaded.minority_label,
        "feature_types": {feature: "numeric" for feature in X.columns},
        "feature_metadata_authority": "Original KEEL input attribute declarations; all declared real",
        "feature_selection": list(X.columns), "excluded_features": [],
        "source_fidelity": document["source_fidelity"], "author_rows_unverified": True,
        "author_folds_unverified": True, "author_unlabeled_pool_protocol_unverified": True,
        "dataset_selection_policy": document["selection_policy"],
        "source_registration_sha256": _file_hash(manifest_path),
        "runner_script_sha256": _file_hash(Path(__file__)), "archive": archive,
        "coordinate_groups": groups.tolist(), "duplicate_audit": duplicate_summary,
        "coordinate_group_policy": document["duplicate_policy"], "requires_group_or_time_split": True,
        "fold_source": "Five stratified-group outer folds repeated five times; identical original raw feature vectors stay together; NumPy SeedSequence([42,repeat,0,1]); author folds unavailable; no redraw",
        "mask_source": "Hide training labels only: class-wise permutation of whole coordinate groups with NumPy SeedSequence([42,repeat,fold,2]); retain group prefixes until max(2,floor(.5*class_training_rows)); preserve original training row order; no redraw; not a verified author protocol",
        "frozen_design": {"folds_sha256": content_hash(folds), "masks_sha256": content_hash(masks),
                          "coordinate_groups_sha256": content_hash(groups),
                          "masks": masks},
        "hidden_label_policy": "Gold labels are used only for experimental-design stratification/conflict checking and evaluator-side retrospective audit; hidden labels never enter learned methods.",
    }
    return name, X, y, provenance, folds, masks


def _progress(name: str, condition: str, expected: int):
    count = 0

    def callback(record: dict, cached: bool) -> None:
        nonlocal count
        count += 1
        if count % 5 == 0 or count == expected or record["status"] != "success":
            print(f"{name}/{condition}: {count}/{expected} jobs; latest={record['status']}; cached={cached}", flush=True)

    return callback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/sshr-small-study"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--report-dir", type=Path, help="Optional directory for shareable aggregate reports; raw job records remain under --output.")
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must contain unique selected names.")
    # Prepare every selected input before any classifier fit or held-out score.
    prepared = [prepare_dataset(name) for name in args.datasets]
    for _, _, _, provenance, _, _ in prepared:
        if _file_hash(ROOT / provenance["source_relative_path"]) != provenance["source_sha256"]:
            raise ValueError("Registered source changed after preparation; no evaluation may run.")
    args.output.mkdir(parents=True, exist_ok=True)
    overview = {"schema_version": 1, "purpose": "two-dataset descriptive controlled SSHR pilot",
                "not_author_paper_scores": True, "no_superiority_claim": True,
                "config": asdict(CONFIG), "ablation_config": asdict(ABLATION_CONFIG),
                "selection_before_scores": True, "sampler_parameters": "Existing sampler defaults and engineering guards retained; no reduced fast settings or performance-driven tuning",
                "datasets": []}
    failed = False
    for name, X, y, provenance, folds, masks in prepared:
        entry = {"dataset_id": name, "source_sha256": provenance["source_sha256"],
                 "rows": len(y), "features": X.shape[1], "minority": int(np.sum(y == 1)),
                 "majority": int(np.sum(y == 0)), "duplicate_audit": provenance["duplicate_audit"],
                 "source_fidelity": provenance["source_fidelity"],
                 "fold_source": provenance["fold_source"], "mask_source": provenance["mask_source"],
                 "folds": folds,
                 "frozen_design": provenance["frozen_design"], "runs": {}}
        overview["datasets"].append(entry)
        for condition, config in (("with-u", CONFIG), ("without-u", ABLATION_CONFIG)):
            started = perf_counter()
            result = evaluate_controlled(
                X, y, dataset_id=name, positive_label=1, provenance=provenance,
                output_dir=args.output / name / condition, config=config,
                supplied_folds=folds, supplied_masks=masks,
                on_record=_progress(name, condition, config.n_splits * config.n_repeats
                                    * len(config.classifiers) * len(config.methods)))
            entry["runs"][condition] = {"run_id": result["run_id"], "output_dir": result["output_dir"],
                                         "reused_jobs": result["reused_jobs"], "coverage": result["coverage"],
                                         "elapsed_seconds": perf_counter() - started}
            failed |= any(record["status"] != "success" for record in result["records"])
            _atomic_json(args.output / "overview.json", overview)
        # Report implementation is imported only for completed runs, allowing
        # the source preparation helpers to be used independently.
        from src.sshr_study_report import (
            export_study_reports, summarize_controlled_run, summarize_unlabeled_ablation)
        report = summarize_controlled_run(args.output / name / "with-u")
        ablation = summarize_unlabeled_ablation(
            args.output / name / "with-u", args.output / name / "without-u")
        report_dir = (args.report_dir / name) if args.report_dir else args.output / name / "reports"
        export_study_reports(report, ablation, report_dir)
        entry["report_dir"] = str(report_dir.resolve())
        _atomic_json(args.output / "overview.json", overview)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
