"""Run a fixed two-dataset, four-method BFCH/UGO comparison on local inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.comparison_report import export_comparison_report, summarize_saved_run
from src.data_loader import load_dataset
from src.evaluation import EvaluationConfig, evaluate_ten_fold
from src.results import _atomic_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/manifests/ugo_74_approximate.json"
DATASETS = ("winequality_red_8_vs_6", "vowel0")
ACOUSTIC_FEATURES = tuple(f"F{index}" for index in range(10))
CONFIG = EvaluationConfig(classifiers=("LogReg", "GNB"),
                          methods=("nores", "smote", "ugo_smote", "bfch_smote"),
                          seed=42, scale=True)


def prepare_dataset(name: str):
    """Verify source identity before any split; keep identifiers out of X."""
    document = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    entry = next((entry for entry in document["datasets"] if entry["dataset_id"] == name), None)
    if entry is None or name not in DATASETS:
        raise ValueError(f"Unknown small-study dataset {name!r}.")
    source = ROOT / entry["relative_path"]
    if not source.is_file():
        raise FileNotFoundError(f"Missing local input {source}; acquire the named KEEL file first.")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Input checksum differs from the registered {name} file.")
    loaded = load_dataset(source, minority_label="positive")
    X, y = loaded.X, loaded.y.to_numpy()
    if (len(y) != entry["rows"] or X.shape[1] != entry["input_features"]
            or int(np.sum(y == 1)) != entry["minority_count"]
            or int(np.sum(y == 0)) != entry["majority_count"]
            or loaded.label_mapping != {"negative": 0, "positive": 1}):
        raise ValueError(f"Loaded {name} does not match registered dimensions or labels.")
    folds = None
    if name == "vowel0":
        groups = X["SpeakerNumber"].to_numpy()
        folds = list(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
                     .split(X, y, groups))
        if any(np.intersect1d(groups[training], groups[testing]).size
               for training, testing in folds):
            raise ValueError("Speaker groups overlap a vowel train/test fold.")
        X = X.loc[:, list(ACOUSTIC_FEATURES)].copy()
        dataset_id = "vowel0_acoustic10_grouped"
        split = "ten stratified-group folds by SpeakerNumber; no speaker appears in both partitions"
        excluded = ["TT", "SpeakerNumber", "Sex"]
    else:
        dataset_id = name
        split = "ten stratified shuffled folds, seed 42"
        excluded = []
    provenance = {
        "kind": "KEEL binary variant from local UGO collection; BFCH name/count match only",
        "source_dataset_id": name, "source_filename": source.name, "source_sha256": digest,
        "source_fidelity": entry["fidelity"], "author_rows_unverified": True,
        "feature_selection": list(X.columns), "excluded_features": excluded,
        "fold_source": split,
    }
    return dataset_id, X, y, provenance, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/bfch-small-study"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    overview = {"purpose": "small descriptive BFCH-versus-UGO comparison",
                "not_author_paper_scores": True, "config": {
                    "classifiers": list(CONFIG.classifiers), "methods": list(CONFIG.methods),
                    "seed": CONFIG.seed, "scale": CONFIG.scale}, "datasets": []}
    for name in args.datasets:
        dataset_id, X, y, provenance, folds = prepare_dataset(name)
        run_dir = args.output / dataset_id
        result = evaluate_ten_fold(
            X, y, dataset_id=dataset_id, positive_label=1, provenance=provenance,
            output_dir=run_dir, config=CONFIG, supplied_folds=folds)
        report = summarize_saved_run(run_dir)
        export_comparison_report(report, run_dir / "comparison.csv")
        overview["datasets"].append({"dataset_id": dataset_id, "source_id": name,
                                      "input_sha256": provenance["source_sha256"],
                                      "rows": len(y), "features": X.shape[1],
                                      "minority": int(np.sum(y == 1)),
                                      "majority": int(np.sum(y == 0)),
                                      "run_id": result["run_id"], "reused_jobs": result["reused_jobs"],
                                      "report_status": report["status"],
                                      "coverage": report["coverage"],
                                      "source_fidelity": provenance["source_fidelity"],
                                      "excluded_features": provenance["excluded_features"],
                                      "fold_source": provenance["fold_source"]})
        print(f"{dataset_id}: {report['status']}, reused {result['reused_jobs']} jobs; "
              f"report at {run_dir / 'comparison.csv'}", flush=True)
    _atomic_json(args.output / "overview.json", overview)


if __name__ == "__main__":
    main()
