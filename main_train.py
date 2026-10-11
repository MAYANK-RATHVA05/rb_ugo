"""Train a reproducible Logistic Regression/UGO pilot on one CSV or KEEL file."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, RANDOM_STATE, RESULTS_DIR
from src.data_loader import DatasetError, load_dataset
from src.training import ExperimentResult, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Extracted .dat file or CSV; not a ZIP archive.")
    parser.add_argument("--target", help="Required target column for CSV; optional KEEL override.")
    parser.add_argument("--minority-label", help="Exact source label to encode as positive class 1.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Stratified test fraction; default 0.25.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--max-iter", type=int, default=100, help="UGO scoring-loop safeguard; not LR's max_iter.")
    parser.add_argument("--cap-policy", choices=["section_3_3", "table_4"], default="section_3_3",
                        help="Explicit interpretation of the paper's conflicting generation limits.")
    parser.add_argument("--output-dir", type=Path,
                        help="New artifact directory; existing directories are never overwritten.")
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=_json_default)
        stream.write("\n")


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision() -> dict[str, Any]:
    """Record source version without requiring Git in a copied Windows folder."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if revision.returncode:
            return {"commit": None, "worktree_dirty": None}
        status = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, check=False,
        )
        return {"commit": revision.stdout.strip(),
                "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except OSError:
        return {"commit": None, "worktree_dirty": None}


def save_experiment(result: ExperimentResult, source: Path, output: Path) -> Path:
    """Persist raw-feature prediction pipelines and audit records in a new folder."""
    source_sha256 = _source_hash(source)
    report = {
        **result.report,
        "run_metadata": {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_filename": source.name,
            "source_sha256": source_sha256,
            "code": _code_revision(),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in
                         ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "joblib"]},
            "model_files": {name: name.lower().replace("-", "_") + ".joblib"
                            for name in result.models},
            "model_input": "Raw feature DataFrame with the original named columns; omit the target.",
        },
    }
    # Catch non-serializable/nonfinite report values before creating output files.
    json.dumps(report, allow_nan=False, default=_json_default)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "run.json", report)
    _write_json(output / "split.json", {
        "index_definition": "Zero-based row positions in the supplied data file, after parsing.",
        "train_indices": result.train_indices,
        "test_indices": result.test_indices,
    })
    metric_rows = []
    prediction_rows = []
    for name, model in result.models.items():
        method = result.report["methods"][name]
        metrics = {key: value for key, value in method["metrics"].items() if key != "confusion_matrix"}
        metric_rows.append({"method": name, "training_rows": method["training_rows"],
                            "synthetic_rows": method["synthetic_rows"], **metrics})
        prediction_rows.extend({"test_row_index": int(index), "y_true": int(truth),
                                "method": name, "prediction": int(prediction),
                                "minority_probability": float(probability)}
                               for index, truth, prediction, probability in zip(
                                   result.test_indices, result.y_test, result.predictions[name],
                                   result.probabilities[name]))
        filename = report["run_metadata"]["model_files"][name]
        joblib.dump(model, output / filename)
        diagnostics = method["ugo_diagnostics"]
        if diagnostics is not None:
            _write_json(output / (Path(filename).stem + "_diagnostics.json"), diagnostics)
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.csv", index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = load_dataset(args.data, args.target, args.minority_label)
        output = args.output_dir
        if output is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = RESULTS_DIR / "ugo_training" / f"{args.data.stem}-{stamp}-{uuid4().hex[:8]}"
        if output.exists():
            raise ValueError(f"Output already exists; choose a new directory: {output}")
        print("UGO single-dataset training pilot; paper assumptions remain documented.")
        result = run_experiment(dataset, test_size=args.test_size, random_state=args.seed,
                                cap_policy=args.cap_policy, max_iter=args.max_iter)
        save_experiment(result, args.data, output)
        print(f"Training rows: {len(result.train_indices)}; untouched test rows: {len(result.test_indices)}")
        print(f"{'Method':<12} {'Added':>7} {'IAM':>9} {'MCC':>9} {'F1':>9} {'Recall':>9}")
        for name, method in result.report["methods"].items():
            metric = method["metrics"]
            print(f"{name:<12} {method['synthetic_rows']:>7} {metric['IAM']:>9.4f} "
                  f"{metric['MCC']:>9.4f} {metric['F1']:>9.4f} {metric['minority_recall']:>9.4f}")
            if method["ugo_diagnostics"] is not None:
                print(f"  Stopping reason: {method['ugo_diagnostics']['stopping_reason']}")
        print(f"Models and audit records saved: {output.resolve()}")
        return 0
    except (DatasetError, OSError, ValueError, TypeError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
