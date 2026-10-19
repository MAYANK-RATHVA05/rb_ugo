"""Check the frozen contamination plan, or explicitly run/resume after authorization."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from scripts.run_rare_subgroups_pilot import _save_predictions
from src.rare_subgroups_contamination import (
    DEFAULT_PROTOCOL, ROOT, case_id, generate_case, load_protocol, planned_jobs, validate_protocol,
)
from src.rare_subgroups_contamination_report import export_reports, summarize_cases
from src.rare_subgroups_contamination_study import candidate_id, evaluate_case
from src.rare_subgroups_protocol import array_hash
from src.results import ResultStore, content_hash


SOURCE_FILES = (
    "scripts/run_rare_subgroups_contamination.py", "src/rare_subgroups_contamination.py",
    "src/rare_subgroups_contamination_study.py", "src/rare_subgroups_contamination_report.py",
    "scripts/run_rare_subgroups_pilot.py", "src/rare_subgroups_protocol.py", "src/rare_subgroups_study.py",
    "src/rare_subgroups_report.py", "src/aw_smote.py", "src/_validation.py", "src/results.py",
    "requirements.txt", "requirements-dev.txt",
)
BASE_VECTORS = {"selection_y", "calibration_y", "test_y", "test_groups", "calibration_ties", "test_ties",
                "fitting_y", "fitting_true_y", "fitting_groups", "fitting_corrupted", "fitting_stable_ids"}


def preflight(config: dict) -> dict:
    validate_protocol(config)
    design = config["scientific_design"]
    jobs, budget = planned_jobs(design), design["budget"]
    grid = len(design["aw_tuning"]["min_samples_grid"]) * len(design["aw_tuning"]["epsilon_percentile_grid"])
    fits = len(jobs) * (len(design["methods"]) - 1 + grid - 1)
    if (len(jobs) != budget["paired_cases"] or fits != budget["max_classifier_fits"]
            or len(jobs) * len(design["methods"]) != budget["planned_report_rows"]
            or len(jobs) * grid != budget["candidate_records"]):
        raise ValueError("Contamination case/model counts disagree with the frozen budget.")
    return {"protocol_id": design["protocol_id"], "design_sha256": config["design_sha256"],
            "execution_enabled": config["execution_enabled"], "cases": len(jobs), "methods": design["methods"],
            "max_classifier_fits": fits, "planned_score_rows": budget["planned_report_rows"],
            "candidate_records": budget["candidate_records"],
            "data_generated": False, "models_fitted": False, "output_created": False}


def _manifest(config: dict, config_path: Path) -> dict:
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Runtime configuration disagrees with the committed configuration file.")
    relative_config = config_path.resolve().relative_to(ROOT).as_posix()
    # Research inputs must be in the committed checkout, including a custom
    # metadata-only configuration. Untracked source cannot masquerade as frozen.
    subprocess.check_output(["git", "ls-files", "--error-unmatch", *SOURCE_FILES, relative_config], cwd=ROOT)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"study": "rare_subgroups_contamination_v1", "config": config,
            "design_sha256": config["design_sha256"],
            "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "git": {"commit": commit, "tracked_changes": bool(dirty)},
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn")},
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
            "jobs": planned_jobs(config["scientific_design"]), "not_author_paper_scores_or_novelty_evidence": True,
            "checkpoint_unit": "complete case; unfinished cases rerun, durable cases verified before fitting"}


def _verify_vectors(directory: Path, record: dict, design: dict) -> None:
    artifact = record["predictions"]
    if artifact["path"] != f"predictions/case-{content_hash(record['job'])}.npz":
        raise ValueError("Unexpected cached vector path.")
    path = directory / artifact["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("Cached probability vectors are missing or corrupt.")
    expected = set(BASE_VECTORS)
    identifiers = {candidate_id(m, p) for m in design["aw_tuning"]["min_samples_grid"]
                   for p in design["aw_tuning"]["epsilon_percentile_grid"]}
    candidates = record["selection_candidates"]
    if len(candidates) != len(identifiers) or {c["candidate_id"] for c in candidates} != identifiers:
        raise ValueError("Cached candidate grid is incomplete or duplicated.")
    for candidate in candidates:
        vector = candidate["selection_vector"]
        if candidate["status"] == "success":
            if vector != "selection__" + candidate["candidate_id"]:
                raise ValueError("Cached selection-vector identity is invalid.")
            expected.add(vector)
        elif vector is not None:
            raise ValueError("Failed candidate has an unexpected selection vector.")
    for row in record["rows"]:
        vectors = row["vectors"]
        if row["status"] == "success":
            required = {"calibration": "calibration__" + row["method"], "test": "test__" + row["method"]}
            if vectors != required:
                raise ValueError("Cached method-vector identity is invalid.")
            expected.update(vectors.values())
        elif vectors:
            raise ValueError("Failed method has unexpected held-out vectors.")
    if set(record["vector_sha256"]) != expected:
        raise ValueError("Cached vector fingerprint schema disagrees with case statuses.")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != expected:
            raise ValueError("Cached vector columns disagree with case statuses.")
        for name in expected:
            if array_hash(saved[name]) != record["vector_sha256"][name]:
                raise ValueError("Cached vector array is corrupt.")
        mapping = {"selection_y": ("selection", "y"), "calibration_y": ("calibration", "y"),
                   "test_y": ("test", "y"), "test_groups": ("test", "groups"),
                   **{"fitting_" + name: ("fitting", name) for name in ("y", "true_y", "groups", "corrupted", "stable_ids")}}
        for key, (partition, field) in mapping.items():
            if record["vector_sha256"][key] != record["partitions_sha256"][partition][field]:
                raise ValueError("Cached truth/label vectors do not match partition fingerprints.")


def run_study(config: dict, config_path: Path, output: Path) -> dict:
    preflight(config)
    if not config["execution_enabled"]:
        raise ValueError("Contamination execution is disabled; this is a software-only stage.")
    manifest = _manifest(config, config_path)
    if manifest["git"]["tracked_changes"]:
        raise ValueError("Freeze the configuration and implementation in a clean commit before research execution.")
    design, output = config["scientific_design"], Path(output)
    store = ResultStore(output, manifest)
    jobs, records = planned_jobs(design), {}
    if {path.name for path in output.glob("job-*.json")} - {store._path(job).name for job in jobs}:
        raise ValueError("Unexpected case checkpoints in output directory.")
    for job in jobs:
        record = store.read(job)
        if record is not None:
            _verify_vectors(output, record, design)
            records[case_id(job)] = record
    summarize_cases(records, design)  # validate every cached row before new fits
    try:
        for job in jobs:
            record = records.get(case_id(job))
            cached = record is not None
            if record is None:
                record, vectors = evaluate_case(generate_case(design, job), design, job)
                record["predictions"] = _save_predictions(output, job, vectors)
                record = store.write(record)
            records[case_id(job)] = record
            export_reports(output / "reports", records, design)
            print(f"{case_id(job)}: {record['case_status']}, fits={record['classifier_fit_attempts']}, cached={cached}", flush=True)
    finally:
        report = export_reports(output / "reports", records, design)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Validate the plan without generating data or outputs (default).")
    modes.add_argument("--run", action="store_true", help="Explicit execution, requiring enabled metadata and a clean commit.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=Path("results/rare-subgroups-contamination-v1"))
    args = parser.parse_args(argv)
    try:
        config = load_protocol(args.config)
        plan = preflight(config)
        if not args.run:
            print(json.dumps(plan, indent=2))
            return 0
        report = run_study(config, args.config, args.output)
        print(f"Reports: {args.output / 'reports'} ({report['status']})")
        return 0 if report["status"] == "complete" else 1
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed cases are checkpointed. Repeat the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
