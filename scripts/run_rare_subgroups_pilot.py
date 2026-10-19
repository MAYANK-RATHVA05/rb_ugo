"""Preflight, or explicitly run/resume, the frozen rare-subgroup diagnostic pilot."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

from src.rare_subgroups_protocol import (DEFAULT_PROTOCOL, ROOT, array_hash, case_id,
                                        design_hash, generate_case, load_protocol, planned_jobs)
from src.rare_subgroups_report import export_reports
from src.rare_subgroups_study import evaluate_case
from src.results import ResultStore, content_hash


SOURCE_FILES = (
    "scripts/run_rare_subgroups_pilot.py", "src/rare_subgroups_protocol.py",
    "src/rare_subgroups_study.py", "src/rare_subgroups_report.py", "src/aw_smote.py",
    "src/_validation.py", "src/results.py", "requirements.txt", "requirements-dev.txt",
)


def preflight(config: dict) -> dict:
    jobs = planned_jobs(config)
    grid_size = len(config["aw_tuning"]["min_samples_grid"]) * len(config["aw_tuning"]["epsilon_percentile_grid"])
    max_fits = len(jobs) * (len(config["methods"]) - 1 + grid_size - 1)
    if (len(jobs) != config["budget"]["paired_cases"] or max_fits != config["budget"]["max_classifier_fits"]
            or len(jobs) * len(config["methods"]) != config["budget"]["planned_report_rows"]):
        raise ValueError("Design budget does not match the planned cases and model reuse.")
    return {"protocol_id": config["protocol_id"], "design_sha256": design_hash(config),
            "execution_enabled": config["execution_enabled"], "cases": len(jobs),
            "methods": config["methods"], "max_classifier_fits": max_fits,
            "planned_score_rows": len(jobs) * len(config["methods"]),
            "data_generated": False, "models_fitted": False, "output_created": False}


def _manifest(config: dict, config_path: Path) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    return {"study": "rare_subgroups_diagnostic_v1", "config": config, "design_sha256": design_hash(config),
            "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "git": {"commit": commit, "tracked_changes": bool(dirty)},
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn")},
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
            "jobs": planned_jobs(config), "not_author_paper_scores": True,
            "checkpoint_unit": "complete case; interrupted cases rerun, completed cases reuse validated records"}


def _save_predictions(directory: Path, job: dict, arrays: dict[str, np.ndarray]) -> dict:
    relative = Path("predictions") / f"case-{content_hash(job)}.npz"
    target = directory / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".pending-", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": relative.as_posix(), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def _verify_predictions(directory: Path, record: dict) -> None:
    artifact = record["predictions"]
    expected = f"predictions/case-{content_hash(record['job'])}.npz"
    if artifact["path"] != expected:
        raise ValueError("Cached prediction artifact has an unexpected path.")
    path = directory / artifact["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("Cached predictions are missing or corrupt; use an intact checkpoint or a new output directory.")
    successes = {r["method"]: r for r in record["rows"] if r["status"] == "success"}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"y", "groups", *successes}:
            raise ValueError("Cached prediction columns do not match successful methods.")
        for name in ("y", "groups"):
            if array_hash(saved[name]) != record["partitions_sha256"]["test"][name]:
                raise ValueError("Cached test metadata do not match their fingerprints.")
        for method, row in successes.items():
            if array_hash(saved[method]) != row["test_scores_sha256"]:
                raise ValueError("Cached score array does not match its fingerprint.")


def run_study(config: dict, config_path: Path, output: Path) -> dict:
    """One writer per output directory; this function enforces the execution gate."""
    if not config["execution_enabled"]:
        raise ValueError("Pilot execution is disabled in the protocol metadata; this stage prepares software only.")
    preflight(config)
    manifest = _manifest(config, config_path)
    if manifest["git"]["tracked_changes"]:
        raise ValueError("Freeze the runner/configuration in a commit before starting a research run.")
    store = ResultStore(output, manifest)
    records = {}
    expected_files = {store._path(job).name for job in planned_jobs(config)}
    if {path.name for path in output.glob("job-*.json")} - expected_files:
        raise ValueError("Output contains unexpected case checkpoints.")
    # Validate all durable checkpoints before fitting anything new. Reports then
    # reflect every saved case even if interruption happens early in a resume.
    for job in planned_jobs(config):
        saved = store.read(job)
        if saved is not None:
            _verify_predictions(output, saved)
            records[case_id(job)] = saved
    try:
        for job in planned_jobs(config):
            record = records.get(case_id(job))
            cached = record is not None
            if record is None:
                record, predictions = evaluate_case(generate_case(config, job), config, job)
                record["predictions"] = _save_predictions(output, job, predictions)
                record = store.write(record)
            records[case_id(job)] = record
            export_reports(output / "reports", records, config)
            print(f"{case_id(job)}: {record['case_status']}, fits={record['classifier_fit_attempts']}, cached={cached}", flush=True)
    finally:
        # Missing cases remain visible after a controlled interruption or failure.
        report = export_reports(output / "reports", records, config)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate and print the plan without generating data or files (default).")
    mode.add_argument("--run", action="store_true", help="Run/resume only when protocol execution_enabled is true.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=Path("results/rare-subgroups-pilot-v1"))
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
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed cases are checkpointed; repeat the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
