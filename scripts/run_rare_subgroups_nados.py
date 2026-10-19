"""Check the frozen NADOS diagnostic, or explicitly run it after authorization."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

from scripts.run_rare_subgroups_pilot import _save_predictions
from src.rare_subgroups_contamination import case_id
from src.rare_subgroups_nados_study import (
    DEFAULT_PROTOCOL,
    ROOT,
    evaluate_case,
    generate_case,
    load_protocol,
    planned_jobs,
    runtime_design,
    validate_protocol,
)
from src.rare_subgroups_protocol import array_hash
from src.results import ResultStore, content_hash


SOURCE_FILES = (
    "scripts/run_rare_subgroups_nados.py",
    "src/rare_subgroups_nados_study.py",
    "src/nados.py",
    "src/rare_subgroups_contamination.py",
    "src/rare_subgroups_protocol.py",
    "src/rare_subgroups_study.py",
    "src/results.py",
    "configs/rare_subgroups_contamination_v1.json",
    "docs/research/rare_subgroups/NADOS_BASELINE.md",
    "docs/research/rare_subgroups/NADOS_DIAGNOSTIC_PROTOCOL.md",
    "requirements.txt",
    "requirements-dev.txt",
)
BASE_VECTORS = {
    "calibration_y", "test_y", "test_groups", "calibration_ties", "test_ties",
    "fitting_y", "fitting_true_y", "fitting_groups", "fitting_corrupted", "fitting_stable_ids",
}


def preflight(config: dict) -> dict:
    validate_protocol(config)
    design = config["scientific_design"]
    jobs = planned_jobs(config)
    budget = design["budget"]
    fits = len(jobs)
    rows = len(jobs) * len(design["methods"])
    if (len(jobs) != budget["paired_cases"] or fits != budget["max_classifier_fits"]
            or rows != budget["planned_report_rows"] or budget["methods_per_case"] != 1):
        raise ValueError("NADOS case/model counts disagree with the frozen budget.")
    return {
        "protocol_id": design["protocol_id"],
        "design_sha256": config["design_sha256"],
        "parent_design_sha256": design["case_source"]["scientific_design_sha256"],
        "execution_enabled": config["execution_enabled"],
        "cases": len(jobs),
        "methods": design["methods"],
        "max_classifier_fits": fits,
        "planned_score_rows": rows,
        "data_generated": False,
        "models_fitted": False,
        "output_created": False,
    }


def _manifest(config: dict, config_path: Path) -> dict:
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Runtime configuration disagrees with the configuration file.")
    relative_config = config_path.resolve().relative_to(ROOT).as_posix()
    subprocess.check_output(
        ["git", "ls-files", "--error-unmatch", *SOURCE_FILES, relative_config], cwd=ROOT
    )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "study": "rare_subgroups_nados_v1",
        "config": config,
        "design_sha256": config["design_sha256"],
        "parent_design_sha256": config["scientific_design"]["case_source"]["scientific_design_sha256"],
        "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git": {"commit": commit, "tracked_changes": bool(dirty)},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn")},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCE_FILES},
        "jobs": planned_jobs(config),
        "not_author_paper_scores_or_novelty_evidence": True,
        "checkpoint_unit": "complete case; unfinished cases rerun, durable cases verified before fitting",
    }


def _verify_vectors(directory: Path, record: dict) -> None:
    artifact = record["predictions"]
    expected_path = f"predictions/case-{content_hash(record['job'])}.npz"
    if artifact["path"] != expected_path:
        raise ValueError("Unexpected cached vector path.")
    path = directory / artifact["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("Cached NADOS vectors are missing or corrupt.")
    row = record["rows"][0]
    expected = set(BASE_VECTORS) | set(row["trace_vectors"].values())
    if row["status"] == "success":
        required = {"calibration": "calibration__NADOS", "test": "test__NADOS"}
        if row["vectors"] != required:
            raise ValueError("Cached NADOS score-vector identity is invalid.")
        expected.update(required.values())
    elif row["vectors"]:
        raise ValueError("Failed NADOS row has unexpected held-out vectors.")
    if set(record["vector_sha256"]) != expected:
        raise ValueError("Cached vector fingerprint schema disagrees with the NADOS row.")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != expected:
            raise ValueError("Cached vector columns disagree with the NADOS row.")
        for name in expected:
            if array_hash(saved[name]) != record["vector_sha256"][name]:
                raise ValueError("Cached NADOS vector array is corrupt.")
        mapping = {
            "calibration_y": ("calibration", "y"),
            "test_y": ("test", "y"),
            "test_groups": ("test", "groups"),
            **{"fitting_" + name: ("fitting", name)
               for name in ("y", "true_y", "groups", "corrupted", "stable_ids")},
        }
        for key, (partition, field) in mapping.items():
            if record["vector_sha256"][key] != record["partitions_sha256"][partition][field]:
                raise ValueError("Cached truth/label vectors do not match partition fingerprints.")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=".pending-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_reports(directory: Path, records: dict[str, dict], config: dict) -> dict:
    """Write transparent progress tables; no screening claim is computed here."""
    jobs = planned_jobs(config)
    rows = []
    eligibility = []
    for job in jobs:
        record = records.get(case_id(job))
        if record is None:
            continue
        row = record["rows"][0]
        # Held-out metrics also contain ``rare_count`` (the fixed test-group
        # denominator). Keep the fitting design count under an unambiguous
        # name instead of allowing the metric to overwrite it.
        flat = {"fit_rare_count": job["rare_count"], "rare_sigma": job["rare_sigma"],
                "flipped_count": job["flipped_count"], "seed": job["seed"],
                "case_id": record["case_id"], "status": row["status"]}
        if row["metrics"] is not None:
            flat.update(row["metrics"])
        rows.append(flat)
        if row["eligibility"] is not None:
            for group, values in row["eligibility"].items():
                eligibility.append({**job, "case_id": record["case_id"], "group": group, **values})

    def csv_text(items: list[dict]) -> str:
        if not items:
            return ""
        fields = list(dict.fromkeys(key for item in items for key in item))
        from io import StringIO
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)
        return stream.getvalue()

    directory = Path(directory)
    _atomic_text(directory / "scores.csv", csv_text(rows))
    _atomic_text(directory / "eligibility.csv", csv_text(eligibility))
    report = {
        "status": "complete" if len(records) == len(jobs) else "incomplete",
        "completed_cases": len(records),
        "planned_cases": len(jobs),
        "successful_rows": sum(item["status"] == "success" for item in rows),
        "screening_status": "not_evaluated_by_runner",
    }
    _atomic_text(directory / "status.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_study(config: dict, config_path: Path, output: Path) -> dict:
    preflight(config)
    if not config["execution_enabled"]:
        raise ValueError("NADOS diagnostic execution is disabled; this is a software-only stage.")
    manifest = _manifest(config, config_path)
    if manifest["git"]["tracked_changes"]:
        raise ValueError("Freeze the NADOS configuration and implementation in a clean commit before execution.")
    output = Path(output)
    store = ResultStore(output, manifest)
    jobs = planned_jobs(config)
    expected_files = {store._path(job).name for job in jobs}
    if {path.name for path in output.glob("job-*.json")} - expected_files:
        raise ValueError("Unexpected case checkpoints in the NADOS output directory.")
    records: dict[str, dict] = {}
    for job in jobs:
        saved = store.read(job)
        if saved is not None:
            _verify_vectors(output, saved)
            records[case_id(job)] = saved
    try:
        for job in jobs:
            identifier = case_id(job)
            record = records.get(identifier)
            cached = record is not None
            if record is None:
                record, vectors = evaluate_case(generate_case(config, job), runtime_design(config), job)
                record["predictions"] = _save_predictions(output, job, vectors)
                record = store.write(record)
            records[identifier] = record
            export_reports(output / "reports", records, config)
            print(f"{identifier}: {record['case_status']}, fits={record['classifier_fit_attempts']}, "
                  f"cached={cached}", flush=True)
    finally:
        report = export_reports(output / "reports", records, config)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true",
                       help="Validate the plan without generating data or outputs (default).")
    modes.add_argument("--run", action="store_true",
                       help="Explicit execution, requiring enabled metadata and a clean commit.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=Path("results/rare-subgroups-nados-v1"))
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
        print("Interrupted; completed NADOS cases are checkpointed. Repeat the command to resume.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
