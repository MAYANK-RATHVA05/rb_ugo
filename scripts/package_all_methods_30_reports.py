"""Package completed 30-study aggregate reports for review, without raw jobs."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data/manifests/all_methods_30.json"
REPORT_FILES = ("scores.csv", "coverage.csv", "diagnostics.csv", "clusters.csv",
                "ablation.csv", "summary.json")
EXPECTED_PAIRS = {(classifier, method)
                  for classifier in ("LogReg", "GNB")
                  for method in ("nores", "smote", "ugo_smote", "bfch_smote",
                                 "ec_ebo", "sshr")}


def selected_names() -> list[str]:
    names = json.loads(SELECTION.read_text(encoding="utf-8"))["datasets"]
    if len(names) != 30 or len(set(names)) != 30:
        raise ValueError("The fixed selection must contain 30 distinct datasets.")
    return names


def package_reports(input_dir: Path, output: Path, names: list[str]) -> dict:
    """Validate coverage and make a portable ZIP of aggregate reports only."""
    combined = input_dir / "all_methods_30_scores.csv"
    overview = input_dir / "overview.json"
    required = [combined, overview]
    for name in names:
        required.extend(input_dir / name / "reports" / report for report in REPORT_FILES)
    missing = [str(path.relative_to(input_dir)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Expected {len(names)} completed dataset report folders; missing "
            f"{len(missing)} file(s): {', '.join(missing[:8])}"
        )

    run = json.loads(overview.read_text(encoding="utf-8"))
    reported = {item["dataset_id"]: item for item in run["datasets"]}
    if (len(reported) != len(names) or set(reported) != set(names)
            or any(set(reported[name]["runs"]) != {"with-u", "without-u"} for name in names)):
        raise ValueError("overview.json does not contain both completed runs for all 30 datasets.")

    with combined.open(newline="", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    score_names = {row["dataset_id"] for row in score_rows}
    if not score_names <= set(names):
        raise ValueError("Combined scores contain a dataset outside the fixed selection.")
    coverage_issues = []
    for name in names:
        with (input_dir / name / "reports" / "coverage.csv").open(
                newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(EXPECTED_PAIRS) or {
                (row["classifier"], row["method"]) for row in rows} != EXPECTED_PAIRS:
            raise ValueError(f"Incomplete or duplicated method coverage rows for {name}.")
        for row in rows:
            if (row["dataset_id"] != name or int(row["expected"]) != 25
                    or int(row["success"]) != 25 or int(row["failed"]) != 0
                    or int(row["unavailable"]) != 0 or int(row["missing"]) != 0
                    or int(row["all_methods_shared_success"]) != 25):
                coverage_issues.append({"dataset_id": name, "classifier": row["classifier"],
                                        "method": row["method"], "success": row["success"],
                                        "failed": row["failed"], "unavailable": row["unavailable"],
                                        "missing": row["missing"],
                                        "shared_folds": row["all_methods_shared_success"]})

    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(required, key=lambda path: path.relative_to(input_dir).as_posix())
    inventory = {path.relative_to(input_dir).as_posix(): sha256(path.read_bytes()).hexdigest()
                 for path in files}
    bundle = {"schema_version": 1,
              "purpose": "Aggregate 30-study review; no raw data, per-fold jobs, or predictions",
              "selected_datasets": names, "dataset_reports": len(names),
              "combined_score_rows": len(score_rows),
              "datasets_missing_combined_scores": sorted(set(names) - score_names),
              "coverage_issues": coverage_issues, "files_sha256": inventory}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        archive.writestr("review_manifest.json", json.dumps(bundle, indent=2) + "\n")
        for path in files:
            archive.write(path, arcname=path.relative_to(input_dir).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Review ZIP failed CRC verification.")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/all-methods-30")
    parser.add_argument("--output", type=Path, default=ROOT / "results/all-methods-30-review.zip")
    args = parser.parse_args()
    bundle = package_reports(args.input, args.output, selected_names())
    print(f"Review ZIP: {args.output}")
    print(f"Datasets with reports: {bundle['dataset_reports']}/30")
    print(f"Combined score rows: {bundle['combined_score_rows']}")
    print(f"Coverage rows needing review: {len(bundle['coverage_issues'])}")
    print(f"Datasets absent from combined scores: "
          f"{', '.join(bundle['datasets_missing_combined_scores']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
