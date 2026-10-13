"""Bundle validation and exclusion of per-fold job records."""
import csv
import json
import zipfile

import pytest

from scripts import package_all_methods_30_reports as package


def _fixture(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "overview.json").write_text(json.dumps({"datasets": [
        {"dataset_id": "sample", "runs": {"with-u": {}, "without-u": {}}}]}))
    with (root / "all_methods_30_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "metric", "mean"])
        writer.writeheader()
        writer.writerow({"dataset_id": "sample", "metric": "f1", "mean": 0.5})
    reports = root / "sample" / "reports"
    reports.mkdir(parents=True)
    for name in package.REPORT_FILES:
        (reports / name).write_text("{}" if name.endswith("json") else "placeholder\n")
    with (reports / "coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset_id", "classifier", "method", "expected", "success", "failed",
            "unavailable", "missing", "all_methods_shared_success"])
        writer.writeheader()
        for classifier, method in sorted(package.EXPECTED_PAIRS):
            writer.writerow(dict(dataset_id="sample", classifier=classifier, method=method,
                                 expected=25, success=25, failed=0, unavailable=0, missing=0,
                                 all_methods_shared_success=25))
    jobs = root / "sample" / "with-u"
    jobs.mkdir()
    (jobs / "secret-job.json").write_text("do not bundle")
    return root


def test_review_bundle_includes_reports_and_excludes_jobs(tmp_path):
    root = _fixture(tmp_path)
    output = tmp_path / "review.zip"
    result = package.package_reports(root, output, ["sample"])
    assert result["dataset_reports"] == 1
    assert result["combined_score_rows"] == 1
    assert result["coverage_issues"] == []
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "review_manifest.json" in names
        assert "sample/reports/coverage.csv" in names
        assert not any("secret-job" in name for name in names)


def test_review_bundle_rejects_incomplete_reports(tmp_path):
    root = _fixture(tmp_path)
    (root / "sample" / "reports" / "scores.csv").unlink()
    with pytest.raises(FileNotFoundError, match="scores.csv"):
        package.package_reports(root, tmp_path / "review.zip", ["sample"])
