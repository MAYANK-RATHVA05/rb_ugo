import csv
import json

import pytest

from src.comparison_report import export_comparison_report, summarize_saved_run
from src.results import ResultStore


def saved_run(tmp_path):
    methods = ["nores", "smote", "bfch_smote"]
    manifest = {"dataset_id": "artificial_comparison_fixture", "positive_label": 1,
                "fold_protocol": "test-only paired folds", "metric_policy": {"version": "test"},
                "config": {"classifiers": ["GNB"], "methods": methods},
                "folds": [{"fold": fold} for fold in range(10)]}
    store = ResultStore(tmp_path, manifest)
    for fold in range(10):
        for method, offset in zip(methods, (0., .05, .1)):
            value = .4 + fold * .01 + offset
            store.write({"job": {"fold": fold, "classifier": "GNB", "method": method},
                         "status": "success",
                         "metrics": {**{metric: value for metric in
                                         ("iam", "mcc", "f1", "gmean", "auc")},
                                     "undefined": {metric: False for metric in
                                                   ("iam", "mcc", "f1", "gmean", "auc")}},
                         "resampler_diagnostics": ({"status": "generated"}
                                                   if method == "bfch_smote" else None),
                         "sample_counts": {"added_rows": 0 if method == "nores" else 5}})
    return store


def test_complete_report_uses_identical_folds_and_writes_table(tmp_path):
    store = saved_run(tmp_path / "run")
    report = summarize_saved_run(store.directory)
    assert report["status"] == "complete"
    assert report["run_id"] == store.run_id
    row = next(row for row in report["rows"]
               if row["method"] == "bfch_smote" and row["metric"] == "auc")
    assert row["shared_fold_ids"] == list(range(10))
    assert row["mean_difference_from_nores"] == pytest.approx(.1)
    assert row["mean"] == pytest.approx(.545)
    assert row["success"] == 10 and row["missing"] == 0
    assert next(entry for entry in report["coverage"] if entry["method"] == "bfch_smote")[
        "sampler_status_counts"] == {"generated": 10}
    assert len(report["rows"]) == 15
    csv_path, json_path = export_comparison_report(report, tmp_path / "report.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 15
    assert json.loads(exported[-1]["shared_fold_ids"]) == list(range(10))
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == store.run_id


def test_partial_report_exposes_failure_missing_job_and_common_subset(tmp_path):
    store = saved_run(tmp_path)
    failed_job = {"fold": 1, "classifier": "GNB", "method": "bfch_smote"}
    store.write({"job": failed_job, "status": "failed", "metrics": None,
                 "error": {"message": "intentional test failure"}})
    missing_job = {"fold": 2, "classifier": "GNB", "method": "smote"}
    store._path(missing_job).unlink()
    report = summarize_saved_run(tmp_path)
    assert report["status"] == "partial"
    coverage = {entry["method"]: entry for entry in report["coverage"]}
    assert coverage["bfch_smote"]["failed"] == 1
    assert coverage["smote"]["missing"] == 1
    for row in report["rows"]:
        assert row["shared_fold_ids"] == [0, 3, 4, 5, 6, 7, 8, 9]
        assert row["shared_fold_count"] == 8
    bfch_auc = next(row for row in report["rows"]
                    if row["method"] == "bfch_smote" and row["metric"] == "auc")
    assert bfch_auc["mean_difference_from_nores"] == pytest.approx(.1)


def test_old_scores_and_corrupt_records_fail_explicitly(tmp_path):
    store = saved_run(tmp_path)
    job = {"fold": 0, "classifier": "GNB", "method": "nores"}
    record = store.read(job)
    del record["metrics"]["auc"]
    store.write(record)
    with pytest.raises(ValueError, match="lacks auc"):
        summarize_saved_run(tmp_path)
    record["metrics"]["auc"] = .4
    store.write(record)
    path = store._path(job)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["record"]["status"] = "failed"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        summarize_saved_run(tmp_path)
