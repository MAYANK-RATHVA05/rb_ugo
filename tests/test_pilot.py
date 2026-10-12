import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import EvaluationConfig
from src.pilot import export_pilot_report, pilot_config, process_peak_memory, run_simulation_pilot, summarize_records


def records():
    result = []
    for fold in range(10):
        for method in ("nores", "ros"):
            value = fold / 10 + (.1 if method == "ros" else 0)
            result.append({"job": {"fold": fold, "classifier": "GNB", "method": method},
                           "status": "success", "metrics": {"iam": value, "mcc": value, "f1": value,
                                                               "undefined": {"iam": False, "mcc": False, "f1": False}},
                           "timings_seconds": {"total": 1.}, "sample_counts": {"added_rows": 2},
                           "resampler_diagnostics": None, "warnings": [], "error": None})
    return result


def test_paired_subset_uses_matching_baseline_folds_and_tracks_failures():
    rows = records()
    rows[1].update(status="failed", metrics=None, error={"message": "failed"})
    rows[2].update(status="unavailable", metrics=None, error={"message": "missing"})
    config = EvaluationConfig(classifiers=("GNB",), methods=("nores", "ros"))
    result = summarize_records(rows, config)[1]
    assert result["success"] == 9 and result["failed"] == 1
    metric = result["metrics"]["iam"]
    assert metric["paired_fold_ids"] == list(range(2, 10))
    assert metric["paired_count"] == 8
    assert metric["nores_mean"] == pytest.approx(.55)
    assert metric["method_mean"] == pytest.approx(.65)
    assert metric["mean_difference"] == pytest.approx(.1)
    assert metric["difference_sample_sd"] == pytest.approx(0)


def test_missing_scores_and_degenerate_zeros_remain_explicit():
    rows = records()
    rows[1]["metrics"]["iam"] = None
    rows[1]["metrics"]["undefined"]["iam"] = True
    rows[3]["metrics"]["mcc"] = 0.
    rows[3]["metrics"]["undefined"]["mcc"] = True
    config = EvaluationConfig(classifiers=("GNB",), methods=("nores", "ros"))
    result = summarize_records(rows, config)[1]["metrics"]
    assert result["iam"]["paired_count"] == 9
    assert result["mcc"]["paired_count"] == 10
    assert result["mcc"]["undefined_method_scores_included"] == 1
    assert result["mcc"]["mean_difference"] == pytest.approx(.08)


@pytest.mark.parametrize("change", ["duplicate", "missing", "nonterminal"])
def test_summary_rejects_wrong_coverage(change):
    rows = records()
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    else:
        rows[0]["status"] = "running"
    with pytest.raises(ValueError):
        summarize_records(rows, EvaluationConfig(classifiers=("GNB",), methods=("nores", "ros")))


def small_spec():
    spec = json.loads((Path(__file__).resolve().parents[1] / "configs/stage4r6_pilot.json").read_text())
    spec["classifiers"] = ["GNB"]
    spec["methods"] = ["nores", "ros"]
    spec["method_parameters"] = {}
    spec["simulations"]["linear"].update(n_majority=80, n_minority=20)
    return spec


def test_real_pilot_report_resume_resources_and_export(tmp_path):
    spec = small_spec()
    root, export = tmp_path / "runs", tmp_path / "report"
    report = run_simulation_pilot(spec, root)
    assert report["current_invocation"]["expected_jobs"] == 40
    assert report["current_invocation"]["executed_jobs"] == 40
    assert report["current_invocation"]["memory_after"]["bytes"] > 0
    assert all(pool["num_threads"] == 1 for pool in report["current_invocation"]["native_thread_pools"])
    assert [d["positive_label"] for d in report["datasets"]] == [1, 0]
    assert all(s["success"] == 10 for d in report["datasets"] for s in d["summaries"])
    export_pilot_report(report, export)
    summary_before = (export / "summary.json").read_bytes()
    csv_before = (export / "paired_scores.csv").read_bytes()
    resumed = run_simulation_pilot(spec, root)
    assert resumed["current_invocation"]["reused_jobs"] == 40
    assert resumed["current_invocation"]["executed_jobs"] == 0
    assert resumed["first_complete_invocation"] == report["first_complete_invocation"]
    export_pilot_report(resumed, export)
    assert (export / "summary.json").read_bytes() == summary_before
    assert (export / "paired_scores.csv").read_bytes() == csv_before
    assert len(csv_before.decode().splitlines()) == 13  # 2 datasets * 2 methods * 3 metrics + header
    changed = copy.deepcopy(spec)
    changed["seed"] = 43
    with pytest.raises(ValueError, match="different configuration"):
        run_simulation_pilot(changed, root)
    report["datasets"][0]["run_id"] = "different"
    with pytest.raises(ValueError, match="different run identities"):
        export_pilot_report(report, export)


def test_unsupported_memory_measurement_is_explicit(monkeypatch):
    monkeypatch.setattr("src.pilot.platform.system", lambda: "unsupported")
    assert process_peak_memory()["bytes"] is None
    assert process_peak_memory()["status"] == "unsupported platform"


def test_fixed_pilot_validation():
    spec = small_spec()
    spec["native_threads"] = 4
    with pytest.raises(ValueError, match="one native thread"):
        pilot_config(spec)
