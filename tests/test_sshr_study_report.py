"""Hand-built saved software evidence; no real-data performance assertions."""
from copy import deepcopy
import csv
import json

import numpy as np
import pytest

from src.metrics import binary_metrics
from src.results import ResultStore, content_hash
from src.sshr_study_report import (
    export_study_reports,
    summarize_controlled_run,
    summarize_unlabeled_ablation,
)


def proxy(rows, majority=0, ties=0):
    return {"rows": rows, "nearest_majority_count": majority,
            "nearest_majority_rate": majority / rows if rows else None,
            "distance_tie_count": ties}


def fixture_run(path, *, use_unlabeled=True, methods=("nores", "sshr"), classifiers=("LogReg",),
                qualities=None, selections=(0, 1, 2, 1), correct=(0, 1, 1, 0)):
    folds, masks = [], []
    for repeat in range(2):
        for fold in range(2):
            testing = list(range(4 * fold, 4 * fold + 4))
            training = sorted(set(range(8)) - set(testing))
            folds.append({"repeat": repeat, "fold": fold, "train_indices": training, "test_indices": testing})
            masks.append({"repeat": repeat, "fold": fold, "labeled_indices": training[:2], "unlabeled_indices": training[2:]})
    manifest = {
        "stage": "SSHR-3", "schema_version": 1, "dataset_id": "hand_software_fixture", "rows": 8,
        "labels": [0, 1], "positive_label": 1, "input_sha256": "fixture_input_hash",
        "provenance": {"kind": "hand-built software fixture; not experimental evidence", "source_sha256": "a" * 64},
        "folds": folds, "masks": masks, "folds_sha256": content_hash(folds), "masks_sha256": content_hash(masks),
        "software": {"source_sha256": "b" * 64}, "preprocessing_policy": {"fit_pool": "observed L only"},
        "metric_policy": {"probability_auc": "probability ROC AUC"},
        "config": {"methods": list(methods), "classifiers": list(classifiers), "n_splits": 2, "n_repeats": 2,
                   "seed": 42, "scale": True, "labeled_fraction": 0.5, "min_labeled_per_class": 2,
                   "use_unlabeled": use_unlabeled, "method_parameters": {}},
    }
    store = ResultStore(path, manifest)
    qualities = qualities or ([0, 1, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 0])
    for index, (fold, mask) in enumerate(zip(folds, masks)):
        for classifier in classifiers:
            for method in methods:
                job = {"repeat": fold["repeat"], "fold": fold["fold"], "classifier": classifier, "method": method}
                prediction = list(qualities[index]) if method == "sshr" else [0, 0, 0, 1]
                probabilities = [[0.2, 0.8] if value else [0.8, 0.2] for value in prediction]
                metrics = binary_metrics([0, 1, 0, 1], prediction, negative_label=0, positive_label=1,
                                         positive_probabilities=[row[1] for row in probabilities])
                metrics["probability_auc"] = metrics["auc"]
                metrics["fpr"] = metrics["counts"]["fp"] / 2
                record = {
                    "job": job, "status": "success", "seed": index + 1,
                    "train_indices": fold["train_indices"], "test_indices": fold["test_indices"],
                    "labeled_indices": mask["labeled_indices"], "unlabeled_indices": mask["unlabeled_indices"],
                    "algorithm_unlabeled_indices": mask["unlabeled_indices"] if method == "sshr" and use_unlabeled else [],
                    "positive_label": 1, "probability_labels": [0, 1], "true_labels": [0, 1, 0, 1],
                    "predictions": prediction, "probabilities": probabilities, "metrics": metrics,
                    "preprocessing": {"training_rows": 2}, "resampler_parameters": {"random_state": index + 1},
                    "pseudo_label_audit": None, "row_lineage": None, "resampler_diagnostics": None,
                    "sample_counts": {"labeled_training": 2, "resampled_training": 2, "retained_labeled": 2,
                                      "selected_pseudo_labeled": 0, "generated_synthetic": 0, "generated_minority": 0,
                                      "generated_majority": 0, "removed_majority": 0, "removed_minority": 0},
                    "training_overlap_proxy": {"synthetic_all": proxy(0), "synthetic_minority": proxy(0), "synthetic_majority": proxy(0)},
                }
                if method == "sshr":
                    selected = selections[index] if use_unlabeled else 0
                    generated = 2 - selected
                    counts = record["sample_counts"]
                    counts.update(selected_pseudo_labeled=selected, generated_synthetic=generated,
                                  generated_smote=generated, generated_minority=generated, resampled_training=4)
                    source_rows = mask["unlabeled_indices"][:selected]
                    record["pseudo_label_audit"] = {
                        "selected_count": selected, "correct_count": correct[index] if selected else 0,
                        "incorrect_count": selected - correct[index] if selected else 0,
                        "accuracy": correct[index] / selected if selected else None, "selected_global_indices": source_rows}
                    record["row_lineage"] = ([{"kind": "labeled"}] * 2
                                             + [{"kind": "pseudo_labeled"}] * selected + [{"kind": "smote"}] * generated)
                    record["training_overlap_proxy"] = {"pseudo_labeled": proxy(selected, int(selected > 0)), "smote": proxy(generated)}
                    record["resampler_diagnostics"] = {
                        "status": "quotas_met", "counts": {}, "shortfalls": [],
                        "quotas": {"weighted_removal_requested": 0, "weighted_removal_actual": 0, "weighted_removal_unmet": 0,
                                   "additions_requested": 2, "pseudo_selected": selected, "smote_generated": generated,
                                   "additions_unmet": 0, "smote_fallback_requested": generated},
                        "clusters": [{"id": 0, "label": 0, "size": 1, "labeled_size": 1, "unlabeled_size": 0,
                                      "observed_majority_count": 1, "observed_minority_count": 0, "unanchored": False, "impure": False},
                                     {"id": 1, "label": 1, "size": 3 if use_unlabeled else 1, "labeled_size": 1,
                                      "unlabeled_size": 2 if use_unlabeled else 0, "observed_majority_count": 0,
                                      "observed_minority_count": 1, "unanchored": False, "impure": False}],
                        "smote_allocations": [{"cluster_id": 1, "pool_size": 1 + selected, "requested": generated,
                                               "generated": generated, "shortage_reason": None}],
                    }
                store.write(record)
    return store


def rewrite(store, method, repeat=0, fold=0, classifier="LogReg", **changes):
    job = {"repeat": repeat, "fold": fold, "classifier": classifier, "method": method}
    record = store.read(job)
    record.update(changes)
    store.write(record)
    return record


def score(report, *, method="sshr", metric="f1", scope="all_methods_shared_success"):
    return next(row for row in report["scores"] if row["method"] == method and row["metric"] == metric and row["scope"] == scope)


def test_hand_arithmetic_repeated_folds_sample_sd_and_paired_nores(tmp_path):
    fixture_run(tmp_path)
    report = summarize_controlled_run(tmp_path)
    row = score(report)
    expected = [1, 2 / 3, 4 / 5, 0]
    assert row["mean"] == pytest.approx(sum(expected) / 4)
    assert row["sample_sd"] == pytest.approx(np.std(expected, ddof=1))
    assert row["nores_mean"] == pytest.approx(2 / 3)
    assert row["paired_mean_difference"] == pytest.approx(sum(expected) / 4 - 2 / 3)
    assert row["fold_ids"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert row["n_folds"] == row["n_paired_with_nores"] == 4
    assert "dependent" in report["policy"]
    assert all(row["success"] == 4 for row in report["coverage"])


def test_common_fold_intersection_and_separate_all_success_baseline(tmp_path):
    store = fixture_run(tmp_path, methods=("nores", "smote", "sshr"))
    rewrite(store, "sshr", status="failed", metrics=None, error={"stage": "resampling", "type": "ValueError", "message": "fixture"})
    store._path({"repeat": 1, "fold": 1, "classifier": "LogReg", "method": "smote"}).unlink()
    report = summarize_controlled_run(tmp_path)
    assert score(report)["fold_ids"] == [[0, 1], [1, 0]]
    assert score(report, method="nores", scope="all_method_success")["n_folds"] == 4
    assert score(report, scope="all_method_success")["n_folds"] == 3
    assert next(row for row in report["coverage"] if row["method"] == "smote")["missing"] == 1
    failed = next(row for row in report["coverage"] if row["method"] == "sshr")
    assert failed["failed"] == 1 and failed["issues"][0]["error"]["message"] == "fixture"


def test_unavailable_and_zero_shared_fold_scores_remain_explicit(tmp_path):
    store = fixture_run(tmp_path)
    for repeat in range(2):
        for fold in range(2):
            rewrite(store, "sshr", repeat, fold, status="unavailable", metrics=None)
    report = summarize_controlled_run(tmp_path)
    row = score(report)
    assert row["mean"] is row["sample_sd"] is row["nores_mean"] is None
    assert row["n_folds"] == 0
    assert score(report, method="nores", scope="all_method_success")["n_folds"] == 4
    assert next(row for row in report["coverage"] if row["method"] == "sshr")["unavailable"] == 4


def test_weighted_pseudo_accuracy_counts_and_overlap_by_classifier(tmp_path):
    fixture_run(tmp_path, classifiers=("LogReg", "CART"), correct=(0, 1, 2, 0))
    report = summarize_controlled_run(tmp_path)
    rows = [row for row in report["diagnostics"] if row["method"] == "sshr"]
    assert len(rows) == 2
    for row in rows:
        assert row["totals"]["selected_pseudo_labeled"] == 4
        assert row["totals"]["generated_synthetic"] == 4
        assert row["pseudo_label_event_weighted_accuracy"] == 0.75
        assert row["pseudo_label_mean_nonempty_fold_accuracy"] == pytest.approx(2 / 3)
        assert row["pseudo_label_nonempty_fold_events"] == 3
        assert row["training_overlap_proxy"]["pseudo_labeled"]["nearest_majority_rate"] == 0.75
        assert "not unique" in row["counts_scope"]
    assert len(report["clusters"]) == 16
    assert all(row["minority_anchored"] for row in report["clusters"] if row["label"] == 1)


def test_sampler_shortfalls_and_stops_include_failed_diagnostics(tmp_path):
    store = fixture_run(tmp_path, methods=("nores", "ugo_smote", "bfch_smote", "ec_ebo", "sshr"))
    rewrite(store, "ugo_smote", resampler_diagnostics={"stopping_reason": "generation_cap"})
    rewrite(store, "bfch_smote", resampler_diagnostics={"status": "insufficient_minority_boundary"})
    rewrite(store, "ec_ebo", resampler_diagnostics={"status": "partial", "shortfall_count": 3,
                                                 "cluster_shortfall_reasons": [{"overlap_rejection": 3}]})
    rewrite(store, "sshr", status="failed", metrics=None,
            resampler_diagnostics={"shortfalls": [{"stage": "weighted_removal", "cause": "no_centroid", "count": 2}]})
    report = summarize_controlled_run(tmp_path)
    d = {row["method"]: row for row in report["diagnostics"]}
    assert d["ugo_smote"]["sampler_stopping_reason_counts"] == {"generation_cap": 1}
    assert d["bfch_smote"]["sampler_status_counts"] == {"insufficient_minority_boundary": 1}
    assert d["ec_ebo"]["totals"]["sampler_shortfall_count"] == 3
    assert d["ec_ebo"]["shortfall_causes_all_saved_jobs"] == {"ec_ebo:overlap_rejection": 3}
    assert d["sshr"]["shortfall_causes_all_saved_jobs"] == {"weighted_removal:no_centroid": 2}


def test_duplicate_counts_and_failed_removals_have_explicit_reporting_coverage(tmp_path):
    store = fixture_run(tmp_path)
    record = store.read({"repeat": 0, "fold": 0, "classifier": "LogReg", "method": "sshr"})
    record["resampler_diagnostics"]["coordinate_duplicates"] = {
        "comparison_policy": "exact coordinate equality; not row provenance",
        "input_labeled_duplicate_coordinates": 2,
        "smote_duplicate_with_retained_or_pseudo_or_prior_smote": 0}
    store.write(record)
    rewrite(store, "sshr", repeat=1, fold=1, status="failed", metrics=None,
            resampler_diagnostics={"status": "failure_class_erased_by_preliminary_cleaning",
                                   "counts": {"removed_majority": 3, "removed_minority": 1}})
    report = summarize_controlled_run(tmp_path)
    row = next(row for row in report["diagnostics"] if row["method"] == "sshr")
    assert row["coordinate_duplicate_fold_event_counts"] == {
        "input_labeled_duplicate_coordinates": 2, "smote_duplicate_with_retained_or_pseudo_or_prior_smote": 0}
    assert row["coordinate_duplicate_reporting_coverage"]["input_labeled_duplicate_coordinates"] == {
        "reported_successful_fold_events": 1, "missing_successful_fold_events": 2}
    assert row["failed_job_removal_totals"] == {"removed_majority": 3, "removed_minority": 1}
    assert row["failed_job_removal_reporting_fold_counts"] == {"removed_majority": 1, "removed_minority": 1}
    assert "retained_labeled" not in row["failed_job_removal_totals"]


def test_majority_synthetic_rows_are_reported_separately(tmp_path):
    store = fixture_run(tmp_path, methods=("nores", "ugo_smote", "sshr"))
    record = store.read({"repeat": 0, "fold": 0, "classifier": "LogReg", "method": "ugo_smote"})
    record["sample_counts"].update(resampled_training=3, generated_synthetic=1, generated_majority=1)
    record["training_overlap_proxy"].update(synthetic_all=proxy(1, 1), synthetic_majority=proxy(1, 1))
    store.write(record)
    report = summarize_controlled_run(tmp_path)
    row = next(row for row in report["diagnostics"] if row["method"] == "ugo_smote")
    assert row["totals"]["generated_minority"] == 0
    assert row["totals"]["generated_majority"] == 1
    assert row["training_overlap_proxy"]["synthetic_minority"]["nearest_majority_rate"] is None


@pytest.mark.parametrize("field,value", [("f1", None), ("mcc", "0.0"), ("fpr", 0.6), ("probability_auc", True)])
def test_successful_score_corruption_is_rejected_even_when_hash_is_rewritten(tmp_path, field, value):
    store = fixture_run(tmp_path)
    record = store.read({"repeat": 0, "fold": 0, "classifier": "LogReg", "method": "sshr"})
    record["metrics"][field] = value
    store.write(record)
    with pytest.raises(ValueError, match="finite numeric|disagrees"):
        summarize_controlled_run(tmp_path)


def test_record_hash_and_manifest_hash_integrity(tmp_path):
    fixture_run(tmp_path)
    path = next(tmp_path.glob("job-*.json"))
    envelope = json.loads(path.read_text())
    envelope["record"]["status"] = "failed"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="corrupt"):
        summarize_controlled_run(tmp_path)
    manifest = tmp_path / "manifest.json"
    envelope = json.loads(manifest.read_text())
    envelope["manifest"]["dataset_id"] = "different"
    manifest.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="manifest hash"):
        summarize_controlled_run(tmp_path)


def test_unexpected_job_is_rejected(tmp_path):
    store = fixture_run(tmp_path)
    store.write({"job": {"repeat": 3, "fold": 0, "classifier": "LogReg", "method": "sshr"}, "status": "failed"})
    with pytest.raises(ValueError, match="Unexpected saved job"):
        summarize_controlled_run(tmp_path)


@pytest.mark.parametrize("fault", ["row_conservation", "pseudo_audit", "pseudo_membership", "lineage", "quota", "proxy", "test_indices", "algorithm_u"])
def test_successful_accounting_and_training_identity_fail_loudly(tmp_path, fault):
    store = fixture_run(tmp_path)
    record = store.read({"repeat": 0, "fold": 1, "classifier": "LogReg", "method": "sshr"})
    if fault == "row_conservation":
        record["sample_counts"]["resampled_training"] += 1
    elif fault == "pseudo_audit":
        record["pseudo_label_audit"]["correct_count"] += 1
    elif fault == "pseudo_membership":
        record["pseudo_label_audit"]["selected_global_indices"] = record["test_indices"][:1]
    elif fault == "lineage":
        record["row_lineage"][0]["kind"] = "smote"
    elif fault == "quota":
        record["resampler_diagnostics"]["quotas"]["additions_requested"] += 1
    elif fault == "proxy":
        record["training_overlap_proxy"]["smote"]["nearest_majority_rate"] = None
    elif fault == "test_indices":
        record["test_indices"] = list(reversed(record["test_indices"]))
    else:
        record["algorithm_unlabeled_indices"] = record["test_indices"]
    store.write(record)
    with pytest.raises(ValueError):
        summarize_controlled_run(tmp_path)


def test_unlabeled_ablation_pairs_and_no_unique_event_pooling(tmp_path):
    fixture_run(tmp_path / "with")
    fixture_run(tmp_path / "without", use_unlabeled=False, qualities=([0, 0, 0, 1],) * 4)
    report = summarize_unlabeled_ablation(tmp_path / "with", tmp_path / "without")
    row = next(row for row in report["rows"] if row["metric"] == "f1")
    assert row["n_paired_folds"] == 4
    assert row["with_u_mean"] == pytest.approx((1 + 2 / 3 + 4 / 5) / 4)
    assert row["without_u_mean"] == pytest.approx(2 / 3)
    assert row["paired_mean_difference"] == pytest.approx(row["with_u_mean"] - row["without_u_mean"])
    assert row["paired_fold_event_counts"]["with_u"]["selected_pseudo_labeled"] == 4
    assert row["paired_fold_event_counts"]["without_u"]["selected_pseudo_labeled"] == 0
    assert report["coverage"][0]["paired_success"] == 4


def test_ablation_failed_pairs_are_visible_and_not_scored(tmp_path):
    fixture_run(tmp_path / "with")
    store = fixture_run(tmp_path / "without", use_unlabeled=False)
    rewrite(store, "sshr", status="failed", metrics=None, error={"stage": "resampling"})
    report = summarize_unlabeled_ablation(tmp_path / "with", tmp_path / "without")
    assert report["rows"][0]["n_paired_folds"] == 3
    assert report["coverage"][0]["issues"][0]["run"] == "without_u"


def test_ablation_missing_duplicated_baseline_is_explicit(tmp_path):
    fixture_run(tmp_path / "with")
    store = fixture_run(tmp_path / "without", use_unlabeled=False)
    job = {"repeat": 0, "fold": 0, "classifier": "LogReg", "method": "nores"}
    store._path(job).unlink()
    report = summarize_unlabeled_ablation(tmp_path / "with", tmp_path / "without")
    assert report["coverage"][0]["identical_saved_nores_pairs"] == 3
    assert report["coverage"][0]["issues"][0]["method"] == "nores"
    assert report["coverage"][0]["issues"][0]["status"] == "missing"


@pytest.mark.parametrize("field", ["probabilities", "metrics", "predictions", "preprocessing", "seed"])
def test_ablation_requires_identical_duplicated_nores_evidence(tmp_path, field):
    fixture_run(tmp_path / "with")
    store = fixture_run(tmp_path / "without", use_unlabeled=False)
    record = store.read({"repeat": 0, "fold": 0, "classifier": "LogReg", "method": "nores"})
    if field == "probabilities":
        record[field] = [[0.9, 0.1] if row[1] == 0.2 else [0.1, 0.9]
                         for row in record[field]]  # same ranking/ties, changed probabilities
    elif field == "metrics":
        record[field]["extra_diagnostic"] = "changed"
    elif field == "predictions":
        # Changing predictions would otherwise violate the score contract.
        record["status"] = "failed"
        record[field] = None
    elif field == "preprocessing":
        record[field]["training_rows"] = 3
    else:
        record[field] += 1
    store.write(record)
    with pytest.raises(ValueError, match="Duplicated nores"):
        summarize_unlabeled_ablation(tmp_path / "with", tmp_path / "without")


@pytest.mark.parametrize("change", ["source", "folds", "parameters", "preprocessing"])
def test_ablation_rejects_different_manifest_identity(tmp_path, change):
    fixture_run(tmp_path / "with")
    fixture_run(tmp_path / "without", use_unlabeled=False)
    manifest_path = tmp_path / "without" / "manifest.json"
    envelope = json.loads(manifest_path.read_text())
    manifest = envelope["manifest"]
    if change == "source":
        manifest["provenance"]["source_sha256"] = "c" * 64
    elif change == "folds":
        manifest["fold_protocol"] = "different authority"
    elif change == "parameters":
        manifest["config"]["method_parameters"] = {"sshr": {"mu": 0.4}}
    else:
        manifest["preprocessing_policy"]["fit_pool"] = "different"
    new_run = content_hash(manifest)
    envelope["run_id"] = new_run
    manifest_path.write_text(json.dumps(envelope))
    # Preserve envelopes so the intended identity check, not a stale record ID,
    # detects this internally consistent but different run.
    for path in (tmp_path / "without").glob("job-*.json"):
        record_envelope = json.loads(path.read_text())
        record_envelope["record"]["run_id"] = new_run
        record_envelope["sha256"] = content_hash(record_envelope["record"])
        path.write_text(json.dumps(record_envelope))
    with pytest.raises(ValueError, match="identities differ|parameters"):
        summarize_unlabeled_ablation(tmp_path / "with", tmp_path / "without")


def test_export_shareable_aggregates_has_no_predictions_or_raw_features(tmp_path):
    fixture_run(tmp_path / "with")
    fixture_run(tmp_path / "without", use_unlabeled=False)
    report = summarize_controlled_run(tmp_path / "with")
    ablation = summarize_unlabeled_ablation(tmp_path / "with", tmp_path / "without")
    paths = export_study_reports(report, ablation, tmp_path / "reports")
    assert set(paths) == {"scores", "coverage", "diagnostics", "clusters", "ablation", "summary"}
    with open(paths["scores"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(rows[0]["fold_ids"]) == [[0, 0], [0, 1], [1, 0], [1, 1]]
    summary = json.loads(open(paths["summary"]).read())
    assert summary["report"]["identity"]["source_sha256"] == "a" * 64
    assert '"predictions"' not in json.dumps(summary)
    assert '"probabilities"' not in json.dumps(summary)
    assert '"true_labels"' not in json.dumps(summary)
    assert '"centroid"' not in json.dumps(summary)
    bad = deepcopy(ablation)
    bad["identity"]["with_u"]["run_id"] = "different"
    with pytest.raises(ValueError, match="different with-U"):
        export_study_reports(report, bad, tmp_path / "bad")


def test_missing_manifest_does_not_create_output(tmp_path):
    path = tmp_path / "absent"
    with pytest.raises(ValueError, match="does not exist"):
        summarize_controlled_run(path)
    assert not path.exists()


def test_single_shared_fold_has_no_sample_sd(tmp_path):
    store = fixture_run(tmp_path)
    for repeat, fold in ((0, 1), (1, 0), (1, 1)):
        rewrite(store, "sshr", repeat, fold, status="failed", metrics=None)
    report = summarize_controlled_run(tmp_path)
    row = score(report)
    assert row["n_folds"] == 1 and row["mean"] == 1
    assert row["sample_sd"] is None
