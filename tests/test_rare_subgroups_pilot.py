"""Leakage probes, hand arithmetic, and reduced artificial software fixtures.

These tests do not execute the frozen 30-case research design or publish scores.
"""
from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from scripts import run_rare_subgroups_pilot as cli
from src.aw_smote import AWUnsupportedError
from src.rare_subgroups_protocol import (METRICS, array_hash, calibrate_threshold, case_id,
                                        evaluate_scores, generate_case, load_protocol,
                                        planned_jobs, positive_scores)
from src.rare_subgroups_report import export_reports, summarize_cases
import src.rare_subgroups_study as study


@pytest.fixture
def fixture_config():
    cfg = deepcopy(load_protocol())
    cfg.update(purpose="artificial_software_fixture_only", seeds=[42, 43])
    d = cfg["distribution"]
    d["fit_counts"].update(majority=30, total_minority=12, rare_grid=[2, 4])
    d["fitting_pool_sizes"].update(common=12, rare=4)
    d["selection_counts"] = {"majority": 20, "common": 6, "rare": 2}
    d["calibration_counts"] = {"majority": 20, "common": 0, "rare": 0}
    d["test_counts"] = {"majority": 40, "common": 12, "rare": 4}
    cfg["aw_conventions"]["generated_count"] = 18
    cfg["classifier"]["n_estimators"] = 3
    return cfg


class GeometrySpy(BaseEstimator):
    """No classifier learning: record fitting arrays and score a fixed geometry."""
    events = []

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fitting_X_ = np.array(X, copy=True)
        self.fitting_y_ = np.array(y, copy=True)
        self.events.append(("fit", len(X)))
        return self

    def predict_proba(self, X):
        self.events.append(("predict", len(X)))
        scores = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack((1 - scores, scores))


@pytest.fixture
def spy_factory(monkeypatch):
    GeometrySpy.events = []
    created = []

    def factory(config, seed, weighted=False):
        model = GeometrySpy()
        created.append((model, weighted))
        return model

    monkeypatch.setattr(study, "_new_classifier", factory)
    return created


def first_case(config):
    job = planned_jobs(config)[0]
    return job, generate_case(config, job)


def test_frozen_design_rejects_changed_analysis_but_allows_progress_metadata(tmp_path):
    config = load_protocol()
    config["implementation_status"] = "software_progress"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    assert load_protocol(path)["implementation_status"] == "software_progress"
    config["seeds"].append(99)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="frozen v1"):
        load_protocol(path)


def test_preflight_and_disabled_run_generate_no_data_fit_or_output(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("No data generation, model fitting, or manifest creation is allowed")
    for name in ("generate_case", "evaluate_case", "_manifest"):
        monkeypatch.setattr(cli, name, forbidden)
    output = tmp_path / "must-not-exist"
    assert cli.main(["--check-only", "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == 30 and plan["max_classifier_fits"] == 420
    assert plan["planned_score_rows"] == 210
    assert not plan["data_generated"] and not plan["models_fitted"]
    assert cli.main(["--run", "--output", str(output)]) == 2
    assert "disabled" in capsys.readouterr().err
    assert not output.exists()


def test_partition_counts_streams_prefix_pairing_and_hidden_metadata(fixture_config):
    cfg = fixture_config
    job, a = first_case(cfg)
    b = generate_case(cfg, {**job, "rare_count": 4})
    c = generate_case(cfg, {**job, "rare_sigma": .9})
    assert a.fitting.X.shape == (42, 2)
    assert len(a.selection.y) == 28 and len(a.calibration.y) == 20 and len(a.test.y) == 56
    assert np.all(a.calibration.y == 0)
    for part in vars(a).values():
        np.testing.assert_array_equal(part.y, (part.groups > 0).astype(int))
        assert part.X.shape[1] == 2
    as_set = lambda values: set(map(tuple, values))
    assert as_set(b.fitting.X[b.fitting.groups == 1]) <= as_set(a.fitting.X[a.fitting.groups == 1])
    assert as_set(a.fitting.X[a.fitting.groups == 2]) <= as_set(b.fitting.X[b.fitting.groups == 2])
    np.testing.assert_array_equal(a.test.X, b.test.X)
    for part_a, part_c in zip(vars(a).values(), vars(c).values()):
        mask = part_a.groups != 2
        np.testing.assert_array_equal(part_a.X[mask], part_c.X[mask])
    mu = np.array([0., 3.])
    np.testing.assert_allclose((a.fitting.X[a.fitting.groups == 2] - mu) / .35,
                               (c.fitting.X[c.fitting.groups == 2] - mu) / .9)
    expected = np.random.default_rng(np.random.SeedSequence([42, 2, 0])).standard_normal((20, 2))[0]
    assert tuple(expected) in as_set(a.selection.X[a.selection.groups == 0])
    again = generate_case(cfg, job)
    for first, second in zip(vars(a).values(), vars(again).values()):
        np.testing.assert_array_equal(first.X, second.X)
    assert not as_set(a.fitting.X) & as_set(a.selection.X)


def test_threshold_uses_lowest_qualifying_unique_score_with_ties():
    calibration = calibrate_threshold(np.array([.1, .2, .2, .7, .9]), .4)
    assert calibration["threshold"] == .7
    assert calibration["calibration_fp"] == 2
    assert calibration["calibration_fpr"] == .4
    assert not calibration["threshold_is_infinite"]
    infinite = calibrate_threshold(np.array([.5, .5]), .05)
    assert infinite["threshold"] is None and infinite["threshold_is_infinite"]
    assert infinite["calibration_fpr"] == 0


def test_metrics_have_hand_calculated_group_denominators_fpr_f1_and_ap():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    scores = np.array([.9, .1, .1, .1, .9, .1, .9, .9])
    calibration = calibrate_threshold(np.array([.1, .5]), .5)
    metrics = evaluate_scores(y, groups, scores, calibration)
    assert metrics["rare_recall"] == 1 and metrics["common_recall"] == .5
    assert metrics["positive_recall"] == metrics["f1"] == .75
    assert metrics["test_fpr"] == .25 and metrics["test_fp"] == 1
    assert metrics["rare_count"] == metrics["common_count"] == 2
    assert metrics["ap"] == pytest.approx(.6875)
    never = evaluate_scores(y, groups, scores, calibrate_threshold(np.array([.5, .5]), .05))
    assert never["f1"] == never["test_fpr"] == never["rare_recall"] == 0
    assert never["ap"] == metrics["ap"]


@pytest.mark.parametrize("scores,target", [([], .05), ([np.nan], .05), ([1.1], .05), ([.5], -.1)])
def test_invalid_threshold_inputs_rejected(scores, target):
    with pytest.raises(ValueError):
        calibrate_threshold(np.array(scores), target)


def test_positive_probability_column_follows_class_label():
    class Reversed:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.array([[.8, .2], [.3, .7]])
    np.testing.assert_array_equal(positive_scores(Reversed(), np.zeros((2, 2))), [.8, .3])


def test_selection_reuses_default_fit_and_never_refits_on_validation(fixture_config, spy_factory):
    job, data = first_case(fixture_config)
    training = study.train_methods(data.fitting.X, data.fitting.y,
                                   data.selection.X, data.selection.y, fixture_config, job["seed"])
    assert training.classifier_fit_attempts == 14 == len(spy_factory)
    assert len(training.candidates) == 9
    assert sum(c["reused_fixed_default"] for c in training.candidates) == 1
    assert training.methods["AW_REMOVE_TUNED"].model is training.methods["AW_REMOVE"].model
    assert training.methods["AW_REMOVE_TUNED"].record["selected_parameters"] == {"min_samples": 2, "eps_percentile": 65}
    assert all(rows != len(data.selection.y) for event, rows in GeometrySpy.events if event == "fit")
    assert all(rows == len(data.selection.y) for event, rows in GeometrySpy.events if event == "predict")
    for model, _ in spy_factory:
        assert model.fitting_X_.shape[1] == 2
    np.testing.assert_array_equal(training.methods["AW_REMOVE"].sampled.X[len(training.methods["AW_REMOVE"].sampled.retained):],
                                  training.methods["AW_SEED_FILTER_RETAIN"].sampled.X[len(data.fitting.X):])
    assert sum(weighted for _, weighted in spy_factory) == 1


def test_calibration_test_labels_and_group_ids_cannot_change_training_selection(fixture_config, spy_factory):
    job, data = first_case(fixture_config)
    first, predictions = study.evaluate_case(data, fixture_config, job)
    events = GeometrySpy.events.copy()
    final_fit = max(i for i, e in enumerate(events) if e[0] == "fit")
    first_heldout = min(i for i, e in enumerate(events) if e[0] == "predict" and e[1] in {20, 56})
    assert final_fit < first_heldout
    np.testing.assert_allclose(first["scaler"]["mean"], data.fitting.X.mean(axis=0))
    assert first["scaler"]["fitted_rows"] == len(data.fitting.X)
    swapped_fit_groups = np.where(data.fitting.groups == 1, 2, np.where(data.fitting.groups == 2, 1, 0))
    rotated_test_groups = np.roll(data.test.groups, 1)
    changed = replace(data,
        fitting=replace(data.fitting, groups=swapped_fit_groups),
        selection=replace(data.selection, groups=np.full(len(data.selection.y), 99)),
        calibration=replace(data.calibration, X=data.calibration.X + 1000),
        test=replace(data.test, groups=rotated_test_groups, y=(rotated_test_groups != 0).astype(np.int8)))
    second, changed_predictions = study.evaluate_case(changed, fixture_config, job)
    for row_a, row_b in zip(first["rows"], second["rows"]):
        assert row_a.get("selected_parameters") == row_b.get("selected_parameters")
        assert row_a["output_X_sha256"] == row_b["output_X_sha256"]
        np.testing.assert_array_equal(predictions[row_a["method"]], changed_predictions[row_a["method"]])
    assert [c["validation_ap"] for c in first["selection_candidates"]] == [c["validation_ap"] for c in second["selection_candidates"]]
    assert first["partitions_sha256"]["fitting"]["groups"] != second["partitions_sha256"]["fitting"]["groups"]
    assert any(a["metrics"]["threshold_is_infinite"] != b["metrics"]["threshold_is_infinite"]
               for a, b in zip(first["rows"], second["rows"]))


def test_tuning_can_select_a_better_nondefault_model_without_refitting(fixture_config, monkeypatch):
    cfg = deepcopy(fixture_config)
    cfg["aw_tuning"]["min_samples_grid"] = [2]
    job, data = first_case(cfg)
    created = []
    class ControlledAP(GeometrySpy):
        def __init__(self, best=False):
            self.best = best
        def predict_proba(self, X):
            assert len(X) == len(data.selection.y)
            # Fixed software predictions give AP=1 only to the final nondefault
            # candidate, epsilon percentile 80. Other candidates reverse ranks.
            score = np.where(data.selection.y == int(self.best), .9, .1)
            return np.column_stack((1 - score, score))
    def factory(config, seed, weighted=False):
        model = ControlledAP(best=len(created) == 7)
        created.append(model)
        return model
    monkeypatch.setattr(study, "_new_classifier", factory)
    training = study.train_methods(data.fitting.X, data.fitting.y, data.selection.X, data.selection.y, cfg, 42)
    chosen = training.methods["AW_REMOVE_TUNED"]
    assert chosen.record["selected_parameters"] == {"min_samples": 2, "eps_percentile": 80}
    assert chosen.record["selected_validation_ap"] == 1
    assert chosen.model is created[7]
    assert training.classifier_fit_attempts == len(created) == 8


def test_positive_calibration_labels_rejected_before_training(fixture_config, spy_factory):
    job, data = first_case(fixture_config)
    bad = replace(data, calibration=replace(data.calibration, y=np.ones_like(data.calibration.y)))
    with pytest.raises(ValueError, match="known negatives"):
        study.evaluate_case(bad, fixture_config, job)
    assert not spy_factory


def test_all_unsupported_aw_candidates_remain_missing_scores_without_fallback(fixture_config, spy_factory, monkeypatch):
    class Unsupported:
        def __init__(self, **kwargs):
            pass
        def fit_resample(self, X, y):
            self.diagnostics_ = {"status": "unsupported", "reason": "no_valid_cluster"}
            raise AWUnsupportedError("no_valid_cluster", "Artificial unsupported fixture")
    monkeypatch.setattr(study, "AWSMOTE", Unsupported)
    job, data = first_case(fixture_config)
    record, predictions = study.evaluate_case(data, fixture_config, job)
    assert record["classifier_fit_attempts"] == 4
    assert record["case_status"] == "partial"
    for row in record["rows"]:
        if row["method"].startswith("AW_"):
            assert row["status"] == "unsupported" and row["metrics"] is None
            assert row["method"] not in predictions
    assert all(c["validation_ap"] is None for c in record["selection_candidates"])


def test_test_prediction_failure_does_not_trigger_reselection(fixture_config, spy_factory, monkeypatch):
    original = GeometrySpy.predict_proba
    def fail_only_test(self, X):
        if len(X) == 56:
            raise ValueError("Artificial prediction failure")
        return original(self, X)
    monkeypatch.setattr(GeometrySpy, "predict_proba", fail_only_test)
    job, data = first_case(fixture_config)
    record, predictions = study.evaluate_case(data, fixture_config, job)
    assert len(spy_factory) == 14
    assert set(predictions) == {"y", "groups"}
    tuned = next(r for r in record["rows"] if r["method"] == "AW_REMOVE_TUNED")
    assert tuned["selected_parameters"] == {"min_samples": 2, "eps_percentile": 65}
    assert tuned["status"] == "failed" and tuned["metrics"] is None


def fake_records(config):
    records = {}
    for job in planned_jobs(config):
        rows = [{"method": method, "status": "success", "metrics": {m: .5 for m in METRICS},
                 "group_retention": {g: {"original": 10, "retained": 10, "removed": 0, "fraction": 1.0}
                                     for g in ("common", "rare")}}
                for method in config["methods"]]
        records[case_id(job)] = {"job": job, "rows": rows}
    return records


def test_summary_uses_identical_successful_seeds_and_keeps_missing_rows(tmp_path):
    cfg = load_protocol()
    records = fake_records(cfg)
    target = records[case_id(planned_jobs(cfg)[1])]
    target["rows"][0]["metrics"]["f1"] = .99
    target["rows"][3].update(status="unsupported", metrics=None)
    removed_id = case_id(planned_jobs(cfg)[-1])
    del records[removed_id]
    report = export_reports(tmp_path, records, cfg)
    assert len(report["scores"]) == 210
    assert sum(r["status"] == "missing" for r in report["scores"]) == 7
    summary = next(r for r in report["summary"] if r["fit_rare_count"] == 4 and r["rare_sigma"] == .35
                   and r["method"] == "NORES" and r["metric"] == "f1")
    assert summary["n_shared_seeds"] == 4 and summary["mean"] == .5
    assert 43 not in json.loads(summary["shared_seeds"])
    assert report["screening"]["status"] == "incomplete_coverage"
    assert (tmp_path / "scores.csv").is_file() and (tmp_path / "status.json").is_file()


def test_sample_sd_is_across_seeds_with_no_silent_single_seed_zero():
    cfg, records = load_protocol(), fake_records(load_protocol())
    for record in records.values():
        for row in record["rows"]:
            row["metrics"]["f1"] = .1 * (record["job"]["seed"] - 41)
    report = summarize_cases(records, cfg)
    summary = next(r for r in report["summary"] if r["metric"] == "f1")
    assert summary["mean"] == pytest.approx(.3)
    assert summary["sample_sd"] == pytest.approx(math.sqrt(.025))
    cfg["seeds"] = [42]
    subset = {case_id(job): records[case_id(job)] for job in planned_jobs(cfg)}
    assert all(row["sample_sd"] is None for row in summarize_cases(subset, cfg)["summary"])


def test_screening_requires_repeated_loss_and_checks_simple_controls():
    cfg = load_protocol()
    records = fake_records(cfg)
    for record in records.values():
        if record["job"]["rare_count"] != 4:
            continue
        by_method = {r["method"]: r for r in record["rows"]}
        by_method["AW_REMOVE"]["metrics"].update(rare_recall=.4, test_fpr=.05)
        by_method["AW_REMOVE"]["group_retention"]["rare"].update(removed=2, retained=8, fraction=.8)
        by_method["AW_SEED_FILTER_RETAIN"]["metrics"].update(rare_recall=.45, common_recall=.8, test_fpr=.06)
        for method in cfg["screening_rules"]["simple_controls"]:
            by_method[method]["metrics"].update(rare_recall=.45, common_recall=.8, test_fpr=.06)
    screen = summarize_cases(records, cfg)["screening"]
    assert screen["qualifying_settings"] == 2
    assert screen["status"] == "simple_controls_suffice"
    for record in records.values():
        for row in record["rows"]:
            if row["method"] in cfg["screening_rules"]["simple_controls"]:
                row["metrics"]["rare_recall"] = .1
    assert summarize_cases(records, cfg)["screening"]["status"] == "follow_up_required"


def checkpoint_config(cfg, tmp_path, monkeypatch):
    cfg = deepcopy(cfg)
    cfg.update(execution_enabled=True, seeds=[42])
    cfg["distribution"]["fit_counts"]["rare_grid"] = [2]
    cfg["distribution"]["rare_minority"]["isotropic_sigma_grid"] = [.35]
    cfg["budget"].update(paired_cases=1, max_classifier_fits=14, planned_report_rows=7)
    path = tmp_path / "software_fixture.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(cli, "_manifest", lambda config, config_path: {
        "git": {"tracked_changes": False}, "software_fixture": True, "config": deepcopy(config)})
    return cfg, path


def test_complete_case_checkpoint_resume_and_prediction_integrity(fixture_config, spy_factory, tmp_path, monkeypatch):
    cfg, path = checkpoint_config(fixture_config, tmp_path, monkeypatch)
    output = tmp_path / "outputs"
    report = cli.run_study(cfg, path, output)
    assert report["status"] == "complete" and len(spy_factory) == 14
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed case must not regenerate or refit")
    monkeypatch.setattr(cli, "generate_case", forbidden)
    again = cli.run_study(cfg, path, output)
    assert again["summary"] == report["summary"] and len(spy_factory) == 14
    prediction = next((output / "predictions").glob("*.npz"))
    prediction.write_bytes(prediction.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        cli.run_study(cfg, path, output)
    assert len(spy_factory) == 14


def test_resume_rejects_different_configuration_before_fitting(fixture_config, spy_factory, tmp_path, monkeypatch):
    cfg, path = checkpoint_config(fixture_config, tmp_path, monkeypatch)
    output = tmp_path / "outputs"
    cli.run_study(cfg, path, output)
    changed = deepcopy(cfg)
    changed["classifier"]["n_estimators"] += 1
    with pytest.raises(ValueError, match="different or corrupt run"):
        cli.run_study(changed, path, output)
    assert len(spy_factory) == 14


def test_interruption_keeps_finished_case_and_resumes_only_missing_case(fixture_config, spy_factory, tmp_path, monkeypatch):
    cfg, path = checkpoint_config(fixture_config, tmp_path, monkeypatch)
    cfg["seeds"] = [42, 43]
    cfg["budget"].update(paired_cases=2, max_classifier_fits=28, planned_report_rows=14)
    real_evaluate = cli.evaluate_case
    def interrupt_second(data, config, job):
        if job["seed"] == 43:
            raise KeyboardInterrupt()
        return real_evaluate(data, config, job)
    monkeypatch.setattr(cli, "evaluate_case", interrupt_second)
    output = tmp_path / "outputs"
    with pytest.raises(KeyboardInterrupt):
        cli.run_study(cfg, path, output)
    status = json.loads((output / "reports/status.json").read_text())
    assert status["saved_cases"] == 1 and status["success_rows"] == 7
    assert len(spy_factory) == 14
    monkeypatch.setattr(cli, "evaluate_case", real_evaluate)
    assert cli.run_study(cfg, path, output)["status"] == "complete"
    assert len(spy_factory) == 28


def test_real_forest_on_small_software_fixture_writes_valid_metrics(fixture_config):
    job, data = first_case(fixture_config)
    record, predictions = study.evaluate_case(data, fixture_config, job)
    assert record["classifier_fit_attempts"] <= 14
    assert len(record["rows"]) == 7
    for row in record["rows"]:
        if row["status"] == "success":
            assert all(np.isfinite(row["metrics"][key]) for key in METRICS)
            assert row["test_scores_sha256"] == array_hash(predictions[row["method"]])
    assert record["rows"][0]["status"] == "success"
