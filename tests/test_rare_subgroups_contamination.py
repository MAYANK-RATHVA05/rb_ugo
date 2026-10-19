"""Artificial software fixtures only; never execute the frozen research design."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score

import scripts.run_rare_subgroups_contamination as cli
import src.rare_subgroups_contamination as protocol
import src.rare_subgroups_contamination_report as report
import src.rare_subgroups_contamination_study as study
import src.rare_subgroups_study as baseline
from src.aw_smote import AWUnsupportedError
from src.rare_subgroups_protocol import METRICS, array_hash, generate_case as clean_case
from src.results import ResultStore


@pytest.fixture
def design():
    d = deepcopy(protocol.load_protocol()["scientific_design"])
    d["seeds"] = [42, 43]
    distribution = d["distribution"]
    distribution["fit_counts"].update(true_majority=30, true_total_minority=12, rare_grid=[2, 4], total_rows=42)
    distribution["fitting_pool_sizes"].update(common=12, rare=4)
    distribution["selection_counts"] = {"majority": 20, "common": 6, "rare": 2}
    distribution["calibration_counts"] = {"majority": 20, "common": 0, "rare": 0}
    distribution["test_counts"] = {"majority": 40, "common": 12, "rare": 4}
    d["label_corruption"]["flipped_majority_count_grid"] = [0, 2, 4]
    d["screening_rules"]["contaminated_levels"] = [2, 4]
    d["threshold"].update(calibration_negatives=20, target_false_positive_count=1)
    d["classifier"]["n_estimators"] = 3
    return d


class GeometrySpy(BaseEstimator):
    events = []

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fit_X_ = X.copy()
        self.fit_y_ = y.copy()
        self.events.append(("fit", id(self), len(X)))
        return self

    def predict_proba(self, X):
        self.events.append(("predict", id(self), len(X)))
        scores = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack((1 - scores, scores))


@pytest.fixture
def models(monkeypatch):
    GeometrySpy.events = []
    created = []
    def factory(config, seed, weighted=False):
        instance = GeometrySpy()
        created.append(instance)
        return instance
    monkeypatch.setattr(baseline, "_new_classifier", factory)
    return created


def first(design, noise=0):
    job = {**protocol.planned_jobs(design)[0], "flipped_count": noise}
    return job, protocol.generate_case(design, job)


def test_frozen_hash_rejects_rehashed_design_change_but_allows_metadata(tmp_path):
    cfg = protocol.load_protocol()
    cfg["runner_status"] = "software_progress"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(cfg))
    assert protocol.load_protocol(path)["runner_status"] == "software_progress"
    cfg["scientific_design"]["seeds"].append(99)
    cfg["design_sha256"] = protocol.design_hash(cfg["scientific_design"])
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="frozen contamination"):
        protocol.load_protocol(path)


@pytest.mark.parametrize("value", [None, 1, "false"])
def test_execution_flag_must_be_boolean(value):
    cfg = protocol.load_protocol()
    cfg["execution_enabled"] = value
    with pytest.raises(ValueError, match="boolean"):
        protocol.validate_protocol(cfg)


def test_check_only_and_disabled_run_do_not_generate_fit_or_create_output(monkeypatch, tmp_path, capsys):
    disabled = protocol.load_protocol()
    disabled["execution_enabled"] = False
    disabled_path = tmp_path / "disabled_protocol.json"
    disabled_path.write_text(json.dumps(disabled))
    def forbidden(*args, **kwargs):
        raise AssertionError("Software stage must not generate research data or create manifests")
    for name in ("generate_case", "evaluate_case", "_manifest"):
        monkeypatch.setattr(cli, name, forbidden)
    output = tmp_path / "never-create"
    assert cli.main(["--config", str(disabled_path), "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == 90 and plan["planned_score_rows"] == 630
    assert plan["max_classifier_fits"] == 1260 and plan["candidate_records"] == 810
    assert not plan["data_generated"] and not plan["execution_enabled"]
    assert cli.main(["--run", "--config", str(disabled_path), "--output", str(output)]) == 2
    assert "disabled" in capsys.readouterr().err
    assert not output.exists()


def test_only_nested_majority_labels_change_and_clean_streams_match(design):
    job, a = first(design)
    _, b = first(design, 2)
    _, c = first(design, 4)
    expected_ids = np.random.default_rng(np.random.SeedSequence([42, 5, 100])).permutation(30)
    assert set(b.fitting.stable_ids[b.fitting.corrupted]) == set(expected_ids[:2])
    assert set(c.fitting.stable_ids[c.fitting.corrupted]) == set(expected_ids[:4])
    assert np.all(~b.fitting.corrupted | c.fitting.corrupted)
    for other, count in ((b, 2), (c, 4)):
        for name in ("X", "true_y", "groups", "stable_ids"):
            np.testing.assert_array_equal(getattr(a.fitting, name), getattr(other.fitting, name))
        assert np.sum(a.fitting.y != other.fitting.y) == count
        assert np.all(other.fitting.true_y[other.fitting.corrupted] == 0)
        assert np.all(other.fitting.y[other.fitting.corrupted] == 1)
        assert np.sum(other.fitting.y == 1) == 12 + count
        for partition in ("selection", "calibration", "test"):
            for field in ("X", "y", "groups"):
                np.testing.assert_array_equal(getattr(getattr(a, partition), field), getattr(getattr(other, partition), field))
    clean_config = deepcopy(design)
    clean_config["distribution"]["fit_counts"].update(majority=30, total_minority=12)
    clean = clean_case(clean_config, {key: job[key] for key in ("seed", "rare_count", "rare_sigma")})
    for name in ("fitting", "selection", "calibration", "test"):
        for key in ("X", "y", "groups"):
            assert array_hash(getattr(getattr(a, name), key)) == array_hash(getattr(getattr(clean, name), key))


def test_corruption_prefix_shared_across_base_cells_and_streams_are_independent(design):
    job, a = first(design, 2)
    b = protocol.generate_case(design, {**job, "rare_count": 4, "rare_sigma": .9})
    np.testing.assert_array_equal(a.fitting.stable_ids[a.fitting.corrupted], b.fitting.stable_ids[b.fitting.corrupted])
    majority_a = a.fitting.X[a.fitting.groups == 0]
    majority_b = b.fitting.X[b.fitting.groups == 0]
    np.testing.assert_array_equal(majority_a, majority_b)
    assert not set(map(tuple, a.fitting.X)) & set(map(tuple, a.selection.X))
    with pytest.raises(ValueError, match="not part"):
        protocol.generate_case(design, {**job, "flipped_count": 3})


def test_hand_calibration_tie_order_and_predictions():
    result = protocol.calibrate_threshold(np.array([.1, .5, .5, .9]), np.array([.2, .3, .8, .1]), 2)
    assert result == {"threshold": .5, "tie_cutoff": .8, "calibration_fp": 2,
                      "calibration_negatives": 4, "calibration_fpr": .5}
    np.testing.assert_array_equal(protocol.predict_with_ties(
        np.array([.5, .5, .9, .4]), np.array([.7, .9, 0, .99]), result), [False, True, True, False])


def test_exact_target_even_if_every_forest_score_is_tied():
    scores = np.full(2000, .25)
    uniforms = np.arange(2000) / 2000
    calibrated = protocol.calibrate_threshold(scores, uniforms, 100)
    assert calibrated["calibration_fp"] == 100 and calibrated["calibration_fpr"] == .05
    assert protocol.predict_with_ties(scores, uniforms, calibrated).sum() == 100
    assert calibrated["threshold"] == .25 and calibrated["tie_cutoff"] == .95


@pytest.mark.parametrize("target", [0, 5, True, 1.5])
def test_calibration_rejects_invalid_count(target):
    with pytest.raises(ValueError, match="target_fp"):
        protocol.calibrate_threshold(np.array([.1, .2, .3, .4]), np.arange(4) / 4, target)


@pytest.mark.parametrize("uniforms", [[.1], [-.1, .2], [.1, 1], [float("nan"), .2]])
def test_calibration_rejects_bad_uniforms(uniforms):
    with pytest.raises(ValueError, match="uniforms"):
        protocol.calibrate_threshold(np.array([.2, .2]), np.array(uniforms), 1)


def test_duplicate_calibration_keys_fail_without_redraw():
    with pytest.raises(ValueError, match="Duplicate"):
        protocol.calibrate_threshold(np.array([.5, .5]), np.array([.2, .2]), 1)


def test_uniform_streams_repeat_and_partitions_differ():
    np.testing.assert_array_equal(protocol.tie_uniforms(42, 3, 10), protocol.tie_uniforms(42, 3, 10))
    assert not np.array_equal(protocol.tie_uniforms(42, 3, 10), protocol.tie_uniforms(42, 4, 10))


def test_hand_metrics_and_ap_ignore_random_tie_keys():
    groups = np.array([0, 0, 1, 1, 2, 2])
    y = (groups != 0).astype(int)
    scores = np.array([.1, .8, .9, .2, .7, .6])
    calibration = {"threshold": .6, "tie_cutoff": .1}
    metrics = protocol.evaluate_scores(y, groups, scores, np.full(6, .2), calibration)
    assert metrics["rare_recall"] == 1 and metrics["common_recall"] == .5
    assert metrics["positive_recall"] == .75 and metrics["test_fpr"] == .5
    assert metrics["f1"] == .75 and metrics["worst_group_recall"] == .5
    assert metrics["ap"] == pytest.approx((1 + 2/3 + 3/4 + 4/5) / 4)
    changed = protocol.evaluate_scores(y, groups, scores, np.zeros(6), calibration)
    assert changed["rare_recall"] == .5 and changed["ap"] == metrics["ap"]


@pytest.mark.parametrize("seed", [3, 42, 999])
def test_smote_instrumentation_matches_library_and_interpolation(seed):
    X = np.arange(36, dtype=float).reshape(18, 2)
    X[-1] = X[-2]  # duplicate features exercise the library's tie behavior
    y = np.r_[np.zeros(12, dtype=int), np.ones(6, dtype=int)]
    sampler = study.TracedSMOTE(k_neighbors=5, random_state=seed)
    for _ in range(2):
        traced = sampler.fit_resample(X, y)
        reference = SMOTE(k_neighbors=5, random_state=seed).fit_resample(X, y)
        for actual, expected in zip(traced, reference):
            np.testing.assert_array_equal(actual, expected)
        seeds, partners, fractions = sampler.generated_seed_indices_, sampler.generated_partner_indices_, sampler.interpolation_fractions_
        reconstructed = X[seeds] + fractions[:, None] * (X[partners] - X[seeds])
        np.testing.assert_array_equal(traced[0][len(X):], reconstructed)
        assert np.all(y[seeds] == 1) and np.all(y[partners] == 1)


def test_ros_trace_preserves_reference_output(design):
    _, data = first(design, 2)
    config = deepcopy(design)
    config["aw_conventions"]["generated_count"] = 14
    actual = study._sample(data.fitting.X, data.fitting.y, config, 42, "ROS")
    ref = RandomOverSampler(random_state=42)
    X, y = ref.fit_resample(data.fitting.X, data.fitting.y)
    np.testing.assert_array_equal(actual.X, X)
    np.testing.assert_array_equal(actual.y, y)
    np.testing.assert_array_equal(actual.trace["duplicated_indices"], ref.sample_indices_[42:])
    np.testing.assert_array_equal(actual.X[42:], data.fitting.X[actual.trace["duplicated_indices"]])


def test_parent_categories_are_exposure_not_oracle_labels():
    corrupted = np.array([True, False, False, True])
    counts = study.parent_diagnostics(corrupted, {"generated_seed_indices": np.array([1, 0, 0]),
                                               "generated_partner_indices": np.array([2, 1, 3])})
    assert counts == {"kind": "interpolation", "generated": 3, "clean_clean": 1, "clean_corrupt": 1, "corrupt_corrupt": 1}
    counts = study.parent_diagnostics(corrupted, {"duplicated_indices": np.array([0, 1, 3])})
    assert counts["corrupted_parents"] == 2 and counts["corrupted_parent_fraction"] == pytest.approx(2/3)


def test_generation_budget_selection_vectors_and_paired_retention(design, models):
    job, data = first(design, 2)
    record, vectors = study.evaluate_case(data, design, job)
    assert record["case_status"] == "complete" and record["classifier_fit_attempts"] == len(models) == 14
    assert record["input_observed_counts"] == {"majority": 28, "minority": 14}
    rows = {row["method"]: row for row in record["rows"]}
    assert all(row["generated_count"] == 14 for method, row in rows.items() if method not in {"NORES", "WEIGHTED"})
    assert rows["AW_REMOVE_TUNED"]["selected_parameters"] == {"min_samples": 2, "eps_percentile": 65}
    for key in rows["AW_REMOVE"]["trace"]:
        np.testing.assert_array_equal(rows["AW_REMOVE"]["trace"][key], rows["AW_SEED_FILTER_RETAIN"]["trace"][key])
    assert rows["AW_SEED_FILTER_RETAIN"]["output_counts"] == {"majority": 28, "minority": 28}
    assert rows["AW_SEED_FILTER_RETAIN"]["group_retention"]["noise"]["fraction"] == 1
    for candidate in record["selection_candidates"]:
        assert candidate["validation_ap"] == average_precision_score(vectors["selection_y"], vectors[candidate["selection_vector"]])
    for row in record["rows"]:
        recalibrated = protocol.calibrate_threshold(vectors[row["vectors"]["calibration"]], vectors["calibration_ties"], 1)
        recomputed = protocol.evaluate_scores(vectors["test_y"], vectors["test_groups"], vectors[row["vectors"]["test"]], vectors["test_ties"], recalibrated)
        assert row["metrics"] == recomputed
    assert all(array_hash(vector) == record["vector_sha256"][name] for name, vector in vectors.items())


def test_hidden_truth_and_heldouts_do_not_change_fitting_or_selection(design, models):
    job, data = first(design, 2)
    first_record, first_vectors = study.evaluate_case(data, design, job)
    events = GeometrySpy.events.copy()
    assert max(i for i, event in enumerate(events) if event[0] == "fit") < min(
        i for i, event in enumerate(events) if event[0] == "predict" and event[2] in {20, 56})
    assert first_record["scaler"]["fitted_rows"] == 42
    np.testing.assert_allclose(first_record["scaler"]["mean"], data.fitting.X.mean(axis=0))
    swapped_groups = np.where(data.fitting.groups == 1, 2, np.where(data.fitting.groups == 2, 1, 0))
    test_groups = np.roll(data.test.groups, 1)
    changed = replace(data, fitting=replace(data.fitting, groups=swapped_groups,
                                          true_y=1-data.fitting.true_y, corrupted=np.roll(data.fitting.corrupted, 1)),
                      selection=replace(data.selection, groups=np.full(len(data.selection.y), 99)),
                      calibration=replace(data.calibration, X=data.calibration.X + 1000),
                      test=replace(data.test, groups=test_groups, y=(test_groups != 0).astype(int)))
    second, second_vectors = study.evaluate_case(changed, design, job)
    for a, b in zip(models[:14], models[14:]):
        np.testing.assert_array_equal(a.fit_X_, b.fit_X_)
        np.testing.assert_array_equal(a.fit_y_, b.fit_y_)
    for candidate_a, candidate_b in zip(first_record["selection_candidates"], second["selection_candidates"]):
        assert candidate_a["validation_ap"] == candidate_b["validation_ap"]
    for a, b in zip(first_record["rows"], second["rows"]):
        assert a.get("selected_parameters") == b.get("selected_parameters")
        np.testing.assert_array_equal(first_vectors[a["vectors"]["test"]], second_vectors[b["vectors"]["test"]])
    assert first_record["rows"][0]["metrics"]["threshold"] != second["rows"][0]["metrics"]["threshold"]


def test_nondefault_selection_is_reused_and_unselected_models_never_score_test(design, monkeypatch):
    job, data = first(design)
    instances = []
    class SelectionSpy(GeometrySpy):
        def __init__(self, best=False):
            self.best = best
        def predict_proba(self, X):
            if len(X) == len(data.selection.y):
                scores = np.where(data.selection.y == int(self.best), .9, .1)
                return np.column_stack((1-scores, scores))
            return super().predict_proba(X)
    def factory(config, seed, weighted=False):
        model = SelectionSpy(best=len(instances) == 7)
        instances.append(model)
        return model
    GeometrySpy.events = []
    monkeypatch.setattr(baseline, "_new_classifier", factory)
    record, _ = study.evaluate_case(data, design, job)
    tuned = record["rows"][-1]
    assert len(instances) == record["classifier_fit_attempts"] == 14
    assert tuned["selected_parameters"] == {"min_samples": 2, "eps_percentile": 80}
    tested = {event[1] for event in GeometrySpy.events if event[0] == "predict" and event[2] == 56}
    assert tested == {id(instances[i]) for i in (0, 1, 2, 3, 4, 5, 7)}


def test_unsupported_aw_has_no_fallback_or_zero_metrics(design, models, monkeypatch):
    class Unsupported:
        def __init__(self, **kwargs):
            pass
        def fit_resample(self, X, y):
            self.diagnostics_ = {"status": "unsupported", "reason": "no_valid_cluster"}
            raise AWUnsupportedError("no_valid_cluster", "Artificial fixture")
    monkeypatch.setattr(baseline, "AWSMOTE", Unsupported)
    job, data = first(design)
    record, vectors = study.evaluate_case(data, design, job)
    assert record["classifier_fit_attempts"] == len(models) == 4
    for row in record["rows"]:
        if row["method"].startswith("AW_"):
            assert row["status"] == "unsupported" and row["metrics"] is None and not row["vectors"]
    assert not any(name.startswith("selection__") for name in vectors)


def test_test_prediction_failure_keeps_selection_locked(design, models, monkeypatch):
    original = GeometrySpy.predict_proba
    def fail_test(self, X):
        if len(X) == 56:
            raise RuntimeError("Artificial test failure")
        return original(self, X)
    monkeypatch.setattr(GeometrySpy, "predict_proba", fail_test)
    job, data = first(design)
    record, vectors = study.evaluate_case(data, design, job)
    assert len(models) == 14 and record["case_status"] == "partial"
    assert record["rows"][-1]["selected_parameters"] == {"min_samples": 2, "eps_percentile": 65}
    assert all(row["status"] == "failed" and row["metrics"] is None for row in record["rows"])
    assert sum(name.startswith("selection__") for name in vectors) == 9


def test_positive_calibration_labels_fail_before_fitting(design, models):
    job, data = first(design)
    changed = replace(data, calibration=replace(data.calibration, y=np.ones_like(data.calibration.y)))
    with pytest.raises(ValueError, match="known negatives"):
        study.evaluate_case(changed, design, job)
    assert not models


def test_zero_noise_retention_is_null_and_scaler_shared_across_noise(design, models):
    job0, data0 = first(design)
    job2, data2 = first(design, 2)
    a, _ = study.evaluate_case(data0, design, job0)
    b, _ = study.evaluate_case(data2, design, job2)
    assert a["rows"][0]["group_retention"]["noise"] == {"original": 0, "retained": 0, "removed": 0, "fraction": None}
    np.testing.assert_array_equal(a["scaler"]["mean"], b["scaler"]["mean"])
    np.testing.assert_array_equal(a["scaler"]["scale"], b["scaler"]["scale"])


def test_real_forest_exercises_small_artificial_case(design):
    job, data = first(design, 2)
    record, vectors = study.evaluate_case(data, design, job)
    assert record["rows"][0]["status"] == "success"
    assert record["classifier_fit_attempts"] <= 14
    assert all(row["metrics"]["calibration_fp"] == 1 for row in record["rows"] if row["status"] == "success")
    assert vectors["test_y"].shape == (56,)


def fake_records(d):
    """Hand-specified software numbers, not trained-model or research results."""
    records = {}
    for job in protocol.planned_jobs(d):
        noise = job["flipped_count"]
        rows = []
        for method in d["methods"]:
            metrics = {metric: .5 for metric in METRICS}
            metrics.update(rare_recall=.6, common_recall=.9, test_fpr=.05, ap=.8)
            if method == "AW_REMOVE":
                metrics.update(rare_recall=.4, common_recall=.8, ap=.75 if noise else .8)
            if method == "AW_SEED_FILTER_RETAIN":
                metrics.update(rare_recall=.5, common_recall=.8, ap=.71 if noise else .8)
            retention = {}
            for name in ("common", "rare", "noise"):
                original = noise if name == "noise" else 10
                removed = int(method == "AW_REMOVE" and original > 0)
                retention[name] = {"original": original, "removed": removed, "retained": original-removed,
                                   "fraction": (original-removed)/original if original else None}
            rows.append({"method": method, "status": "success", "metrics": metrics, "group_retention": retention})
        records[protocol.case_id(job)] = {"job": job, "rows": rows}
    return records


def by_method(record, method):
    return next(row for row in record["rows"] if row["method"] == method)


def test_reports_hand_ap_interaction_counts_and_simple_control_stop(tmp_path):
    d = protocol.load_protocol()["scientific_design"]
    result = report.export_reports(tmp_path, fake_records(d), d)
    assert len(result["scores"]) == 630 and len(result["coverage"]) == 126
    assert len(result["summary"]) == 1260 and len(result["paired_changes"]) == 420
    zero_noise = next(row for row in result["summary"] if row["metric"] == "noise_retained_fraction" and row["flipped_count"] == 0)
    assert zero_noise["mean"] is None and zero_noise["sample_sd"] is None and zero_noise["n_values"] == 0
    assert zero_noise["n_shared_seeds"] == 5
    noisy = next(row for row in result["summary"] if row["metric"] == "noise_retained_fraction"
                 and row["flipped_count"] == 5 and row["method"] == "AW_REMOVE")
    assert noisy["mean"] == .8 and noisy["sample_sd"] == 0
    screen = result["screening"]
    assert screen["status"] == "simple_controls_suffice" and screen["qualifying_base_settings"] == 6
    for cell in screen["settings"]:
        assert cell["mean_AP_interaction"] == pytest.approx(-.04)
        assert cell["negative_seed_interactions"] == cell["positive_seed_differences"] == 5
        assert cell["rare_protection"] and cell["noise_tradeoff"] and cell["qualifies"]
    change = next(row for row in result["paired_changes"] if row["method"] == "AW_SEED_FILTER_RETAIN")
    assert change["ap_noisy_minus_clean"] == pytest.approx(-.09)
    assert (tmp_path / "paired_changes.csv").is_file()


def test_two_noise_levels_in_one_base_setting_do_not_pass_repeatability_gate():
    d = protocol.load_protocol()["scientific_design"]
    records = fake_records(d)
    for record in records.values():
        job = record["job"]
        if job["rare_count"] != 4 or job["rare_sigma"] != .35:
            by_method(record, "AW_SEED_FILTER_RETAIN")["metrics"]["ap"] = .8
    screen = report.summarize_cases(records, d)["screening"]
    assert sum(cell["qualifies"] for cell in screen["settings"]) == 2
    assert screen["qualifying_base_settings"] == 1 and screen["status"] == "tradeoff_trigger_not_met"


def test_rare_gain_without_noise_specific_cost_does_not_qualify():
    d = protocol.load_protocol()["scientific_design"]
    records = fake_records(d)
    for record in records.values():
        by_method(record, "AW_SEED_FILTER_RETAIN")["metrics"]["ap"] = by_method(record, "AW_REMOVE")["metrics"]["ap"]
    screen = report.summarize_cases(records, d)["screening"]
    assert all(cell["rare_protection"] and not cell["noise_tradeoff"] for cell in screen["settings"])
    assert screen["status"] == "tradeoff_trigger_not_met"


def test_screen_boundaries_mean_absolute_fpr_and_four_seed_signs():
    d = protocol.load_protocol()["scientific_design"]
    records = fake_records(d)
    for record in records.values():
        job = record["job"]
        if job["flipped_count"]:
            retain = by_method(record, "AW_SEED_FILTER_RETAIN")["metrics"]
            # Four positive differences .0625 and one zero => mean .05.
            # Four interactions -.025 and one zero => mean -.02.
            retain.update(rare_recall=.4 + (.0625 if job["seed"] != 46 else 0),
                          common_recall=.78, test_fpr=.06 if job["seed"] % 2 else .04,
                          ap=.725 if job["seed"] != 46 else .75)
    screen = report.summarize_cases(records, d)["screening"]
    assert all(cell["qualifies"] and cell["positive_seed_differences"] == 4
               and cell["negative_seed_interactions"] == 4 for cell in screen["settings"])
    for record in records.values():
        if record["job"]["flipped_count"]:
            by_method(record, "AW_SEED_FILTER_RETAIN")["metrics"]["test_fpr"] = .06001
    screen = report.summarize_cases(records, d)["screening"]
    assert not any(cell["rare_protection"] for cell in screen["settings"])


def test_ordinary_control_must_cover_all_noise_levels_without_oracle_switching():
    d = protocol.load_protocol()["scientific_design"]
    records = fake_records(d)
    for record in records.values():
        noise = record["job"]["flipped_count"]
        for method in d["screening_rules"]["simple_controls"]:
            row = by_method(record, method)
            row["metrics"]["ap"] = .1
            if (method == "NORES" and noise == 0) or (method == "WEIGHTED" and noise > 0):
                row["metrics"]["ap"] = .8
    screen = report.summarize_cases(records, d)["screening"]
    assert screen["status"] == "residual_tradeoff_requires_replication"
    assert not any(item["covers_all_noise_levels"] for item in screen["ordinary_controls"])


def test_missing_coverage_precedes_favorable_screen_and_uses_shared_seeds(tmp_path):
    d = protocol.load_protocol()["scientific_design"]
    records = fake_records(d)
    target = records[protocol.case_id(protocol.planned_jobs(d)[1])]
    by_method(target, "SMOTE").update(status="unsupported", metrics=None)
    by_method(target, "NORES")["metrics"]["f1"] = .99
    del records[protocol.case_id(protocol.planned_jobs(d)[-1])]
    result = report.export_reports(tmp_path, records, d)
    assert result["screening"]["status"] == "incomplete_coverage"
    assert len(result["scores"]) == 630
    assert sum(row["status"] == "missing" for row in result["scores"]) == 7
    summary = next(row for row in result["summary"] if row["fit_rare_count"] == 4 and row["rare_sigma"] == .35
                   and row["flipped_count"] == 0 and row["method"] == "NORES" and row["metric"] == "f1")
    assert summary["n_shared_seeds"] == 4 and summary["mean"] == .5
    assert 43 not in json.loads(summary["shared_seeds"])
    missing = next(row for row in result["scores"] if row["status"] == "missing")
    assert missing["f1"] is None
    assert any(not row["paired_complete"] and row["ap_noisy_minus_clean"] is None for row in result["paired_changes"])


def test_mean_and_sample_sd_use_seeds_and_single_seed_sd_is_null():
    d = protocol.load_protocol()["scientific_design"]
    records = fake_records(d)
    for record in records.values():
        for row in record["rows"]:
            row["metrics"]["f1"] = .1 * (record["job"]["seed"] - 41)
    f1 = next(row for row in report.summarize_cases(records, d)["summary"] if row["metric"] == "f1")
    assert f1["mean"] == pytest.approx(.3) and f1["sample_sd"] == pytest.approx(math.sqrt(.025))
    d["seeds"] = [42]
    subset = {protocol.case_id(job): records[protocol.case_id(job)] for job in protocol.planned_jobs(d)}
    assert all(row["sample_sd"] is None for row in report.summarize_cases(subset, d)["summary"])


def test_report_rejects_duplicate_methods_and_nonfinite_success_metrics():
    d = protocol.load_protocol()["scientific_design"]
    records = fake_records(d)
    record = next(iter(records.values()))
    record["rows"][0]["metrics"]["ap"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        report.summarize_cases(records, d)
    record["rows"][0]["metrics"]["ap"] = .8
    record["rows"][0] = deepcopy(record["rows"][1])
    with pytest.raises(ValueError, match="duplicate"):
        report.summarize_cases(records, d)


def checkpoint_config(design, tmp_path, monkeypatch):
    cfg = deepcopy(protocol.load_protocol())
    d = deepcopy(design)
    d["seeds"] = [42]
    d["distribution"]["fit_counts"]["rare_grid"] = [2]
    d["distribution"]["rare_minority"]["isotropic_sigma_grid"] = [.35]
    d["label_corruption"]["flipped_majority_count_grid"] = [0, 2]
    d["screening_rules"]["contaminated_levels"] = [2]
    d["budget"].update(paired_cases=2, max_classifier_fits=28, planned_report_rows=14, candidate_records=18)
    cfg.update(scientific_design=d, execution_enabled=True, design_sha256=protocol.design_hash(d))
    # Only this reduced artificial fixture may pass the frozen loader in tests.
    monkeypatch.setattr(protocol, "FROZEN_DESIGN_SHA256", cfg["design_sha256"])
    path = tmp_path / "artificial_fixture.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(cli, "_manifest", lambda config, config_path: {
        "git": {"tracked_changes": False}, "artificial_software_fixture": True, "config": deepcopy(config)})
    return cfg, path


def test_complete_checkpoint_resume_preserves_all_audit_vectors(design, models, tmp_path, monkeypatch):
    cfg, path = checkpoint_config(design, tmp_path, monkeypatch)
    directory = tmp_path / "outputs"
    first_report = cli.run_study(cfg, path, directory)
    assert first_report["status"] == "complete" and len(models) == 28
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed case must not regenerate or refit")
    monkeypatch.setattr(cli, "generate_case", forbidden)
    second_report = cli.run_study(cfg, path, directory)
    assert second_report["summary"] == first_report["summary"] and len(models) == 28
    with np.load(next((directory / "predictions").glob("*.npz")), allow_pickle=False) as vectors:
        assert sum(name.startswith("selection__") for name in vectors.files) == 9
        assert sum(name.startswith("calibration__") for name in vectors.files) == 7
        assert sum(name.startswith("test__") for name in vectors.files) == 7
        assert "fitting_corrupted" in vectors and "test_ties" in vectors
    artifact = next((directory / "predictions").glob("*.npz"))
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        cli.run_study(cfg, path, directory)
    assert len(models) == 28


def test_vector_fingerprints_detect_changed_array_even_with_new_file_checksum(design, models, tmp_path, monkeypatch):
    cfg, path = checkpoint_config(design, tmp_path, monkeypatch)
    directory = tmp_path / "outputs"
    cli.run_study(cfg, path, directory)
    store = ResultStore(directory, cli._manifest(cfg, path))
    job = protocol.planned_jobs(cfg["scientific_design"])[0]
    record = store.read(job)
    artifact = directory / record["predictions"]["path"]
    with np.load(artifact, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["selection_y"][0] = 1 - arrays["selection_y"][0]
    with artifact.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    record["predictions"]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    store.write(record)
    with pytest.raises(ValueError, match="array is corrupt"):
        cli.run_study(cfg, path, directory)
    assert len(models) == 28


def test_run_identity_and_dirty_tree_reject_before_new_fits(design, models, tmp_path, monkeypatch):
    cfg, path = checkpoint_config(design, tmp_path, monkeypatch)
    directory = tmp_path / "outputs"
    cli.run_study(cfg, path, directory)
    changed = deepcopy(cfg)
    changed["metadata_marker"] = "different_configuration"
    with pytest.raises(ValueError, match="different or corrupt run"):
        cli.run_study(changed, path, directory)
    monkeypatch.setattr(cli, "_manifest", lambda *args: {"git": {"tracked_changes": True}})
    with pytest.raises(ValueError, match="clean commit"):
        cli.run_study(cfg, path, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists() and len(models) == 28


def test_interruption_reports_missing_case_then_resumes_only_that_case(design, models, tmp_path, monkeypatch):
    cfg, path = checkpoint_config(design, tmp_path, monkeypatch)
    directory = tmp_path / "outputs"
    real_evaluate = cli.evaluate_case
    def interrupt_second(data, d, job):
        if job["flipped_count"] == 2:
            raise KeyboardInterrupt()
        return real_evaluate(data, d, job)
    monkeypatch.setattr(cli, "evaluate_case", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        cli.run_study(cfg, path, directory)
    status = json.loads((directory / "reports/status.json").read_text())
    assert status["saved_cases"] == 1 and status["success_rows"] == 7
    assert status["screening_status"] == "incomplete_coverage" and len(models) == 14
    monkeypatch.setattr(cli, "evaluate_case", real_evaluate)
    assert cli.run_study(cfg, path, directory)["status"] == "complete"
    assert len(models) == 28


def test_all_selection_prediction_failures_leave_tuned_method_failed(design, models, monkeypatch):
    original = GeometrySpy.predict_proba
    def fail_selection(self, X):
        if len(X) == 28:
            raise RuntimeError("Artificial selection failure")
        return original(self, X)
    monkeypatch.setattr(GeometrySpy, "predict_proba", fail_selection)
    job, data = first(design)
    record, vectors = study.evaluate_case(data, design, job)
    assert len(models) == 14
    assert record["rows"][-1]["status"] == "failed" and record["rows"][-1]["error"]["reason"] == "no_eligible_candidate"
    assert all(candidate["validation_ap"] is None for candidate in record["selection_candidates"])
    assert not any(name.startswith("selection__") for name in vectors)
    assert all(row["status"] == "success" for row in record["rows"][:-1])


def test_duplicate_keys_fail_methods_without_test_prediction_or_seed_retry(design, models, monkeypatch):
    def tied(self, X):
        self.events.append(("predict", id(self), len(X)))
        return np.full((len(X), 2), .5)
    monkeypatch.setattr(GeometrySpy, "predict_proba", tied)
    monkeypatch.setattr(study, "tie_uniforms", lambda seed, partition, rows: np.zeros(rows))
    job, data = first(design)
    record, _ = study.evaluate_case(data, design, job)
    assert record["case_status"] == "partial" and len(models) == 14
    assert all(row["status"] == "failed" and "Duplicate" in row["error"]["message"] for row in record["rows"])
    assert not any(event[0] == "predict" and event[2] == 56 for event in GeometrySpy.events)


@pytest.mark.parametrize("kind", ["third_feature", "nonfinite", "wrong_labels"])
def test_training_rejects_invalid_inputs_before_any_fit(design, models, kind):
    _, data = first(design)
    X, y = data.fitting.X.copy(), data.fitting.y.copy()
    if kind == "third_feature":
        X = np.column_stack((X, np.zeros(len(X))))
    elif kind == "nonfinite":
        X[0, 0] = float("nan")
    else:
        y[y == 1] = 2
    with pytest.raises(ValueError):
        study.train_methods(X, y, data.selection.X, data.selection.y, design, 42)
    assert not models


def test_manifest_rejects_runtime_execution_flag_that_disagrees_with_file():
    cfg = protocol.load_protocol()
    cfg["execution_enabled"] = not cfg["execution_enabled"]
    with pytest.raises(ValueError, match="committed configuration file"):
        cli._manifest(cfg, protocol.DEFAULT_PROTOCOL)
