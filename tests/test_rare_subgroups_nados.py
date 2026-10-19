"""Artificial software fixtures only; the disabled 90-case study is never run."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
from sklearn.base import BaseEstimator

import scripts.run_rare_subgroups_nados as cli
from scripts.run_rare_subgroups_pilot import _save_predictions
from src.nados import NADOSUnsupportedError
import src.rare_subgroups_contamination as parent
import src.rare_subgroups_nados_study as study
import src.rare_subgroups_study as baseline
from src.results import ResultStore


@pytest.fixture
def design():
    d = study.runtime_design(study.load_protocol())
    d["seeds"] = [42]
    distribution = d["distribution"]
    distribution["fit_counts"].update(
        true_majority=30, true_total_minority=12, rare_grid=[4], total_rows=42
    )
    distribution["fitting_pool_sizes"].update(common=12, rare=4)
    distribution["selection_counts"] = {"majority": 20, "common": 6, "rare": 2}
    distribution["calibration_counts"] = {"majority": 20, "common": 0, "rare": 0}
    distribution["test_counts"] = {"majority": 40, "common": 12, "rare": 4}
    d["label_corruption"]["flipped_majority_count_grid"] = [0, 2]
    d["threshold"].update(calibration_negatives=20, target_false_positive_count=1)
    d["classifier"]["n_estimators"] = 3
    return d


def first(design, noise=2):
    job = {"rare_count": 4, "rare_sigma": .35, "flipped_count": noise, "seed": 42}
    return job, parent.generate_case(design, job)


class GeometrySpy(BaseEstimator):
    fits = []

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fit_X_ = np.asarray(X).copy()
        self.fit_y_ = np.asarray(y).copy()
        self.fits.append(self)
        return self

    def predict_proba(self, X):
        scores = 1 / (1 + np.exp(-np.asarray(X)[:, 0]))
        return np.column_stack((1 - scores, scores))


def test_frozen_protocol_reuses_exact_parent_90_jobs_and_rejects_design_change(tmp_path):
    config = study.load_protocol()
    assert study.planned_jobs(config) == parent.planned_jobs(parent.load_protocol()["scientific_design"])
    assert len(study.planned_jobs(config)) == 90
    assert config["scientific_design"]["case_source"]["scientific_design_sha256"] == parent.FROZEN_DESIGN_SHA256
    changed = deepcopy(config)
    changed["scientific_design"]["nados_conventions"]["reliability_threshold"] = .4
    changed["design_sha256"] = study.design_hash(changed["scientific_design"])
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen NADOS"):
        study.load_protocol(path)


@pytest.mark.parametrize("value", [None, 1, "false"])
def test_execution_flag_must_be_boolean(value):
    config = study.load_protocol()
    config["execution_enabled"] = value
    with pytest.raises(ValueError, match="boolean"):
        study.validate_protocol(config)


def test_check_only_and_disabled_run_do_no_work(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled preflight must not generate cases, fit models or create a manifest")
    monkeypatch.setattr(cli, "generate_case", forbidden)
    monkeypatch.setattr(cli, "evaluate_case", forbidden)
    monkeypatch.setattr(cli, "_manifest", forbidden)
    output = tmp_path / "never-create"
    assert cli.main(["--check-only", "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == plan["max_classifier_fits"] == plan["planned_score_rows"] == 90
    assert not plan["execution_enabled"] and not plan["data_generated"] and not output.exists()
    assert cli.main(["--run", "--output", str(output)]) == 2
    assert "disabled" in capsys.readouterr().err and not output.exists()


def test_sampler_retains_originals_reconstructs_rows_and_meets_budget(design):
    job, data = first(design)
    sampled = study.sample_nados(data.fitting.X, data.fitting.y, design, job["seed"])
    assert sampled.metadata["status"] == "success"
    assert sampled.metadata["generated_count"] == 14
    np.testing.assert_array_equal(sampled.X[:len(data.fitting.X)], data.fitting.X)
    np.testing.assert_array_equal(sampled.y[:len(data.fitting.y)], data.fitting.y)
    trace = sampled.trace
    expected = (data.fitting.X[trace["generated_seed_indices"]]
                + trace["interpolation_fractions"][:, None]
                * (data.fitting.X[trace["generated_partner_indices"]]
                   - data.fitting.X[trace["generated_seed_indices"]]))
    np.testing.assert_array_equal(sampled.X[len(data.fitting.X):], expected)
    assert set(trace["generated_seed_indices"]) <= set(trace["eligible_indices"])


def test_training_api_cannot_receive_hidden_metadata(design):
    X = np.arange(84, dtype=float).reshape(42, 2)
    y = np.r_[np.zeros(30, dtype=int), np.ones(12, dtype=int)]
    with pytest.raises(TypeError):
        study.train_nados(X, y, design, 42, groups=np.zeros(42))


def test_hand_eligibility_and_parent_exposure_arithmetic():
    fitting = type("Fitting", (), {
        "groups": np.array([1, 2, 0, 1, 2]),
        "corrupted": np.array([False, False, True, False, False]),
    })()
    trace = {
        "minority_indices": np.array([0, 1, 2, 3, 4]),
        "eligible_mask": np.array([True, False, True, False, True]),
        "reliability_scores": np.array([.8, .2, .7, .4, .9]),
        "difficulty_scores": np.array([.3, .6, .5, .7, .1]),
        "generated_seed_indices": np.array([0, 2, 2, 4]),
        "generated_partner_indices": np.array([3, 0, 2, 1]),
    }
    groups = study.eligibility_diagnostics(fitting, trace)
    assert groups["common"] == {"original": 2, "eligible": 1, "excluded": 1, "fraction": .5,
                                "mean_reliability": pytest.approx(.6),
                                "mean_difficulty": pytest.approx(.5)}
    assert groups["rare"]["eligible"] == 1 and groups["rare"]["fraction"] == .5
    assert groups["noise"]["eligible"] == 1 and groups["noise"]["fraction"] == 1
    parents = study.parent_diagnostics(fitting.corrupted, trace)
    assert parents == {"kind": "interpolation", "generated": 4, "clean_clean": 2,
                       "clean_corrupt": 1, "corrupt_corrupt": 1,
                       "any_corrupted_parent": 2, "corrupted_parent_fraction": .5}


def test_hidden_fitting_truth_cannot_change_training_trace_or_scores(design, monkeypatch):
    GeometrySpy.fits = []
    monkeypatch.setattr(baseline, "_new_classifier", lambda config, seed, weighted=False: GeometrySpy())
    job, data = first(design)
    a, av = study.evaluate_case(data, design, job)
    changed_fitting = replace(
        data.fitting,
        true_y=1 - data.fitting.true_y,
        groups=np.roll(data.fitting.groups, 3),
        corrupted=np.roll(data.fitting.corrupted, 5),
        stable_ids=np.roll(data.fitting.stable_ids, 7),
    )
    b, bv = study.evaluate_case(replace(data, fitting=changed_fitting), design, job)
    assert len(GeometrySpy.fits) == 2
    np.testing.assert_array_equal(GeometrySpy.fits[0].fit_X_, GeometrySpy.fits[1].fit_X_)
    np.testing.assert_array_equal(GeometrySpy.fits[0].fit_y_, GeometrySpy.fits[1].fit_y_)
    for key in a["rows"][0]["trace_vectors"].values():
        np.testing.assert_array_equal(av[key], bv[key])
    np.testing.assert_array_equal(av["calibration__NADOS"], bv["calibration__NADOS"])
    np.testing.assert_array_equal(av["test__NADOS"], bv["test__NADOS"])
    assert a["scaler"]["fitted_rows"] == len(data.fitting.X)
    np.testing.assert_allclose(a["scaler"]["mean"], data.fitting.X.mean(axis=0))


def test_calibration_target_trace_vectors_and_real_small_forest(design):
    job, data = first(design)
    record, vectors = study.evaluate_case(data, design, job)
    row = record["rows"][0]
    assert record["case_status"] == "complete" and record["classifier_fit_attempts"] == 1
    assert row["status"] == "success" and row["metrics"]["calibration_fp"] == 1
    assert row["metrics"]["calibration_fpr"] == .05
    assert row["eligibility"]["common"]["original"] == 8
    assert row["eligibility"]["rare"]["original"] == 4
    assert row["eligibility"]["noise"]["original"] == 2
    assert all(study.array_hash(value) == record["vector_sha256"][name]
               for name, value in vectors.items())


def test_unsupported_geometry_has_no_fallback_or_classifier_fit(design, monkeypatch):
    class Unsupported:
        def __init__(self, **kwargs):
            pass
        def fit_resample(self, X, y):
            raise NADOSUnsupportedError("no_positive_synthesis_priority", "Artificial fixture")
    monkeypatch.setattr(study, "NADOS", Unsupported)
    sampled = study.sample_nados(np.zeros((8, 2)), np.r_[np.zeros(5), np.ones(3)], design, 42)
    assert sampled.X is None and sampled.metadata["status"] == "unsupported"
    assert sampled.metadata["error"]["reason"] == "no_positive_synthesis_priority"
    fitted = study.train_nados(np.zeros((8, 2)), np.r_[np.zeros(5), np.ones(3)], design, 42)
    assert fitted.model is None and not fitted.record["classifier_fit_attempted"]


def test_checkpoint_vectors_verify_and_corruption_is_rejected(design, tmp_path):
    job, data = first(design)
    record, vectors = study.evaluate_case(data, design, job)
    store = ResultStore(tmp_path, {"fixture": "nados"})
    record["predictions"] = _save_predictions(tmp_path, job, vectors)
    saved = store.write(record)
    cli._verify_vectors(tmp_path, saved)
    path = tmp_path / saved["predictions"]["path"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        cli._verify_vectors(tmp_path, saved)


def test_resume_reuses_verified_case_without_refitting(monkeypatch, tmp_path, design):
    config = study.load_protocol()
    config["execution_enabled"] = True
    job, data = first(design)
    calls = {"evaluate": 0}
    monkeypatch.setattr(cli, "preflight", lambda cfg: {})
    monkeypatch.setattr(cli, "_manifest", lambda cfg, path: {
        "fixture": "resume", "git": {"tracked_changes": False}
    })
    monkeypatch.setattr(cli, "planned_jobs", lambda cfg: [job])
    monkeypatch.setattr(cli, "generate_case", lambda cfg, current: data)
    monkeypatch.setattr(cli, "runtime_design", lambda cfg: design)
    original = study.evaluate_case
    def counted(case, current_design, current_job):
        calls["evaluate"] += 1
        return original(case, current_design, current_job)
    monkeypatch.setattr(cli, "evaluate_case", counted)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "results"
    assert cli.run_study(config, config_path, output)["status"] == "complete"
    assert cli.run_study(config, config_path, output)["status"] == "complete"
    assert calls["evaluate"] == 1
    assert len(list(output.glob("job-*.json"))) == 1
