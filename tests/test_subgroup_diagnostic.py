from copy import deepcopy
import inspect
import json
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.noise_detection import noise_pre_detect
from src.results import ResultStore
from src.subgroup_diagnostic import (
    _TracedUGO, _describe_arrays, _prepared, _read_arrays, _save_arrays,
    array_hash, load_protocol, make_model, make_partition, paired_interval,
    resample_arm, run_mechanism, run_pilot, select_validation, test_metrics as evaluate_metrics,
    threshold_at_fpr, trace_groups, write_summary,
)
from src.ugo import UGO


def test_exact_strata_independent_splits_and_reproducible_features():
    p = load_protocol()
    F = make_partition(p, 42, "fit", "clean")
    again = make_partition(p, 42, "fit", "clean")
    V = make_partition(p, 42, "validation", "clean")
    assert np.array_equal(F.X, again.X)
    assert np.bincount(F.y).tolist() == [540, 60]
    assert [(F.groups == i).sum() for i in range(3)] == [42, 15, 3]
    assert [(V.groups == i).sum() for i in range(3)] == [140, 50, 10]
    assert not set(map(tuple, F.X)) & set(map(tuple, V.X))
    assert np.all(F.groups[F.y == 0] == -1)
    centers = np.array(p["centers"])
    distance = np.linalg.norm(F.X[:, None]-centers, axis=2).min(axis=1)
    assert (distance[F.y == 1] <= .1+1e-12).all()
    assert (distance[F.y == 0] >= .25-1e-12).all()


def test_invalid_protocol_and_resampler_fail_explicitly(tmp_path):
    p = load_protocol()
    p["seeds"] = [42, 42]
    path = tmp_path / "p.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="unique"):
        load_protocol(path)
    with pytest.raises(ValueError, match="Unknown"):
        resample_arm(np.zeros((2, 2)), np.array([0, 1]), None, "ugo_unknown", load_protocol(), 42)


@pytest.mark.parametrize("seed", [42, 43])
def test_tracing_preserves_original_ugo_arrays_and_decisions(seed):
    X, y = make_classification(n_samples=80, n_features=3, n_redundant=0,
                               weights=[.8, .2], class_sep=.8, random_state=seed)
    model = make_model("LogReg", seed)
    parameters = dict(variant="smote", random_state=seed, allow_provisional=True, phi=1., max_iter=3)
    original = UGO(model, **parameters)
    traced = _TracedUGO(model, **parameters)
    with pytest.warns(UserWarning):
        a, b = original.fit_resample(X, y)
    with pytest.warns(UserWarning):
        c, d = traced.fit_resample(X, y)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert original.diagnostics_["stopping_reason"] == traced.diagnostics_["stopping_reason"]
    for old, new in zip(original.diagnostics_["history"], traced.diagnostics_["history"]):
        assert old["selected_class"] == new["selected_class"]
        assert old["seed_indices"] == new["seed_indices"]
        assert old["welch"] == new["welch"]


def test_filter_is_eligibility_not_training_removal_and_ablations_are_distinct():
    p = load_protocol()
    F = make_partition(p, 42, "fit", "clean")
    _, X = _prepared(F)
    results = {}
    for arm in ["ugo_k5", "ugo_generation_filter_disabled", "ugo_prefilter_disabled"]:
        with pytest.warns(UserWarning):
            a, b, meta, sampler = resample_arm(X, F.y, make_model("LogReg", 42), arm, p, 42)
        np.testing.assert_array_equal(a[:len(X)], X)
        np.testing.assert_array_equal(b[:len(X)], F.y)
        results[arm] = meta
    original = results["ugo_k5"]
    generation = results["ugo_generation_filter_disabled"]
    full = results["ugo_prefilter_disabled"]
    np.testing.assert_array_equal(original["pre_noise_mask"], generation["pre_noise_mask"])
    assert np.asarray(original["pre_noise_mask"])[F.groups == 2].all()
    assert len(generation["eligible_indices"]) == 60
    assert not np.asarray(full["pre_noise_mask"]).any()
    assert len(full["scoring_eligible_indices"]) == 60
    assert noise_pre_detect is __import__("src.ugo", fromlist=["noise_pre_detect"]).noise_pre_detect


def test_disabled_filter_scope_restores_on_failure():
    p = load_protocol()
    with patch.object(_TracedUGO, "_validate_parameters", side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match="injected"):
            resample_arm(np.zeros((2, 2)), np.array([0, 1]), None, "ugo_prefilter_disabled", p, 42)
    import src.ugo as module
    assert module.noise_pre_detect is noise_pre_detect


def test_no_generation_does_not_count_as_active_mechanism():
    meta = {"eligible_indices": [0], "scoring_eligible_indices": [0],
            "pre_noise_mask": np.array([False, True]), "eligibility_stage": "test",
            "diagnostics": {"history": [{"generated_count": 0}],
                            "stopping_reason": "welch_fail_to_reject", "generated_noisy_samples_relocated": 0}}
    out = trace_groups(meta, np.array([0, 2]))
    assert out["pre_excluded_by_group"] == [0, 0, 1]
    assert out["minority_generation_rounds"] == out["ranked_rare_filter_opportunity_rounds"] == 0


def test_threshold_ties_and_reached_fpr_are_measured():
    y = np.array([0]*20+[1]*3)
    scores = np.r_[np.arange(20)/20, [.1, .94, .99]]
    chosen = threshold_at_fpr(y, scores, .05)
    assert chosen["threshold"] == .9
    assert chosen["fpr"] == .05
    assert chosen["tpr"] == pytest.approx(2/3)
    constant = threshold_at_fpr(y, np.full(23, .5), .05)
    assert constant["fpr"] == constant["tpr"] == 0
    with pytest.raises(ValueError):
        threshold_at_fpr(np.zeros(3), np.ones(3))


def test_test_groups_and_test_outcomes_cannot_select_neighborhood():
    records = {"a": {"status": "success", "validation": {"balanced_accuracy": .6}, "metrics": {"worst_recall": 1}},
               "b": {"status": "success", "validation": {"balanced_accuracy": .7}, "metrics": {"worst_recall": 0}}}
    assert select_validation(records, ["a", "b"]) == "b"
    records["a"]["metrics"]["worst_recall"] = -100
    assert select_validation(records, ["a", "b"]) == "b"
    records["b"]["validation"]["balanced_accuracy"] = .6
    assert select_validation(records, ["a", "b"]) == "a"
    assert "groups" not in inspect.signature(resample_arm).parameters
    assert "test" not in inspect.signature(resample_arm).parameters


def test_metrics_keep_absent_groups_undefined_and_ap_score_based():
    y = np.array([0, 0, 1, 1])
    scores = np.array([.1, .6, .4, .9])
    g = np.array([-1, -1, 0, 2])
    out = evaluate_metrics(y, scores, g, .5)
    assert out["subgroup_recall"] == [0, None, 1]
    assert out["worst_recall"] is None
    assert out["fpr"] == .5
    assert out["average_precision"] == pytest.approx(5/6)
    assert out["subgroup_denominators"] == [1, 0, 1]


def test_cache_checksum_and_run_identity_protect_outputs(tmp_path):
    store = ResultStore(tmp_path, {"frozen": 1})
    X, y = np.arange(8).reshape(4, 2), np.array([0, 0, 1, 1])
    job = {"a": 1}
    arrays = _save_arrays(store, job, X, y, np.ones(4))
    record = {"arrays": arrays}
    a, b, weights = _read_arrays(store, record)
    np.testing.assert_array_equal(a, X)
    assert weights.sum() == 4
    with (tmp_path / arrays["file"]).open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        _read_arrays(store, record)
    with pytest.raises(ValueError, match="different"):
        ResultStore(tmp_path, {"frozen": 2})


def test_test_phase_requires_fitting_only_gate(tmp_path):
    store = ResultStore(tmp_path, {"frozen": 1})
    with patch("src.subgroup_diagnostic.make_partition", side_effect=AssertionError("should not construct data")):
        with pytest.raises(ValueError, match="mechanism"):
            run_pilot(store, load_protocol(), progress=lambda _: None)


def test_resampling_and_final_models_are_group_blind_and_validation_precedes_test(tmp_path):
    p = deepcopy(load_protocol())
    p["seeds"], p["regimes"], p["classifiers"] = [987], {"clean": 0.}, ["LogReg"]
    p["ugo"]["max_iter"] = 1
    store = ResultStore(tmp_path, {"unit_test_protocol": p})
    constructed = []
    original = make_partition
    def track(*args):
        constructed.append(args[2])
        return original(*args)
    with patch("src.subgroup_diagnostic.make_partition", side_effect=track):
        mechanisms = run_mechanism(store, p, progress=lambda _: None)
    assert set(constructed) == {"fit"}
    assert all(r["status"] == "success" for r in mechanisms), [r.get("error") for r in mechanisms]
    # Simulate an interrupted/incomplete fitting-only run: it must not enter T.
    first = mechanisms[0]
    saved_bytes = store._path(first["job"]).read_bytes()
    store._path(first["job"]).unlink()
    with patch("src.subgroup_diagnostic.make_partition", side_effect=AssertionError("no data yet")):
        with pytest.raises(ValueError, match="Incomplete"):
            run_pilot(store, p, progress=lambda _: None)
    store._path(first["job"]).write_bytes(saved_bytes)
    constructed.clear()
    selected = []
    def select(records, candidates):
        assert "test" not in constructed
        selected.append(True)
        return select_validation(records, candidates)
    with patch("src.subgroup_diagnostic.make_partition", side_effect=track), \
         patch("src.subgroup_diagnostic.select_validation", side_effect=select):
        records = run_pilot(store, p, progress=lambda _: None)
    assert len(selected) == 2
    assert all(r["status"] == "success" for r in records), [r.get("error") for r in records]
    again = run_pilot(store, p, progress=lambda _: None)
    assert records == again
    summary = write_summary(store, records, p)
    assert summary["status_counts"] == {"success": len(records)}
    assert (tmp_path / "metrics.csv").exists()


def test_paired_intervals_use_whole_replicates():
    p = load_protocol()
    out = paired_interval([.1]*10, p)
    assert out["n"] == 10
    assert out["low"] == pytest.approx(.1)
    assert out["high"] == pytest.approx(.1)
    assert paired_interval([], p)["mean_difference"] is None
