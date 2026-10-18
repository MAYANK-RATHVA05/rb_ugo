"""Structural/reference/metric checks; mock learners never train classifiers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import scripts.diagnose_ppeo_geometry as diagnostic
from src.ppeo_reference import allocate_anchor_quotas, fit_phhc, generate_large_smote


def _config():
    fit = {"main_minority": 80, "rare_minority_each": 8,
           "main_majority": 240, "rare_majority_each_side": 60}
    return {"base_seed": 42, "blocks": [{"elongation": 1., "angle_deg": 0.,
                                        "majority_offset": 2.4}],
            "sample_counts": {key: {name: value * scale for name, value in fit.items()}
                              for key, scale in (("fit", 1), ("validation", 5), ("test", 50))},
            "sphere_scales": [.25, .5, .75], "methods": list(diagnostic.METHODS)}


def test_phhc_rollback_and_small_prototype_graph_are_from_observed_geometry():
    components = diagnostic.make_components(1., 0., 2.4)
    stream = np.random.SeedSequence(42).spawn(8)[0].spawn(7)[0]
    X, y, _ = diagnostic.make_partition(components, 1, np.random.default_rng(stream))
    scaled = StandardScaler().fit_transform(X)
    fitted = fit_phhc(scaled, y)
    assert fitted.diagnostics["restored_previous_model"]
    assert fitted.diagnostics["f1_history"] == [pytest.approx(.8465608465608465)]
    assert fitted.diagnostics["prototype_sizes"] == [8, 8, 80]
    assert fitted.small_flags.tolist() == [True, True, False]
    centers = np.asarray(fitted.diagnostics["final_cluster_centers"])
    np.testing.assert_allclose(centers, np.array([scaled[y == label].mean(axis=0)
                                                for label in (0, 1)]))
    repeated = fit_phhc(scaled, y)
    np.testing.assert_array_equal(fitted.prototype_ids, repeated.prototype_ids)


def test_phhc_rejects_invalid_or_one_class_inputs():
    for X, y in (([[0.], [1.]], [1, 1]), ([[0.], [np.nan]], [0, 1]),
                 ([[0.], [1.]], [0, 1, 1])):
        with pytest.raises(ValueError):
            fit_phhc(X, y)


def test_phhc_opposite_label_coordinate_ties_fail_explicitly_instead_of_looping():
    with pytest.raises(RuntimeError, match="repeated an observed partition"):
        fit_phhc(np.array([[0., 0.], [0., 0.]]), np.array([0, 1]))


def test_fixed_quota_allocation_uses_original_prototype_members_only():
    ids = np.r_[np.zeros(80, dtype=int), np.ones(8, dtype=int), np.full(8, 2)]
    quotas = allocate_anchor_quotas(ids, 384, seed=42)
    np.testing.assert_array_equal(quotas, allocate_anchor_quotas(ids, 384, seed=42))
    assert len(quotas) == len(ids) and quotas.sum() == 384
    assert quotas[80:].sum() > quotas[:80].sum()
    with pytest.raises(ValueError):
        allocate_anchor_quotas([0, -1], 10, seed=42)


def test_common_large_smote_uses_within_prototype_original_neighbors_and_quotas():
    X = np.array([[0., 0.], [1., 0.], [20., 20.], [100., 100.]])
    ids, quotas, flags = np.array([0, 0, 1, 2]), np.array([4, 3, 2, 5]), np.array([False, True, False])
    rows, anchors, info = generate_large_smote(X, ids, quotas, flags, seed=42)
    assert rows.shape == (12, 2) and info["singleton_copies"] == 5
    assert 2 not in anchors
    np.testing.assert_array_equal(np.bincount(anchors, minlength=4), [4, 3, 0, 5])
    np.testing.assert_array_equal(rows[anchors == 3], np.full((5, 2), 100.))
    assert np.all((rows[anchors < 2, 0] >= 0) & (rows[anchors < 2, 0] <= 1))


@pytest.mark.parametrize("elongation,angle", [(1, 0), (1, 60), (6, 0), (6, 60)])
def test_controlled_overlap_is_affine_invariant_and_other_class_supports_do_not_intersect(elongation, angle):
    for offset, expected_pairs in ((2.4, 0), (1.2, 4)):
        components = diagnostic.make_components(elongation, angle, offset)
        intersections = diagnostic.check_allowed_intersections(components, offset)
        assert sum(row["different_labels"] for row in intersections) == expected_pairs
        rare = next(c for c in components if c.name == "minority_rare_1")
        neighbor = next(c for c in components if c.name == "majority_rare_1_+1")
        displacement = np.linalg.solve(rare.transform, neighbor.center - rare.center)
        assert np.linalg.norm(displacement) == pytest.approx(offset)
    assert diagnostic.disk_overlap_fraction(2.4) == 0
    assert diagnostic.disk_overlap_fraction(1.2) == pytest.approx(.284756979865294)


def test_true_supports_and_split_proportions_are_evaluator_only():
    components = diagnostic.make_components(6, 60, 1.2)
    for multiplier in (1, 5, 50):
        X, y, groups = diagnostic.make_partition(components, multiplier, np.random.default_rng(multiplier))
        assert X.shape == (576 * multiplier, 2)
        assert (y == 1).sum() == 96 * multiplier
        assert (y == 0).sum() == 480 * multiplier
        for group in (1, 2):
            assert (groups == group).sum() == 8 * multiplier
        assert diagnostic.support_membership(X[y == 1], components, 1).all()
        assert diagnostic.support_membership(X[y == 0], components, 0).all()
    assert not diagnostic.support_membership(np.array([[1e6, 1e6]]), components, 1)[0]


@pytest.mark.parametrize("scores", [np.array([.2, .2, .2, .9, .9, .7]),
                                   np.array([.1, .2, .3, .4, .5, .6]),
                                   np.ones(6)])
def test_independent_average_precision_agrees_with_grouped_sklearn_metric(scores):
    labels = np.array([1, 0, 1, 0, 0, 1])
    assert diagnostic.average_precision(labels, scores) == pytest.approx(
        average_precision_score(labels, scores))


def test_average_precision_no_positive_convention_and_invalid_score_failure():
    assert diagnostic.average_precision([0, 0], [.1, .2]) == 0
    with pytest.raises(ValueError):
        diagnostic.average_precision([0, 1], [.1, np.nan])


def test_validation_threshold_uses_lowest_acceptable_threshold_and_preserves_ties():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([.1, .2, .2, .8, .7, .9])
    assert diagnostic.choose_threshold(labels, scores, .25) == .7
    assert diagnostic.choose_threshold(labels, scores, 0) == .9
    threshold = diagnostic.choose_threshold(labels, np.ones(6), .05)
    assert threshold == np.nextafter(1., np.inf)
    with pytest.raises(ValueError, match="negatives"):
        diagnostic.choose_threshold([1, 1], [.1, .2], .05)


def test_metrics_are_counted_independently_and_require_rare_evaluation_rows():
    labels = np.array([0, 0, 1, 1, 1, 1])
    scores = np.array([.1, .9, .8, .4, .7, .3])
    groups = np.array([-1, -1, 1, 1, 2, 2])
    metrics = diagnostic.evaluate_scores(labels, scores, groups, .5)
    assert metrics["tp"] == 2 and metrics["fp"] == 1
    assert metrics["rare_macro_recall"] == .5 and metrics["fpr"] == .5
    with pytest.raises(ValueError, match="rare subgroup"):
        diagnostic.evaluate_scores(labels, scores, [-1, -1, 0, 0, 0, 0], .5)


def test_frozen_budget_counts_selected_sphere_once_and_rejects_changed_counts():
    assert diagnostic.fit_budget(8, 8, [.25, .5, .75]) == {
        "final_models": 64, "total_fits_including_selection": 80}
    with pytest.raises(ValueError):
        diagnostic.fit_budget(13, 8, [.25, .5, .75])
    config = _config()
    config["sample_counts"]["test"]["rare_minority_each"] = 800
    with pytest.raises(ValueError, match="counts"):
        diagnostic._validate_config(config)


def test_actual_generation_checker_checks_both_constraints_and_anchor_means():
    X_minority, X_majority = np.array([[0., 0.]]), np.array([[1., 0.], [0., 1.]])
    rows = np.array([[.2, .1], [-.2, -.1], [0., 0.]])
    info = diagnostic.validate_generation(rows, np.zeros(3, dtype=int), X_minority,
                                          X_majority, "symmetric_clip", .9)
    assert info["maximum_anchor_mean_error"] == 0
    assert info["minimum_constraint_slack"] == pytest.approx(.5)
    assert info["strict_original_majority_certificate_rows"] == 3
    with pytest.raises(AssertionError):
        diagnostic.validate_generation(np.array([[.8, 0.], [-.8, 0.]]), np.zeros(2, dtype=int),
                                       X_minority, X_majority, "symmetric_clip", .9)


class MockLearner:
    fits = 0

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        type(self).fits += 1
        return self

    def predict_proba(self, X):
        positive = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - positive, positive])


def _mock_run(monkeypatch, tmp_path, factory):
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(_config()))
    monkeypatch.setattr(diagnostic, "verify_identity", lambda *args: {"mock_software_check": True})
    monkeypatch.setattr(diagnostic, "_sha", lambda path: "mock-software-test-identity")
    output = tmp_path / "mock-records"
    output.mkdir()
    monkeypatch.setattr(diagnostic, "_ensure_ignored_output", lambda path: output)
    return diagnostic.run(config_path, tmp_path / "identity.json", output,
                          learner_factory=factory), output


def test_mock_runner_accounts_for_search_without_refitting_selected_model(monkeypatch, tmp_path):
    MockLearner.fits = 0
    summary, output = _mock_run(monkeypatch, tmp_path, MockLearner)
    assert MockLearner.fits == summary["actual_fits"] == 10
    assert summary["final_records"] == 8
    assert len(summary["sphere_candidates"]) == 3
    assert len(list(output.glob("record-*.json"))) == 8
    assert {record["quotas_sha256"] for record in summary["records"]}.__len__() == 1
    assert all(r["generated_count"] == (0 if r["method"] == "nores" else 384)
               for r in summary["records"])


def test_mock_runner_retains_failure_record_and_does_not_retry(monkeypatch, tmp_path):
    class FailingMock(MockLearner):
        def fit(self, X, y):
            raise RuntimeError("deliberate mock fit failure")
    with pytest.raises(RuntimeError, match="deliberate mock"):
        _mock_run(monkeypatch, tmp_path, FailingMock)
    saved = json.loads((tmp_path / "mock-records/record-00-spsdo.json").read_text())
    assert saved["status"] == "failed" and saved["actual_fits_so_far"] == 0


def test_identity_mismatch_stops_before_learner_fit(tmp_path):
    config, identity = tmp_path / "protocol.json", tmp_path / "identity.json"
    config.write_text("{}")
    identity.write_text(json.dumps({"protocol_sha256": "wrong"}))
    with pytest.raises(ValueError, match="protocol hash"):
        diagnostic.verify_identity(config, identity)
