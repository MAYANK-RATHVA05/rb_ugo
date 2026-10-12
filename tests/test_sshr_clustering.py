"""Independent numerical checks for the audited SSHC clustering reproduction.

Artificial geometry is used only to test software. No resampling, hidden-label
evaluation, final classifier, or real-data study belongs to this stage.
"""

import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.sshr_clustering import SSHC, clustering_objective


def simple_training_geometry():
    # Class centroids before unlabelled assignment: majority 12, minority 1.
    # The unlabelled points alter the final means without changing membership.
    return (np.array([[0.], [2.], [10.], [12.], [14.]]),
            np.array([1, 1, 0, 0, 0]), np.array([[3.], [15.]]))


def assert_partition_is_conserved(model, X_labeled, y_labeled, X_unlabeled):
    combined = np.vstack((X_labeled, X_unlabeled))
    members = np.concatenate(model.clusters_)
    np.testing.assert_array_equal(np.sort(members), np.arange(len(combined)))
    assert len(np.unique(model.cluster_ids_)) == len(model.cluster_ids_)
    assert len(model.centroids_) == len(model.clusters_) == len(model.cluster_labels_)
    assignments = np.r_[model.labeled_cluster_ids_, model.unlabeled_cluster_ids_]
    for cluster_id, members, label in zip(
            model.cluster_ids_, model.clusters_, model.cluster_labels_):
        np.testing.assert_array_equal(assignments[members], np.repeat(cluster_id, len(members)))
        labelled_members = members[members < len(y_labeled)]
        np.testing.assert_array_equal(model.labeled_predictions_[labelled_members],
                                      np.repeat(label, len(labelled_members)))
    assert np.isfinite(model.centroids_).all()
    assert np.isfinite(model.objective_)
    # Dumps with allow_nan=False catches both nonportable scalar types and NaNs.
    json.loads(json.dumps(model.diagnostics_, allow_nan=False))


def test_printed_equation_is_literal_and_flags_value_larger_than_one():
    truth = np.r_[np.ones(10, dtype=int), np.zeros(100, dtype=int)]
    score = clustering_objective(truth, truth, minority_label=1)
    assert score["counts"] == {"TP": 10, "TN": 100, "FP": 0, "FN": 0}
    assert score["accuracy"] == 1
    assert score["f1"] == pytest.approx(20 / 11)
    assert score["objective"] == pytest.approx(31 / 22)
    assert score["defined"] is True
    assert score["anomaly"] is True


def test_standard_minority_f1_is_explicit_and_differs_from_printed_equation():
    truth = np.array([1, 1, 0, 0, 0])
    predictions = np.array([1, 0, 1, 0, 0])
    printed = clustering_objective(truth, predictions, 1)
    standard = clustering_objective(truth, predictions, 1,
                                    objective_mode="standard_minority_f1")
    assert printed["counts"] == {"TP": 1, "TN": 2, "FP": 1, "FN": 1}
    assert printed["accuracy"] == standard["accuracy"] == pytest.approx(3 / 5)
    assert printed["f1"] == pytest.approx(2 / 3)
    assert printed["objective"] == pytest.approx(19 / 30)
    assert standard["f1"] == pytest.approx(1 / 2)
    assert standard["objective"] == pytest.approx(11 / 20)
    assert standard["anomaly"] is False


@pytest.mark.parametrize("mu,expected", [(0, 2 / 3), (1, 3 / 5), (.25, .65)])
def test_objective_mu_weights_the_two_verified_terms(mu, expected):
    score = clustering_objective([1, 1, 0, 0, 0], [1, 0, 1, 0, 0], 1, mu=mu)
    assert score["objective"] == pytest.approx(expected)


def test_undefined_printed_denominator_is_recorded_without_nan_or_clipping():
    score = clustering_objective([1, 0], [0, 1], 1)
    assert score["counts"] == {"TP": 0, "TN": 0, "FP": 1, "FN": 1}
    assert score["defined"] is False
    assert score["f1"] is None
    assert score["objective"] is None
    json.dumps(score, allow_nan=False)


def test_initial_class_means_nearest_assignment_and_all_member_refinement():
    X, y, U = simple_training_geometry()
    model = SSHC().fit(X, y, U)
    assert model.fit(X, y, U) is model
    initial = dict(zip(model.diagnostics_["initial_cluster_ids"],
                       model.diagnostics_["initial_centroids"]))
    # np.unique orders numeric labels as 0 then 1; persistent IDs start at zero.
    np.testing.assert_allclose(initial[0], [12.])
    np.testing.assert_allclose(initial[1], [1.])
    np.testing.assert_array_equal(model.diagnostics_["initial_assignment"],
                                  [1, 1, 0, 0, 0, 1, 0])
    by_label = dict(zip(model.cluster_labels_, model.centroids_))
    np.testing.assert_allclose(by_label[0], [51 / 4])
    np.testing.assert_allclose(by_label[1], [5 / 3])
    np.testing.assert_array_equal(model.labeled_predictions_, y)
    assert model.objective_ == pytest.approx(1.1)
    assert_partition_is_conserved(model, X, y, U)


def test_unlabelled_features_affect_geometry_but_do_not_become_observed_labels():
    X, y, _ = simple_training_geometry()
    near = SSHC().fit(X, y, [[3.], [15.]])
    farther = SSHC().fit(X, y, [[4.], [16.]])
    assert not np.array_equal(near.centroids_, farther.centroids_)
    assert len(near.labeled_predictions_) == len(y)
    assert sum(item["labeled_size"] for item in near.diagnostics_["clusters"]) == len(y)
    assert sum(item["unlabeled_size"] for item in near.diagnostics_["clusters"]) == 2
    assert "y_unlabeled" not in inspect.signature(SSHC.fit).parameters
    assert not any("hidden" in name for name in vars(near))


def test_inputs_are_not_mutated_and_explicit_empty_pool_matches_none():
    X, y, U = simple_training_geometry()
    originals = [array.copy() for array in (X, y, U)]
    SSHC().fit(X, y, U)
    for array, original in zip((X, y, U), originals):
        np.testing.assert_array_equal(array, original)
    absent = SSHC().fit(X, y)
    empty = SSHC().fit(X, y, np.empty((0, 1)))
    np.testing.assert_array_equal(absent.centroids_, empty.centroids_)
    np.testing.assert_array_equal(absent.labeled_cluster_ids_, empty.labeled_cluster_ids_)
    assert len(absent.unlabeled_cluster_ids_) == 0
    assert_partition_is_conserved(empty, X, y, np.empty((0, 1)))


def test_identical_points_have_stable_distance_ties_and_stop_without_fake_purity():
    X = np.zeros((5, 2))
    y = np.array([1, 1, 0, 0, 0])
    U = np.zeros((3, 2))
    first = SSHC(max_split_rounds=10).fit(X, y, U)
    second = SSHC(max_split_rounds=10).fit(X, y, U)
    np.testing.assert_array_equal(first.diagnostics_["initial_assignment"], np.zeros(8))
    # The duplicate initial centroids cannot separate contradictory labels.
    assert len(first.cluster_ids_) == 1
    assert first.cluster_ids_[0] == 0
    assert first.cluster_labels_[0] == 0
    np.testing.assert_array_equal(first.labeled_predictions_, np.zeros(5))
    assert first.objective_ == pytest.approx(.3)
    assert first.diagnostics_["clusters"][0]["impure"] is True
    assert first.diagnostics_["stop_reasons"]["split"] == "no_progress_partition"
    assert first.diagnostics_["guard_reached"] is False
    assert sum(item["phase"] == "split" for item in first.diagnostics_["history"]) == 1
    assert first.diagnostics_ == second.diagnostics_
    assert_partition_is_conserved(first, X, y, U)


def test_equal_observed_votes_prefer_birth_label_over_fixed_class_order():
    X = np.array([[0.], [2.], [0.], [2.], [10.]])
    y = np.array([1, 1, 0, 0, 0])
    model = SSHC().fit(X, y)
    # Four rows share the minority-centred cluster: 2 observed minority and
    # 2 majority. Its birth label 1 wins the vote tie despite classes_[0] == 0.
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.cluster_labels_, [0, 1])
    np.testing.assert_array_equal(model.labeled_predictions_, [1, 1, 1, 1, 0])
    assert model.objective_ == pytest.approx(.5)
    cluster = next(item for item in model.diagnostics_["clusters"] if item["id"] == 1)
    assert cluster["accuracy"] == pytest.approx(.5)
    assert cluster["impure"] is True
    assert_partition_is_conserved(model, X, y, np.empty((0, 1)))


def test_midpoint_unlabelled_distance_tie_prefers_persistent_id():
    X = np.array([[0.], [2.], [2.]])
    y = np.array([1, 0, 0])
    model = SSHC().fit(X, y, [[1.]])
    # The first source row is minority, but class order births majority ID 0.
    assert model.diagnostics_["initial_assignment"][-1] == 0
    assert model.unlabeled_cluster_ids_[0] == 0
    assert_partition_is_conserved(model, X, y, np.array([[1.]]))


def test_initial_geometric_assignment_does_not_pin_observed_labels():
    X = np.array([[0.], [10.], [4.], [5.], [6.], [7.], [8.]])
    y = np.array([1, 1, 0, 0, 0, 0, 0])
    model = SSHC().fit(X, y)
    # Means: class 0 is 6, class 1 is 5. Observed minority row 10 is closer
    # to the majority centroid, despite its known label.
    assert model.diagnostics_["initial_assignment"][1] == 0
    assert model.diagnostics_["initial_assignment"][2] == 1
    assert_partition_is_conserved(model, X, y, np.empty((0, 1)))


def test_real_impure_cluster_split_is_accepted_and_separates_two_minority_regions():
    X = np.array([[0.], [.5], [1.], [-10.], [9.]])
    y = np.array([0, 0, 0, 1, 1])
    U = np.array([[-9.], [9.5]])
    model = SSHC().fit(X, y, U)
    np.testing.assert_array_equal(model.diagnostics_["initial_assignment"],
                                  [0, 0, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(model.cluster_ids_, [0, 1, 2])
    np.testing.assert_allclose(model.centroids_, [[.5], [-9.5], [9.25]])
    np.testing.assert_array_equal(model.cluster_labels_, [0, 1, 1])
    np.testing.assert_array_equal(model.labeled_predictions_, y)
    assert model.objective_ == pytest.approx(1.1)
    assert any(item["phase"] == "split" and item["accepted"]
               for item in model.diagnostics_["history"])
    assert_partition_is_conserved(model, X, y, U)


def test_same_member_groups_with_changed_cluster_labels_are_meaningful_split_progress():
    X = np.array([[4.9], [4.5], [1.7], [-4.4], [-3.3], [-3.5],
                  [1.7], [-1.3], [5.]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    model = SSHC().fit(X, y)
    accepted = [event for event in model.diagnostics_["history"]
                if event["phase"] == "split" and event["accepted"]]
    assert len(accepted) >= 2
    first, second = accepted[:2]
    assert first["objective"]["objective"] == pytest.approx(2 / 3)
    assert second["objective"]["objective"] == pytest.approx(23 / 33)

    def grouping(snapshot):
        assignments = np.asarray(snapshot["assignments"])
        return {tuple(np.flatnonzero(assignments == cluster_id))
                for cluster_id in snapshot["cluster_ids"]}

    def row_labels(snapshot):
        mapping = dict(zip(snapshot["cluster_ids"], snapshot["labels"]))
        return [mapping[cluster_id] for cluster_id in snapshot["assignments"]]

    # Child birth-label voting changes predictions on duplicate coordinates.
    # Equating only row groups would throw away this legitimate improvement.
    assert grouping(first["snapshot"]) == grouping(second["snapshot"])
    assert row_labels(first["snapshot"]) != row_labels(second["snapshot"])
    assert_partition_is_conserved(model, X, y, np.empty((0, 1)))


def test_unlabelled_only_clusters_keep_birth_labels_and_have_no_observed_accuracy():
    X = np.array([[-7.], [-6.], [5.], [6.], [-8.], [8.]])
    y = np.array([0, 0, 0, 0, 1, 1])
    U = np.array([[-.5], [0.]])
    model = SSHC().fit(X, y, U)
    unanchored = [item for item in model.diagnostics_["clusters"]
                  if item["labeled_size"] == 0]
    assert len(unanchored) == 2
    for cluster in unanchored:
        assert cluster["unanchored"] is True
        assert cluster["accuracy"] is None
        assert cluster["label"] == cluster["birth_label"]
        assert cluster["unlabeled_size"] == 1
    # The contradictory labels in the two anchored clusters are still impure.
    assert sum(item["impure"] for item in model.diagnostics_["clusters"]) == 2
    assert_partition_is_conserved(model, X, y, U)


def test_real_split_objective_decline_rolls_back_every_model_component():
    X = np.array([[0.], [1.], [0.], [2.], [3.], [4.]])
    y = np.array([1, 1, 0, 0, 0, 0])
    model = SSHC().fit(X, y)
    initial = model.diagnostics_["history"][0]
    rejected = next(item for item in model.diagnostics_["history"]
                    if item["phase"] == "split")
    # A majority child at zero sacrifices one minority true positive. Printed
    # J changes from 11/12 to 31/36, so the initial complete state is restored.
    assert rejected["accepted"] is False
    assert rejected["reason"] == "objective_declined"
    assert rejected["objective"]["objective"] == pytest.approx(11 / 12)
    assert rejected["candidate_objective"]["objective"] == pytest.approx(31 / 36)
    assert rejected["snapshot"] == initial["snapshot"]
    np.testing.assert_array_equal(model.cluster_ids_, [0, 1])
    np.testing.assert_array_equal(model.birth_labels_, [0, 1])
    np.testing.assert_array_equal(model.labeled_cluster_ids_, [1, 1, 1, 0, 0, 0])
    assert_partition_is_conserved(model, X, y, np.empty((0, 1)))


@pytest.mark.parametrize("tol", [0, 1e6])
def test_refinement_rescores_new_assignment_and_rolls_back_decline_even_with_large_tol(tol):
    X = np.array([[0.], [1.], [2.], [10.], [11.]])
    y = np.array([0, 0, 0, 1, 1])
    U = np.array([[100.]])
    model = SSHC(tol=tol).fit(X, y, U)
    rejection = next(item for item in model.diagnostics_["history"]
                     if item["phase"] == "refine")
    # Updating minority centroid to 121/3 attracts only U; all observed rows
    # become majority. Evaluating stale assignments would incorrectly accept.
    assert rejection["accepted"] is False
    assert rejection["reason"] == "objective_declined"
    assert rejection["objective"]["objective"] == pytest.approx(1.1)
    assert rejection["candidate_objective"]["objective"] == pytest.approx(.3)
    np.testing.assert_allclose(model.centroids_, [[1.], [10.5]])
    np.testing.assert_array_equal(model.cluster_ids_, [0, 1])
    np.testing.assert_array_equal(model.cluster_labels_, [0, 1])
    np.testing.assert_array_equal(model.birth_labels_, [0, 1])
    np.testing.assert_array_equal(model.labeled_predictions_, y)
    assert model.objective_ == pytest.approx(1.1)
    assert_partition_is_conserved(model, X, y, U)


def test_pruning_scope_conflict_is_exposed_and_uses_unweighted_observed_accuracy():
    X = np.array([[0.], [2.], [0.], [10.], [11.], [12.]])
    y = np.array([1, 1, 0, 0, 0, 0])
    majority_only = SSHC(prune_scope="majority_only").fit(X, y)
    all_clusters = SSHC(prune_scope="all").fit(X, y)
    pruning = next(item for item in all_clusters.diagnostics_["history"]
                   if item["phase"] == "prune")["pruning"]
    # Majority cluster accuracy 1; minority cluster accuracy 2/3. Average is
    # 5/6. The latter is pruned only under the literal 'all' interpretation.
    assert pruning["average_accuracy"] == pytest.approx(5 / 6)
    assert pruning["removed_ids"] == [1]
    assert pruning["accepted"] is True
    assert len(majority_only.cluster_ids_) == 2
    assert len(all_clusters.cluster_ids_) == 1
    np.testing.assert_array_equal(all_clusters.cluster_labels_, [0])
    np.testing.assert_allclose(all_clusters.centroids_, [[35 / 6]])
    assert all_clusters.objective_ == pytest.approx(1 / 3)
    assert_partition_is_conserved(all_clusters, X, y, np.empty((0, 1)))


def test_single_observed_minority_anchor_is_valid_for_clustering():
    X = np.array([[0.], [10.], [11.]])
    y = np.array([1, 0, 0])
    model = SSHC().fit(X, y, [[1.]])
    np.testing.assert_allclose(model.centroids_, [[10.5], [.5]])
    np.testing.assert_array_equal(model.labeled_predictions_, y)
    assert_partition_is_conserved(model, X, y, np.array([[1.]]))


def test_refinement_guard_is_reported_separately_from_centroid_convergence():
    X, y, U = simple_training_geometry()
    guarded = SSHC(max_refine_iter=1, tol=0).fit(X, y, U)
    converged = SSHC(max_refine_iter=1, tol=2).fit(X, y, U)
    assert guarded.diagnostics_["stop_reasons"]["refine"] == "max_refine_iter"
    assert guarded.diagnostics_["guard_reached"] is True
    assert converged.diagnostics_["stop_reasons"]["refine"] == "centroids_converged"
    assert converged.diagnostics_["guard_reached"] is False
    # Both preserve the accepted updated state, despite different stopping.
    np.testing.assert_array_equal(guarded.centroids_, converged.centroids_)


def test_split_guard_does_not_claim_no_impurity_convergence():
    X = np.array([[0.], [.5], [1.], [-10.], [9.]])
    y = np.array([0, 0, 0, 1, 1])
    model = SSHC(max_split_rounds=1).fit(X, y)
    assert model.diagnostics_["stop_reasons"]["split"] == "max_split_rounds"
    assert model.diagnostics_["guard_reached"] is True
    assert len(model.cluster_ids_) == 3


def test_undefined_candidate_is_rejected_without_changing_prior_snapshot():
    class UndefinedSplitObjective(SSHC):
        def _evaluate(self, y, state):
            score = super()._evaluate(y, state)
            if len(state.ids) > 2:
                return {**score, "defined": False, "f1": None, "objective": None}
            return score

    X = np.array([[0.], [.5], [1.], [-10.], [9.]])
    y = np.array([0, 0, 0, 1, 1])
    model = UndefinedSplitObjective().fit(X, y)
    initial, split = model.diagnostics_["history"][:2]
    assert split["phase"] == "split"
    assert split["accepted"] is False
    assert split["reason"] == "undefined_candidate_objective"
    assert split["snapshot"] == initial["snapshot"]
    assert split["candidate_objective"]["objective"] is None
    # Later pruning is a separate paper step and may change this restored
    # state; the rollback contract concerns the rejected split itself.
    assert split["snapshot"]["cluster_ids"] == [0, 1]
    assert_partition_is_conserved(model, X, y, np.empty((0, 1)))


def test_undefined_initial_objective_raises_and_clears_partial_fitted_state():
    class UndefinedInitialObjective(SSHC):
        def _evaluate(self, y, state):
            score = super()._evaluate(y, state)
            return {**score, "defined": False, "f1": None, "objective": None}

    X, y, U = simple_training_geometry()
    model = UndefinedInitialObjective()
    with pytest.raises(ValueError, match="Initial SSHC objective is undefined"):
        model.fit(X, y, U)
    assert not any(name.endswith("_") for name in vars(model))


def test_string_labels_preserve_original_class_domain():
    X, y, U = simple_training_geometry()
    names = np.where(y == 1, "rare", "common")
    model = SSHC(minority_label="rare").fit(X, names, U)
    assert model.minority_label_ == "rare"
    assert model.majority_label_ == "common"
    np.testing.assert_array_equal(model.labeled_predictions_, names)
    assert set(model.cluster_labels_) <= {"rare", "common"}
    assert_partition_is_conserved(model, X, names, U)


def test_equal_counts_require_an_explicit_minority_identity():
    X = np.array([[0.], [1.], [10.], [11.]])
    y = np.array([1, 1, 0, 0])
    with pytest.raises(ValueError):
        SSHC().fit(X, y)
    model = SSHC(minority_label=1).fit(X, y)
    assert model.minority_label_ == 1
    assert model.majority_label_ == 0
    np.testing.assert_array_equal(model.labeled_predictions_, y)


def test_clone_parameters_and_repeated_fits_are_deterministic():
    X, y, U = simple_training_geometry()
    estimator = SSHC(mu=.25, objective_mode="standard_minority_f1",
                     prune_scope="all", tol=1e-6, max_split_rounds=7,
                     max_refine_iter=9)
    another = clone(estimator)
    assert another.get_params() == estimator.get_params()
    first, second = estimator.fit(X, y, U), another.fit(X, y, U)
    np.testing.assert_array_equal(first.centroids_, second.centroids_)
    np.testing.assert_array_equal(first.cluster_labels_, second.cluster_labels_)
    np.testing.assert_array_equal(first.labeled_cluster_ids_, second.labeled_cluster_ids_)
    assert first.diagnostics_ == second.diagnostics_


@pytest.mark.parametrize("name,value", [
    ("mu", -.1), ("mu", 1.1), ("mu", np.nan), ("mu", True),
    ("objective_mode", "f1"), ("prune_scope", "minority_only"),
    ("objective_mode", []), ("prune_scope", {}),
    ("tol", -1), ("tol", np.inf), ("tol", True),
    ("max_split_rounds", 0), ("max_split_rounds", 1.5),
    ("max_split_rounds", True), ("max_refine_iter", 0),
    ("max_refine_iter", np.inf), ("max_refine_iter", True),
])
def test_invalid_parameters_are_rejected(name, value):
    X, y, U = simple_training_geometry()
    with pytest.raises(ValueError):
        SSHC(**{name: value}).fit(X, y, U)


@pytest.mark.parametrize("X,y,U", [
    ([], [], None), ([[0.]], [0], None),
    ([[0.], [1.], [2.]], [0, 1, 2], None),
    ([[0.], [1.], [2.]], [0, 1], None),
    ([[0.], [1.], [2.]], [[0], [1], [0]], None),
    ([[0.], [np.nan], [2.]], [0, 1, 0], None),
    ([[0.], [1.], [2.]], [0, 1, 0], [[np.inf]]),
    ([[0.], [1.], [2.]], [0, 1, 0], [[1., 2.]]),
    ([[0.], [1.], [2.]], [0, 1, 0], np.empty((0, 2))),
    (np.empty((3, 0)), [0, 1, 0], None),
    ([0., 1., 2.], [0, 1, 0], None),
    ([[False], [True], [False]], [0, 1, 0], None),
    ([[0.], [1.], [2.]], [0, 1, 0], [[True]]),
    ([["0"], ["1"], ["2"]], [0, 1, 0], None),
    (np.array([[0.], [1.], [2.]], dtype=object), [0, 1, 0], None),
    (np.array([[0j], [1j], [2j]]), [0, 1, 0], None),
    ([[0.], [1.], [2.]], [0, None, 1], None),
    ([[0.], [1.], [2.]], [0, np.inf, 1], None),
    ([[0.], [1.], [2.]], np.array([0j, 1j, 0j]), None),
    ([[0.], [1.], [2.]], np.array([b"a", b"b", b"a"]), None),
    ([[0.], [1.], [2.]], np.array(["2020-01-01", "2020-01-02", "2020-01-01"],
                                 dtype="datetime64[D]"), None),
])
def test_invalid_features_labels_alignment_and_dimensions_are_rejected(X, y, U):
    with pytest.raises(ValueError):
        SSHC().fit(X, y, U)


def test_dataframe_categorical_and_boolean_source_features_are_rejected():
    frame = pd.DataFrame({"numeric": [0., 1., 2.],
                          "category": pd.Categorical([0, 1, 0])})
    with pytest.raises(ValueError):
        SSHC().fit(frame, [0, 1, 0])
    with pytest.raises(ValueError):
        SSHC().fit(frame.assign(category=[False, True, False]), [0, 1, 0])
    with pytest.raises(ValueError):
        SSHC().fit(pd.DataFrame({"complex": [0j, 1j, 2j]}), [0, 1, 0])


def test_missing_explicit_class_is_rejected_and_source_identity_can_survive_count_inversion():
    X, y, U = simple_training_geometry()
    with pytest.raises(ValueError):
        SSHC(minority_label=7).fit(X, y, U)
    # A future reduced-label training fold may invert source-domain counts.
    # Explicit identity is respected and is not inferred again from this fold.
    model = SSHC(minority_label=0).fit(X, y, U)
    assert model.minority_label_ == 0
    assert model.majority_label_ == 1


def test_failed_refit_removes_stale_fitted_geometry_and_diagnostics():
    X, y, U = simple_training_geometry()
    model = SSHC().fit(X, y, U)
    assert hasattr(model, "centroids_")
    with pytest.raises(ValueError):
        model.set_params(mu=2).fit(X, y, U)
    for attr in ("centroids_", "clusters_", "cluster_ids_", "cluster_labels_",
                 "birth_labels_", "labeled_predictions_", "diagnostics_",
                 "objective_", "n_features_in_"):
        assert not hasattr(model, attr)
