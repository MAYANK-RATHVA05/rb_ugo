"""Hand-calculated EC-EBO external-constraint checks; no real data or sampling."""
import numpy as np
import pytest

from src.ec_ebo_clustering import ExternallyConstrainedClustering


def test_majority_barrier_keeps_two_minority_subclusters_separate():
    X = np.array([[0., 0.], [1., 0.], [4., 0.], [5., 0.],
                  [2.5, 0.], [8., 0.], [9., 0.], [-3., 0.], [-4., 0.]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    X_before, y_before = X.copy(), y.copy()

    fitted = ExternallyConstrainedClustering().fit(X, y)

    assert [members.tolist() for members in fitted.clusters_] == [[0, 1], [2, 3]]
    np.testing.assert_allclose(fitted.centroids_, [[0.5, 0.], [4.5, 0.]])
    np.testing.assert_allclose(fitted.nearest_majority_distances_, [1.5, 1.5])
    np.testing.assert_array_equal(fitted.cluster_labels_, [0, 0, 1, 1, -1, -1, -1, -1, -1])
    assert fitted.minority_label_ == 1 and fitted.majority_label_ == 0
    assert len(fitted.merge_history_) == 2
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


def test_merge_inequality_is_strict_at_the_majority_distance():
    # d_cent=1; the right minority point's nearest-majority distance is 1.
    fitted = ExternallyConstrainedClustering().fit(
        np.array([[0.], [1.], [2.], [3.], [4.]]), [1, 1, 0, 0, 0])
    assert [members.tolist() for members in fitted.clusters_] == [[0], [1]]
    assert fitted.merge_history_ == ()


def test_tied_pairs_use_original_row_order_then_recompute_centroid():
    fitted = ExternallyConstrainedClustering().fit(
        np.array([[0.], [0.2], [0.4], [1.], [2.], [3.], [4.]]),
        [1, 1, 1, 0, 0, 0, 0])
    assert fitted.merge_history_[0]["left_indices"] == (0,)
    assert fitted.merge_history_[0]["right_indices"] == (1,)
    assert fitted.merge_history_[0]["centroid_distance"] == pytest.approx(0.2)
    assert fitted.merge_history_[1]["centroid_distance"] == pytest.approx(0.3)
    assert [members.tolist() for members in fitted.clusters_] == [[0, 1, 2]]
    np.testing.assert_allclose(fitted.centroids_, [[0.2]])
    np.testing.assert_allclose(fitted.nearest_majority_distances_, [0.6])


def test_coincident_majority_prevents_merge_but_keeps_singletons():
    fitted = ExternallyConstrainedClustering().fit(
        np.array([[0.], [0.], [0.], [2.], [3.]]), [1, 1, 0, 0, 0])
    assert [members.tolist() for members in fitted.clusters_] == [[0], [1]]
    np.testing.assert_allclose(fitted.nearest_majority_distances_, [0., 0.])


def test_explicit_minority_label_and_invalid_inputs():
    X = np.array([[0.], [0.2], [2.], [3.], [4.]])
    y = np.array(["rare", "rare", "common", "common", "common"])
    fitted = ExternallyConstrainedClustering(minority_label="rare").fit(X, y)
    assert fitted.minority_label_ == "rare"
    with pytest.raises(ValueError, match="smaller class"):
        ExternallyConstrainedClustering(minority_label="common").fit(X, y)
    with pytest.raises(ValueError, match="tied class count"):
        ExternallyConstrainedClustering().fit([[0.], [1.], [2.], [3.]], [0, 0, 1, 1])
    with pytest.raises(ValueError, match="finite numerical"):
        ExternallyConstrainedClustering().fit([[0.], [np.nan], [2.]], [0, 1, 1])


def test_failed_refit_clears_previous_geometry():
    model = ExternallyConstrainedClustering().fit(
        [[0.], [0.2], [2.], [3.], [4.]], [1, 1, 0, 0, 0])
    assert model.clusters_
    with pytest.raises(ValueError, match="tied class count"):
        model.fit([[0.], [1.], [2.], [3.]], [1, 1, 0, 0])
    assert not hasattr(model, "clusters_")
    assert not hasattr(model, "merge_history_")
