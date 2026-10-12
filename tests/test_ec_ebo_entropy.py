"""EC-EBO mean entropy and exact quota tests on artificial fitting rows."""
import numpy as np
import pytest

from src.ec_ebo_entropy import (EntropyBalancedAllocation, allocate_entropy_budget,
                                binary_entropy, mean_cluster_entropies)


def test_binary_entropy_is_symmetric_and_finite_at_probability_limits():
    result = binary_entropy([0., 0.25, 0.5, 0.75, 1.])
    np.testing.assert_allclose(result[1], result[3])
    np.testing.assert_allclose(result[2], np.log(2.0))
    assert np.isfinite(result).all() and (result >= 0).all()
    assert result[0] == pytest.approx(result[4])
    with pytest.raises(ValueError, match="probabilities"):
        binary_entropy([0.5, 1.1])


def test_mean_entropy_does_not_favor_the_large_cluster_by_size_alone():
    # The paper's printed sum would allocate 20:2; its prose's mean gives 1:1.
    entropies = np.full(22, 0.4)
    means = mean_cluster_entropies(entropies, [np.arange(20), np.arange(20, 22)])
    np.testing.assert_allclose(means, [0.4, 0.4])
    np.testing.assert_array_equal(allocate_entropy_budget(means, 22), [11, 11])
    with pytest.raises(ValueError, match="nonoverlapping"):
        mean_cluster_entropies(entropies, [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="unique"):
        mean_cluster_entropies(entropies, [[0, 0]])


def test_largest_remainder_exact_count_stable_ties_and_zero_fallback():
    np.testing.assert_array_equal(allocate_entropy_budget([2., 1., 1.], 7), [3, 2, 2])
    np.testing.assert_array_equal(allocate_entropy_budget([1., 1., 1.], 2), [1, 1, 0])
    np.testing.assert_array_equal(allocate_entropy_budget([0., 0., 0.], 5), [2, 2, 1])
    np.testing.assert_array_equal(allocate_entropy_budget([0., 1.], 0), [0, 0])
    with pytest.raises(ValueError, match="nonnegative integer"):
        allocate_entropy_budget([1., 2.], True)
    with pytest.raises(ValueError, match="nonnegative vector"):
        allocate_entropy_budget([1., -1.], 3)


def test_fitted_entropy_uses_training_rows_and_correct_probability_column():
    X = np.array([[0., 0.], [0.2, 0.], [4., 0.], [4.2, 0.],
                  [2., 0.], [8., 0.], [9., 0.], [-3., 0.], [-4., 0.]])
    y = np.array(["a_rare"] * 4 + ["z_common"] * 5)
    original_X, original_y = X.copy(), y.copy()
    fitted = EntropyBalancedAllocation(minority_label="a_rare").fit(X, y)

    assert fitted.logistic_model_.classes_[0] == "a_rare"
    np.testing.assert_allclose(fitted.minority_probabilities_,
                               fitted.logistic_model_.predict_proba(X)[:, 0])
    np.testing.assert_allclose(fitted.sample_entropies_,
                               binary_entropy(fitted.minority_probabilities_))
    for index, members in enumerate(fitted.clustering_.clusters_):
        assert fitted.cluster_entropies_[index] == pytest.approx(
            fitted.sample_entropies_[members].mean())
    assert fitted.requested_count_ == 1
    assert fitted.allocated_counts_.sum() == 1
    assert fitted.diagnostics_["cluster_sizes"] == [2, 2]
    assert fitted.diagnostics_["status"] == "allocation_only"
    assert fitted.diagnostics_["logistic_policy"]["class_weight"] is None
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


def test_repeated_fit_is_deterministic_and_failed_refit_clears_state():
    X = np.array([[0.], [0.2], [2.], [3.], [4.]])
    y = [1, 1, 0, 0, 0]
    first = EntropyBalancedAllocation(random_state=42).fit(X, y)
    second = EntropyBalancedAllocation(random_state=42).fit(X, y)
    np.testing.assert_allclose(first.cluster_entropies_, second.cluster_entropies_)
    np.testing.assert_array_equal(first.allocated_counts_, second.allocated_counts_)
    with pytest.raises(ValueError, match="tied class count"):
        first.fit([[0.], [1.], [2.], [3.]], [1, 1, 0, 0])
    assert not hasattr(first, "clustering_")
    assert not hasattr(first, "allocated_counts_")


def test_invalid_entropy_parameters_are_rejected():
    X = [[0.], [0.2], [2.], [3.], [4.]]
    y = [1, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="random_state"):
        EntropyBalancedAllocation(random_state=-1).fit(X, y)
    with pytest.raises(ValueError, match="max_iter"):
        EntropyBalancedAllocation(max_iter=0).fit(X, y)
    with pytest.raises(ValueError, match="entropy_clip"):
        EntropyBalancedAllocation(entropy_clip=0.5).fit(X, y)
