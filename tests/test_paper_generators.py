"""Source equation checks for MCSMOTE and LEAO, not performance claims."""
import numpy as np
import pytest
from sklearn.base import clone

from src.comparators import NoResampling, make_comparator
from src.generators import generate_batch
from src.leao import (LEAOOversampler, adversarial_potential_and_gradient,
                      entropy_allocation, positive_binary_entropy)
from src.mcsmote import MCSMOTEOversampler, transition_matrix


def test_two_state_gaussian_sinkhorn_has_expected_transition_probabilities():
    matrix, info = transition_matrix([[0.], [2.]], q=0.1)
    off_diagonal = np.exp(-0.5) / (1 + np.exp(-0.5))
    np.testing.assert_allclose(matrix, [[1 - off_diagonal, off_diagonal],
                                       [off_diagonal, 1 - off_diagonal]])
    np.testing.assert_allclose(info["bandwidths"], [2., 2.])
    np.testing.assert_allclose(matrix.sum(axis=0), [1., 1.], atol=1e-4)
    np.testing.assert_allclose(matrix.sum(axis=1), [1., 1.], atol=1e-4)


def test_mcsmote_interpolation_is_convex_equation_one_not_pseudocode_minus():
    X = np.array([[0.], [2.], [100.]])
    samples, info = generate_batch("mcsmote", X, [0, 0, 1], [0], 0,
                                    [False] * 3, 1000, np.random.default_rng(42))
    targets = np.array(info["target_indices"])
    steps = np.array(info["interpolation_steps"])
    expected = (1 - steps) * X[0, 0] + steps * X[targets, 0]
    np.testing.assert_allclose(samples[:, 0], expected)
    assert (targets == 1).any() and (targets == 0).any()
    assert ((samples >= 0) & (samples <= 2)).all()
    assert info["interpolation_policy"] == "equation_1_convex"


def test_mcsmote_duplicate_and_singleton_pools_have_defined_transitions():
    matrix, info = transition_matrix([[1.], [1.], [1.]])
    np.testing.assert_allclose(matrix, np.full((3, 3), 1 / 3))
    assert info["bandwidths"] == [0., 0., 0.]
    singleton, _ = transition_matrix([[5.]])
    np.testing.assert_array_equal(singleton, [[1.]])


@pytest.mark.parametrize("kwargs", [{"q": 0}, {"q": 1.1},
                                    {"tolerance": 0}, {"max_iter": 0}])
def test_mcsmote_rejects_invalid_transition_parameters(kwargs):
    with pytest.raises(ValueError):
        transition_matrix([[0.], [2.]], **kwargs)


def test_mcsmote_nonconvergence_is_explicit():
    with pytest.raises(ValueError, match="did not converge"):
        transition_matrix([[0.], [1.], [10.]], q=1, tolerance=1e-12, max_iter=1)


def test_standalone_mcsmote_balances_strings_and_preserves_fitting_rows():
    X = np.array([[0.], [1.], [2.], [3.], [10.], [12.]])
    y = np.array(["common"] * 4 + ["rare"] * 2)
    sampler = clone(MCSMOTEOversampler(allow_provisional=True))
    with pytest.warns(UserWarning, match="Provisional MCSMOTE"):
        augmented, labels = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(augmented[:6], X)
    np.testing.assert_array_equal(labels[:6], y)
    assert np.unique(labels, return_counts=True)[1].tolist() == [4, 4]
    assert sampler.diagnostics_["generation"]["transition_reference_indices"] == [4, 5]
    assert sampler.diagnostics_["number_of_synthetic_samples"] == 2


def test_positive_entropy_and_exact_integer_allocation():
    entropies = positive_binary_entropy(np.array([0., 0.5, 1.]))
    assert (entropies > 0).all()
    assert entropies[1] > entropies[0]
    np.testing.assert_array_equal(entropy_allocation(np.array([1., 2., 3.]), 7), [1, 2, 4])
    np.testing.assert_array_equal(entropy_allocation(np.zeros(3), 2), [1, 1, 0])


def test_adversarial_gradient_matches_finite_difference_away_from_references():
    x = np.array([0.3, 0.2])
    minority = np.array([0., 0.])
    majority = np.array([2., 1.])
    inverse = np.diag([1., 2.])
    _, gradient = adversarial_potential_and_gradient(x, minority, majority,
                                                     .6, .3, inverse, 1.)
    numerical = []
    for coordinate in range(2):
        offset = np.eye(2)[coordinate] * 1e-6
        plus, _ = adversarial_potential_and_gradient(x + offset, minority, majority,
                                                       .6, .3, inverse, 1.)
        minus, _ = adversarial_potential_and_gradient(x - offset, minority, majority,
                                                        .6, .3, inverse, 1.)
        numerical.append((plus - minus) / 2e-6)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)


def test_leao_generates_exact_balance_with_fitting_only_probability_model():
    X = np.array([[0., 0.], [1., 0.], [2., 1.], [3., 1.], [6., 3.], [7., 4.]])
    y = np.array([0, 0, 0, 0, 1, 1])
    sampler = clone(LEAOOversampler(iterations=5, allow_provisional=True))
    with pytest.warns(UserWarning, match="Provisional LEAO"):
        augmented, labels = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(augmented[:6], X)
    np.testing.assert_array_equal(labels[:6], y)
    assert len(augmented) == 8 and np.sum(labels == 1) == 4
    assert sampler.probability_model_.n_features_in_ == 2
    assert sum(sampler.diagnostics_["allocation"]) == 2
    assert set(sampler.diagnostics_["selected_majority_indices"]) <= {0, 1, 2, 3}
    other = LEAOOversampler(iterations=5, allow_provisional=True)
    with pytest.warns(UserWarning):
        repeated, _ = other.fit_resample(X, y)
    np.testing.assert_array_equal(augmented, repeated)


def test_leao_balanced_is_copy_and_failed_refit_clears_state():
    sampler = LEAOOversampler(allow_provisional=True)
    X, y = np.array([[0.], [1.]]), np.array([0, 1])
    with pytest.warns(UserWarning):
        points, labels = sampler.fit_resample(X, y)
    assert not np.shares_memory(points, X) and not np.shares_memory(labels, y)
    with pytest.raises(ValueError):
        sampler.fit_resample([[np.nan], [1.]], y)
    assert not hasattr(sampler, "diagnostics_")


@pytest.mark.parametrize("name", ["nores", "rus", "ros", "smote", "rwo", "mcsmote", "leao"])
def test_all_paper_comparators_are_constructible(name):
    comparator = make_comparator(name)
    assert callable(comparator.fit_resample)


@pytest.mark.parametrize("name", ["nores", "rus", "ros", "smote", "rwo", "mcsmote", "leao"])
def test_standalone_comparators_run_on_a_fitting_fold(name):
    X = np.arange(14., dtype=float).reshape(-1, 1)
    y = np.array([0] * 8 + [1] * 6)
    estimator = make_comparator(name, random_state=42)
    if name in {"rwo", "mcsmote", "leao"}:
        with pytest.warns(UserWarning):
            augmented, labels = estimator.fit_resample(X, y)
    else:
        augmented, labels = estimator.fit_resample(X, y)
    assert augmented.shape[1] == 1 and len(augmented) == len(labels)
    if name == "nores":
        np.testing.assert_array_equal(augmented, X)
        assert np.bincount(labels).tolist() == [8, 6]
    elif name == "rus":
        assert np.bincount(labels).tolist() == [6, 6]
    else:
        np.testing.assert_array_equal(augmented[:len(X)], X)
        assert np.bincount(labels).tolist() == [8, 8]


def test_nores_copies_original_fitting_data():
    X, y = np.array([[0.], [1.]]), np.array([0, 1])
    points, labels = NoResampling().fit_resample(X, y)
    np.testing.assert_array_equal(points, X)
    assert not np.shares_memory(points, X) and not np.shares_memory(labels, y)
