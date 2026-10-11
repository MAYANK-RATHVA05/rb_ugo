"""Check paper equations and declared simulation choices, never model scores."""
import numpy as np
import pytest
from scipy.special import expit

from src.simulations import make_linear_simulation, make_nonlinear_simulation


def test_linear_default_relabels_majority_to_match_figure_counts():
    simulated = make_linear_simulation()
    assert simulated.X.shape == (1050, 2)
    assert np.bincount(simulated.pre_noise_y).tolist() == [1000, 50]
    assert np.bincount(simulated.y).tolist() == [985, 65]
    assert len(simulated.relabeled_indices) == 15
    assert (simulated.pre_noise_y[simulated.relabeled_indices] == 0).all()
    assert (simulated.y[simulated.relabeled_indices] == 1).all()
    assert simulated.metadata["figure_counts_match_by_declared_inference"]
    assert simulated.metadata["normal_second_parameter"] == "sd"


def test_linear_variance_interpretation_is_distinct_and_reproducible():
    sd = make_linear_simulation(seed=7)
    variance = make_linear_simulation(seed=7, normal_second_parameter="variance")
    repeated = make_linear_simulation(seed=7, normal_second_parameter="variance")
    np.testing.assert_array_equal(variance.X, repeated.X)
    np.testing.assert_array_equal(variance.y, repeated.y)
    np.testing.assert_allclose(variance.X[:1000] / sd.X[:1000], 1 / np.sqrt(3))
    np.testing.assert_array_equal(sd.y, variance.y)


def test_nonlinear_equation_and_majority_to_minority_noise():
    simulated = make_nonlinear_simulation()
    x1, x2 = simulated.X.T
    z = np.sin(x1) ** 2 + np.cos(x2) + np.sin(x1) * np.cos(x2)
    expected = np.rint(expit(z)).astype(int)
    np.testing.assert_array_equal(simulated.pre_noise_y, expected)
    assert simulated.majority_label == 1 and simulated.minority_label == 0
    assert np.bincount(simulated.pre_noise_y).tolist() == [58, 942]
    assert np.bincount(simulated.y).tolist() == [75, 925]
    assert len(simulated.relabeled_indices) == 17
    assert (expected[simulated.relabeled_indices] == 1).all()
    assert (simulated.y[simulated.relabeled_indices] == 0).all()
    repeated = make_nonlinear_simulation()
    np.testing.assert_array_equal(simulated.X, repeated.X)
    np.testing.assert_array_equal(simulated.y, repeated.y)
    # Figure 7's 85/915 is a source illustration, not our seed-42 outcome.
    assert simulated.metadata["figure_counts_are_source_example_not_asserted_output"] == [915, 85]


@pytest.mark.parametrize("parameters", [{"noise_fraction": -0.1}, {"noise_fraction": 1.1},
                                         {"seed": True}, {"seed": None},
                                         {"n_majority": 0}, {"n_minority": 0},
                                         {"normal_second_parameter": "unknown"}])
def test_linear_invalid_or_ambiguous_inputs_fail(parameters):
    with pytest.raises(ValueError):
        make_linear_simulation(**parameters)


def test_noise_count_exceeding_majority_pool_is_explicit():
    with pytest.raises(ValueError, match="exceeds"):
        make_linear_simulation(n_majority=2, n_minority=10, noise_fraction=1)


def test_nonlinear_degenerate_small_draw_is_explicit():
    with pytest.raises(ValueError, match="fewer than two"):
        make_nonlinear_simulation(n_samples=1)
