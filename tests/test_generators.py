"""Generator contracts and numerical checks; no research performance claims."""
import numpy as np
import pytest
from sklearn.base import clone

from src.generators import RandomWalkOversampler, generate_batch, generator_metadata


@pytest.fixture
def pool():
    # The noisy row and other class would greatly distort RWO spread if used.
    return (np.array([[0., 7.], [2., 7.], [4., 7.], [100., 8.], [1000., 9.]]),
            np.array([0, 0, 0, 0, 1]), np.array([False, False, False, True, False]))


def test_rwo_uses_population_spread_of_clean_class_not_only_seeds(pool):
    X, y, noise = pool
    before = X.copy(), y.copy(), noise.copy()
    samples, info = generate_batch("rwo", X, y, [0], 0, noise, 20000,
                                    np.random.default_rng(42))
    # Population variance of [0,2,4] is 8/3; dividing std by sqrt(3)
    # gives sqrt(8)/3, rather than sample std/sqrt(3) or seed-only zero.
    scale = np.sqrt(8) / 3
    np.testing.assert_allclose(info["perturbation_scale"], [scale, 0])
    assert info["spread_reference_indices"] == [0, 1, 2]
    assert info["std_ddof"] == 0 and info["zero_spread_features"] == [1]
    assert set(info["drawn_seed_indices"]) == {0}
    assert abs(samples[:, 0].mean()) < 0.03
    assert abs(samples[:, 0].std() - scale) < 0.03
    assert (samples[:, 0] < 0).any()  # Gaussian RWO is not convex SMOTE.
    np.testing.assert_array_equal(samples[:, 1], np.full(20000, 7.))
    for actual, expected in zip((X, y, noise), before):
        np.testing.assert_array_equal(actual, expected)


def test_singleton_rwo_has_zero_spread_without_method_switch(pool):
    X, y, noise = pool
    samples, info = generate_batch("rwo", X, y, [4], 1, noise, 3, np.random.default_rng(42))
    np.testing.assert_array_equal(samples, np.tile(X[4], (3, 1)))
    assert info["spread_reference_indices"] == [4]
    assert info["zero_spread_features"] == [0, 1]


@pytest.mark.parametrize("variant", ["ros", "smote", "rwo", "mcsmote"])
def test_generators_repeat_with_same_seed_and_use_only_fitting_references(pool, variant):
    X, y, noise = pool
    first, info = generate_batch(variant, X, y, [0, 1], 0, noise, 9, np.random.default_rng(42))
    second, again = generate_batch(variant, X, y, [0, 1], 0, noise, 9, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)
    assert info == again and first.shape == (9, 2)
    assert set(info["drawn_seed_indices"]) <= {0, 1}
    if variant == "smote":
        assert set(info["neighbor_indices"]) <= {0, 1, 2}
        assert info["effective_k_smote"] == 2
    if variant == "mcsmote":
        assert set(info["target_indices"]) <= {0, 1, 2}
        assert info["transition_reference_indices"] == [0, 1, 2]


@pytest.mark.parametrize("seeds", [[-1], [5], [3], [4], [], [0.5], [[0]]])
def test_generator_rejects_invalid_noisy_or_wrong_class_seeds(pool, seeds):
    X, y, noise = pool
    with pytest.raises(ValueError, match="Seed|seed"):
        generate_batch("rwo", X, y, seeds, 0, noise, 2, np.random.default_rng(42))


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": True},
                                    {"batch_size": 1.5}, {"k_smote": 0},
                                    {"rng": 42}, {"pre_noise": [0, 0, 0, 1, 0]}])
def test_generator_rejects_invalid_contract_parameters(pool, kwargs):
    X, y, noise = pool
    args = dict(variant="rwo", X=X, y=y, seeds=[0], selected=0,
                pre_noise=noise, batch_size=2, rng=np.random.default_rng(42))
    args.update(kwargs)
    with pytest.raises(ValueError):
        generate_batch(**args)


def test_nonfinite_rwo_perturbation_fails_explicitly():
    with pytest.raises(ValueError, match="nonfinite"):
        generate_batch("rwo", [[1e308], [-1e308], [0.]], [0, 0, 1], [0], 0,
                       [False, False, False], 2, np.random.default_rng(42))


def test_mcsmote_metadata_records_paper_conflict():
    assert generator_metadata("mcsmote")["source_id"] == "MCSMOTE-PDF"


def test_rwo_metadata_identifies_original_source():
    assert generator_metadata("rwo")["source_id"] == "RWO-PDF"


def test_standalone_rwo_balances_minority_preserves_inputs_and_is_cloneable():
    X = np.array([[0.], [1.], [2.], [3.], [10.], [12.]])
    y = np.array(["common"] * 4 + ["rare"] * 2)
    sampler = clone(RandomWalkOversampler(allow_provisional=True))
    with pytest.warns(UserWarning, match="Provisional RWO"):
        augmented, labels = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(augmented[:6], X)
    np.testing.assert_array_equal(labels[:6], y)
    assert np.unique(labels, return_counts=True)[1].tolist() == [4, 4]
    assert sampler.diagnostics_["generation"]["spread_reference_indices"] == [4, 5]
    assert set(sampler.diagnostics_["generation"]["drawn_seed_indices"]) <= {4, 5}
    assert sampler.diagnostics_["number_of_synthetic_samples"] == 2
    assert sampler.diagnostics_["generation"]["drawn_seed_indices"] == [4, 5]


def test_balanced_standalone_rwo_returns_copies():
    X, y = np.array([[0.], [1.]]), np.array([0, 1])
    sampler = RandomWalkOversampler(allow_provisional=True)
    with pytest.warns(UserWarning):
        augmented, labels = sampler.fit_resample(X, y)
    assert not np.shares_memory(augmented, X) and not np.shares_memory(labels, y)
    assert sampler.diagnostics_["number_of_synthetic_samples"] == 0
    assert sampler.diagnostics_["stopping_reason"] == "balanced_counts"


@pytest.mark.parametrize("kwargs", [{}, {"allow_provisional": "yes"},
                                    {"allow_provisional": True, "random_state": None},
                                    {"allow_provisional": True, "random_state": True}])
def test_standalone_rwo_requires_opt_in_and_valid_seed(kwargs):
    with pytest.raises(ValueError):
        RandomWalkOversampler(**kwargs).fit_resample([[0.], [1.]], [0, 1])


def test_failed_standalone_refit_clears_old_diagnostics():
    sampler = RandomWalkOversampler(allow_provisional=True)
    with pytest.warns(UserWarning):
        sampler.fit_resample([[0.], [1.]], [0, 1])
    with pytest.raises(ValueError):
        sampler.fit_resample([[np.nan], [1.]], [0, 1])
    assert not hasattr(sampler, "diagnostics_")
    assert not hasattr(sampler, "classes_")
