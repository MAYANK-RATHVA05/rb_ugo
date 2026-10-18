"""Geometry/software checks for the opt-in generator diagnostic, not a benchmark."""

import numpy as np
import pytest

from src.ppeo_geometry import (
    METHODS, clip_displacement, constraint_slack, estimate_shape, generate,
)
import src.ppeo_geometry as geometry


@pytest.fixture
def observations():
    minority = np.array([[-0.2, 0.0], [0.2, 0.0], [0.0, 0.1], [0.0, -0.1]])
    majority = np.array([[0.8, 0.0], [-0.9, 0.0], [0.0, 0.7], [0.0, -1.0]])
    prototypes = np.zeros(4, dtype=int)
    quotas = np.array([7, 6, 5, 0])
    return minority, majority, prototypes, quotas


def test_known_scale_both_signs_and_zero_denominators():
    anchor = np.zeros(2)
    majority = np.array([[1.0, 0.0], [0.0, 3.0]])
    displacement, scale = clip_displacement(np.array([2.0, 0.0]), anchor, majority, eta=0.8)
    assert scale == pytest.approx(0.2)
    np.testing.assert_allclose(displacement, [0.4, 0])
    np.testing.assert_allclose(constraint_slack(displacement, anchor, majority, 0.8), [0, 7.2])
    for sign in (-1, 1):
        sample = anchor + sign * displacement
        distances = np.sum((majority - sample) ** 2, axis=1)
        bound = np.sum(displacement ** 2) + 0.2 * np.sum(majority ** 2, axis=1)
        assert np.all(distances >= bound - 1e-14)
    perpendicular, unchanged = clip_displacement(np.array([0.0, 10.0]), anchor, majority[:1])
    assert unchanged == 1
    np.testing.assert_array_equal(perpendicular, [0, 10])


def test_every_majority_point_constrains_not_only_the_nearest():
    # The closest majority is perpendicular to the proposed displacement.  The
    # farther majority is the active constraint and must not be omitted.
    majority = np.array([[1.0, 0.0], [0.0, 1.1]])
    clipped, scale = clip_displacement(np.array([0.0, 1.4]), np.zeros(2), majority, 0.8)
    assert scale < 1
    np.testing.assert_allclose(clipped, [0, 0.44])


def test_random_high_dimensional_clip_obeys_voronoi_distance_bound():
    rng = np.random.default_rng(13)
    for dimension in (1, 2, 5):
        anchor = rng.normal(size=dimension)
        majority = rng.normal(size=(41, dimension)) + 0.2
        for _ in range(20):
            displacement, _ = clip_displacement(rng.normal(size=dimension) * 10, anchor, majority)
            assert np.min(constraint_slack(displacement, anchor, majority)) >= -1e-12
            squared_a = np.sum((majority - anchor) ** 2, axis=1)
            for sign in (-1, 1):
                distances = np.sum((anchor + sign * displacement - majority) ** 2, axis=1)
                expected_lower = np.sum(displacement ** 2) + 0.1 * squared_a
                assert np.all(distances >= expected_lower - 1e-12)


def test_covariance_trace_factor_and_rotation_equivariance():
    rng = np.random.default_rng(17)
    points = rng.normal(size=(40, 2)) * [3, 0.2] + [3, -2]
    angle = 0.61
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    original = estimate_shape(points)
    rotated = estimate_shape(points @ rotation.T)
    assert original.fallback is None and rotated.fallback is None
    assert 0 <= original.shrinkage <= 1
    assert np.trace(original.covariance) == pytest.approx(2)
    np.testing.assert_allclose(original.factor @ original.factor.T, original.covariance, atol=1e-13)
    np.testing.assert_allclose(rotated.covariance, rotation @ original.covariance @ rotation.T, atol=1e-13)
    np.testing.assert_allclose(rotated.factor, rotation @ original.factor @ rotation.T, atol=1e-13)
    majority = np.array([[1.0, 3.0], [-2.0, 0.5]])
    anchor = np.array([0.2, -0.1])
    vector = np.array([4.0, 1.0])
    v1, c1 = clip_displacement(vector, anchor, majority)
    v2, c2 = clip_displacement(rotation @ vector, rotation @ anchor, majority @ rotation.T)
    assert c1 == pytest.approx(c2)
    np.testing.assert_allclose(v2, rotation @ v1)


@pytest.mark.parametrize("points,reason", [
    (np.empty((0, 2)), "too_few_observations"),
    (np.array([[0, 0], [1, 1]]), "too_few_observations"),
    (np.ones((4, 2)), "zero_trace"),
    (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), "deficient_empirical_rank"),
])
def test_shape_fallbacks_are_explicit(points, reason):
    result = estimate_shape(points)
    assert result.fallback == reason
    assert result.covariance is None and result.factor is None


def test_finite_but_extreme_geometry_records_numerical_fallback(observations):
    points = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]) * 1e200
    assert estimate_shape(points).fallback == "nonfinite_geometry"
    result = generate(points, np.array([[3e200, 2e200]]), np.zeros(4, dtype=int),
                      np.array([2, 0, 0, 0]), method="symmetric_clip")
    assert result.diagnostics["fallback_counts"] == {"nonfinite_geometry": 2}
    assert np.isfinite(result.samples).all()


@pytest.mark.parametrize("failure", ["fitting", "eigendecomposition"])
def test_numerical_estimator_failures_are_recorded(monkeypatch, observations, failure):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("declared numerical failure")

    if failure == "fitting":
        monkeypatch.setattr(geometry.LedoitWolf, "fit", fail)
    else:
        monkeypatch.setattr(geometry.np.linalg, "eigh", fail)
    result = generate(*observations, method="symmetric_clip")
    assert result.diagnostics["fallback_counts"] == {"nonfinite_geometry": 18}
    np.testing.assert_array_equal(result.samples, observations[0][result.anchor_indices])


@pytest.mark.parametrize("method", sorted(METHODS))
def test_exact_quotas_reproducible_finite_outputs_and_no_input_mutation(observations, method):
    before = tuple(array.copy() for array in observations)
    first = generate(*observations, method=method, seed=31)
    second = generate(*observations, method=method, seed=31)
    assert first.samples.shape == (18, 2)
    assert np.isfinite(first.samples).all()
    np.testing.assert_array_equal(np.bincount(first.anchor_indices, minlength=4), observations[3])
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.diagnostics == second.diagnostics
    for actual, expected in zip(observations, before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", ["symmetric_clip", "paired_rejection"])
def test_paired_mean_is_exact_per_anchor_including_odd_quotas(observations, method):
    minority, majority, _, quotas = observations
    result = generate(*observations, method=method, seed=8)
    assert result.diagnostics["odd_copy_rows"] == 2
    for index in np.flatnonzero(quotas):
        samples = result.samples[result.anchor_indices == index]
        np.testing.assert_allclose(samples.mean(axis=0), minority[index], atol=1e-15)
        if quotas[index] % 2:
            np.testing.assert_array_equal(samples[-1], minority[index])
        for sample in samples:
            slack = constraint_slack(sample - minority[index], minority[index], majority)
            assert np.min(slack) >= -1e-12


def test_unequal_anchor_quotas_do_not_preserve_global_original_mean(observations):
    minority, _, _, quotas = observations
    result = generate(*observations, method="symmetric_clip")
    np.testing.assert_allclose(result.samples.mean(axis=0), np.average(minority, axis=0, weights=quotas))
    assert not np.allclose(result.samples.mean(axis=0), minority.mean(axis=0))


@pytest.mark.parametrize("method", sorted(METHODS))
def test_conflicting_original_labels_force_flagged_uncertified_copies(observations, method):
    minority, majority, prototypes, quotas = observations
    majority = np.vstack([majority, minority[0]])
    result = generate(minority, majority, prototypes, quotas, method=method)
    np.testing.assert_array_equal(result.samples[result.anchor_indices == 0], np.tile(minority[0], (7, 1)))
    assert result.diagnostics["fallback_counts"]["conflicting_original_labels"] == 7
    assert result.diagnostics["anchor_details"][0]["label_conflict"] is True
    # An equal-coordinate majority cannot be strictly farther than the anchor.
    np.testing.assert_array_equal(constraint_slack(np.zeros(2), minority[0], majority)[-1], 0)


def test_shape_fallback_does_not_silently_change_spsdo_generation():
    minority = np.array([[0.0, 0.0]])
    majority = np.array([[2.0, 0.0]])
    args = minority, majority, np.array([0]), np.array([4])
    original = generate(*args, method="spsdo")
    fallback = generate(*args, method="symmetric_clip")
    assert np.any(original.samples != minority[0])
    np.testing.assert_array_equal(fallback.samples, np.zeros((4, 2)))
    assert fallback.diagnostics["fallback_counts"] == {"too_few_observations": 4}
    assert fallback.diagnostics["anchor_details"][0]["shape_fallback"] == "too_few_observations"


def test_paired_rejection_has_bounded_attempts_and_counted_fallback(observations):
    minority, majority, prototypes, _ = observations
    result = generate(minority, majority, prototypes, np.array([4, 0, 0, 0]),
                      method="paired_rejection", eta=1e-20, max_attempts=3)
    assert result.diagnostics["attempts"] == 6
    assert result.diagnostics["fallback_counts"] == {"paired_rejection_exhausted": 4}
    np.testing.assert_array_equal(result.samples, np.tile(minority[0], (4, 1)))


def test_screening_uses_original_observations_only():
    minority = np.array([[0, -0.4], [0.02, -0.1], [-0.02, 0.1], [0, 0.4]], dtype=float)
    majority = np.array([[0.15, -0.2], [0.15, 0.0], [0.15, 0.2]])
    result = generate(minority, majority, np.zeros(4, dtype=int), np.full(4, 60),
                      method="screened_ellipsoid", seed=5)
    assert result.diagnostics["attempts"] > len(result.samples)
    for sample in result.samples:
        nearest_minority = np.min(np.sum((minority - sample) ** 2, axis=1))
        nearest_majority = np.min(np.sum((majority - sample) ** 2, axis=1))
        assert nearest_minority < nearest_majority


def test_spsdo_radius_is_uniform_not_uniform_ball_volume():
    result = generate(np.zeros((1, 2)), np.array([[1.0, 0.0]]), np.array([0]), np.array([8000]),
                      method="spsdo", seed=7)
    radii = np.linalg.norm(result.samples, axis=1)
    assert np.max(radii) <= 1
    assert radii.mean() == pytest.approx(0.5, abs=0.015)
    assert np.mean(radii ** 2) == pytest.approx(1 / 3, abs=0.015)
    assert result.diagnostics["fallback_generated_rows"] == 0


@pytest.mark.parametrize("kwargs", [
    {"eta": 0}, {"eta": 1}, {"eta": np.nan}, {"sphere_scale": 0},
    {"sphere_scale": 1.1}, {"max_attempts": 0}, {"max_attempts": 1.5},
    {"method": "full_ppeo"},
])
def test_invalid_parameters_rejected(observations, kwargs):
    with pytest.raises(ValueError):
        generate(*observations, **({"method": "symmetric_clip"} | kwargs))


@pytest.mark.parametrize("position,bad", [
    (0, np.array([[np.nan, 0.0]])),
    (1, np.empty((0, 2))),
    (2, np.zeros(4, dtype=float)),
    (3, np.array([1.0, 2.0, 3.0, 0.0])),
    (3, np.array([1, -1, 2, 0])),
    (3, np.array([1, 2])),
])
def test_malformed_inputs_rejected(observations, position, bad):
    inputs = list(observations)
    inputs[position] = bad
    with pytest.raises(ValueError):
        generate(*inputs, method="symmetric_clip")


def test_zero_quotas_and_empty_minority_are_supported(observations):
    minority, majority, prototypes, _ = observations
    empty = generate(minority, majority, prototypes, np.zeros(4, dtype=int), method="symmetric_clip")
    assert empty.samples.shape == (0, 2)
    assert empty.anchor_indices.shape == (0,)
    empty = generate(np.empty((0, 2)), majority, np.empty(0, dtype=int), np.empty(0, dtype=int), method="spsdo")
    assert empty.samples.shape == (0, 2)
