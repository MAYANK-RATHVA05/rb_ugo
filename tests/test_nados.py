import numpy as np
import pytest

from src.nados import NADOS, NADOSUnsupportedError


def mixed_fixture():
    # Minority rows are 0, 2 and 10. Majority rows create mixed k=3
    # neighborhoods and a two-row generation gap.
    X = np.array([[0.0], [2.0], [10.0], [1.0], [3.0], [11.0], [12.0], [13.0]])
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    return X, y


def test_hand_reliability_difficulty_and_eligibility():
    X, y = mixed_fixture()
    sampler = NADOS(k_neighbors=3, random_state=7)
    sampler.fit_resample(X, y)
    np.testing.assert_array_equal(sampler.minority_indices_, [0, 1, 2])
    np.testing.assert_array_equal(sampler.minority_neighbor_counts_, [1, 1, 0])
    np.testing.assert_allclose(sampler.minority_support_, [1 / 3, 1 / 3, 0])
    # Row 0: minority distance 2; majority distances 1 and 3 -> evidence .5.
    assert sampler.relative_distance_evidence_[0] == pytest.approx(.5)
    assert sampler.reliability_scores_[0] == pytest.approx(5 / 12)
    assert sampler.boundary_scores_[0] == pytest.approx(2 / 3)
    assert sampler.difficulty_scores_[0] == pytest.approx(2 / 3)
    assert sampler.synthesis_priorities_[0] == pytest.approx(5 / 18)
    # The distant minority row has weaker distance evidence and misses tau=.35.
    assert sampler.reliability_scores_[2] == 0
    assert sampler.reliability_scores_[1] == pytest.approx(1 / 3)
    np.testing.assert_array_equal(sampler.eligible_indices_, [0])
    np.testing.assert_array_equal(sampler.excluded_indices_, [1, 2])


def test_originals_retained_and_generated_rows_reconstruct_exactly():
    X, y = mixed_fixture()
    sampler = NADOS(k_neighbors=3, random_state=19)
    out_X, out_y = sampler.fit_resample(X, y)
    np.testing.assert_array_equal(out_X[:len(X)], X)
    np.testing.assert_array_equal(out_y[:len(y)], y)
    np.testing.assert_array_equal(sampler.retained_indices_, np.arange(len(X)))
    assert len(out_X) == 10 and np.sum(out_y == 0) == np.sum(out_y == 1) == 5
    expected = (X[sampler.generated_seed_indices_] + sampler.interpolation_fractions_[:, None]
                * (X[sampler.generated_partner_indices_] - X[sampler.generated_seed_indices_]))
    np.testing.assert_array_equal(out_X[len(X):], expected)
    assert set(sampler.generated_seed_indices_) <= set(sampler.eligible_indices_)
    for seed, partner in zip(sampler.generated_seed_indices_, sampler.generated_partner_indices_):
        position = int(np.flatnonzero(sampler.minority_indices_ == seed)[0])
        assert partner in sampler.partner_pools_[position]
        assert y[partner] == 1
    assert sampler.diagnostics_["generated_count"] == 2


def test_fixed_seed_is_repeatable_and_different_seed_changes_trace():
    X, y = mixed_fixture()
    a, b, c = (NADOS(k_neighbors=3, random_state=seed) for seed in (42, 42, 43))
    a_out, b_out, c_out = a.fit_resample(X, y), b.fit_resample(X, y), c.fit_resample(X, y)
    for left, right in zip(a_out, b_out):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a.generated_seed_indices_, b.generated_seed_indices_)
    assert not np.array_equal(a_out[0][len(X):], c_out[0][len(X):])


def test_distance_ties_resolve_by_original_row_index():
    X = np.array([[0.0], [-1.0], [1.0], [-2.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    sampler = NADOS(k_neighbors=3)
    sampler.fit_resample(X, y)
    np.testing.assert_array_equal(sampler.neighbor_indices_[0], [1, 2, 3])


def test_no_local_minority_and_no_local_majority_conventions():
    X = np.array([[0.0], [10.0], [10.1], [10.2], [1.0], [2.0], [3.0], [20.0], [21.0]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    sampler = NADOS(k_neighbors=2, reliability_threshold=0.0, target_ratio=.8)
    sampler.fit_resample(X, y)
    # Isolated minority at zero sees two majority rows and has no partner.
    assert sampler.minority_neighbor_counts_[0] == 0
    assert sampler.relative_distance_evidence_[0] == 0
    assert not sampler.eligible_mask_[0]
    # The three adjacent minority rows see no majority in their k-neighborhood.
    np.testing.assert_array_equal(sampler.minority_neighbor_counts_[1:], [2, 2, 2])
    np.testing.assert_array_equal(sampler.relative_distance_evidence_[1:], [1, 1, 1])
    assert np.isnan(sampler.mean_majority_distance_[1:]).all()
    assert np.isfinite(sampler.reliability_scores_).all()


def test_dense_safe_minority_with_zero_difficulty_fails_without_fallback():
    X = np.r_[np.arange(6, dtype=float), 100 + np.arange(10, dtype=float)][:, None]
    y = np.r_[np.ones(6, dtype=int), np.zeros(10, dtype=int)]
    with pytest.raises(NADOSUnsupportedError, match="No eligible") as caught:
        NADOS(k_neighbors=5).fit_resample(X, y)
    assert caught.value.reason == "no_positive_synthesis_priority"


def test_coincident_cross_class_local_geometry_is_explicitly_unsupported():
    X = np.array([[0.0], [0.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.array([1, 1, 0, 0, 0, 0])
    with pytest.raises(NADOSUnsupportedError, match="coincident") as caught:
        NADOS(k_neighbors=2).fit_resample(X, y)
    assert caught.value.reason == "zero_cross_class_distance_denominator"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k_neighbors": 0}, "positive integer"),
        ({"reliability_weight": 1.1}, r"\[0, 1\]"),
        ({"difficulty_weight": -0.1}, r"\[0, 1\]"),
        ({"reliability_threshold": 2.0}, r"\[0, 1\]"),
        ({"target_ratio": 0.0}, r"\(0, 1\]"),
        ({"random_state": True}, "integer"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, message):
    X, y = mixed_fixture()
    parameters = {"k_neighbors": 3, **kwargs}
    with pytest.raises(ValueError, match=message):
        NADOS(**parameters).fit_resample(X, y)


def test_invalid_inputs_and_minority_label_are_rejected():
    X, y = mixed_fixture()
    with pytest.raises(ValueError, match="finite"):
        NADOS(k_neighbors=3).fit_resample(np.r_[X[:-1], [[np.nan]]], y)
    with pytest.raises(ValueError, match="Exactly two"):
        NADOS(k_neighbors=3).fit_resample(X, np.ones(len(X)))
    with pytest.raises(ValueError, match="minority_label"):
        NADOS(k_neighbors=3, minority_label=9).fit_resample(X, y)
    with pytest.raises(NADOSUnsupportedError, match="strictly smaller"):
        NADOS(k_neighbors=3, minority_label=0).fit_resample(X, y)


def test_sampler_api_cannot_receive_hidden_groups_or_corruption_flags():
    X, y = mixed_fixture()
    with pytest.raises(TypeError):
        NADOS(k_neighbors=3).fit_resample(X, y, groups=np.zeros(len(y)))
