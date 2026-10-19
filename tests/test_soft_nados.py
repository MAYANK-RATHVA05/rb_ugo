import numpy as np
import pytest

from src.soft_nados import SoftNADOS


def mixed_fixture():
    X = np.array([[0.0], [2.0], [10.0], [1.0], [3.0], [11.0], [12.0], [13.0]])
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    return X, y


def test_soft_priority_matches_declared_equation_and_softens_cutoff():
    X, y = mixed_fixture()
    sampler = SoftNADOS(k_neighbors=3, temperature=.05, random_state=7)
    sampler.fit_resample(X, y)
    expected_gate = 1 / (1 + np.exp(-(sampler.reliability_scores_ - .35) / .05))
    expected = (sampler.reliability_scores_ * sampler.difficulty_scores_ * expected_gate)
    expected[sampler.minority_neighbor_counts_ == 0] = 0
    np.testing.assert_allclose(sampler.soft_gate_weights_, expected_gate)
    np.testing.assert_allclose(sampler.synthesis_priorities_, expected)

    # Row 1 has local minority support but R=1/3, just below the hard cutoff.
    assert sampler.reliability_scores_[1] == pytest.approx(1 / 3)
    assert not sampler.hard_gate_mask_[1]
    assert sampler.synthesis_priorities_[1] > 0
    # Row 2 has no interpolation partner and remains inactive despite softening.
    assert sampler.synthesis_priorities_[2] == 0


def test_originals_retained_generation_reconstructs_and_seed_is_repeatable():
    X, y = mixed_fixture()
    a = SoftNADOS(k_neighbors=3, temperature=.05, random_state=19)
    b = SoftNADOS(k_neighbors=3, temperature=.05, random_state=19)
    a_out, b_out = a.fit_resample(X, y), b.fit_resample(X, y)
    for left, right in zip(a_out, b_out):
        np.testing.assert_array_equal(left, right)
    out_X, out_y = a_out
    np.testing.assert_array_equal(out_X[:len(X)], X)
    np.testing.assert_array_equal(out_y[:len(y)], y)
    expected = (X[a.generated_seed_indices_] + a.interpolation_fractions_[:, None]
                * (X[a.generated_partner_indices_] - X[a.generated_seed_indices_]))
    np.testing.assert_array_equal(out_X[len(X):], expected)
    assert set(a.generated_seed_indices_) <= set(a.active_seed_indices_)
    assert a.diagnostics_["generated_count"] == 2


@pytest.mark.parametrize("temperature", [0, -0.1, np.inf, True])
def test_invalid_temperature_is_rejected(temperature):
    X, y = mixed_fixture()
    with pytest.raises(ValueError, match="temperature"):
        SoftNADOS(k_neighbors=3, temperature=temperature).fit_resample(X, y)


def test_hidden_group_metadata_cannot_enter_sampler():
    X, y = mixed_fixture()
    with pytest.raises(TypeError):
        SoftNADOS(k_neighbors=3).fit_resample(X, y, groups=np.zeros(len(y)))
