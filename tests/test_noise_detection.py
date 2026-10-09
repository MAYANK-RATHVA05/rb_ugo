"""Check self exclusion, votes, and actual relocation coordinates."""
import numpy as np
import pytest

from src.noise_detection import noise_final_detect_and_relocate, noise_pre_detect


def test_obvious_noise() -> None:
    X = np.array([[0.0], [0.1], [0.2], [0.15], [10.0], [10.1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    mask = noise_pre_detect(X, y, 3)
    assert mask.tolist() == [False, False, False, True, False, False]


def test_self_exclusion_and_small_k() -> None:
    # If self were allowed each point would vote for itself and look clean.
    assert noise_pre_detect([[0.0], [10.0]], [0, 1], 5).tolist() == [True, True]


def test_identical_distinct_points_remain_neighbors() -> None:
    assert noise_pre_detect([[0.0], [0.0]], [0, 1], 1).tolist() == [True, True]


def test_vote_tie_preserves_true_label() -> None:
    X = [[0.0], [1.0], [2.0]]
    assert not noise_pre_detect(X, [0, 1, 0], 2)[0]


def test_relocation_uses_clean_original_same_class() -> None:
    reference = np.array([[0.0], [0.1], [0.2], [0.15], [10.0], [10.1]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    generated = np.array([[0.0], [10.05]])
    original_generated = generated.copy()
    original_reference = reference.copy()
    corrected, diagnostics = noise_final_detect_and_relocate(
        generated, 1, reference, labels, k_pre=3,
    )
    # The nearer minority point at 0.15 is pre-noisy and must not be a destination.
    np.testing.assert_allclose(corrected, [[10.0], [10.05]])
    assert diagnostics["noise_mask"].tolist() == [True, False]
    assert diagnostics["relocation_reference_indices"].tolist() == [4, -1]
    assert diagnostics["relocated_count"] == 1
    np.testing.assert_array_equal(generated, original_generated)
    np.testing.assert_array_equal(reference, original_reference)


def test_no_eligible_relocation_reference_is_error() -> None:
    with pytest.raises(ValueError, match="No original non-noisy"):
        noise_final_detect_and_relocate([[0.0]], 1, [[0.0], [10.0]], [0, 1],
                                        reference_noise_mask=np.array([True, True]))


def test_final_vote_tie_and_k_clamp() -> None:
    corrected, diagnostics = noise_final_detect_and_relocate(
        [[5.0]], 1, [[0.0], [10.0]], [0, 1], k_after=9,
    )
    np.testing.assert_array_equal(corrected, [[5.0]])
    assert diagnostics["relocated_count"] == 0
    assert diagnostics["effective_k_after"] == 2


def test_empty_generated_batch() -> None:
    corrected, diagnostics = noise_final_detect_and_relocate(
        np.empty((0, 1)), 1, [[0.0], [1.0]], [0, 1],
    )
    assert corrected.shape == (0, 1)
    assert diagnostics["relocated_count"] == 0


@pytest.mark.parametrize("X,y,k,message", [
    ([[0.0]], [0], 1, "two classes"),
    ([[0.0], [1.0]], [0], 1, "aligned"),
    ([[0.0], [np.nan]], [0, 1], 1, "finite"),
    ([[0.0], [1.0]], [0, 1], 0, "positive integer"),
    ([[0.0], [1.0]], [0, 1], 1.5, "positive integer"),
])
def test_invalid_inputs(X, y, k, message) -> None:
    with pytest.raises(ValueError, match=message):
        noise_pre_detect(X, y, k)
