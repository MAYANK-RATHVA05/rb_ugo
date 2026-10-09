"""Hand-calculated MMA checks, independent of estimator training."""
import numpy as np
import pytest

from src.uncertainty import modified_margin_uncertainty


def test_exact_values() -> None:
    p = [[1, 0], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0, 1]]
    scores = modified_margin_uncertainty(p, [0] * 5, [0, 1])
    np.testing.assert_allclose(scores, [-1, -0.8, 0, 0.6, 1])
    np.testing.assert_allclose(scores, 1 - 2 * np.array(p)[:, 0])


def test_actual_column_order_and_string_labels() -> None:
    scores = modified_margin_uncertainty([[0.8, 0.2], [0.3, 0.7]],
                                         ["rare", "normal"], ["rare", "normal"])
    np.testing.assert_allclose(scores, [-0.6, -0.4])
    reversed_scores = modified_margin_uncertainty([[0.2, 0.8], [0.7, 0.3]],
                                                  ["rare", "normal"], ["normal", "rare"])
    np.testing.assert_allclose(scores, reversed_scores)


@pytest.mark.parametrize("p,y,classes,message", [
    ([[0.5, 0.5]], [2], [0, 1], "map"),
    ([[0.5, 0.5]], [0], [0, 0], "distinct"),
    ([[0.5, 0.5]], [0], [0, None], "nonmissing"),
    ([[0.5, 0.5]], [None], [0, 1], "map"),
    ([[0.5, 0.5]], [[0]], [0, 1], "one-dimensional"),
    ([[0.5, 0.5]], [], [0, 1], "aligned"),
    ([[0.1, 0.2, 0.7]], [0], [0, 1], "matrix"),
    ([0.5, 0.5], [0], [0, 1], "matrix"),
    ([[np.nan, 0.5]], [0], [0, 1], "finite"),
    ([[np.inf, 0]], [0], [0, 1], "finite"),
    ([[-0.1, 1.1]], [0], [0, 1], "in \\[0, 1\\]"),
    ([[0.4, 0.4]], [0], [0, 1], "sum to one"),
    ([["0.5", "0.5"]], [0], [0, 1], "numerical"),
])
def test_invalid_probability_inputs(p, y, classes, message) -> None:
    with pytest.raises(ValueError, match=message):
        modified_margin_uncertainty(p, y, classes)
