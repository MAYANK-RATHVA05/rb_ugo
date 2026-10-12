import math

import numpy as np
import pytest
from sklearn.metrics import f1_score, matthews_corrcoef

from src.metrics import binary_metrics, metrics_from_confusion


def test_hand_computed_iam_and_standard_fp_fn_orientation():
    result = metrics_from_confusion([[8, 2], [1, 3]])
    assert result["iam"] == pytest.approx(((8 - 2) / 10 + (3 - 2) / 5) / 2)
    assert result["mcc"] == pytest.approx((24 - 2) / math.sqrt(5 * 4 * 10 * 9))
    assert result["f1"] == pytest.approx(6 / 9)
    assert result["counts"] == {"tn": 8, "fp": 2, "fn": 1, "tp": 3}
    assert not any(result["undefined"].values())


@pytest.mark.parametrize("matrix,iam", [([[8, 0], [0, 2]], 1.),
                                       ([[0, 8], [2, 0]], -1.),
                                       ([[8, 0], [2, 0]], -.2)])
def test_iam_extremes_and_single_prediction_class(matrix, iam):
    assert metrics_from_confusion(matrix)["iam"] == pytest.approx(iam)


def test_class_permutation_and_transpose_invariance_for_iam_mcc():
    matrix = np.array([[8, 2], [1, 3]])
    base = metrics_from_confusion(matrix)
    for changed in (matrix[::-1, ::-1], matrix.T):
        result = metrics_from_confusion(changed)
        assert result["iam"] == pytest.approx(base["iam"])
        assert result["mcc"] == pytest.approx(base["mcc"])


def test_absent_class_is_declared_undefined_not_perfect():
    result = metrics_from_confusion([[10, 0], [0, 0]])
    assert result["iam"] is None
    assert result["f1"] == result["mcc"] == 0.
    assert all(result["undefined"].values())


def test_source_positive_zero_matches_independent_sklearn_metrics():
    truth = np.array([1] * 10 + [0] * 4)
    prediction = np.array([1] * 8 + [0] * 2 + [1] + [0] * 3)
    result = binary_metrics(truth, prediction, negative_label=1, positive_label=0)
    assert result["confusion_matrix"] == [[8, 2], [1, 3]]
    assert result["f1"] == pytest.approx(f1_score(truth, prediction, pos_label=0))
    assert result["mcc"] == pytest.approx(matthews_corrcoef(truth, prediction))


@pytest.mark.parametrize("matrix", [[[1., 0.], [0., 1.]], [[-1, 0], [0, 2]],
                                    [[0, 0], [0, 0]], [[1, 2, 3]]])
def test_invalid_matrix_rejected(matrix):
    with pytest.raises(ValueError):
        metrics_from_confusion(matrix)


def test_unknown_or_misaligned_predictions_rejected():
    for truth, prediction in (([0, 1], [0, 2]), ([0, 1], [0]), ([0, None], [0, 1])):
        with pytest.raises(ValueError):
            binary_metrics(truth, prediction, negative_label=0, positive_label=1)
