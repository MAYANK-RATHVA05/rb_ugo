import math

import numpy as np
import pytest
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

from src.metrics import binary_metrics, metrics_from_confusion


def test_hand_computed_iam_and_standard_fp_fn_orientation():
    result = metrics_from_confusion([[8, 2], [1, 3]])
    assert result["iam"] == pytest.approx(((8 - 2) / 10 + (3 - 2) / 5) / 2)
    assert result["mcc"] == pytest.approx((24 - 2) / math.sqrt(5 * 4 * 10 * 9))
    assert result["f1"] == pytest.approx(6 / 9)
    assert result["gmean"] == pytest.approx(math.sqrt((3 / 4) * (8 / 10)))
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
    assert result["gmean"] is None
    assert all(result["undefined"].values())


def test_source_positive_zero_matches_independent_sklearn_metrics():
    truth = np.array([1] * 10 + [0] * 4)
    prediction = np.array([1] * 8 + [0] * 2 + [1] + [0] * 3)
    result = binary_metrics(truth, prediction, negative_label=1, positive_label=0)
    assert result["confusion_matrix"] == [[8, 2], [1, 3]]
    assert result["f1"] == pytest.approx(f1_score(truth, prediction, pos_label=0))
    assert result["mcc"] == pytest.approx(matthews_corrcoef(truth, prediction))
    assert result["auc"] is None and result["undefined"]["auc"]


def test_auc_uses_explicit_positive_label_probability_and_ties():
    truth = np.array([1, 1, 0, 0])
    predictions = np.array([1, 1, 1, 0])
    probability_of_zero = np.array([0.1, 0.4, 0.35, 0.8])
    result = binary_metrics(truth, predictions, negative_label=1, positive_label=0,
                            positive_probabilities=probability_of_zero)
    assert result["auc"] == pytest.approx(0.75)
    assert result["auc"] == pytest.approx(roc_auc_score(truth == 0, probability_of_zero))
    assert not result["undefined"]["auc"]
    tied = binary_metrics(truth, predictions, negative_label=1, positive_label=0,
                          positive_probabilities=np.full(4, 0.5))
    assert tied["auc"] == pytest.approx(0.5)


def test_absent_true_class_leaves_auc_and_gmean_undefined():
    result = binary_metrics([0, 0, 0], [0, 0, 0], negative_label=0, positive_label=1,
                            positive_probabilities=[0.2, 0.3, 0.4])
    assert result["auc"] is None and result["undefined"]["auc"]
    assert result["gmean"] is None and result["undefined"]["gmean"]


@pytest.mark.parametrize("scores", [[0.1, 0.2], [0.1, float("nan"), 0.2, 0.3],
                                    [-0.1, 0.2, 0.3, 0.4], [1.1, 0.2, 0.3, 0.4]])
def test_invalid_auc_probabilities_fail(scores):
    with pytest.raises(ValueError, match="probabilities"):
        binary_metrics([0, 0, 1, 1], [0, 0, 1, 1], negative_label=0,
                       positive_label=1, positive_probabilities=scores)


@pytest.mark.parametrize("matrix", [[[1., 0.], [0., 1.]], [[-1, 0], [0, 2]],
                                    [[0, 0], [0, 0]], [[1, 2, 3]]])
def test_invalid_matrix_rejected(matrix):
    with pytest.raises(ValueError):
        metrics_from_confusion(matrix)


def test_unknown_or_misaligned_predictions_rejected():
    for truth, prediction in (([0, 1], [0, 2]), ([0, 1], [0]), ([0, None], [0, 1])):
        with pytest.raises(ValueError):
            binary_metrics(truth, prediction, negative_label=0, positive_label=1)
