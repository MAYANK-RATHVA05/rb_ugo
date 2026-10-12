"""Paper IAM and standard binary MCC/F1 with fixed label meanings."""
from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd


METRIC_POLICY = {
    "orientation": "true rows, predicted columns; label order negative, positive",
    "iam": "mean((diagonal - max(off-diagonal row, off-diagonal column))/max(row total, column total))",
    "f1_zero_denominator": "0 with an undefined flag",
    "mcc_zero_denominator": "0 with an undefined flag",
    "iam_absent_class": "undefined (null) if a class has zero row and column totals",
}


def metrics_from_confusion(matrix: Any) -> dict[str, Any]:
    """Accept [[TN, FP], [FN, TP]]; never copy Table 6's swapped FP/FN."""
    counts = np.asarray(matrix)
    if (counts.shape != (2, 2) or counts.dtype.kind not in "iu"
            or (counts < 0).any() or not counts.any()):
        raise ValueError("Confusion matrix must be a nonempty 2x2 matrix of nonnegative integer counts.")
    tn, fp, fn, tp = (int(v) for v in counts.ravel())
    rows = [tn + fp, fn + tp]
    columns = [tn + fn, fp + tp]
    iam_terms = []
    for i, diagonal in enumerate((tn, tp)):
        denominator = max(rows[i], columns[i])
        iam_terms.append((diagonal - max(rows[i] - diagonal, columns[i] - diagonal)) / denominator
                         if denominator else None)
    iam = sum(iam_terms) / 2 if all(v is not None for v in iam_terms) else None
    f1_denominator = 2 * tp + fp + fn
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "iam": iam,
        "mcc": (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "undefined": {"iam": iam is None, "mcc": not bool(mcc_denominator),
                      "f1": not bool(f1_denominator)},
        "confusion_matrix": counts.tolist(),
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def binary_metrics(y_true: Any, y_pred: Any, *, negative_label: Any,
                   positive_label: Any) -> dict[str, Any]:
    """The caller supplies source/domain labels; test frequencies never choose them."""
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if (truth.ndim != 1 or truth.shape != prediction.shape or not len(truth)
            or positive_label == negative_label or pd.isna(truth).any() or pd.isna(prediction).any()):
        raise ValueError("Nonempty aligned labels and distinct positive/negative labels are required.")
    if not np.isin(truth, [negative_label, positive_label]).all() or not np.isin(
            prediction, [negative_label, positive_label]).all():
        raise ValueError("Truth/prediction contains an unknown label.")
    matrix = [[int(np.sum((truth == actual) & (prediction == predicted)))
               for predicted in (negative_label, positive_label)]
              for actual in (negative_label, positive_label)]
    return {**metrics_from_confusion(matrix), "labels": [negative_label, positive_label]}
