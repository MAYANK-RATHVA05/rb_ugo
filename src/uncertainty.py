"""Binary Modified Margin Uncertainty (MMA), using explicit class ordering."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def modified_margin_uncertainty(
    probabilities: Any, true_labels: Any, classes: Any,
) -> NDArray[np.float64]:
    """Return P(other | x) - P(true | x) in classifier.classes_ order.

    Rows must be valid binary probability distributions (sum tolerance 1e-8).
    Uses the general subtraction formula, equivalent to 1 - 2*P(true) when
    probabilities sum to one. Larger values mean more disagreement with the
    known fitting label, not entropy or an unlabeled prediction margin.
    """
    p = np.asarray(probabilities)
    labels = np.asarray(true_labels)
    order = np.asarray(classes)
    if p.ndim != 2 or p.shape[1] != 2 or p.dtype.kind not in "fiu":
        raise ValueError("Probabilities must be a numerical (n_samples, 2) matrix.")
    if labels.ndim != 1 or len(labels) != len(p):
        raise ValueError("True labels must be one-dimensional and aligned with probabilities.")
    if order.ndim != 1 or len(order) != 2 or pd.isna(order).any() or order[0] == order[1]:
        raise ValueError("classes must contain two distinct nonmissing labels in probability-column order.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Each probability row must sum to one.")
    matches = labels[:, None] == order[None, :]
    if pd.isna(labels).any() or not np.all(matches.sum(axis=1) == 1):
        raise ValueError("Every true label must map to exactly one classifier class.")
    columns = np.argmax(matches, axis=1)
    rows = np.arange(len(p))
    return np.asarray(p[rows, 1 - columns] - p[rows, columns], dtype=np.float64)
