"""Numerical MCSMOTE from Sağlam and Cengiz (2025), with declared conventions."""
from __future__ import annotations

from numbers import Integral
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer


def validate_transition_parameters(q: Any, tolerance: Any, max_iter: Any) -> None:
    if not 0 < finite_real(q, "q") <= 1:
        raise ValueError("q must lie in (0, 1].")
    if not 0 < finite_real(tolerance, "tolerance") < 1:
        raise ValueError("tolerance must lie in (0, 1).")
    positive_integer(max_iter, "max_sinkhorn_iter")


def transition_matrix(X: Any, q: float = 0.1, tolerance: float = 1e-4,
                      max_iter: int = 1000) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Gaussian row quantiles and alternating Sinkhorn row/column scaling."""
    validate_transition_parameters(q, tolerance, max_iter)
    points = numeric_matrix(X)
    n = len(points)
    if n == 1:
        return np.ones((1, 1)), {"bandwidths": [0.0], "iterations": 0,
                                 "max_marginal_error": 0.0}
    distances = cdist(points, points, metric="euclidean")
    if not np.isfinite(distances).all():
        raise ValueError("MCSMOTE distances are nonfinite.")
    bandwidths = np.empty(n)
    for i in range(n):
        positive = distances[i][distances[i] > 0]
        bandwidths[i] = np.quantile(positive, q) if len(positive) else 0.0
    # The paper does not resolve self-distance zeros. Exclude zero distances
    # from the bandwidth quantile; a wholly coincident pool is uniform.
    if np.all(bandwidths == 0):
        kernel = np.ones((n, n))
    else:
        if np.any(bandwidths == 0):
            raise ValueError("MCSMOTE has undefined row bandwidths.")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            kernel = np.exp(-0.5 * np.square(distances / bandwidths[:, None]))
    if not np.isfinite(kernel).all():
        raise ValueError("MCSMOTE Gaussian kernel is nonfinite.")
    matrix = kernel.copy()
    for iteration in range(1, max_iter + 1):
        row_sums = matrix.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("MCSMOTE transition matrix has a zero row.")
        matrix /= row_sums[:, None]
        col_sums = matrix.sum(axis=0)
        if np.any(col_sums <= 0):
            raise ValueError("MCSMOTE transition matrix has a zero column.")
        matrix /= col_sums[None, :]
        error = float(max(np.max(np.abs(matrix.sum(axis=0) - 1)),
                          np.max(np.abs(matrix.sum(axis=1) - 1))))
        if error < tolerance:
            return matrix, {"bandwidths": bandwidths.tolist(),
                            "iterations": iteration, "max_marginal_error": error}
    raise ValueError("MCSMOTE Sinkhorn normalization did not converge.")


def mcsmote_batch(points: NDArray, eligible_indices: NDArray, draws: NDArray,
                  rng: np.random.Generator, *, q: float, tolerance: float,
                  max_iter: int) -> tuple[NDArray, dict[str, Any]]:
    """Transition from each seed and interpolate using source Equation 1."""
    matrix, details = transition_matrix(points[eligible_indices], q, tolerance, max_iter)
    local = {int(index): position for position, index in enumerate(eligible_indices)}
    # Sinkhorn accepts marginals within tolerance, but numpy choice requires
    # each sampling row to sum to one within a much tighter float tolerance.
    targets = np.array([eligible_indices[rng.choice(len(eligible_indices),
                       p=matrix[local[int(seed)]] / matrix[local[int(seed)]].sum())]
                       for seed in draws], dtype=int)
    steps = rng.random((len(draws), 1))
    generated = (1 - steps) * points[draws] + steps * points[targets]
    if not np.isfinite(generated).all():
        raise ValueError("MCSMOTE interpolation produced nonfinite values.")
    return generated, {"transition_reference_indices": eligible_indices.tolist(),
                       "target_indices": targets.tolist(),
                       "interpolation_steps": steps[:, 0].tolist(),
                       "transition": details, "q": q,
                       "interpolation_policy": "equation_1_convex"}


class MCSMOTEOversampler(BaseEstimator):
    """Original-source numerical MCSMOTE full-balance comparator."""

    def __init__(self, q: float = 0.1, tolerance: float = 1e-4,
                 max_sinkhorn_iter: int = 1000, random_state: int = 42,
                 allow_provisional: bool = False) -> None:
        self.q = q
        self.tolerance = tolerance
        self.max_sinkhorn_iter = max_sinkhorn_iter
        self.random_state = random_state
        self.allow_provisional = allow_provisional

    def fit_resample(self, X: Any, y: Any) -> tuple[NDArray, NDArray]:
        for attr in ["diagnostics_", "classes_", "n_features_in_"]:
            if hasattr(self, attr):
                delattr(self, attr)
        if not isinstance(self.allow_provisional, bool) or not self.allow_provisional:
            raise ValueError("MCSMOTE source conflicts remain; set allow_provisional=True.")
        validate_transition_parameters(self.q, self.tolerance, self.max_sinkhorn_iter)
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2 ** 32):
            raise ValueError("random_state must be an integer in [0, 2**32).")
        points = numeric_matrix(X)
        labels, classes = binary_labels(y, len(points))
        counts = np.array([(labels == label).sum() for label in classes])
        count = int(counts.max() - counts.min())
        warnings.warn("Provisional MCSMOTE: paper pseudocode conflicts are documented.",
                      UserWarning, stacklevel=2)
        self.classes_ = classes.copy()
        self.n_features_in_ = points.shape[1]
        self.diagnostics_ = {"source_id": "MCSMOTE-PDF",
                             "reproduction_status": "original_source_with_declared_conventions",
                             "number_of_synthetic_samples": count,
                             "random_state": int(self.random_state), "q": self.q,
                             "stopping_reason": "balanced_counts" if count == 0 else "minority_balanced"}
        if count == 0:
            return points, labels
        selected = classes[int(np.argmin(counts))]
        eligible = np.flatnonzero(labels == selected)
        rng = np.random.default_rng(self.random_state)
        draws = rng.choice(eligible, size=count, replace=True)
        generated, details = mcsmote_batch(points, eligible, draws, rng,
                                            q=self.q, tolerance=self.tolerance,
                                            max_iter=self.max_sinkhorn_iter)
        self.diagnostics_["generation"] = {"drawn_seed_indices": draws.tolist(), **details}
        return np.vstack((points, generated)), np.concatenate(
            (labels, np.full(count, selected, dtype=labels.dtype)))
