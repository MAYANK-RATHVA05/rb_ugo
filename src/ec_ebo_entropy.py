"""Training-only EC-EBO sample entropy and minority-cluster allocation.

The paper's prose specifies *mean* cluster entropy, while Eq. 7 prints a
sum. The declared mean interpretation and largest-remainder count rounding
are documented in docs/EC_EBO_METHOD_DECISIONS.md. This module does not
generate samples or evaluate a classifier.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from src._validation import finite_real, numeric_matrix, positive_integer
from src.ec_ebo_clustering import ExternallyConstrainedClustering


def binary_entropy(probabilities: Any, *, clip: float = 1e-12) -> np.ndarray:
    """Eq. 6, using the minority probability for rows of either class."""
    epsilon = finite_real(clip, "clip")
    if not 0 < epsilon < 0.5:
        raise ValueError("clip must lie strictly between 0 and 0.5.")
    values = np.asarray(probabilities)
    if (values.ndim != 1 or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
        raise ValueError("probabilities must be a finite one-dimensional vector in [0, 1].")
    safe = np.clip(values.astype(float), epsilon, 1.0 - epsilon)
    return -safe * np.log(safe) - (1.0 - safe) * np.log1p(-safe)


def mean_cluster_entropies(sample_entropies: Any,
                           clusters: Sequence[Any]) -> np.ndarray:
    """Average member entropy so cluster size alone does not set the weight."""
    values = np.asarray(sample_entropies)
    if (values.ndim != 1 or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("sample_entropies must be finite nonnegative values.")
    if len(clusters) == 0:
        raise ValueError("At least one minority cluster is required.")
    means = []
    seen: set[int] = set()
    for members in clusters:
        indices = np.asarray(members)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
                or (indices < 0).any() or (indices >= len(values)).any()
                or len(np.unique(indices)) != len(indices)
                or any(int(index) in seen for index in indices)):
            raise ValueError("Clusters must contain unique, valid, nonoverlapping row indices.")
        seen.update(map(int, indices))
        means.append(float(np.mean(values[indices])))
    return np.asarray(means, dtype=float)


def allocate_entropy_budget(weights: Any, budget: int) -> np.ndarray:
    """Allocate exactly `budget` rows; stable cluster order breaks equal remainders."""
    values = np.asarray(weights)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("weights must be a nonempty finite nonnegative vector.")
    if (isinstance(budget, bool) or not isinstance(budget, Integral)
            or budget < 0 or budget > np.iinfo(np.int64).max):
        raise ValueError("budget must be a nonnegative integer within int64 range.")
    if budget == 0:
        return np.zeros(len(values), dtype=int)
    total = float(np.sum(values, dtype=float))
    if not np.isfinite(total):
        raise ValueError("Total entropy weight overflowed.")
    proportions = np.full(len(values), 1.0 / len(values)) if total == 0 else values / total
    exact = int(budget) * proportions
    allocated = np.floor(exact).astype(int)
    remaining = int(budget) - int(allocated.sum())
    if remaining:
        order = np.argsort(-(exact - allocated), kind="stable")
        allocated[order[:remaining]] += 1
    return allocated


class EntropyBalancedAllocation(BaseEstimator):
    """Fit logistic entropy and assign quotas to externally constrained clusters.

    This estimator only consumes the rows supplied to `fit`. The caller must
    pass a fitting partition, never held-out test rows. It does not oversample.
    """

    def __init__(self, *, minority_label: Any = None, random_state: int = 42,
                 max_iter: int = 1000, entropy_clip: float = 1e-12):
        self.minority_label = minority_label
        self.random_state = random_state
        self.max_iter = max_iter
        self.entropy_clip = entropy_clip

    def fit(self, X: Any, y: Any) -> EntropyBalancedAllocation:
        for name in ("clustering_", "logistic_model_", "minority_probabilities_",
                     "sample_entropies_", "cluster_entropies_", "allocated_counts_",
                     "requested_count_", "diagnostics_", "n_features_in_"):
            if hasattr(self, name):
                delattr(self, name)
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2**32):
            raise ValueError("random_state must be an integer in [0, 2**32).")
        positive_integer(self.max_iter, "max_iter")
        epsilon = finite_real(self.entropy_clip, "entropy_clip")
        if not 0 < epsilon < 0.5:
            raise ValueError("entropy_clip must lie strictly between 0 and 0.5.")

        points = numeric_matrix(X)
        clustering = ExternallyConstrainedClustering(
            minority_label=self.minority_label).fit(points, y)
        model = LogisticRegression(max_iter=int(self.max_iter),
                                   random_state=int(self.random_state))
        model.fit(points, y)
        minority_column = int(np.flatnonzero(model.classes_ == clustering.minority_label_)[0])
        probabilities = model.predict_proba(points)[:, minority_column]
        sample_entropies = binary_entropy(probabilities, clip=epsilon)
        cluster_entropies = mean_cluster_entropies(sample_entropies, clustering.clusters_)
        budget = len(clustering.majority_indices_) - len(clustering.minority_indices_)
        allocated = allocate_entropy_budget(cluster_entropies, budget)

        self.clustering_ = clustering
        self.logistic_model_ = model
        self.minority_probabilities_ = probabilities.copy()
        self.sample_entropies_ = sample_entropies
        self.cluster_entropies_ = cluster_entropies
        self.allocated_counts_ = allocated
        self.requested_count_ = budget
        self.n_features_in_ = points.shape[1]
        self.diagnostics_ = {
            "status": "allocation_only", "requested_count": budget,
            "cluster_sizes": [len(members) for members in clustering.clusters_],
            "cluster_mean_entropies": cluster_entropies.tolist(),
            "allocated_counts": allocated.tolist(),
            "zero_entropy_equal_fallback": bool(not cluster_entropies.any()),
            "entropy_policy": "mean member binary entropy; logistic minority probability",
            "logistic_policy": {"class_weight": None, "max_iter": int(self.max_iter),
                                "random_state": int(self.random_state)},
            "entropy_clip": epsilon,
        }
        return self
