"""Paper-based AW-SMOTE with the audited rare-subgroup pilot conventions.

Yan, Guo & Su (2026), doi:10.3390/math14173238, Eqs. (1)--(12).
Independently written; not a port of the author's released code and not an
exact verified reproduction. See docs/research/rare_subgroups/PILOT_PROTOCOL.md.
Only numerical fitting X/y belong here; scaling, hidden groups, validation,
classifier training, and performance evaluation are outside this sampler.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer


class AWUnsupportedError(RuntimeError):
    """A valid fitting case cannot meet the declared AW generation rules.

    ``reason`` is a stable code; ``diagnostics`` is a snapshot for reporting.
    No unchanged-data or alternative-sampler result is returned on failure.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.diagnostics: dict[str, Any] = {}


def _nearest_other_rows(points: np.ndarray, query_ids: np.ndarray,
                        reference_ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Original row IDs, ordered by Euclidean distance then original row ID.

    Exclude only the query's own ID, retaining other rows at identical locations.
    Callers supply unique valid IDs into points; no labels or group IDs are used.
    """
    reference_ids = np.sort(reference_ids)
    available = len(reference_ids) - np.isin(query_ids, reference_ids).astype(int)
    if np.any(available < k):
        raise AWUnsupportedError("too_few_neighbors", f"Need {k} other reference rows per query.")
    distances = cdist(points[query_ids], points[reference_ids], metric="euclidean")
    if not np.isfinite(distances).all():
        raise AWUnsupportedError("nonfinite_distances", "Distances overflowed; scale fitting data first.")
    distances[query_ids[:, None] == reference_ids[None, :]] = np.inf
    # Sorting references by ID first makes stable distance ties deterministic.
    chosen = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return reference_ids[chosen], np.take_along_axis(distances, chosen, axis=1)


def _allocate_quotas(weights: np.ndarray, count: int) -> np.ndarray:
    """Paper half-up rounding and correction at the first largest weight.

    All-zero weights use the disclosed uniform fallback. The paper's correction
    can produce a negative quota; diagnose that case rather than change the rule.
    """
    if (not len(weights) or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise AWUnsupportedError("invalid_quota", "Cluster weights must be finite and nonnegative.")
    effective = weights.astype(float) if np.any(weights) else np.ones(len(weights))
    quotas = np.floor(effective / effective.sum() * count + 0.5).astype(np.int64)
    quotas[int(np.argmax(effective))] += count - int(quotas.sum())
    if np.any(quotas < 0) or int(quotas.sum()) != count:
        raise AWUnsupportedError("invalid_quota", "The paper's quota correction cannot meet nonnegative totals.")
    return quotas


class AWSMOTE(BaseEstimator):
    """AW-paper-conventions for binary numerical data with minority label 1.

    ``noise_policy='remove'`` excludes ineligible minority originals from the
    output. ``'retain_originals'`` restores those originals but uses identical
    seeds and synthetic samples. It is a seed-filter-only retention control,
    not the paper's incompletely specified KeepNoise algorithm.

    ``n_samples=None`` uses the ORIGINAL count difference, before cleaning.
    Therefore removal need not produce final 1:1 class totals. Output originals
    preserve input order; synthetic rows follow them. A zero request is a no-op.
    Each call resets RNG state from the integer seed. No preprocessing is fitted.
    """

    def __init__(self, *, min_samples: int = 2, k_neighbors: int = 5,
                 eps_percentile: float = 65, theta: float = 0.2,
                 gamma: float = 5.0, n_samples: int | None = None,
                 noise_policy: str = "remove", random_state: int = 42):
        self.min_samples = min_samples
        self.k_neighbors = k_neighbors
        self.eps_percentile = eps_percentile
        self.theta = theta
        self.gamma = gamma
        self.n_samples = n_samples
        self.noise_policy = noise_policy
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if positive_integer(self.min_samples, "min_samples") < 2:
            raise ValueError("min_samples must be at least 2.")
        positive_integer(self.k_neighbors, "k_neighbors")
        percentile = finite_real(self.eps_percentile, "eps_percentile")
        if not 0 <= percentile <= 100:
            raise ValueError("eps_percentile must lie in [0, 100].")
        if not 0 <= finite_real(self.theta, "theta") <= 1:
            raise ValueError("theta must lie in [0, 1].")
        if finite_real(self.gamma, "gamma") <= 0:
            raise ValueError("gamma must be positive.")
        if self.n_samples is not None and (isinstance(self.n_samples, bool)
                or not isinstance(self.n_samples, Integral) or self.n_samples < 0):
            raise ValueError("n_samples must be a nonnegative integer or None.")
        if (not isinstance(self.noise_policy, str)
                or self.noise_policy not in {"remove", "retain_originals"}):
            raise ValueError("noise_policy must be 'remove' or 'retain_originals'.")
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2**32):
            raise ValueError("random_state must be an integer in [0, 2**32).")

    def fit_resample(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        # Clear previous results even when a subsequent parameter/input check fails.
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        self._validate_parameters()
        points = numeric_matrix(X)
        labels, classes = binary_labels(y, len(points))
        if labels.dtype.kind not in "biuf" or not np.array_equal(classes, [0, 1]):
            raise ValueError("AW-paper-conventions requires binary numerical labels 0 and 1.")
        minority = np.flatnonzero(labels == 1)
        majority = np.flatnonzero(labels == 0)
        if len(minority) > len(majority):
            raise ValueError("Label 1 must be the minority class (or tied).")
        target = int(len(majority) - len(minority) if self.n_samples is None else self.n_samples)
        self.n_features_in_ = points.shape[1]
        self.minority_indices_ = minority
        self.majority_indices_ = majority
        self.generated_samples_ = np.empty((0, points.shape[1]))
        self.generated_seed_indices_ = np.empty(0, dtype=int)
        self.generated_partner_indices_ = np.empty(0, dtype=int)
        self.generated_cluster_labels_ = np.empty(0, dtype=int)
        self.interpolation_fractions_ = np.empty(0)
        self.diagnostics_ = {
            "status": "pending", "implementation": "AW-paper-conventions",
            "fidelity": "paper_based_with_declared_conventions",
            "noise_policy": self.noise_policy, "requested_count": target,
            "generated_count": 0, "input_counts": {"majority": len(majority), "minority": len(minority)},
            "count_reference": "original_before_cleaning", "random_state": int(self.random_state),
            "rng_policy": "fresh_default_rng_per_call", "excluded_seed_count": 0,
        }
        if target == 0:
            self.retained_indices_ = np.arange(len(points))
            self.diagnostics_.update(status="zero_requested", removed_original_count=0,
                                     output_counts=dict(self.diagnostics_["input_counts"]))
            return points, labels
        try:
            return self._generate(points, labels, minority, majority, target)
        except AWUnsupportedError as exc:
            self.diagnostics_.update(status="unsupported", reason=exc.reason)
            exc.diagnostics = deepcopy(self.diagnostics_)
            raise

    def _generate(self, points: np.ndarray, labels: np.ndarray, minority: np.ndarray,
                  majority: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
        _, distances = _nearest_other_rows(points, minority, minority, int(self.k_neighbors))
        self.epsilon_ = float(np.percentile(distances[:, -1], self.eps_percentile, method="linear"))
        self.diagnostics_["epsilon"] = self.epsilon_
        if self.epsilon_ <= 0:
            raise AWUnsupportedError("nonpositive_epsilon", "The fitted k-distance percentile is nonpositive.")
        clustered = DBSCAN(eps=self.epsilon_, min_samples=int(self.min_samples),
                           metric="euclidean", algorithm="brute", n_jobs=1).fit_predict(points[minority])
        self.cluster_labels_ = np.full(len(points), -2, dtype=int)
        self.cluster_labels_[minority] = clustered
        cluster_ids = [int(c) for c in np.unique(clustered) if c >= 0
                       and np.count_nonzero(clustered == c) >= 2]
        self.clusters_ = tuple(minority[clustered == c] for c in cluster_ids)
        self.eligible_indices_ = minority[np.isin(clustered, cluster_ids)]
        self.noise_indices_ = minority[clustered == -1]
        self.excluded_indices_ = np.setdiff1d(minority, self.eligible_indices_)
        self.cleaned_indices_ = np.sort(np.r_[majority, self.eligible_indices_])
        self.diagnostics_.update(cluster_count=len(cluster_ids),
                                 cluster_labels=cluster_ids,
                                 cluster_sizes=[len(c) for c in self.clusters_],
                                 excluded_seed_count=len(self.excluded_indices_))
        if not self.clusters_:
            raise AWUnsupportedError("no_valid_cluster", "No minority cluster of size at least two survived.")

        self.boundary_neighbors_, _ = _nearest_other_rows(
            points, self.eligible_indices_, self.cleaned_indices_, int(self.k_neighbors))
        self.boundary_counts_ = np.full(len(points), -1, dtype=int)
        self.boundary_counts_[self.eligible_indices_] = (labels[self.boundary_neighbors_] == 0).sum(axis=1)
        self.cluster_weights_ = np.array([self.boundary_counts_[c].max() for c in self.clusters_])
        self.diagnostics_["uniform_cluster_fallback"] = bool(not self.cluster_weights_.any())
        self.cluster_quotas_ = _allocate_quotas(self.cluster_weights_, target)
        self.diagnostics_.update(cluster_weights=self.cluster_weights_.tolist(),
                                 cluster_quotas=self.cluster_quotas_.tolist())
        self.sample_probabilities_ = np.zeros(len(points))
        rng = np.random.default_rng(self.random_state)
        generated, seeds, partners, fractions, generated_clusters = [], [], [], [], []
        for cluster_id, members, quota in zip(cluster_ids, self.clusters_, self.cluster_quotas_):
            boundary = self.boundary_counts_[members].astype(float)
            boundary = boundary / boundary.sum() if boundary.any() else np.full(len(members), 1 / len(members))
            pair_distances = cdist(points[members], points[members], metric="euclidean")
            mean_distances = pair_distances.sum(axis=1) / (len(members) - 1)
            density = 1 / (mean_distances + 1e-8)
            if not np.isfinite(mean_distances).all() or not np.isfinite(density).all():
                raise AWUnsupportedError("nonfinite_distances", "Within-cluster distances overflowed.")
            density /= density.sum()
            alpha = expit(self.gamma * (boundary - self.theta))
            fused = alpha * boundary + (1 - alpha) * density
            probabilities = fused / fused.sum()
            self.sample_probabilities_[members] = probabilities
            # Draw seed counts first, then interpolate in stable original-row order.
            allocation = np.bincount(rng.choice(len(members), size=int(quota), p=probabilities),
                                     minlength=len(members))
            for local_id, count in enumerate(allocation):
                seed_id = int(members[local_id])
                other_members = np.delete(members, local_id)
                for _ in range(int(count)):
                    partner_id = int(rng.choice(other_members))
                    fraction = float(rng.uniform(0.0, 1.0))
                    point = (1 - fraction) * points[seed_id] + fraction * points[partner_id]
                    generated.append(point)
                    seeds.append(seed_id)
                    partners.append(partner_id)
                    fractions.append(fraction)
                    generated_clusters.append(cluster_id)
        if len(generated) != target:
            raise AWUnsupportedError("requested_generation_count_not_met", "Synthetic count did not match N.")
        synthetic = np.asarray(generated).reshape(target, points.shape[1])
        if not np.isfinite(synthetic).all():
            raise AWUnsupportedError("nonfinite_generation", "Interpolation produced nonfinite features.")
        # Publish completed generation only, never partial samples after a failure.
        self.generated_samples_ = synthetic
        self.generated_seed_indices_ = np.asarray(seeds, dtype=int)
        self.generated_partner_indices_ = np.asarray(partners, dtype=int)
        self.generated_cluster_labels_ = np.asarray(generated_clusters, dtype=int)
        self.interpolation_fractions_ = np.asarray(fractions)
        self.retained_indices_ = (self.cleaned_indices_.copy() if self.noise_policy == "remove"
                                  else np.arange(len(points)))
        self.diagnostics_.update(status="generated", generated_count=target,
                                 removed_original_count=len(points) - len(self.retained_indices_),
                                 output_counts={"majority": len(majority),
                                                "minority": int(np.sum(labels[self.retained_indices_] == 1)) + target})
        return (np.vstack((points[self.retained_indices_], synthetic)),
                np.concatenate((labels[self.retained_indices_], np.full(target, 1, dtype=labels.dtype))))
