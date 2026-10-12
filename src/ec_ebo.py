"""Audited EC-EBO interpretation for numerical binary fitting data.

The paper's printed score sign and stopping order conflict with its prose.
This sampler follows the declared minority-side rule in
docs/EC_EBO_METHOD_DECISIONS.md. A positive score is only a local proxy;
generation does not guarantee population-level absence of class overlap.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer
from src.ec_ebo_entropy import EntropyBalancedAllocation


def entropy_balance_score(point: np.ndarray, centroid: np.ndarray,
                          majority_point: np.ndarray, cluster_entropy: float,
                          majority_entropy: float, alpha: float) -> float:
    """Paper Eq. 8: minority influence minus majority influence."""
    return float(cluster_entropy * np.exp(-alpha * np.linalg.norm(point - centroid))
                 - majority_entropy * np.exp(-alpha * np.linalg.norm(point - majority_point)))


def _minority_step(points: np.ndarray, fraction: float) -> float:
    """Median nearest different-location minority distance, scaled by fraction."""
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    if len(unique) < 2:
        return 0.0
    model = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(unique)
    distances = model.kneighbors(unique, return_distance=True)[0][:, 1]
    if not np.isfinite(distances).all():
        raise ValueError("Minority distances overflowed; scale fitting data first.")
    return float(fraction * np.median(distances[inverse]))


class ECEBO(BaseEstimator):
    """Generate minority samples using fitted EC-EBO clusters and entropy.

    `fit_resample` expects numerical training features on a comparable scale.
    It returns unchanged originals first, followed by successfully generated
    minority rows. Unmet cluster quotas are reported and never reassigned.
    """

    def __init__(self, *, minority_label: Any = None, alpha: float = 1.0,
                 k_majority: int = 7, n_iterations: int = 60,
                 stop_probability: float = 0.2, step_fraction: float = 0.1,
                 random_state: int = 42, logistic_max_iter: int = 1000,
                 entropy_clip: float = 1e-12):
        self.minority_label = minority_label
        self.alpha = alpha
        self.k_majority = k_majority
        self.n_iterations = n_iterations
        self.stop_probability = stop_probability
        self.step_fraction = step_fraction
        self.random_state = random_state
        self.logistic_max_iter = logistic_max_iter
        self.entropy_clip = entropy_clip

    def _validate_parameters(self) -> None:
        positive_integer(self.k_majority, "k_majority")
        positive_integer(self.n_iterations, "n_iterations")
        positive_integer(self.logistic_max_iter, "logistic_max_iter")
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2**32):
            raise ValueError("random_state must be an integer in [0, 2**32).")
        for name in ("alpha", "step_fraction"):
            if finite_real(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive.")
        probability = finite_real(self.stop_probability, "stop_probability")
        if not 0 <= probability <= 1:
            raise ValueError("stop_probability must lie in [0, 1].")
        clip = finite_real(self.entropy_clip, "entropy_clip")
        if not 0 < clip < 0.5:
            raise ValueError("entropy_clip must lie strictly between 0 and 0.5.")

    def fit_resample(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        for name in ("allocation_", "generated_samples_", "generated_info_",
                     "diagnostics_", "step_", "minority_label_", "majority_label_"):
            if hasattr(self, name):
                delattr(self, name)
        self._validate_parameters()
        points = numeric_matrix(X)
        labels, classes = binary_labels(y, len(points))
        counts = np.array([(labels == label).sum() for label in classes])
        if counts[0] == counts[1]:
            self.generated_samples_ = np.empty((0, points.shape[1]), dtype=float)
            self.generated_info_ = ()
            self.step_ = 0.0
            self.diagnostics_ = {"status": "already_balanced", "requested_count": 0,
                                 "generated_count": 0, "shortfall_count": 0}
            return points, labels

        allocation = EntropyBalancedAllocation(
            minority_label=self.minority_label, random_state=self.random_state,
            max_iter=self.logistic_max_iter, entropy_clip=self.entropy_clip).fit(points, labels)
        clustering = allocation.clustering_
        minority = clustering.minority_indices_
        majority = clustering.majority_indices_
        requested = allocation.requested_count_
        step = _minority_step(points[minority], float(self.step_fraction))
        generated: list[np.ndarray] = []
        info: list[dict[str, Any]] = []
        generated_per_cluster: list[int] = []
        shortfall_reasons: list[dict[str, int]] = []
        rng = np.random.default_rng(self.random_state)

        if step > 0:
            for cluster_id, (members, centroid, cluster_entropy, quota) in enumerate(zip(
                    clustering.clusters_, clustering.centroids_,
                    allocation.cluster_entropies_, allocation.allocated_counts_)):
                created = 0
                reasons: Counter[str] = Counter()
                distances = np.linalg.norm(points[majority] - centroid, axis=1)
                if not np.isfinite(distances).all():
                    raise ValueError("Majority-reference distances overflowed; scale fitting data first.")
                nearest = np.argsort(distances, kind="stable")[:min(int(self.k_majority), len(majority))]
                reference_pool = majority[nearest]
                for _ in range(int(quota)):
                    reference_index = int(rng.choice(reference_pool))
                    reference = points[reference_index]
                    majority_entropy = float(allocation.sample_entropies_[reference_index])
                    initial_scores = np.array([
                        entropy_balance_score(points[index], centroid, reference,
                                              float(cluster_entropy), majority_entropy,
                                              float(self.alpha)) for index in members])
                    positive_seeds = members[initial_scores > 0]
                    if not len(positive_seeds):
                        reasons["no_positive_seed"] += 1
                        continue
                    seed_index = int(rng.choice(positive_seeds))
                    current = points[seed_index].copy()
                    initial_score = entropy_balance_score(
                        current, centroid, reference, float(cluster_entropy),
                        majority_entropy, float(self.alpha))
                    current_score = initial_score
                    accepted = 0
                    stop_reason = "proposal_limit"
                    proposals = 0
                    for proposals in range(1, int(self.n_iterations) + 1):
                        coordinate = int(rng.integers(points.shape[1]))
                        direction = -1 if rng.integers(2) == 0 else 1
                        proposal = current.copy()
                        proposal[coordinate] += direction * step
                        if not np.isfinite(proposal).all():
                            continue
                        proposed_score = entropy_balance_score(
                            proposal, centroid, reference, float(cluster_entropy),
                            majority_entropy, float(self.alpha))
                        if 0 < proposed_score < current_score:
                            current, current_score = proposal, proposed_score
                            accepted += 1
                            if rng.random() < self.stop_probability:
                                stop_reason = "probability"
                                break
                    if not accepted:
                        reasons["no_accepted_move"] += 1
                        continue
                    generated.append(current)
                    info.append({"cluster_id": cluster_id, "seed_index": seed_index,
                                 "majority_reference_index": reference_index,
                                 "accepted_steps": accepted, "proposals": proposals,
                                 "initial_score": initial_score, "final_score": current_score,
                                 "stop_reason": stop_reason})
                    created += 1
                generated_per_cluster.append(created)
                shortfall_reasons.append(dict(sorted(reasons.items())))
        else:
            generated_per_cluster = [0] * len(clustering.clusters_)
            shortfall_reasons = [({"no_positive_step": int(quota)} if quota else {})
                                 for quota in allocation.allocated_counts_]

        generated_array = (np.vstack(generated) if generated
                           else np.empty((0, points.shape[1]), dtype=float))
        seen = {tuple(row) for row in points}
        duplicate_count = 0
        for row in generated_array:
            key = tuple(row)
            duplicate_count += key in seen
            seen.add(key)
        count = len(generated_array)
        status = ("generated" if count == requested else
                  "partial_generation" if count else
                  "no_positive_step" if step == 0 else "no_valid_generation")
        self.allocation_ = allocation
        self.generated_samples_ = generated_array
        self.generated_info_ = tuple(info)
        self.step_ = step
        self.minority_label_ = clustering.minority_label_
        self.majority_label_ = clustering.majority_label_
        self.diagnostics_ = {
            "status": status, "interpretation": "audited_mean_entropy_minority_side",
            "requested_count": requested, "generated_count": count,
            "shortfall_count": requested - count,
            "cluster_allocations": allocation.allocated_counts_.tolist(),
            "cluster_generated_counts": generated_per_cluster,
            "cluster_shortfall_reasons": shortfall_reasons,
            "duplicate_generated_count": duplicate_count,
            "step": step, "step_fraction": float(self.step_fraction),
            "alpha": float(self.alpha), "k_majority": int(self.k_majority),
            "n_iterations": int(self.n_iterations),
            "stop_probability": float(self.stop_probability),
            "random_state": int(self.random_state),
            "allocation": allocation.diagnostics_,
        }
        synthetic_labels = np.repeat(self.minority_label_, count)
        return np.vstack((points, generated_array)), np.concatenate((labels, synthetic_labels))
