"""Independent paper-aligned implementation of the NADOS sampler.

NADOS separates whether an observed minority row is reliable enough to seed
synthetic generation from how much generation priority an eligible seed gets.
The implementation intentionally accepts only observed ``X`` and ``y``. Hidden
subgroup identities and corruption indicators belong to the evaluator.

The paper does not release executable code and leaves several edge cases
implicit. The exact conventions used here are documented in
``docs/research/rare_subgroups/NADOS_BASELINE.md``. They are explicit failure
or boundary conventions, not claims about the authors' implementation.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer


class NADOSUnsupportedError(ValueError):
    """The published procedure is undefined for the observed geometry."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class NADOS:
    """Reliability-Gated Difficulty-Aware Oversampling.

    Parameters follow the fixed main-paper configuration by default: ``k=5``,
    reliability and difficulty weights ``0.5``, eligibility threshold ``0.35``
    and a target minority/majority ratio of ``1.0``.
    """

    def __init__(
        self,
        *,
        k_neighbors: int = 5,
        reliability_weight: float = 0.5,
        difficulty_weight: float = 0.5,
        reliability_threshold: float = 0.35,
        target_ratio: float = 1.0,
        minority_label: Any = 1,
        random_state: int = 42,
    ):
        self.k_neighbors = k_neighbors
        self.reliability_weight = reliability_weight
        self.difficulty_weight = difficulty_weight
        self.reliability_threshold = reliability_threshold
        self.target_ratio = target_ratio
        self.minority_label = minority_label
        self.random_state = random_state

    def _parameters(self) -> tuple[int, float, float, float, float, int]:
        k = positive_integer(self.k_neighbors, "k_neighbors")
        lam = finite_real(self.reliability_weight, "reliability_weight")
        eta = finite_real(self.difficulty_weight, "difficulty_weight")
        tau = finite_real(self.reliability_threshold, "reliability_threshold")
        ratio = finite_real(self.target_ratio, "target_ratio")
        if not 0 <= lam <= 1:
            raise ValueError("reliability_weight must lie in [0, 1].")
        if not 0 <= eta <= 1:
            raise ValueError("difficulty_weight must lie in [0, 1].")
        if not 0 <= tau <= 1:
            raise ValueError("reliability_threshold must lie in [0, 1].")
        if not 0 < ratio <= 1:
            raise ValueError("target_ratio must lie in (0, 1].")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral):
            raise ValueError("random_state must be an integer.")
        return k, lam, eta, tau, ratio, int(self.random_state)

    @staticmethod
    def _stable_neighbors(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """All-class kNN, excluding each query by row identity.

        Stable sorting resolves equal distances by original fitting-row index.
        This avoids backend-dependent tie ordering.
        """
        if len(X) <= k:
            raise NADOSUnsupportedError(
                "too_few_rows", "NADOS requires at least k_neighbors + 1 fitting rows."
            )
        neighbors = np.empty((len(X), k), dtype=int)
        distances = np.empty((len(X), k), dtype=float)
        for row in range(len(X)):
            delta = X - X[row]
            all_distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            all_distances[row] = np.inf
            order = np.argsort(all_distances, kind="stable")[:k]
            neighbors[row] = order
            distances[row] = all_distances[order]
        return neighbors, distances

    def fit_resample(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        X = numeric_matrix(X)
        y, classes = binary_labels(y, len(X))
        k, lam, eta, tau, target_ratio, seed = self._parameters()
        if self.minority_label not in classes:
            raise ValueError("minority_label must be one of the two observed labels.")

        minority_mask = y == self.minority_label
        minority_indices = np.flatnonzero(minority_mask)
        majority_count = int((~minority_mask).sum())
        minority_count = len(minority_indices)
        if minority_count >= majority_count:
            raise NADOSUnsupportedError(
                "minority_not_underrepresented",
                "NADOS requires minority_label to identify the strictly smaller class.",
            )

        neighbor_indices, neighbor_distances = self._stable_neighbors(X, k)
        local_indices = neighbor_indices[minority_indices]
        local_distances = neighbor_distances[minority_indices]
        local_minority = minority_mask[local_indices]
        counts = local_minority.sum(axis=1).astype(int)
        support = counts / k

        mean_minority = np.full(minority_count, np.nan)
        mean_majority = np.full(minority_count, np.nan)
        distance_evidence = np.empty(minority_count, dtype=float)
        partner_pools: list[np.ndarray] = []
        for position in range(minority_count):
            min_distances = local_distances[position, local_minority[position]]
            maj_distances = local_distances[position, ~local_minority[position]]
            partners = local_indices[position, local_minority[position]]
            partner_pools.append(partners.copy())
            if not len(min_distances):
                # No local minority support also means there is no valid partner.
                distance_evidence[position] = 0.0
                continue
            mean_minority[position] = float(min_distances.mean())
            if not len(maj_distances):
                # No majority in the declared local neighborhood is maximally
                # favorable distance evidence. Difficulty is still computed
                # independently from support, as in the paper.
                distance_evidence[position] = 1.0
                continue
            mean_majority[position] = float(maj_distances.mean())
            denominator = mean_minority[position] + mean_majority[position]
            if denominator == 0:
                raise NADOSUnsupportedError(
                    "zero_cross_class_distance_denominator",
                    "NADOS relative distance evidence is undefined for coincident local classes.",
                )
            distance_evidence[position] = mean_majority[position] / denominator

        reliability = lam * support + (1 - lam) * distance_evidence
        boundary = 1 - np.abs(2 * support - 1)
        difficulty = eta * boundary + (1 - eta) * (1 - support)
        eligible = (reliability >= tau) & (counts > 0)
        priority = np.where(eligible, reliability * difficulty, 0.0)

        target_minority = int(np.floor(target_ratio * majority_count))
        generated_count = max(0, target_minority - minority_count)
        priority_sum = float(priority.sum())
        if generated_count and (not np.isfinite(priority_sum) or priority_sum <= 0):
            raise NADOSUnsupportedError(
                "no_positive_synthesis_priority",
                "No eligible NADOS seed has positive reliability-times-difficulty priority.",
            )
        probabilities = priority / priority_sum if priority_sum > 0 else np.zeros_like(priority)

        generated = np.empty((generated_count, X.shape[1]), dtype=float)
        generated_seeds = np.empty(generated_count, dtype=int)
        generated_partners = np.empty(generated_count, dtype=int)
        fractions = np.empty(generated_count, dtype=float)
        rng = np.random.default_rng(seed)
        minority_positions = np.arange(minority_count)
        for index in range(generated_count):
            seed_position = int(rng.choice(minority_positions, p=probabilities))
            seed_index = int(minority_indices[seed_position])
            partners = partner_pools[seed_position]
            if not len(partners):  # Defensive: eligibility already excludes this.
                raise RuntimeError("Eligible NADOS seed has no minority partner.")
            partner_index = int(rng.choice(partners))
            fraction = float(rng.random())
            generated[index] = X[seed_index] + fraction * (X[partner_index] - X[seed_index])
            generated_seeds[index] = seed_index
            generated_partners[index] = partner_index
            fractions[index] = fraction

        out_X = np.vstack((X, generated))
        out_y = np.concatenate((y, np.full(generated_count, self.minority_label, dtype=y.dtype)))

        self.minority_indices_ = minority_indices
        self.neighbor_indices_ = neighbor_indices
        self.neighbor_distances_ = neighbor_distances
        self.minority_neighbor_counts_ = counts
        self.minority_support_ = support
        self.mean_minority_distance_ = mean_minority
        self.mean_majority_distance_ = mean_majority
        self.relative_distance_evidence_ = distance_evidence
        self.reliability_scores_ = reliability
        self.boundary_scores_ = boundary
        self.difficulty_scores_ = difficulty
        self.eligible_mask_ = eligible
        self.eligible_indices_ = minority_indices[eligible]
        self.excluded_indices_ = minority_indices[~eligible]
        self.synthesis_priorities_ = priority
        self.seed_probabilities_ = probabilities
        self.partner_pools_ = tuple(partner_pools)
        self.generated_seed_indices_ = generated_seeds
        self.generated_partner_indices_ = generated_partners
        self.interpolation_fractions_ = fractions
        self.retained_indices_ = np.arange(len(X))
        self.diagnostics_ = {
            "status": "success",
            "implementation": "independent_paper_aligned_with_declared_edge_conventions",
            "minority_label": self.minority_label,
            "input_counts": {"majority": majority_count, "minority": minority_count},
            "target_ratio": target_ratio,
            "target_minority_count": target_minority,
            "generated_count": generated_count,
            "eligible_seed_count": int(eligible.sum()),
            "excluded_seed_count": int((~eligible).sum()),
            "output_counts": {"majority": majority_count, "minority": minority_count + generated_count},
        }
        return out_X, out_y
