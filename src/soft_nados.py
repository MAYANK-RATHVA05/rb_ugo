"""Proposed soft-gated extension of the independently implemented NADOS.

The original NADOS sampler remains unchanged in :mod:`src.nados`.  This
module implements one author-suggested future direction: replace the hard
reliability cutoff with a smooth logistic gate.  It is research software, not
part of the paper-aligned NADOS baseline.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from src._validation import binary_labels, finite_real, numeric_matrix
from src.nados import NADOS, NADOSUnsupportedError


class SoftNADOS:
    """NADOS priority with a smooth reliability gate.

    The proposed priority is ``R * D * sigmoid((R - tau) / temperature)``.
    A seed must still have at least one local minority interpolation partner.
    All original rows are retained, as in NADOS.
    """

    _SCORE_ATTRIBUTES = (
        "minority_indices_", "neighbor_indices_", "neighbor_distances_",
        "minority_neighbor_counts_", "minority_support_",
        "mean_minority_distance_", "mean_majority_distance_",
        "relative_distance_evidence_", "reliability_scores_",
        "boundary_scores_", "difficulty_scores_", "partner_pools_",
    )

    def __init__(
        self,
        *,
        k_neighbors: int = 5,
        reliability_weight: float = 0.5,
        difficulty_weight: float = 0.5,
        reliability_threshold: float = 0.35,
        temperature: float = 0.05,
        target_ratio: float = 1.0,
        minority_label: Any = 1,
        random_state: int = 42,
    ):
        self.k_neighbors = k_neighbors
        self.reliability_weight = reliability_weight
        self.difficulty_weight = difficulty_weight
        self.reliability_threshold = reliability_threshold
        self.temperature = temperature
        self.target_ratio = target_ratio
        self.minority_label = minority_label
        self.random_state = random_state

    def _validated_parameters(self) -> tuple[int, float, float, float, float, float, int]:
        # Reuse the baseline's exact validation without changing its source.
        k, lam, eta, tau, ratio, seed = NADOS(
            k_neighbors=self.k_neighbors,
            reliability_weight=self.reliability_weight,
            difficulty_weight=self.difficulty_weight,
            reliability_threshold=self.reliability_threshold,
            target_ratio=self.target_ratio,
            minority_label=self.minority_label,
            random_state=self.random_state,
        )._parameters()
        temperature = finite_real(self.temperature, "temperature")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral):
            raise ValueError("random_state must be an integer.")
        return k, lam, eta, tau, temperature, ratio, seed

    @staticmethod
    def _logistic(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -709.0, 709.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit_resample(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        X = numeric_matrix(X)
        y, classes = binary_labels(y, len(X))
        k, lam, eta, tau, temperature, target_ratio, seed = self._validated_parameters()
        if self.minority_label not in classes:
            raise ValueError("minority_label must be one of the two observed labels.")

        minority_mask = y == self.minority_label
        minority_count = int(minority_mask.sum())
        majority_count = int((~minority_mask).sum())
        if minority_count >= majority_count:
            raise NADOSUnsupportedError(
                "minority_not_underrepresented",
                "SoftNADOS requires minority_label to identify the strictly smaller class.",
            )

        # Ask the unchanged NADOS baseline to calculate the paper-aligned local
        # evidence without generating samples.  Generation below uses only the
        # proposed soft priority.
        scorer = NADOS(
            k_neighbors=k,
            reliability_weight=lam,
            difficulty_weight=eta,
            reliability_threshold=tau,
            target_ratio=minority_count / majority_count,
            minority_label=self.minority_label,
            random_state=seed,
        )
        scorer.fit_resample(X, y)
        for name in self._SCORE_ATTRIBUTES:
            value = getattr(scorer, name)
            if name == "partner_pools_":
                value = tuple(pool.copy() for pool in value)
            else:
                value = np.asarray(value).copy()
            setattr(self, name, value)

        reliability = self.reliability_scores_
        difficulty = self.difficulty_scores_
        supported = self.minority_neighbor_counts_ > 0
        gate = self._logistic((reliability - tau) / temperature)
        priority = np.where(supported, reliability * difficulty * gate, 0.0)
        active = priority > 0

        target_minority = int(np.floor(target_ratio * majority_count))
        generated_count = max(0, target_minority - minority_count)
        priority_sum = float(priority.sum())
        if generated_count and (not np.isfinite(priority_sum) or priority_sum <= 0):
            raise NADOSUnsupportedError(
                "no_positive_synthesis_priority",
                "No supported SoftNADOS seed has positive soft-gated priority.",
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
            seed_index = int(self.minority_indices_[seed_position])
            partners = self.partner_pools_[seed_position]
            if not len(partners):
                raise RuntimeError("An active SoftNADOS seed has no minority partner.")
            partner_index = int(rng.choice(partners))
            fraction = float(rng.random())
            generated[index] = X[seed_index] + fraction * (X[partner_index] - X[seed_index])
            generated_seeds[index] = seed_index
            generated_partners[index] = partner_index
            fractions[index] = fraction

        out_X = np.vstack((X, generated))
        out_y = np.concatenate((y, np.full(generated_count, self.minority_label, dtype=y.dtype)))

        self.hard_gate_mask_ = (reliability >= tau) & supported
        self.soft_gate_weights_ = gate
        self.active_seed_mask_ = active
        self.active_seed_indices_ = self.minority_indices_[active]
        self.inactive_seed_indices_ = self.minority_indices_[~active]
        self.synthesis_priorities_ = priority
        self.seed_probabilities_ = probabilities
        self.generated_seed_indices_ = generated_seeds
        self.generated_partner_indices_ = generated_partners
        self.interpolation_fractions_ = fractions
        self.retained_indices_ = np.arange(len(X))
        self.diagnostics_ = {
            "status": "success",
            "implementation": "proposed_soft_gated_nados",
            "priority_equation": "R*D*sigmoid((R-tau)/temperature)",
            "minority_label": self.minority_label,
            "temperature": temperature,
            "input_counts": {"majority": majority_count, "minority": minority_count},
            "target_ratio": target_ratio,
            "target_minority_count": target_minority,
            "generated_count": generated_count,
            "hard_gate_seed_count": int(self.hard_gate_mask_.sum()),
            "active_seed_count": int(active.sum()),
            "output_counts": {
                "majority": majority_count,
                "minority": minority_count + generated_count,
            },
        }
        return out_X, out_y
