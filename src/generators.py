"""Fitting-only batch generation shared by provisional UGO variants.

RWO's numerical perturbation follows the inspected smote-variants 1.0.1 source,
not verified author code. See docs/reproduction/GENERATOR_ALIGNMENT.md.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from src._validation import binary_labels, numeric_matrix, positive_integer
from src.noise_detection import nearest_indices


def generator_metadata(variant: str) -> dict[str, str]:
    """Expose evidence separately from UGO's core reproduction status."""
    if not isinstance(variant, str) or variant not in {"ros", "smote", "rwo"}:
        if variant == "mcsmote":
            raise ValueError("MCSMOTE is unavailable: its full specification or verified code is required.")
        raise ValueError("variant must be 'ros', 'smote', or 'rwo'.")
    if variant == "rwo":
        return {"method": variant, "source_id": "RWO-SMOTE-VARIANTS-1.0.1",
                "status": "third_party_numeric_reference_original_unverified"}
    return {"method": variant, "source_id": "UGO-PDF",
            "status": "source_audited_with_pool_and_draw_conventions"}


def generate_batch(
    variant: str, X: Any, y: Any, seeds: Any, selected: Any,
    pre_noise: Any, batch_size: int, rng: np.random.Generator, *, k_smote: int = 5,
) -> tuple[NDArray | None, dict[str, Any]]:
    """Generate exactly one batch from explicit original fitting pools.

    Seeds must be clean rows of the selected class. SMOTE partners and RWO
    spread use all clean original rows of that class, including nonseed rows.
    No estimator, labels of held-out data, or augmented pool is accepted.
    None means insufficient SMOTE partners; there is no method substitution.
    """
    generator_metadata(variant)
    matrix = numeric_matrix(X)
    labels, _ = binary_labels(y, len(matrix))
    noise = np.asarray(pre_noise)
    seed_indices = np.asarray(seeds)
    if noise.shape != labels.shape or noise.dtype.kind != "b":
        raise ValueError("pre_noise must be an aligned boolean vector.")
    if seed_indices.ndim != 1 or not len(seed_indices) or seed_indices.dtype.kind not in "iu":
        raise ValueError("seeds must be a nonempty integer index vector.")
    if (seed_indices < 0).any() or (seed_indices >= len(matrix)).any():
        raise ValueError("Seed indices must refer to original fitting rows.")
    if np.any(noise[seed_indices]) or np.any(labels[seed_indices] != selected):
        raise ValueError("Seeds must be clean original rows of the selected class.")
    positive_integer(batch_size, "batch_size")
    positive_integer(k_smote, "k_smote")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator.")
    draws = rng.choice(seed_indices, size=batch_size, replace=True)
    info: dict[str, Any] = {"drawn_seed_indices": draws.tolist()}
    if variant == "ros":
        return matrix[draws].copy(), info
    eligible = np.flatnonzero((labels == selected) & ~noise)
    if variant == "rwo":
        # Third-party source uses population std (ddof=0), not sample std.
        with np.errstate(over="ignore", invalid="ignore"):
            scale = np.std(matrix[eligible], axis=0, ddof=0) / np.sqrt(len(eligible))
            generated = matrix[draws] + rng.normal(size=(batch_size, matrix.shape[1])) * scale
        if not np.isfinite(scale).all() or not np.isfinite(generated).all():
            raise ValueError("RWO perturbation produced nonfinite values.")
        info.update({"spread_reference_indices": eligible.tolist(),
                     "perturbation_scale": scale.tolist(), "std_ddof": 0,
                     "zero_spread_features": np.flatnonzero(scale == 0).tolist()})
        return generated, info
    if len(eligible) < 2:
        return None, {"effective_k_smote": 0}
    k = min(k_smote, len(eligible) - 1)
    neighborhoods = nearest_indices(matrix[eligible], matrix[eligible], k, exclude_self=True)
    seed_to_local = {int(index): local for local, index in enumerate(eligible)}
    partners = np.array([
        eligible[rng.choice(neighborhoods[seed_to_local[int(seed)]])]
        for seed in draws
    ])
    steps = rng.random((batch_size, 1))
    generated = (1 - steps) * matrix[draws] + steps * matrix[partners]
    if not np.isfinite(generated).all():
        raise ValueError("Synthetic interpolation produced nonfinite values.")
    info.update({"neighbor_indices": partners.tolist(), "interpolation_steps": steps[:, 0].tolist(),
                 "effective_k_smote": k})
    return generated, info


class RandomWalkOversampler(BaseEstimator):
    """Provisional numeric RWO baseline; balance the original minority count.

    Uses all original minority rows as seeds/spread references, with no UGO
    noise filtering, uncertainty selection, stopping test, or relocation.
    Dense numerical features only; discrete-feature RWO is not implemented.
    """

    def __init__(self, random_state: int = 42, allow_provisional: bool = False) -> None:
        self.random_state = random_state
        self.allow_provisional = allow_provisional

    def fit_resample(self, X: Any, y: Any) -> tuple[NDArray, NDArray]:
        for attr in ["diagnostics_", "classes_", "n_features_in_"]:
            if hasattr(self, attr):
                delattr(self, attr)
        if not isinstance(self.allow_provisional, bool) or not self.allow_provisional:
            raise ValueError("Original RWO source is unverified; set allow_provisional=True.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral) or not 0 <= self.random_state < 2 ** 32:
            raise ValueError("random_state must be an integer in [0, 2**32).")
        matrix = numeric_matrix(X)
        labels, classes = binary_labels(y, len(matrix))
        counts = np.array([(labels == label).sum() for label in classes])
        count = int(counts.max() - counts.min())
        warnings.warn("Provisional RWO: third-party numeric source; original specification unverified.",
                      UserWarning, stacklevel=2)
        self.classes_ = classes.copy()
        self.n_features_in_ = matrix.shape[1]
        self.diagnostics_ = {"generator": generator_metadata("rwo"),
                             "random_state": int(self.random_state),
                             "number_of_synthetic_samples": count,
                             "stopping_reason": "balanced_counts" if count == 0 else "minority_balanced"}
        if count == 0:
            return matrix, labels
        selected = classes[int(np.argmin(counts))]
        seeds = np.flatnonzero(labels == selected)
        generated, info = generate_batch("rwo", matrix, labels, seeds, selected,
                                         np.zeros(len(matrix), dtype=bool), count,
                                         np.random.default_rng(self.random_state))
        self.diagnostics_["generation"] = info
        return (np.vstack((matrix, generated)),
                np.concatenate((labels, np.full(count, selected, dtype=labels.dtype))))
