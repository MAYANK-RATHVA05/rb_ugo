"""kNN pre-detection and generated-point relocation on fitting data only.

Tie handling and reference eligibility are provisional until the paper's tables
can be inspected. Equal-distance neighbors are ordered by reference row index.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances

from src._validation import binary_labels, numeric_matrix, positive_integer


def nearest_indices(
    queries: NDArray, reference: NDArray, k: int, *, exclude_self: bool = False,
) -> NDArray[np.int64]:
    """Exact Euclidean neighbors with stable row-index tie breaking.

    The full distance matrix is intended for small reproduction checks. Self
    exclusion masks the diagonal by index, not by zero distance: coincident
    but distinct samples remain valid neighbors.
    """
    distances = pairwise_distances(queries, reference, metric="euclidean")
    if not np.isfinite(distances).all():
        raise ValueError("Neighbor distances overflowed; rescale fitting data externally.")
    if exclude_self:
        np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :k]


def _unsupported_labels(neighbor_labels: NDArray, assigned: Any) -> NDArray[np.bool_]:
    # A tied binary vote retains the assigned label: disagreement needs a strict
    # opposing majority. This convention is recorded as a provisional choice.
    support = (neighbor_labels == assigned).sum(axis=1)
    return support < neighbor_labels.shape[1] / 2


def noise_pre_detect(X: Any, y: Any, k_pre: int = 5) -> NDArray[np.bool_]:
    """Mark strict neighbor-majority disagreement; preserve X and y.

    Exclude the sample itself by row index. Effective k=min(k_pre, n-1), with
    a minimum of one neighbor; ties retain the true label. Binary only.
    """
    features = numeric_matrix(X)
    labels, _ = binary_labels(y, len(features))
    k = min(positive_integer(k_pre, "k_pre"), len(features) - 1)
    neighbors = nearest_indices(features, features, k, exclude_self=True)
    support = (labels[neighbors] == labels[:, None]).sum(axis=1)
    return support < k / 2


def noise_final_detect_and_relocate(
    X_generated: Any,
    generated_label: Any,
    X_reference: Any,
    y_reference: Any,
    k_after: int = 1,
    *,
    reference_noise_mask: Any | None = None,
    k_pre: int = 5,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Relocate unsupported samples to nearest clean original same-class point.

    X_reference must be original fitting data, never augmented or held-out data.
    Detection votes use all original reference rows; relocation uses only rows
    not marked by pre-detection. Supply reference_noise_mask to reuse the UGO
    pre-pass; otherwise recompute it using k_pre. Missing eligible references
    raise an error; no sample is discarded. Returns corrected copy and masks,
    destination indices (-1 if unchanged), and effective-neighbor diagnostics.
    """
    generated = numeric_matrix(X_generated, allow_empty=True)
    reference = numeric_matrix(X_reference)
    labels, classes = binary_labels(y_reference, len(reference))
    if np.ndim(generated_label) != 0 or not np.any(classes == generated_label):
        raise ValueError("generated_label must be one scalar reference class.")
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("Generated and reference feature dimensions must match.")
    k = min(positive_integer(k_after, "k_after"), len(reference))
    if reference_noise_mask is None:
        pre_noise = noise_pre_detect(reference, labels, k_pre)
    else:
        pre_noise = np.asarray(reference_noise_mask)
        if pre_noise.dtype.kind != "b" or pre_noise.shape != (len(reference),):
            raise ValueError("reference_noise_mask must be an aligned boolean vector.")
    noise = np.zeros(len(generated), dtype=bool)
    destinations = np.full(len(generated), -1, dtype=np.int64)
    if len(generated):
        neighbors = nearest_indices(generated, reference, k)
        noise = _unsupported_labels(labels[neighbors], generated_label)
    if noise.any():
        eligible = np.flatnonzero((labels == generated_label) & ~pre_noise)
        if not len(eligible):
            raise ValueError("No original non-noisy same-class reference is available for relocation.")
        local = nearest_indices(generated[noise], reference[eligible], 1)[:, 0]
        destinations[noise] = eligible[local]
        generated[noise] = reference[eligible[local]]
    return generated, {
        "noise_mask": noise, "relocation_reference_indices": destinations,
        "relocated_count": int(noise.sum()), "effective_k_after": k,
        "reference_policy": "all_original_for_vote; original_pre_clean_same_class_for_relocation",
    }
