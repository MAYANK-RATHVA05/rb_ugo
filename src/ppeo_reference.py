"""Opt-in paper interpretations for the bounded SPSDO geometry diagnostic.

This is not the incomplete public PPEO implementation, its evolutionary
allocation, or its final prototype-aware classifier.  No subgroup identities
are accepted by either prototype fitting or quota allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class PrototypeFit:
    """IDs index minority rows in their original order; flags index prototypes."""

    prototype_ids: np.ndarray
    small_flags: np.ndarray
    diagnostics: dict


def _f1(y: np.ndarray, predicted: np.ndarray) -> float:
    tp = np.count_nonzero((y == 1) & (predicted == 1))
    fp = np.count_nonzero((y == 0) & (predicted == 1))
    fn = np.count_nonzero((y == 1) & (predicted == 0))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # numpy argmin supplies the explicitly declared lowest-index tie policy.
    return np.argmin(cdist(X, centers, metric="sqeuclidean"), axis=1)


def fit_phhc(X: np.ndarray, y: np.ndarray) -> PrototypeFit:
    """Deterministic interpretation of PPEO Algorithm 1 and Eqs. (4)/(7).

    Existing centroids are recomputed after off-label rows move to appended
    children; empty centroids retain their old position.  A worsening split is
    reverted in full and terminates.  Equal F1 is accepted.  A repeated observed
    partition/predicted-label state raises rather than imposing an unreported
    iteration limit (unused appended centroids do not hide a repeated state).
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if (X.ndim != 2 or not len(X) or y.shape != (len(X),)
            or not np.isfinite(X).all() or set(np.unique(y)) != {0, 1}):
        raise ValueError("PHHC requires finite fitting rows and both binary classes.")
    centers = np.array([X[y == label].mean(axis=0) for label in (0, 1)])
    labels = np.array([0, 1], dtype=int)
    assignment = _assign(X, centers)
    score = _f1(y, labels[assignment])
    accepted_scores, seen, restored, attempts = [score], set(), False, 0
    while True:
        # Canonicalize occupied cluster IDs so appended unused/renumbered
        # centroids cannot conceal a fixed observed partition, e.g. when
        # opposite-label rows have identical coordinates and assignment ties.
        canonical_ids = {}
        partition = np.asarray([canonical_ids.setdefault(int(cluster), len(canonical_ids))
                                for cluster in assignment], dtype=int)
        signature = hashlib.sha256(partition.tobytes()
                                   + labels[assignment].tobytes()).hexdigest()
        if signature in seen:
            raise RuntimeError("PHHC repeated an observed partition; interpretation did not converge.")
        seen.add(signature)
        new_centers, new_labels, split = [], [], []
        for cluster, label in enumerate(labels):
            members = np.flatnonzero(assignment == cluster)
            off = members[y[members] != label]
            own = members[y[members] == label]
            is_impure = bool(len(own) and len(off))
            kept = own if is_impure else members
            new_centers.append(X[kept].mean(axis=0) if len(kept) else centers[cluster])
            new_labels.append(int(label))
            if is_impure:
                split.append((X[off].mean(axis=0), int(1 - label)))
        if not split:
            stop = "no_new_clusters"
            break
        attempts += 1
        # Keep existing indices stable before appending all new children.
        candidate_centers = np.asarray(new_centers + [row for row, _ in split])
        candidate_labels = np.asarray(new_labels + [label for _, label in split])
        candidate_assignment = _assign(X, candidate_centers)
        candidate_score = _f1(y, candidate_labels[candidate_assignment])
        if candidate_score < score:
            restored, stop = True, "f1_decreased_restored"
            break
        centers, labels, assignment, score = (candidate_centers, candidate_labels,
                                               candidate_assignment, candidate_score)
        accepted_scores.append(score)
    minority_indices = np.flatnonzero(y == 1)
    inverse_minority = np.full(len(y), -1, dtype=int)
    inverse_minority[minority_indices] = np.arange(len(minority_indices))
    ids = np.full(len(minority_indices), -1, dtype=int)
    flags, sizes = [], []
    majority = X[y == 0]
    for cluster, label in enumerate(labels):
        members = np.flatnonzero((assignment == cluster) & (y == 1))
        if not len(members):
            continue
        if label == 1:
            groups = [members]
        else:
            minority = X[members]
            radii = cdist(minority, majority).min(axis=1)
            distances = cdist(minority, minority)
            adjacency = distances <= np.maximum(radii[:, None], radii[None, :])
            count, component = connected_components(csr_matrix(adjacency), directed=False)
            groups = [members[component == index] for index in range(count)]
        for group in groups:
            ids[inverse_minority[group]] = len(flags)
            flags.append(label == 0)
            sizes.append(len(group))
    if np.any(ids < 0):
        raise RuntimeError("PHHC failed to assign every minority fitting observation.")
    return PrototypeFit(ids, np.asarray(flags, dtype=bool), {
        "implementation": "paper_Algorithm_1_declared_interpretation_not_author_code",
        "f1_history": accepted_scores, "split_attempts": attempts,
        "restored_previous_model": restored, "stop_reason": stop,
        "prototype_sizes": sizes, "prototype_small_flags": flags,
        "final_cluster_labels": labels.tolist(),
        "final_cluster_centers": centers.tolist(),
        "final_assignment": assignment.tolist(),
        "tie_policy": "nearest centroid; smallest existing index wins",
        "empty_cluster_policy": "retain centroid and label",
    })


def allocate_anchor_quotas(prototype_ids: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    """Fixed inverse-prototype-size weights, then uniformly draw original anchors.

    These declared diagnostic weights are NOT PPEO's evolutionary optimization.
    """
    ids = np.asarray(prototype_ids)
    if ids.ndim != 1 or not len(ids) or count < 0 or int(count) != count:
        raise ValueError("Allocation needs prototype IDs and a nonnegative integer count.")
    unique, sizes = np.unique(ids, return_counts=True)
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("Prototype IDs must be nonnegative integers.")
    probabilities = 1.0 / sizes
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    drawn_prototypes = rng.choice(unique, size=count, p=probabilities)
    quotas = np.zeros(len(ids), dtype=int)
    for prototype in drawn_prototypes:
        candidates = np.flatnonzero(ids == prototype)
        quotas[rng.choice(candidates)] += 1
    return quotas


def generate_large_smote(X_minority: np.ndarray, prototype_ids: np.ndarray,
                         quotas: np.ndarray, small_flags: np.ndarray, *, seed: int):
    """Generate common large-prototype rows once; reuse them across geometry arms."""
    X = np.asarray(X_minority, dtype=float)
    ids, quotas = np.asarray(prototype_ids), np.asarray(quotas)
    rng = np.random.default_rng(seed)
    samples, anchors, copies = [], [], 0
    for anchor, quota in enumerate(quotas):
        if small_flags[ids[anchor]]:
            continue
        members = np.flatnonzero(ids == ids[anchor])
        others = members[members != anchor]
        k = min(int(np.floor(np.sqrt(len(members)))), len(others))
        if k:
            order = np.argsort(np.sum((X[others] - X[anchor]) ** 2, axis=1), kind="stable")
            neighbors = others[order[:k]]
        for _ in range(int(quota)):
            if k:
                target = rng.choice(neighbors)
                row = X[anchor] + rng.random() * (X[target] - X[anchor])
            else:
                row = X[anchor].copy()
                copies += 1
            samples.append(row)
            anchors.append(anchor)
    return (np.asarray(samples, dtype=float).reshape(-1, X.shape[1]),
            np.asarray(anchors, dtype=int), {"singleton_copies": copies})
