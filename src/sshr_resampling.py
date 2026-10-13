"""SSHR Algorithm 2 resampling from an already-fitted SSHC partition.

Selected unlabelled rows are existing observations assigned pseudo-labels;
only the within-cluster SMOTE fallback creates new feature vectors. Author
code remains unverified. Source conflicts and operational interpretations are
recorded in ``docs/SSHR_METHOD_DECISIONS.md``.

The caller must supply the SAME ordered fitting pools used by the clustering.
This functional core checks geometry, labels, and partition coherence, but
those structural checks cannot establish provenance of external input rows.
Stage 3's estimator will fit a fresh clustering on its own supplied pools.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.special import expit

from ._validation import binary_labels, finite_real, positive_integer
from .sshr_clustering import _feature_matrix, _scalar


@dataclass(frozen=True)
class SSHRResamplingResult:
    """Independent output arrays, original pool positions, and JSON diagnostics."""

    X_resampled: NDArray[np.float64]
    y_resampled: NDArray
    retained_labeled_indices: NDArray[np.int64]
    selected_unlabeled_indices: NDArray[np.int64]
    synthetic_samples: NDArray[np.float64]
    row_lineage: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class SSHRResamplingError(ValueError):
    """Diagnosed algorithm failure, distinct from malformed-input validation."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = deepcopy(diagnostics)


def _check_label_scalars(values: NDArray) -> None:
    for value in values:
        scalar = _scalar(value)
        if (not isinstance(scalar, (str, int, float, bool))
                or isinstance(scalar, float) and not math.isfinite(scalar)):
            raise ValueError("Labels must be finite JSON-compatible real numeric, boolean, or string scalars.")


def _canonical_label(value: Any, classes: NDArray, name: str) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar observed class.")
    scalar = array.item()
    _check_label_scalars(np.asarray([scalar]))
    matching = np.flatnonzero(classes == scalar)
    if not len(matching):
        raise ValueError(f"{name} must be an observed class.")
    return _scalar(classes[int(matching[0])])


def _distances(value: Any, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numerical distances.")
    result = np.array(array, dtype=float, copy=True)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must contain finite nonnegative distances.")
    return result


def majority_removal_weight(own_distance: Any, minority_distance: Any) -> Any:
    """Eq. 5, stably evaluated as expit(distance-to-own minus minority)."""
    own = _distances(own_distance, "own_distance")
    minority = _distances(minority_distance, "minority_distance")
    try:
        own, minority = np.broadcast_arrays(own, minority)
    except ValueError as exc:
        raise ValueError("Distance arrays must have broadcast-compatible shapes.") from exc
    result = expit(own - minority)
    return float(result) if result.ndim == 0 else result


def minority_confidence(observed_minority_count: int, cluster_size: int,
                        own_distance: float, total_distance: float) -> float | None:
    """Eq. 7 using observed minority anchors, never unobserved true labels.

    A zero denominator returns ``None`` and must be skipped, not ranked as
    confidence zero. Finite zero scores remain legitimate ranking candidates.
    """
    if (isinstance(observed_minority_count, bool)
            or not isinstance(observed_minority_count, Integral)
            or observed_minority_count < 0):
        raise ValueError("observed_minority_count must be a nonnegative integer.")
    size = positive_integer(cluster_size, "cluster_size")
    if observed_minority_count > size:
        raise ValueError("observed_minority_count cannot exceed cluster_size.")
    own = finite_real(own_distance, "own_distance")
    total = finite_real(total_distance, "total_distance")
    if not math.isfinite(own) or not math.isfinite(total) or own < 0 or total < 0 or own > total:
        raise ValueError("Distances must satisfy 0 <= own_distance <= total_distance.")
    if total == 0:
        return None
    return float((int(observed_minority_count) / size) * (1 - own / total))


@dataclass
class _Partition:
    centroids: NDArray[np.float64]
    ids: NDArray[np.int64]
    labels: NDArray
    labeled_ids: NDArray[np.int64]
    unlabeled_ids: NDArray[np.int64]
    minority: Any
    majority: Any
    clustering_diagnostics: dict[str, Any] | None


def _indices(value: Any, name: str, length: int | None = None) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must have length {length}.")
    if np.any(array < 0) or np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains an out-of-range cluster or row ID.")
    return np.array(array, dtype=np.int64, copy=True)


def _validate_partition(clustering: Any, X_l: NDArray, y: NDArray,
                        X_u: NDArray, classes: NDArray) -> _Partition:
    """Validate a fitted SSHC or an equivalent read-only structural object.

    Required fields are centroids_, cluster_ids_, cluster_labels_, classes_,
    minority_label_, majority_label_, labeled_cluster_ids_,
    unlabeled_cluster_ids_, and n_features_in_. Optional clusters_ and
    labeled_predictions_ are cross-checked if present. No field is mutated.
    """
    required = ("centroids_", "cluster_ids_", "cluster_labels_", "classes_",
                "minority_label_", "majority_label_", "labeled_cluster_ids_",
                "unlabeled_cluster_ids_", "n_features_in_")
    if any(not hasattr(clustering, name) for name in required):
        raise ValueError("clustering must provide a complete fitted SSHC partition.")
    width = positive_integer(clustering.n_features_in_, "clustering.n_features_in_")
    if width != X_l.shape[1]:
        raise ValueError("Clustering feature width differs from the supplied fitting pools.")
    centroids = _feature_matrix(clustering.centroids_)
    if centroids.shape[1] != width:
        raise ValueError("Clustering centroids have an inconsistent feature width.")
    ids = _indices(clustering.cluster_ids_, "cluster_ids_", len(centroids))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Cluster IDs must be unique.")
    labels = np.asarray(clustering.cluster_labels_)
    if labels.ndim != 1 or len(labels) != len(ids):
        raise ValueError("Cluster labels must align with cluster IDs and use the observed label domain.")
    _check_label_scalars(labels)
    labels = np.asarray([_canonical_label(label, classes, "cluster label") for label in labels], dtype=classes.dtype)
    fitted_classes = np.asarray(clustering.classes_)
    if fitted_classes.ndim != 1 or len(fitted_classes) != 2:
        raise ValueError("Clustering classes differ from the supplied observed label domain.")
    _check_label_scalars(fitted_classes)
    try:
        _, fitted_domain = binary_labels(fitted_classes, 2)
    except (ValueError, TypeError) as exc:
        raise ValueError("Clustering classes must form a valid comparable binary label domain.") from exc
    if not np.array_equal(fitted_domain, classes):
        raise ValueError("Clustering classes differ from the supplied observed label domain.")
    minority = _canonical_label(clustering.minority_label_, classes, "minority_label_")
    majority = _canonical_label(clustering.majority_label_, classes, "majority_label_")
    if minority == majority:
        raise ValueError("Clustering minority and majority identities must be distinct observed classes.")
    labeled_ids = _indices(clustering.labeled_cluster_ids_, "labeled_cluster_ids_", len(y))
    unlabeled_ids = _indices(clustering.unlabeled_cluster_ids_, "unlabeled_cluster_ids_", len(X_u))
    combined_ids = np.concatenate((labeled_ids, unlabeled_ids))
    if not np.isin(combined_ids, ids).all():
        raise ValueError("Every assigned cluster ID must name a fitted centroid.")
    if any(not np.any(combined_ids == cluster_id) for cluster_id in ids):
        raise ValueError("Fitted partitions must not contain empty clusters.")
    if hasattr(clustering, "clusters_"):
        members = clustering.clusters_
        if len(members) != len(ids):
            raise ValueError("clusters_ must align with cluster_ids_.")
        for cluster_id, rows in zip(ids, members):
            checked = _indices(rows, "clusters_ members")
            expected = np.flatnonzero(combined_ids == cluster_id)
            if not np.array_equal(np.sort(checked), expected):
                raise ValueError("clusters_ is inconsistent with the complete row assignments.")
    order = np.argsort(ids, kind="stable")
    centroids, ids, labels = centroids[order], ids[order], labels[order].copy()
    all_X = np.concatenate((X_l, X_u), axis=0)
    distances = cdist(all_X, centroids)
    if not np.isfinite(distances).all():
        raise ValueError("Clustering distances overflowed; rescale fitting features.")
    nearest_ids = ids[np.argmin(distances, axis=1)]
    if not np.array_equal(nearest_ids, combined_ids):
        raise ValueError("Supplied row assignments violate nearest-centroid/persistent-ID tie rules.")
    for cluster_id, label in zip(ids, labels):
        observed = y[labeled_ids == cluster_id]
        if len(observed):
            votes = [np.count_nonzero(observed == observed_label) for observed_label in classes]
            tied = classes[np.asarray(votes) == max(votes)]
            if not np.any(tied == label):
                raise ValueError("Anchored cluster labels must be among their dominant observed labels.")
    if hasattr(clustering, "labeled_predictions_"):
        predictions = labels[np.searchsorted(ids, labeled_ids)]
        if not np.array_equal(np.asarray(clustering.labeled_predictions_), predictions):
            raise ValueError("Clustering labelled predictions disagree with its row assignments.")
    diagnostics = deepcopy(getattr(clustering, "diagnostics_", None))
    if diagnostics is not None:
        try:
            json.dumps(diagnostics, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("Clustering diagnostics must be finite JSON-compatible data.") from exc
    return _Partition(centroids, ids, labels, labeled_ids, unlabeled_ids,
                      minority, majority, diagnostics)


def _class_counts(y: NDArray, partition: _Partition) -> dict[str, int]:
    return {"majority": int(np.count_nonzero(y == partition.majority)),
            "minority": int(np.count_nonzero(y == partition.minority))}


def _removal_record(indices: NDArray, y: NDArray, partition: _Partition) -> dict[str, Any]:
    return {"indices": indices.tolist(), "majority_count": int(np.count_nonzero(y[indices] == partition.majority)),
            "minority_count": int(np.count_nonzero(y[indices] == partition.minority))}


def _cluster_records(partition: _Partition, y: NDArray,
                     retained: NDArray[np.bool_], surviving: NDArray[np.bool_]) -> list[dict[str, Any]]:
    records = []
    for position in np.flatnonzero(surviving):
        cluster_id = partition.ids[position]
        observed = y[retained & (partition.labeled_ids == cluster_id)]
        n_u = int(np.count_nonzero(partition.unlabeled_ids == cluster_id))
        records.append({
            "id": int(cluster_id), "label": _scalar(partition.labels[position]),
            "centroid": partition.centroids[position].tolist(),
            "size": len(observed) + n_u, "labeled_size": len(observed), "unlabeled_size": n_u,
            "observed_majority_count": int(np.count_nonzero(observed == partition.majority)),
            "observed_minority_count": int(np.count_nonzero(observed == partition.minority)),
            "unanchored": len(observed) == 0, "impure": len(np.unique(observed)) > 1,
        })
    return records


def _interpolate(anchor: NDArray, neighbor: NDArray, coefficient: float) -> NDArray:
    """Convex interpolation without overflowing opposite-sign subtraction."""
    same_sign = np.signbit(anchor) == np.signbit(neighbor)
    result = np.empty_like(anchor)
    result[same_sign] = anchor[same_sign] + coefficient * (neighbor[same_sign] - anchor[same_sign])
    result[~same_sign] = ((1 - coefficient) * anchor[~same_sign]
                          + coefficient * neighbor[~same_sign])
    if not np.isfinite(result).all():
        raise ValueError("SMOTE interpolation produced nonfinite values; rescale fitting features.")
    return result


def _coordinate_duplicates(X_l: NDArray, X_u: NDArray, retained: NDArray,
                           selected: NDArray, synthetic: NDArray) -> dict[str, Any]:
    # Hashable real-valued tuples treat -0 and +0 as the same coordinates and
    # take linear storage; there is no pairwise n-by-n distance matrix here.
    def keys(rows: NDArray) -> list[tuple[float, ...]]:
        return [tuple(row) for row in rows]
    labeled_keys, unlabeled_keys = keys(X_l), keys(X_u)
    retained_keys = {labeled_keys[int(index)] for index in retained}
    pseudo_keys = [unlabeled_keys[int(index)] for index in selected]
    pseudo_prior: set[tuple[float, ...]] = set()
    repeated_pseudo = 0
    for key in pseudo_keys:
        repeated_pseudo += key in pseudo_prior
        pseudo_prior.add(key)
    seen = retained_keys | pseudo_prior
    synthetic_repeats = 0
    for key in keys(synthetic):
        synthetic_repeats += key in seen
        seen.add(key)
    return {
        "comparison_policy": "exact_coordinate_equality_distinct_from_observation_provenance",
        "input_labeled_duplicate_coordinates": len(labeled_keys) - len(set(labeled_keys)),
        "input_unlabeled_duplicate_coordinates": len(unlabeled_keys) - len(set(unlabeled_keys)),
        "pseudo_duplicate_with_retained_labeled": sum(key in retained_keys for key in pseudo_keys),
        "pseudo_duplicate_with_prior_pseudo": int(repeated_pseudo),
        "smote_duplicate_with_retained_or_pseudo_or_prior_smote": int(synthetic_repeats),
    }


def resample_from_clusters(
    X_labeled: Any,
    y_labeled: Any,
    X_unlabeled: Any,
    clustering: Any,
    *,
    delta: float = 0.1,
    removal_fraction: float = 0.3,
    k_neighbors: int = 5,
    random_state: int = 42,
    smote_pool_policy: str = "observed_and_selected",
) -> SSHRResamplingResult:
    """Apply audited SSHR cleaning, existing-row selection, and SMOTE fallback.

    There is no hidden-label argument and no estimator/runner integration. A
    no-minority-centroid condition yields explicit unmet quotas, not invented
    confidence or a different fallback. Deleted clusters are never reassigned.
    """
    X_l = _feature_matrix(X_labeled)
    y, classes = binary_labels(y_labeled, len(X_l))
    _check_label_scalars(classes)
    X_u = (np.empty((0, X_l.shape[1]), dtype=float) if X_unlabeled is None
           else _feature_matrix(X_unlabeled, allow_empty=True))
    if X_u.shape[1] != X_l.shape[1]:
        raise ValueError("Labelled and unlabelled matrices must have the same feature width.")
    size_fraction = finite_real(delta, "delta")
    fraction = finite_real(removal_fraction, "removal_fraction")
    if not 0 <= size_fraction <= 1:
        raise ValueError("delta must lie in [0, 1].")
    if not 0 <= fraction <= 1:
        raise ValueError("removal_fraction must lie in [0, 1].")
    neighbors = positive_integer(k_neighbors, "k_neighbors")
    if (isinstance(random_state, bool) or not isinstance(random_state, Integral)
            or not 0 <= random_state < 2**32):
        raise ValueError("random_state must be an integer in [0, 2**32).")
    seed = int(random_state)
    if (not isinstance(smote_pool_policy, str)
            or smote_pool_policy not in {"observed_and_selected", "observed_only"}):
        raise ValueError("smote_pool_policy must be 'observed_and_selected' or 'observed_only'.")
    partition = _validate_partition(clustering, X_l, y, X_u, classes)
    keep = np.ones(len(y), dtype=bool)
    surviving = np.ones(len(partition.ids), dtype=bool)
    initial_records = _cluster_records(partition, y, keep, surviving)
    majority_records = [record for record in initial_records if record["label"] == partition.majority]
    mean_majority_size = float(np.mean([record["size"] for record in majority_records])) if majority_records else None
    threshold = size_fraction * mean_majority_size if mean_majority_size is not None else None
    deleted_ids = np.array([record["id"] for record in majority_records
                            if record["size"] < threshold], dtype=np.int64)
    surviving &= ~np.isin(partition.ids, deleted_ids)
    small_removed = np.flatnonzero(np.isin(partition.labeled_ids, deleted_ids))
    keep[small_removed] = False
    assigned_labels = partition.labels[np.searchsorted(partition.ids, partition.labeled_ids)]
    minority_cleaned = np.flatnonzero(keep & (y == partition.majority)
                                       & (assigned_labels == partition.minority))
    keep[minority_cleaned] = False
    post_clean = _class_counts(y[keep], partition)
    diagnostics: dict[str, Any] = {
        "implementation_status": "paper_rules_with_declared_interpretations",
        "author_code_verified": False,
        "parameters": {"delta": size_fraction, "removal_fraction": fraction, "k_neighbors": neighbors,
                       "random_state": seed, "smote_pool_policy": smote_pool_policy,
                       "minority_label": partition.minority, "majority_label": partition.majority},
        "policies": {
            "cluster_size_basis": "assigned_labeled_plus_unlabeled",
            "cleaning": "simultaneous_small_majority_clusters_all_observed_rows_then_minority_cluster_majority_rows",
            "clustering_after_cleaning": "frozen_centroids_and_assignments_remove_deleted_centers_exclude_their_unlabeled_rows",
            "quota_timing": "after_preliminary_cleaning",
            "quota_rounding": "weighted_floor_fraction_gap_additions_remaining_gap",
            "weighted_ranking": "descending_eq5_stable_original_labeled_position",
            "weighted_reference": "all_surviving_minority_centroids_including_unanchored",
            "confidence_counts": "surviving_observed_minority_over_surviving_labeled_plus_original_assigned_unlabeled",
            "pseudo_ranking": "descending_eq7_stable_original_unlabeled_position_no_threshold",
            "smote_allocation": "eligible_pool_proportional_largest_remainder_cluster_id_ties",
            "smote_neighbor_ties": "stable_pool_order_labeled_then_selected_unlabeled_self_index_excluded",
            "output_order": "retained_labeled_original_order_pseudo_confidence_order_smote_cluster_id_order",
            "input_provenance": "caller_must_supply_same_ordered_clustering_fitting_pools",
        },
        "clustering_diagnostics": partition.clustering_diagnostics,
        "initial_clusters": initial_records,
        "small_cluster_threshold": {"mean_majority_size": mean_majority_size, "threshold": threshold,
                                    "deleted_cluster_ids": deleted_ids.tolist()},
        "removals": {
            "small_majority_clusters": _removal_record(small_removed, y, partition),
            "minority_cluster_majority": _removal_record(minority_cleaned, y, partition),
            "weighted_majority": _removal_record(np.empty(0, dtype=np.int64), y, partition),
        },
        "counts": {"input_labeled": len(y), "input_unlabeled": len(X_u),
                   "original": _class_counts(y, partition), "post_preclean": post_clean},
        "shortfalls": [],
    }
    if post_clean["majority"] == 0 or post_clean["minority"] == 0:
        diagnostics["status"] = "failure_class_erased_by_preliminary_cleaning"
        diagnostics["clusters"] = _cluster_records(partition, y, keep, surviving)
        diagnostics["counts"].update({"removed_majority": int(np.count_nonzero(~keep & (y == partition.majority))),
                                      "removed_minority": int(np.count_nonzero(~keep & (y == partition.minority))),
                                      "retained_labeled": int(np.count_nonzero(keep))})
        raise SSHRResamplingError("SSHR preliminary cleaning erased an observed class; class identities are not swapped.", diagnostics)
    gap = max(0, post_clean["majority"] - post_clean["minority"])
    requested_weighted = int(np.floor(fraction * gap))
    requested_additions = gap - requested_weighted
    survivor_positions = np.flatnonzero(surviving)
    survivor_ids = partition.ids[survivor_positions]
    survivor_centroids = partition.centroids[survivor_positions]
    minority_positions = np.flatnonzero(surviving & (partition.labels == partition.minority))
    weighted_candidates = []
    weighted_removed = np.empty(0, dtype=np.int64)
    if requested_weighted:
        if len(minority_positions):
            major_rows = np.flatnonzero(keep & (y == partition.majority))
            own_positions = np.searchsorted(partition.ids, partition.labeled_ids[major_rows])
            own_distances = np.linalg.norm(X_l[major_rows] - partition.centroids[own_positions], axis=1)
            nearest_minority = np.min(cdist(X_l[major_rows], partition.centroids[minority_positions]), axis=1)
            if not np.isfinite(own_distances).all() or not np.isfinite(nearest_minority).all():
                raise ValueError("Resampling distances overflowed; rescale fitting features.")
            weights = majority_removal_weight(own_distances, nearest_minority)
            ranking = np.argsort(-weights, kind="stable")
            weighted_candidates = [{"labeled_index": int(major_rows[index]),
                                    "cluster_id": int(partition.labeled_ids[major_rows[index]]),
                                    "weight": float(weights[index]), "own_distance": float(own_distances[index]),
                                    "minority_distance": float(nearest_minority[index])} for index in ranking]
            weighted_removed = major_rows[ranking[:requested_weighted]]
            keep[weighted_removed] = False
        else:
            diagnostics["shortfalls"].append({"stage": "weighted_removal", "cause": "no_surviving_minority_centroid",
                                               "count": requested_weighted})
    diagnostics["weighted_candidates"] = weighted_candidates
    diagnostics["weighted_reference_unanchored_count"] = sum(
        not np.any(keep & (partition.labeled_ids == partition.ids[position])) for position in minority_positions)
    diagnostics["removals"]["weighted_majority"] = _removal_record(weighted_removed, y, partition)
    cluster_records = _cluster_records(partition, y, keep, surviving)
    diagnostics["clusters"] = cluster_records
    by_id = {record["id"]: record for record in cluster_records}
    pseudo_candidates: list[dict[str, Any]] = []
    skipped_undefined: list[int] = []
    excluded_deleted: list[int] = []
    excluded_unanchored: list[int] = []
    for index, cluster_id in enumerate(partition.unlabeled_ids):
        cluster_id = int(cluster_id)
        if cluster_id not in by_id:
            excluded_deleted.append(index)
            continue
        record = by_id[cluster_id]
        if record["label"] != partition.minority:
            continue
        if record["observed_minority_count"] == 0:
            excluded_unanchored.append(index)
            continue
        distances = cdist(X_u[index:index + 1], survivor_centroids)[0]
        if not np.isfinite(distances).all():
            raise ValueError("Confidence distances overflowed; rescale fitting features.")
        own_position = int(np.searchsorted(survivor_ids, cluster_id))
        scale = float(np.max(distances))
        if scale == 0:
            confidence = minority_confidence(record["observed_minority_count"], record["size"], 0., 0.)
        else:
            normalized = distances / scale
            confidence = minority_confidence(record["observed_minority_count"], record["size"],
                                             float(normalized[own_position]), float(np.sum(normalized)))
        if confidence is None:
            skipped_undefined.append(index)
            continue
        pseudo_candidates.append({"unlabeled_index": index, "cluster_id": cluster_id,
                                  "confidence": confidence, "observed_minority_count": record["observed_minority_count"],
                                  "cluster_size": record["size"]})
    pseudo_candidates.sort(key=lambda record: (-record["confidence"], record["unlabeled_index"]))
    selected_records = pseudo_candidates[:requested_additions]
    selected = np.array([record["unlabeled_index"] for record in selected_records], dtype=np.int64)
    diagnostics["pseudo_candidates"] = pseudo_candidates
    diagnostics["selected_pseudo_rows"] = selected_records
    diagnostics["confidence_exclusions"] = {"undefined_denominator_indices": skipped_undefined,
                                            "deleted_cluster_indices": excluded_deleted,
                                            "unanchored_minority_cluster_indices": excluded_unanchored}
    deficit = requested_additions - len(selected)
    pools: list[dict[str, Any]] = []
    for record in cluster_records:
        if record["label"] != partition.minority:
            continue
        cluster_id = record["id"]
        observed_rows = np.flatnonzero(keep & (y == partition.minority) & (partition.labeled_ids == cluster_id))
        pseudo_rows = selected[partition.unlabeled_ids[selected] == cluster_id] if smote_pool_policy == "observed_and_selected" else np.empty(0, dtype=np.int64)
        points = np.concatenate((X_l[observed_rows], X_u[pseudo_rows]), axis=0)
        sources = ([{"kind": "labeled", "index": int(index)} for index in observed_rows]
                   + [{"kind": "unlabeled", "index": int(index)} for index in pseudo_rows])
        pools.append({"cluster_id": cluster_id, "points": points, "sources": sources, "allocation": 0})
    eligible = [pool for pool in pools if len(pool["points"]) >= 2]
    if deficit and eligible:
        total_size = sum(len(pool["points"]) for pool in eligible)
        for pool in eligible:
            pool["allocation"] = deficit * len(pool["points"]) // total_size
            pool["remainder"] = deficit * len(pool["points"]) % total_size
        leftover = deficit - sum(pool["allocation"] for pool in eligible)
        for pool in sorted(eligible, key=lambda pool: (-pool["remainder"], pool["cluster_id"]))[:leftover]:
            pool["allocation"] += 1
    elif deficit:
        cause = "no_surviving_minority_centroid" if not len(minority_positions) else "no_within_cluster_smote_pool_with_two_rows"
        diagnostics["shortfalls"].append({"stage": "additions", "cause": cause, "count": deficit})
    rng = np.random.default_rng(seed)
    synthetic_rows = []
    synthetic_lineage = []
    allocations = []
    for pool in pools:
        points = pool["points"]
        effective_k = min(neighbors, max(0, len(points) - 1))
        requested = pool["allocation"]
        neighbors_cache: dict[int, NDArray] = {}
        for _ in range(requested):
            anchor_index = int(rng.integers(len(points)))
            if anchor_index not in neighbors_cache:
                distances = cdist(points[anchor_index:anchor_index + 1], points)[0]
                if not np.isfinite(distances).all():
                    raise ValueError("SMOTE neighbor distances overflowed; rescale fitting features.")
                candidates = np.arange(len(points))
                candidates = candidates[candidates != anchor_index]
                neighbors_cache[anchor_index] = candidates[np.argsort(distances[candidates], kind="stable")[:effective_k]]
            neighbor_index = int(neighbors_cache[anchor_index][int(rng.integers(effective_k))])
            coefficient = float(rng.random())
            row = _interpolate(points[anchor_index], points[neighbor_index], coefficient)
            synthetic_rows.append(row)
            synthetic_lineage.append({"kind": "smote", "cluster_id": pool["cluster_id"],
                                      "assigned_label": partition.minority, "anchor": dict(pool["sources"][anchor_index]),
                                      "neighbor": dict(pool["sources"][neighbor_index]), "coefficient": coefficient})
        allocations.append({"cluster_id": pool["cluster_id"], "pool_size": len(points),
                            "pool_sources": pool["sources"], "requested": requested, "generated": requested,
                            "effective_k": effective_k, "k_clamped": len(points) >= 2 and effective_k < neighbors,
                            "eligible": len(points) >= 2,
                            "shortage_reason": None if len(points) >= 2 else "fewer_than_two_pool_rows"})
    synthetic = np.stack(synthetic_rows) if synthetic_rows else np.empty((0, X_l.shape[1]), dtype=float)
    retained_indices = np.flatnonzero(keep)
    X_out = np.concatenate((X_l[retained_indices], X_u[selected], synthetic), axis=0)
    y_out = np.concatenate((y[retained_indices], np.full(len(selected) + len(synthetic), partition.minority, dtype=y.dtype)))
    lineage = [{"kind": "labeled", "original_index": int(index),
                "cluster_id": int(partition.labeled_ids[index]), "assigned_label": _scalar(y[index])}
               for index in retained_indices]
    lineage.extend({"kind": "pseudo_labeled", "original_index": record["unlabeled_index"],
                    "cluster_id": record["cluster_id"], "assigned_label": partition.minority,
                    "confidence": record["confidence"]} for record in selected_records)
    lineage.extend(synthetic_lineage)
    removed_majority = int(np.count_nonzero(~keep & (y == partition.majority)))
    removed_minority = int(np.count_nonzero(~keep & (y == partition.minority)))
    weighted_unmet = requested_weighted - len(weighted_removed)
    additions_unmet = requested_additions - len(selected) - len(synthetic)
    final_counts = _class_counts(y_out, partition)
    remaining_signed_gap = final_counts["majority"] - final_counts["minority"]
    diagnostics["counts"].update({"retained_labeled": len(retained_indices), "removed_majority": removed_majority,
                                  "removed_minority": removed_minority, "selected_pseudo": len(selected),
                                  "generated_smote": len(synthetic), "output_rows": len(y_out), "final": final_counts})
    diagnostics["quotas"] = {"initial_gap_after_preclean": gap, "weighted_removal_requested": requested_weighted,
                              "weighted_removal_actual": len(weighted_removed), "weighted_removal_unmet": weighted_unmet,
                              "additions_requested": requested_additions, "pseudo_selected": len(selected),
                              "smote_generated": len(synthetic), "additions_unmet": additions_unmet,
                              "eligible_pseudo_candidates": len(pseudo_candidates), "smote_fallback_requested": deficit,
                              "remaining_signed_class_gap": remaining_signed_gap}
    diagnostics["smote_allocations"] = allocations
    diagnostics["coordinate_duplicates"] = _coordinate_duplicates(X_l, X_u, retained_indices, selected, synthetic)
    diagnostics["accounting"] = {
        "output_rows_conserved": len(y_out) == len(y) - removed_majority - removed_minority + len(selected) + len(synthetic),
        "addition_quota_conserved": requested_additions == len(selected) + len(synthetic) + additions_unmet,
        "weighted_quota_conserved": requested_weighted == len(weighted_removed) + weighted_unmet,
        "remaining_gap_matches_unmet_quotas": gap == 0 or remaining_signed_gap == weighted_unmet + additions_unmet,
    }
    diagnostics["status"] = ("shortfall" if weighted_unmet or additions_unmet else
                              "designated_majority_not_larger_cleaning_only" if gap == 0 else "quotas_met")
    diagnostics["balanced"] = final_counts["majority"] == final_counts["minority"]
    json.dumps(diagnostics, allow_nan=False)
    return SSHRResamplingResult(X_out, y_out, retained_indices.copy(), selected.copy(), synthetic.copy(), lineage, diagnostics)


__all__ = ["SSHRResamplingResult", "SSHRResamplingError", "resample_from_clusters",
           "majority_removal_weight", "minority_confidence"]
