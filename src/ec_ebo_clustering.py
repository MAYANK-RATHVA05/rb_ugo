"""EC-EBO's externally constrained minority clustering on fitting data.

Equations 1–5 of Tao et al. define the merge constraint and centroid update.
The paper leaves pair-selection order open; the smallest eligible centroid
distance, then original row order, is the declared choice in
docs/EC_EBO_METHOD_DECISIONS.md. This module does not oversample.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from src._validation import binary_labels, numeric_matrix


@dataclass(frozen=True)
class _Cluster:
    members: tuple[int, ...]
    centroid: np.ndarray
    nearest_majority_distance: float


class ExternallyConstrainedClustering(BaseEstimator):
    """Fit pure minority clusters using the paper's strict external constraint.

    `clusters_` contains input-row indices, sorted within and across clusters.
    `cluster_labels_` marks majority rows as -1. This is a fitting-data-only
    component; callers must supply only their training partition.
    """

    def __init__(self, *, minority_label: Any = None):
        self.minority_label = minority_label

    def fit(self, X: Any, y: Any) -> ExternallyConstrainedClustering:
        # A failed refit must not expose geometry from a previous input.
        for name in ("clusters_", "centroids_", "nearest_majority_distances_",
                     "cluster_labels_", "merge_history_", "minority_indices_",
                     "majority_indices_", "minority_label_", "majority_label_",
                     "n_features_in_"):
            if hasattr(self, name):
                delattr(self, name)

        points = numeric_matrix(X)
        labels, classes = binary_labels(y, len(points))
        counts = np.array([(labels == label).sum() for label in classes])
        if counts[0] == counts[1]:
            raise ValueError("A tied class count has no identified minority class.")
        minority = classes[np.argmin(counts)]
        majority = classes[np.argmax(counts)]
        if self.minority_label is not None and self.minority_label != minority:
            raise ValueError("minority_label must identify the smaller class.")

        minority_indices = np.flatnonzero(labels == minority)
        majority_indices = np.flatnonzero(labels == majority)
        neighbor_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nearest_distances = neighbor_model.fit(points[majority_indices]).kneighbors(
            points[minority_indices], return_distance=True)[0][:, 0]
        if not np.isfinite(nearest_distances).all():
            raise ValueError("Majority distances overflowed; scale fitting data first.")

        active = {
            int(index): _Cluster((int(index),), points[index].copy(), float(distance))
            for index, distance in zip(minority_indices, nearest_distances)
        }
        eligible: list[tuple[float, int, int, int, int]] = []

        def offer(left_id: int, right_id: int) -> None:
            left, right = active[left_id], active[right_id]
            distance = float(np.linalg.norm(left.centroid - right.centroid))
            if not np.isfinite(distance):
                raise ValueError("Centroid distances overflowed; scale fitting data first.")
            if distance < min(left.nearest_majority_distance,
                              right.nearest_majority_distance):
                first, second = sorted((left.members[0], right.members[0]))
                heappush(eligible, (distance, first, second, left_id, right_id))

        initial_ids = list(active)
        for offset, left_id in enumerate(initial_ids):
            for right_id in initial_ids[offset + 1:]:
                offer(left_id, right_id)

        history: list[dict[str, Any]] = []
        next_id = len(points)
        while eligible:
            distance, _, _, left_id, right_id = heappop(eligible)
            if left_id not in active or right_id not in active:
                continue
            left, right = active.pop(left_id), active.pop(right_id)
            members = tuple(sorted((*left.members, *right.members)))
            centroid = ((len(left.members) * left.centroid + len(right.members) * right.centroid)
                        / len(members))
            nearest_majority = min(left.nearest_majority_distance,
                                   right.nearest_majority_distance)
            history.append({"left_indices": left.members, "right_indices": right.members,
                            "centroid_distance": distance,
                            "left_majority_distance": left.nearest_majority_distance,
                            "right_majority_distance": right.nearest_majority_distance,
                            "merged_indices": members})
            active[next_id] = _Cluster(members, centroid, nearest_majority)
            for other_id in list(active):
                if other_id != next_id:
                    offer(other_id, next_id)
            next_id += 1

        ordered = sorted(active.values(), key=lambda cluster: cluster.members[0])
        self.clusters_ = tuple(np.asarray(cluster.members, dtype=int) for cluster in ordered)
        self.centroids_ = np.vstack([cluster.centroid for cluster in ordered])
        self.nearest_majority_distances_ = np.asarray(
            [cluster.nearest_majority_distance for cluster in ordered], dtype=float)
        self.cluster_labels_ = np.full(len(points), -1, dtype=int)
        for cluster_id, members in enumerate(self.clusters_):
            self.cluster_labels_[members] = cluster_id
        self.merge_history_ = tuple(history)
        self.minority_indices_ = minority_indices.copy()
        self.majority_indices_ = majority_indices.copy()
        self.minority_label_ = minority.item() if hasattr(minority, "item") else minority
        self.majority_label_ = majority.item() if hasattr(majority, "item") else majority
        self.n_features_in_ = points.shape[1]
        return self
