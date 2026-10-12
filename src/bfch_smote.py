"""Boundary-fusion convex-hull minority oversampling on fitting data only.

The source paper leaves RF fitting and k unspecified, and its rank test is
insufficient for triangle membership. See docs/BFCH_METHOD.md for the exact
conventions used here. No population-level non-overlap guarantee is implied.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer


def _nearest(source: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    """Indices within reference; callers ensure both arrays are nonempty."""
    model = NearestNeighbors(n_neighbors=min(k, len(reference)), metric="euclidean")
    return model.fit(reference).kneighbors(source, return_distance=False)


def _contains_majority(vertices: np.ndarray, majority: np.ndarray,
                       *, tolerance: float = 1e-10) -> bool:
    """Whether an observed majority point is in/on a nondegenerate triangle.

    The residual test matters in more than two feature dimensions: a point
    can project inside the triangle while lying outside its affine plane.
    """
    origin = vertices[0]
    edges = (vertices[1:] - origin).T
    coefficients, _, _, _ = np.linalg.lstsq(edges, (majority - origin).T, rcond=None)
    residual = (majority - origin).T - edges @ coefficients
    scale = max(1.0, float(np.linalg.norm(edges, axis=0).max()))
    in_plane = np.linalg.norm(residual, axis=0) <= tolerance * scale
    weights = np.vstack((1.0 - coefficients.sum(axis=0), coefficients))
    in_simplex = np.all(weights >= -tolerance, axis=0)
    return bool(np.any(in_plane & in_simplex))


def _triangle_area(vertices: np.ndarray) -> float:
    """Gram determinant area, algebraically equivalent to Heron's formula."""
    u, v = vertices[1] - vertices[0], vertices[2] - vertices[0]
    return float(0.5 * np.sqrt(max(0.0, np.dot(u, u) * np.dot(v, v) - np.dot(u, v) ** 2)))


def _allocate_areas(areas: np.ndarray, count: int) -> np.ndarray:
    """Largest-remainder allocation, with stable index-order tie breaking."""
    exact = count * areas / areas.sum()
    allocated = np.floor(exact).astype(int)
    remainder = count - int(allocated.sum())
    if remainder:
        allocated[np.argsort(-(exact - allocated), kind="stable")[:remainder]] += 1
    return allocated


class BFCHSMOTE(BaseEstimator):
    """Generate minority samples in observed-majority-free fused triangles.

    A no-op with a diagnosed reason is returned when valid triangles cannot
    be built; the algorithm never silently substitutes another sampler.
    """

    def __init__(self, *, k_neighbors: int = 5, theta_min: float = 0.3,
                 theta_maj: float = 0.8, fusion: float = 0.5,
                 n_samples: int | None = None, random_state: int = 42,
                 n_estimators: int = 100):
        self.k_neighbors = k_neighbors
        self.theta_min = theta_min
        self.theta_maj = theta_maj
        self.fusion = fusion
        self.n_samples = n_samples
        self.random_state = random_state
        self.n_estimators = n_estimators

    def _validate_parameters(self) -> None:
        positive_integer(self.k_neighbors, "k_neighbors")
        positive_integer(self.n_estimators, "n_estimators")
        if self.n_samples is not None and (isinstance(self.n_samples, bool)
                                           or not isinstance(self.n_samples, Integral)
                                           or self.n_samples < 0):
            raise ValueError("n_samples must be a nonnegative integer or None.")
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2**32):
            raise ValueError("random_state must be an integer in [0, 2**32).")
        for name in ("theta_min", "theta_maj", "fusion"):
            value = finite_real(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1].")

    def fit_resample(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "diagnostics_"):
            del self.diagnostics_
        self._validate_parameters()
        points = numeric_matrix(X)
        labels, classes = binary_labels(y, len(points))
        if points.shape[1] < 2:
            raise ValueError("BFCH-SMOTE needs at least two numerical features for triangles.")
        counts = np.array([(labels == label).sum() for label in classes])
        if counts[0] == counts[1]:
            if self.n_samples not in (None, 0):
                raise ValueError("A tied class count has no identified minority class.")
            self.diagnostics_ = {"status": "already_balanced", "generated_count": 0}
            return points, labels
        minority = classes[np.argmin(counts)]
        majority = classes[np.argmax(counts)]
        target = int(counts.max() - counts.min() if self.n_samples is None else self.n_samples)
        self.diagnostics_ = {
            "status": "pending", "minority_label": minority.item() if hasattr(minority, "item") else minority,
            "majority_label": majority.item() if hasattr(majority, "item") else majority,
            "requested_count": target, "generated_count": 0,
            "rf_policy": "in-sample own-class probabilities; class_weight=balanced",
            "majority_validation": "all original majority rows; barycentric triangle membership",
        }
        if target == 0:
            self.diagnostics_["status"] = "zero_requested"
            return points, labels
        if counts.min() < 2:
            self.diagnostics_["status"] = "too_few_minority_rows"
            return points, labels

        forest = RandomForestClassifier(n_estimators=self.n_estimators,
                                        class_weight="balanced", random_state=self.random_state,
                                        n_jobs=1)
        forest.fit(points, labels)
        probabilities = forest.predict_proba(points)
        minority_indices = np.flatnonzero(labels == minority)
        majority_indices = np.flatnonzero(labels == majority)
        retained_min = minority_indices[probabilities[minority_indices,
                                                        np.flatnonzero(forest.classes_ == minority)[0]] >= self.theta_min]
        retained_maj = majority_indices[probabilities[majority_indices,
                                                        np.flatnonzero(forest.classes_ == majority)[0]] >= self.theta_maj]
        self.diagnostics_["retained_counts"] = {"minority": len(retained_min),
                                                 "majority": len(retained_maj)}
        if len(retained_min) < 2 or len(retained_maj) < 2:
            self.diagnostics_["status"] = "insufficient_filtered_anchors"
            return points, labels

        first = _nearest(points[retained_min], points[retained_maj], self.k_neighbors)
        majority_votes = np.bincount(first.ravel(), minlength=len(retained_maj))
        boundary_maj = retained_maj[majority_votes >= 2]
        if len(boundary_maj) < 2:
            self.diagnostics_["status"] = "insufficient_majority_boundary"
            return points, labels
        second = _nearest(points[boundary_maj], points[retained_min], self.k_neighbors)
        minority_votes = np.bincount(second.ravel(), minlength=len(retained_min))
        boundary_min = retained_min[minority_votes >= 2]
        self.diagnostics_["boundary_counts"] = {"minority": len(boundary_min),
                                                 "majority": len(boundary_maj)}
        if len(boundary_min) < 2:
            self.diagnostics_["status"] = "insufficient_minority_boundary"
            return points, labels

        # Paper describes a set of hulls; deduplicate reciprocal nearest pairs.
        min_neighbors = _nearest(points[boundary_min], points[boundary_min], 2)
        maj_neighbors = NearestNeighbors(n_neighbors=1).fit(points[boundary_maj])
        valid, areas, seen = [], [], set()
        rejected_degenerate = rejected_occupied = 0
        for i, neighbors in enumerate(min_neighbors):
            j = next((int(candidate) for candidate in neighbors if candidate != i), None)
            if j is None:
                continue
            pair = tuple(sorted((int(boundary_min[i]), int(boundary_min[j]))))
            if pair in seen:
                continue
            seen.add(pair)
            base = points[list(pair)]
            midpoint = base.mean(axis=0)
            majority_index = boundary_maj[maj_neighbors.kneighbors(midpoint[None, :],
                                                                     return_distance=False)[0, 0]]
            fused = (1 - self.fusion) * midpoint + self.fusion * points[majority_index]
            vertices = np.vstack((base, fused))
            area = _triangle_area(vertices)
            if area <= 1e-12 * max(1.0, float(np.linalg.norm(base[1] - base[0]) ** 2)):
                rejected_degenerate += 1
                continue
            if _contains_majority(vertices, points[majority_indices]):
                rejected_occupied += 1
                continue
            valid.append(vertices)
            areas.append(area)
        self.diagnostics_.update(candidate_hulls=len(seen), valid_hulls=len(valid),
                                 rejected_degenerate=rejected_degenerate,
                                 rejected_occupied=rejected_occupied)
        if not valid:
            self.diagnostics_["status"] = "no_valid_hulls"
            return points, labels

        allocation = _allocate_areas(np.asarray(areas), target)
        rng = np.random.default_rng(self.random_state)
        generated = np.vstack([rng.dirichlet(np.ones(3), size=int(n)) @ vertices
                               for vertices, n in zip(valid, allocation) if n])
        synthetic_labels = np.repeat(minority, target)
        self.diagnostics_.update(status="generated", generated_count=target,
                                 total_valid_area=float(np.sum(areas)),
                                 allocation=allocation.tolist())
        return np.vstack((points, generated)), np.concatenate((labels, synthetic_labels))
