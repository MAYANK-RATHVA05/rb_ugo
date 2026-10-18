"""Opt-in generator-only PPEO geometry diagnostic, not a full PPEO reproduction.

All inputs are *original fitting observations* in a preprocessing space fitted on
those observations.  Prototype assignment, small-prototype selection and anchor
quotas are deliberately external: this module changes generation only.  No true
synthetic-data subgroup identifiers, validation rows or test rows enter this API.

The clipping region is an anchor-specific intersection of majority bisector
slabs.  It certifies distances to observed majority points, not true labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class ShapeEstimate:
    covariance: np.ndarray | None
    factor: np.ndarray | None
    fallback: str | None
    shrinkage: float | None


@dataclass(frozen=True)
class GenerationResult:
    samples: np.ndarray
    anchor_indices: np.ndarray
    diagnostics: dict[str, Any]


METHODS = frozenset({
    "spsdo", "tuned_sphere", "ellipsoid", "screened_ellipsoid",
    "symmetric_clip", "paired_rejection", "ros",
})
_ELLIPSOID_METHODS = METHODS - {"spsdo", "tuned_sphere", "ros"}
_PAIRED_METHODS = {"symmetric_clip", "paired_rejection"}


def _points(values: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric two-dimensional array") from exc
    if result.ndim != 2 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric two-dimensional array")
    return result


def _eta(value: float) -> float:
    if (not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not 0 < value < 1):
        raise ValueError("eta must satisfy 0 < eta < 1")
    return float(value)


def estimate_shape(points: np.ndarray) -> ShapeEstimate:
    """Ledoit--Wolf shape, trace-normalized to dimension, or declared duplication.

    This conservative pilot requires at least max(3, d+1) original observations
    and full empirical rank.  Shrinkage mathematically permits lower rank input;
    the stricter rule is a declared diagnostic fallback, not a property of LW.
    The symmetric matrix square root avoids a coordinate-order convention.
    """
    points = _points(points, "points")
    n, d = points.shape
    if n < max(3, d + 1):
        return ShapeEstimate(None, None, "too_few_observations", None)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = points - points.mean(axis=0)
    if not np.isfinite(centered).all():
        return ShapeEstimate(None, None, "nonfinite_geometry", None)
    if not np.any(centered):
        return ShapeEstimate(None, None, "zero_trace", None)
    try:
        rank = np.linalg.matrix_rank(centered)
    except np.linalg.LinAlgError:
        return ShapeEstimate(None, None, "nonfinite_geometry", None)
    if rank < d:
        return ShapeEstimate(None, None, "deficient_empirical_rank", None)
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fitted = LedoitWolf().fit(points)
            covariance = fitted.covariance_
            trace = float(np.trace(covariance))
    except (ValueError, np.linalg.LinAlgError):
        return ShapeEstimate(None, None, "nonfinite_geometry", None)
    if not np.isfinite(covariance).all() or not np.isfinite(trace):
        return ShapeEstimate(None, None, "nonfinite_geometry", None)
    if trace <= 0:
        return ShapeEstimate(None, None, "zero_trace", None)
    covariance = (covariance / trace) * d
    covariance = (covariance + covariance.T) / 2
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return ShapeEstimate(None, None, "nonfinite_geometry", float(fitted.shrinkage_))
    if np.min(eigenvalues) <= 0 or not np.isfinite(eigenvalues).all():
        return ShapeEstimate(None, None, "nonpositive_shape", float(fitted.shrinkage_))
    factor = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    return ShapeEstimate(covariance, factor, None, float(fitted.shrinkage_))


def _geometry_inputs(
    displacement: np.ndarray, anchor: np.ndarray, X_majority: np.ndarray, eta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    eta = _eta(eta)
    majority = _points(X_majority, "X_majority")
    displacement = np.asarray(displacement, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    d = majority.shape[1]
    if displacement.shape != (d,) or anchor.shape != (d,):
        raise ValueError("anchor and displacement must match the feature dimension")
    if not np.isfinite(displacement).all() or not np.isfinite(anchor).all():
        raise ValueError("anchor and displacement must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        differences = majority - anchor
    if not np.isfinite(differences).all():
        raise ValueError("majority-minus-anchor differences must be finite")
    return displacement, differences, eta


def _slack(displacement: np.ndarray, differences: np.ndarray, eta: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        squared = np.einsum("ij,ij->i", differences, differences)
        dot = differences @ displacement
        result = eta * squared - 2 * np.abs(dot)
    if not np.isfinite(result).all():
        raise ValueError("geometry exceeds finite floating-point range")
    return result


def constraint_slack(
    displacement: np.ndarray, anchor: np.ndarray, X_majority: np.ndarray,
    eta: float = 0.9,
) -> np.ndarray:
    """Return eta||y-x||² - 2|v·(y-x)|, one value per original majority row.

    Nonnegative values certify both signs of a displacement.  A coincident
    majority anchor has zero slack for every displacement and does *not*
    establish strict nearest-anchor separation.
    """
    displacement, differences, eta = _geometry_inputs(
        displacement, anchor, X_majority, eta,
    )
    return _slack(displacement, differences, eta)


def _clip(
    displacement: np.ndarray, differences: np.ndarray, eta: float,
) -> tuple[np.ndarray, float]:
    with np.errstate(over="ignore", invalid="ignore"):
        squared = np.einsum("ij,ij->i", differences, differences)
        denominators = 2 * np.abs(differences @ displacement)
    if not np.isfinite(squared).all() or not np.isfinite(denominators).all():
        raise ValueError("geometry exceeds finite floating-point range")
    nonzero = denominators != 0  # The specification ignores exactly zero dots.
    if np.any(nonzero):
        with np.errstate(over="ignore", divide="ignore"):
            scale = min(1.0, float(np.min(eta * squared[nonzero] / denominators[nonzero])))
    else:
        scale = 1.0
    return displacement * scale, scale


def clip_displacement(
    displacement: np.ndarray, anchor: np.ndarray, X_majority: np.ndarray,
    eta: float = 0.9,
) -> tuple[np.ndarray, float]:
    """Apply the specified shared radial clipping to a pair's displacement.

    The caller must handle conflicting labels at exactly equal coordinates;
    ``generate`` flags such anchors and emits explicitly uncertified copies.
    """
    displacement, differences, eta = _geometry_inputs(
        displacement, anchor, X_majority, eta,
    )
    return _clip(displacement, differences, eta)


def _integers(values: Any, name: str, length: int, *, nonnegative: bool) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (length,) or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array of length {length}")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def generate(
    X_minority: np.ndarray,
    X_majority: np.ndarray,
    prototype_ids: np.ndarray,
    quotas: np.ndarray,
    *,
    method: str,
    seed: int = 42,
    eta: float = 0.9,
    sphere_scale: float = 1.0,
    max_attempts: int = 64,
) -> GenerationResult:
    """Generate exactly the externally fixed per-anchor quotas.

    ``spsdo`` implements the paper-specified generator only: a uniform unit
    direction and a radius uniform on [0, nearest original majority distance].
    The ellipsoid controls use the same radius with an LW prototype factor.
    Screening uses only nearest *original* minority/majority observations.
    Screening retries per row; paired rejection retries per pair.  Exhaustion,
    shape degeneracy and conflicting labels all cause recorded duplication.

    ``attempts`` counts proposed vectors (proposed pairs for paired methods),
    and ``clip_count`` counts clipped pairs.  ``fallback_counts`` counts copied
    rows by reason; it excludes intentional ROS and odd paired-quota copies.
    """
    minority = _points(X_minority, "X_minority")
    majority = _points(X_majority, "X_majority")
    if minority.shape[1] != majority.shape[1] or not len(majority):
        raise ValueError("minority and nonempty majority must have equal feature dimension")
    prototype_ids = _integers(prototype_ids, "prototype_ids", len(minority), nonnegative=False)
    quotas = _integers(quotas, "quotas", len(minority), nonnegative=True)
    eta = _eta(eta)
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"unknown generation method: {method!r}")
    if (not isinstance(sphere_scale, (int, float, np.integer, np.floating))
            or not np.isfinite(sphere_scale)
            or not 0 < sphere_scale <= 1):
        raise ValueError("sphere_scale must satisfy 0 < sphere_scale <= 1")
    if (isinstance(max_attempts, (bool, np.bool_))
            or not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1):
        raise ValueError("max_attempts must be a positive integer")
    rng = np.random.default_rng(seed)
    d = minority.shape[1]
    shapes: dict[int, ShapeEstimate] = {}
    diagnostics: dict[str, Any] = {
        "method": method, "eta": eta, "sphere_scale": float(sphere_scale),
        "max_attempts": int(max_attempts), "generated_count": 0,
        "fallback_generated_rows": 0, "fallback_counts": {},
        "duplication_rows": 0, "odd_copy_rows": 0, "attempts": 0,
        "clip_count": 0, "clip_scales": [], "prototype_shapes": [],
        "anchor_details": [],
    }
    if method in _ELLIPSOID_METHODS:
        for raw_id in np.unique(prototype_ids[quotas > 0]):
            prototype_id = int(raw_id)
            points = minority[prototype_ids == raw_id]
            shape = estimate_shape(points)
            shapes[prototype_id] = shape
            diagnostics["prototype_shapes"].append({
                "prototype_id": prototype_id, "n_observations": len(points),
                "fallback": shape.fallback, "shrinkage": shape.shrinkage,
            })

    rows: list[np.ndarray] = []
    anchors: list[int] = []

    def append_copy(anchor: np.ndarray, index: int, count: int, reason: str | None,
                    details: dict[str, Any]) -> None:
        for _ in range(count):
            rows.append(anchor.copy())
            anchors.append(index)
        diagnostics["duplication_rows"] += count
        if reason is not None:
            diagnostics["fallback_generated_rows"] += count
            previous = diagnostics["fallback_counts"].get(reason, 0)
            diagnostics["fallback_counts"][reason] = previous + count
            details["fallback_rows"] += count
            details["fallback_reasons"][reason] = details["fallback_reasons"].get(reason, 0) + count

    for index, (anchor, raw_id, raw_quota) in enumerate(zip(minority, prototype_ids, quotas)):
        quota = int(raw_quota)
        if not quota:
            continue
        prototype_id = int(raw_id)
        shape = shapes.get(prototype_id)
        details = {
            "anchor_index": index, "prototype_id": prototype_id, "quota": quota,
            "shape_fallback": None if shape is None else shape.fallback,
            "label_conflict": bool(np.any(np.all(majority == anchor, axis=1))),
            "fallback_rows": 0, "fallback_reasons": {}, "attempts": 0,
            "clipped_pairs": 0, "odd_copy_rows": 0,
        }
        diagnostics["anchor_details"].append(details)
        if details["label_conflict"]:
            append_copy(anchor, index, quota, "conflicting_original_labels", details)
            continue
        if method == "ros":
            append_copy(anchor, index, quota, None, details)
            continue
        if shape is not None and shape.fallback is not None:
            append_copy(anchor, index, quota, shape.fallback, details)
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            differences = majority - anchor
            radius = float(np.min(np.linalg.norm(differences, axis=1)))
        if not np.isfinite(radius) or radius <= 0:
            append_copy(anchor, index, quota, "nonfinite_geometry", details)
            continue

        def propose() -> np.ndarray:
            # A zero Gaussian vector has probability zero, but handle it explicitly.
            direction = rng.normal(size=d)
            norm = float(np.linalg.norm(direction))
            while norm == 0:
                direction = rng.normal(size=d)
                norm = float(np.linalg.norm(direction))
            direction /= norm
            delta = rng.uniform(0, radius)
            if method == "tuned_sphere":
                delta *= sphere_scale
            if shape is not None:
                direction = shape.factor @ direction
            diagnostics["attempts"] += 1
            details["attempts"] += 1
            return delta * direction

        if method in _PAIRED_METHODS:
            for _ in range(quota // 2):
                displacement = None
                if method == "symmetric_clip":
                    try:
                        displacement, scale = _clip(propose(), differences, eta)
                    except ValueError:
                        append_copy(anchor, index, 2, "nonfinite_geometry", details)
                        continue
                    diagnostics["clip_scales"].append(scale)
                    if scale < 1:
                        diagnostics["clip_count"] += 1
                        details["clipped_pairs"] += 1
                else:
                    failure_reason = "paired_rejection_exhausted"
                    for _ in range(max_attempts):
                        candidate = propose()
                        try:
                            accepted = bool(np.all(_slack(candidate, differences, eta) >= 0))
                        except ValueError:
                            failure_reason = "nonfinite_geometry"
                            break
                        if accepted:
                            displacement = candidate
                            break
                    if displacement is None:
                        append_copy(anchor, index, 2, failure_reason, details)
                        continue
                pair = np.array([anchor + displacement, anchor - displacement])
                if not np.isfinite(pair).all():
                    append_copy(anchor, index, 2, "nonfinite_geometry", details)
                    continue
                rows.extend(pair)
                anchors.extend((index, index))
            if quota % 2:
                append_copy(anchor, index, 1, None, details)
                diagnostics["odd_copy_rows"] += 1
                details["odd_copy_rows"] += 1
        else:
            for _ in range(quota):
                sample = None
                failure_reason = ("screening_exhausted" if method == "screened_ellipsoid"
                                  else "nonfinite_geometry")
                for _ in range(max_attempts if method == "screened_ellipsoid" else 1):
                    candidate = anchor + propose()
                    if not np.isfinite(candidate).all():
                        failure_reason = "nonfinite_geometry"
                        break
                    if method == "screened_ellipsoid":
                        with np.errstate(over="ignore", invalid="ignore"):
                            nearest_minority = np.min(np.sum((minority - candidate) ** 2, axis=1))
                            nearest_majority = np.min(np.sum((majority - candidate) ** 2, axis=1))
                        if not np.isfinite(nearest_minority) or not np.isfinite(nearest_majority):
                            failure_reason = "nonfinite_geometry"
                            break
                        if nearest_minority >= nearest_majority:
                            continue
                    sample = candidate
                    break
                if sample is None:
                    append_copy(anchor, index, 1, failure_reason, details)
                else:
                    rows.append(sample)
                    anchors.append(index)
    samples = np.asarray(rows, dtype=float).reshape(-1, d)
    anchor_indices = np.asarray(anchors, dtype=int)
    expected = sum(int(quota) for quota in quotas)
    if len(samples) != expected or not np.isfinite(samples).all():
        raise RuntimeError("generation failed its exact-count/finite-output invariant")
    diagnostics["generated_count"] = expected
    return GenerationResult(samples, anchor_indices, diagnostics)
