"""LEAO comparator from Tao (2026), with explicit paper ambiguity choices."""
from __future__ import annotations

from numbers import Integral
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer


def positive_binary_entropy(probabilities: NDArray, epsilon: float = 1e-12) -> NDArray:
    """Positive entropy used by the paper's text and sampling weights.

    The printed Equation 1 omits the leading minus sign. Clipping prevents
    negative/infinite log arguments at probability zero or one.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1 - epsilon)
    if not np.isfinite(p).all():
        raise ValueError("LEAO class probabilities must be finite.")
    return -p * np.log(p) - (1 - p) * np.log1p(-p)


def entropy_allocation(entropies: NDArray, count: int) -> NDArray[np.int64]:
    """Allocate exactly count synthetic rows by largest remainder."""
    if not np.isfinite(entropies).all() or (entropies < 0).any() or not len(entropies):
        raise ValueError("LEAO entropy weights must be finite and nonnegative.")
    weights = entropies / entropies.sum() if entropies.sum() > 0 else np.full(len(entropies), 1 / len(entropies))
    quotas = weights * count
    allocations = np.floor(quotas).astype(np.int64)
    missing = count - int(allocations.sum())
    if missing:
        order = np.argsort(-(quotas - allocations), kind="stable")
        allocations[order[:missing]] += 1
    return allocations


def adversarial_potential_and_gradient(
    x: NDArray, minority: NDArray, majority: NDArray, h_min: float,
    h_maj: float, inverse_covariance: NDArray, alpha: float,
) -> tuple[float, NDArray]:
    """Paper Equations 2–3; define the gradient as zero at distance zero."""
    potential = 0.0
    gradient = np.zeros_like(x)
    for reference, entropy, sign in ((minority, h_min, 1.0), (majority, h_maj, -1.0)):
        difference = x - reference
        squared = float(difference @ inverse_covariance @ difference)
        distance = np.sqrt(max(0.0, squared))
        component = float(entropy * np.exp(-alpha * distance))
        potential += sign * component
        if distance > 0:
            gradient += -sign * alpha * component * (inverse_covariance @ difference) / distance
    return potential, gradient


class LEAOOversampler(BaseEstimator):
    """Numerical LEAO full-balance comparator with a fitting-only bagging model."""

    def __init__(self, iterations: int = 50, alpha: float = 1.0, k: int = 3,
                 stop_probability: float = 0.3, step: float = 1.0,
                 random_state: int = 42, allow_provisional: bool = False) -> None:
        self.iterations = iterations
        self.alpha = alpha
        self.k = k
        self.stop_probability = stop_probability
        self.step = step
        self.random_state = random_state
        self.allow_provisional = allow_provisional

    def fit_resample(self, X: Any, y: Any) -> tuple[NDArray, NDArray]:
        for attr in ["diagnostics_", "classes_", "n_features_in_", "probability_model_"]:
            if hasattr(self, attr):
                delattr(self, attr)
        if not isinstance(self.allow_provisional, bool) or not self.allow_provisional:
            raise ValueError("LEAO source ambiguities remain; set allow_provisional=True.")
        for name in ("iterations", "k"):
            positive_integer(getattr(self, name), name)
        if not 0 < finite_real(self.alpha, "alpha"):
            raise ValueError("alpha must be positive.")
        if not 0 <= finite_real(self.stop_probability, "stop_probability") < 1:
            raise ValueError("stop_probability must lie in [0, 1).")
        if not 0 < finite_real(self.step, "step"):
            raise ValueError("step must be positive.")
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2 ** 32):
            raise ValueError("random_state must be an integer in [0, 2**32).")
        points = numeric_matrix(X)
        labels, classes = binary_labels(y, len(points))
        counts = np.array([(labels == label).sum() for label in classes])
        count = int(counts.max() - counts.min())
        warnings.warn("Provisional LEAO: source equation/default ambiguities are documented.",
                      UserWarning, stacklevel=2)
        self.classes_ = classes.copy()
        self.n_features_in_ = points.shape[1]
        self.diagnostics_ = {"source_id": "LEAO-PDF",
                             "reproduction_status": "original_source_with_declared_conventions",
                             "number_of_synthetic_samples": count,
                             "random_state": int(self.random_state),
                             "stopping_reason": "balanced_counts" if count == 0 else "minority_balanced",
                             "conventions": ["positive_entropy_sign", "largest_remainder_allocation",
                                             "bagging_10_default_trees", "pseudoinverse_covariance",
                                             "mahalanobis_majority_neighbors", "uniform_step_in_0_to_step",
                                             "zero_gradient_at_reference", "final_iterate_if_no_boundary"]}
        if count == 0:
            return points, labels
        rng = np.random.default_rng(self.random_state)
        minority_label = classes[int(np.argmin(counts))]
        minority_indices = np.flatnonzero(labels == minority_label)
        majority_indices = np.flatnonzero(labels != minority_label)
        model = BaggingClassifier(n_estimators=10, random_state=self.random_state)
        model.fit(points.copy(), labels.copy())
        self.probability_model_ = model
        minority_column = np.flatnonzero(model.classes_ == minority_label)[0]
        entropies = positive_binary_entropy(model.predict_proba(points.copy())[:, minority_column])
        allocations = entropy_allocation(entropies[minority_indices], count)
        covariance = np.atleast_2d(np.cov(points, rowvar=False, ddof=0))
        inverse = np.linalg.pinv(covariance)
        inverse = (inverse + inverse.T) / 2
        if not np.isfinite(inverse).all():
            raise ValueError("LEAO covariance inverse is nonfinite.")
        differences = points[minority_indices, None, :] - points[majority_indices][None, :, :]
        distances = np.sqrt(np.maximum(0.0, np.einsum("ijk,kl,ijl->ij", differences, inverse, differences)))
        neighbors = np.argsort(distances, axis=1, kind="stable")[:, :min(self.k, len(majority_indices))]
        generated = []
        selected_maj = []
        steps_taken = []
        for local, amount in enumerate(allocations):
            seed_index = minority_indices[local]
            for _ in range(int(amount)):
                maj_index = majority_indices[rng.choice(neighbors[local])]
                eta = rng.uniform(0, self.step)
                x = points[seed_index].copy()
                steps = 0
                for _ in range(self.iterations):
                    if rng.random() < self.stop_probability:
                        break
                    _, gradient = adversarial_potential_and_gradient(
                        x, points[seed_index], points[maj_index],
                        entropies[seed_index], entropies[maj_index], inverse, self.alpha)
                    x = x - eta * gradient
                    if not np.isfinite(x).all():
                        raise ValueError("LEAO gradient update produced nonfinite values.")
                    steps += 1
                    potential, _ = adversarial_potential_and_gradient(
                        x, points[seed_index], points[maj_index],
                        entropies[seed_index], entropies[maj_index], inverse, self.alpha)
                    if potential <= 0:
                        break
                generated.append(x)
                selected_maj.append(int(maj_index))
                steps_taken.append(steps)
        self.diagnostics_["allocation"] = allocations.tolist()
        self.diagnostics_["minority_reference_indices"] = minority_indices.tolist()
        self.diagnostics_["selected_majority_indices"] = selected_maj
        self.diagnostics_["gradient_steps"] = steps_taken
        return np.vstack((points, np.array(generated))), np.concatenate(
            (labels, np.full(count, minority_label, dtype=labels.dtype)))
