"""Gaussian-kernel naive Bayes with training-only unbiased-CV bandwidths."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src._validation import binary_labels, numeric_matrix, positive_integer


def _ucv_objective(log_bandwidth: float, centers: np.ndarray,
                   counts: np.ndarray, n: int) -> float:
    bandwidth = float(np.exp(log_bandwidth))
    differences = centers[:, None] - centers[None, :]
    pairs = counts[:, None] * counts[None, :]
    first = np.sum(pairs * np.exp(-differences**2 / (4 * bandwidth**2)))
    off_diagonal = pairs.copy()
    off_diagonal.flat[::len(centers) + 1] -= counts
    second = np.sum(off_diagonal * np.exp(-differences**2 / (2 * bandwidth**2)))
    return (first / (n*n*bandwidth*np.sqrt(4*np.pi))
            - 2*second / (n*(n-1)*bandwidth*np.sqrt(2*np.pi)))


class GaussianKernelNB(ClassifierMixin, BaseEstimator):
    """Product Gaussian KDEs; bandwidth minimizes binned UCV per class/feature."""

    def __init__(self, n_bins: int = 128, bandwidth_lower: float = 0.05,
                 bandwidth_upper: float = 2.0, prediction_chunk_size: int = 256):
        self.n_bins = n_bins
        self.bandwidth_lower = bandwidth_lower
        self.bandwidth_upper = bandwidth_upper
        self.prediction_chunk_size = prediction_chunk_size

    def fit(self, X: Any, y: Any):
        for key in ("classes_", "models_", "bandwidths_", "n_features_in_"):
            self.__dict__.pop(key, None)
        Xv = numeric_matrix(X); yv, classes = binary_labels(y, len(Xv))
        bins = positive_integer(self.n_bins, "n_bins")
        chunk = positive_integer(self.prediction_chunk_size, "prediction_chunk_size")
        lower, upper = float(self.bandwidth_lower), float(self.bandwidth_upper)
        if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
            raise ValueError("Bandwidth multipliers must satisfy 0 < lower < upper.")
        models, bandwidths, boundary = [], np.empty((2, Xv.shape[1])), []
        priors = []
        for class_index, label in enumerate(classes):
            values = Xv[yv == label]; priors.append(len(values) / len(Xv)); class_models = []
            if len(values) < 2:
                raise ValueError("GKNB unbiased-CV bandwidth requires two rows in every class.")
            for feature in range(Xv.shape[1]):
                column = values[:, feature]
                scale = min(np.std(column, ddof=1),
                            (np.subtract(*np.percentile(column, [75, 25])) / 1.349))
                if not np.isfinite(scale) or scale <= 0:
                    scale = np.std(column, ddof=1)
                floor = max(np.finfo(float).eps * max(1.0, np.max(np.abs(column))), 1e-12)
                scale = max(float(scale), floor)
                if len(column) <= bins:
                    centers, counts = np.unique(column, return_counts=True)
                else:
                    counts, edges = np.histogram(column, bins=bins)
                    centers = (edges[:-1] + edges[1:]) / 2
                    present = counts > 0; centers, counts = centers[present], counts[present]
                lo, hi = np.log(lower * scale), np.log(upper * scale)
                result = minimize_scalar(_ucv_objective, bounds=(lo, hi), method="bounded",
                                         args=(centers, counts.astype(float), len(column)),
                                         options={"xatol": 1e-5})
                bandwidth = float(np.exp(result.x))
                bandwidths[class_index, feature] = bandwidth
                boundary.append(bool(abs(result.x-lo) < 1e-4 or abs(result.x-hi) < 1e-4))
                class_models.append((centers.astype(float), counts.astype(float)))
            models.append(class_models)
        self.classes_, self.n_features_in_ = classes, Xv.shape[1]
        self.models_, self.bandwidths_, self.class_prior_ = models, bandwidths, np.asarray(priors)
        self.diagnostics_ = {"kernel": "Gaussian", "bandwidth": "binned unbiased CV",
                             "n_bins": bins, "prediction_chunk_size": chunk,
                             "boundary_solutions": int(sum(boundary))}
        return self

    def predict_log_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "models_")
        Xv = numeric_matrix(X)
        if Xv.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature width differs from fitting data.")
        output = np.empty((len(Xv), 2))
        chunk = positive_integer(self.prediction_chunk_size, "prediction_chunk_size")
        for start in range(0, len(Xv), chunk):
            rows = Xv[start:start+chunk]
            scores = np.tile(np.log(self.class_prior_), (len(rows), 1))
            for class_index in range(2):
                for feature, (centers, counts) in enumerate(self.models_[class_index]):
                    bandwidth = self.bandwidths_[class_index, feature]
                    standardized = ((rows[:, feature, None]-centers[None, :])/bandwidth)
                    log_kernel = -0.5*np.square(np.clip(standardized, -1e150, 1e150))
                    log_kernel += np.log(counts)[None, :] - np.log(counts.sum())
                    log_kernel -= np.log(bandwidth*np.sqrt(2*np.pi))
                    scores[:, class_index] += logsumexp(log_kernel, axis=1)
            output[start:start+len(rows)] = scores - logsumexp(scores, axis=1)[:, None]
        return output

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
