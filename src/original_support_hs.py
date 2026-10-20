"""Count-only HS adaptation for single oversampled decision trees.

Independent implementation of Agarwal et al. (ICML 2022), Eq. (1).
Original support is a proposed substitution, not an author-verified method.
See docs/research/original_support_hs/SOURCE_AUDIT.md.
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from src._validation import finite_real, numeric_matrix


def original_node_counts(estimator: DecisionTreeClassifier, X_original) -> np.ndarray:
    """Count actual original fitting rows visiting every node, without ancestry."""
    if not isinstance(estimator, DecisionTreeClassifier):
        raise ValueError("Only single DecisionTreeClassifier estimators are supported.")
    check_is_fitted(estimator)
    X = numeric_matrix(X_original)
    return np.asarray(estimator.decision_path(X).sum(axis=0)).ravel().astype(np.int64)


def shrink_node_probabilities(children_left, children_right, raw_probabilities,
                              support, reg_param: float) -> np.ndarray:
    """Apply parent-count Eq. (1) without mutating input arrays.

    Zero support: factor 0 for positive lambda, factor 1 for lambda=0.
    Nonincreasing support on paths ensures convex ancestor probabilities.
    """
    lam = finite_real(reg_param, "reg_param")
    if lam < 0:
        raise ValueError("reg_param must be nonnegative.")
    left, right = np.asarray(children_left), np.asarray(children_right)
    raw = np.asarray(raw_probabilities, dtype=float)
    counts = np.asarray(support, dtype=float)
    n = len(left)
    if (left.ndim != 1 or right.shape != (n,) or raw.ndim != 2 or len(raw) != n
            or counts.shape != (n,) or n == 0 or raw.shape[1] < 2
            or not np.isfinite(raw).all() or not np.isfinite(counts).all()
            or (counts < 0).any() or (raw < 0).any() or (raw > 1).any()
            or not np.allclose(raw.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Invalid tree probabilities, support or array shapes.")
    output = np.empty_like(raw)
    output[0] = raw[0]
    stack, visited = [0], {0}
    while stack:
        parent = stack.pop()
        if left[parent] == right[parent] == -1:
            continue
        factor = 1.0 if lam == 0 else counts[parent] / (counts[parent] + lam)
        for child in (left[parent], right[parent]):
            if (not isinstance(child, (int, np.integer)) or child < 0 or child >= n
                    or child in visited or counts[child] > counts[parent] + 1e-12):
                raise ValueError("Invalid topology or support increasing down a path.")
            visited.add(int(child))
            output[child] = output[parent] + factor * (raw[child] - raw[parent])
            stack.append(int(child))
    if len(visited) != n:
        raise ValueError("Tree contains unreachable nodes.")
    if (output < -1e-12).any() or (output > 1 + 1e-12).any():
        raise ValueError("Shrinkage produced invalid probabilities.")
    output = np.clip(output, 0.0, 1.0)
    return output / output.sum(axis=1, keepdims=True)


class SupportAwareHS:
    """Post-fit prediction view sharing one immutable single-tree structure.

    ``fit`` here does not grow/refit the estimator. Standard mode uses its
    weighted augmented counts. Original mode routes only X_original. Both use
    the same snapshot of augmented, pre-shrinkage node predictions.
    """

    def __init__(self, estimator: DecisionTreeClassifier, *, reg_param: float = 0,
                 support_mode: str = "augmented"):
        self.estimator = estimator
        self.reg_param = reg_param
        self.support_mode = support_mode

    def fit(self, X_original=None):
        if not isinstance(self.estimator, DecisionTreeClassifier):
            raise ValueError("Only single DecisionTreeClassifier estimators are supported.")
        check_is_fitted(self.estimator)
        if self.estimator.n_outputs_ != 1 or self.estimator.n_classes_ != 2:
            raise ValueError("The pilot supports single-output binary classification only.")
        tree = self.estimator.tree_
        values = tree.value[:, 0, :].copy()
        raw = values / values.sum(axis=1, keepdims=True)
        if self.support_mode == "original":
            if X_original is None:
                raise ValueError("X_original is required for original support.")
            counts = original_node_counts(self.estimator, X_original)
        elif self.support_mode == "augmented":
            counts = tree.weighted_n_node_samples.copy()
        else:
            raise ValueError("support_mode must be augmented or original.")
        self.raw_probabilities_ = raw
        self.support_counts_ = counts
        self.node_probabilities_ = shrink_node_probabilities(
            tree.children_left, tree.children_right, raw, counts, self.reg_param)
        self.classes_ = self.estimator.classes_.copy()
        self.n_features_in_ = self.estimator.n_features_in_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "node_probabilities_")
        return self.node_probabilities_[self.estimator.apply(numeric_matrix(X))]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
