"""Semi-supervised hierarchical clustering for the SSHR reproduction.

This implements Algorithm 1 of Jiang et al. (2023) using the source conflicts
and declared interpretations recorded in ``docs/SSHR_METHOD_DECISIONS.md``.
The published author implementation was not accessible. In particular, the
default objective is the literal printed Eq. 4, not conventional minority F1.
Only observed fitting labels are used; there is no hidden-label input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator

from ._validation import binary_labels, finite_real, numeric_matrix, positive_integer


def _scalar(value: Any) -> Any:
    """Return a JSON-compatible scalar for the supported scalar labels."""
    return value.item() if isinstance(value, np.generic) else value


def _feature_matrix(value: Any, *, allow_empty: bool = False) -> NDArray[np.float64]:
    # The shared validator casts DataFrames to float before checking array kind;
    # reject their complex source columns before that cast can discard values.
    if isinstance(value, pd.DataFrame) and any(getattr(dtype, "kind", None) == "c"
                                             for dtype in value.dtypes):
        raise ValueError("Features must be real numerical values; complex input is unsupported.")
    return numeric_matrix(value, allow_empty=allow_empty)


def clustering_objective(
    y_true: Any,
    y_pred: Any,
    minority_label: Any,
    mu: float = 0.5,
    objective_mode: str = "paper_printed",
) -> dict[str, Any]:
    """Evaluate Eq. 3 on observed labels and expose Eq. 4's anomaly.

    ``paper_printed`` uses ``2 TP TN / [TP(TP+FP)+TN(TP+FN)]``.
    ``standard_minority_f1`` is an explicit interpretation, not verified author
    behavior. Undefined ratios return ``defined=False`` and ``None`` values;
    callers decide whether an initial model fails or a proposal is rejected.
    """
    weight = finite_real(mu, "mu")
    if not 0 <= weight <= 1:
        raise ValueError("mu must lie in [0, 1].")
    if not isinstance(objective_mode, str) or objective_mode not in {"paper_printed", "standard_minority_f1"}:
        raise ValueError("objective_mode must be 'paper_printed' or 'standard_minority_f1'.")
    raw_true = np.asarray(y_true)
    truth, classes = binary_labels(raw_true, len(raw_true) if raw_true.ndim else 0)
    predicted = np.asarray(y_pred)
    if predicted.ndim != 1 or len(predicted) != len(truth):
        raise ValueError("y_pred must be one-dimensional and aligned with y_true.")
    if not np.isin(predicted, classes).all():
        raise ValueError("Predictions must use only the observed input classes.")
    if np.asarray(minority_label).ndim != 0 or not np.any(classes == minority_label):
        raise ValueError("minority_label must be one of the observed classes.")
    positive = truth == minority_label
    predicted_positive = predicted == minority_label
    tp = int(np.count_nonzero(positive & predicted_positive))
    tn = int(np.count_nonzero(~positive & ~predicted_positive))
    fp = int(np.count_nonzero(~positive & predicted_positive))
    fn = int(np.count_nonzero(positive & ~predicted_positive))
    accuracy = (tp + tn) / len(truth)
    if objective_mode == "paper_printed":
        numerator = 2 * tp * tn
        denominator = tp * (tp + fp) + tn * (tp + fn)
    else:
        numerator = 2 * tp
        denominator = 2 * tp + fp + fn
    defined = denominator != 0
    f1 = numerator / denominator if defined else None
    objective = weight * accuracy + (1 - weight) * f1 if defined else None
    anomaly = bool(defined and f1 > 1)
    return {
        "defined": defined,
        "accuracy": float(accuracy),
        "f1": float(f1) if defined else None,
        "objective": float(objective) if defined else None,
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "anomaly": anomaly,
        "anomalies": ["printed_f1_exceeds_one"] if anomaly else [],
    }


@dataclass
class _Model:
    """Complete candidate model; IDs, rather than positions, name clusters."""

    centroids: NDArray[np.float64]
    ids: NDArray[np.int64]
    labels: NDArray
    births: NDArray
    assignments: NDArray[np.int64]
    score: dict[str, Any] | None = None

    def clone(self) -> "_Model":
        return _Model(
            self.centroids.copy(), self.ids.copy(), self.labels.copy(),
            self.births.copy(), self.assignments.copy(),
            None if self.score is None else {
                **self.score,
                "counts": self.score["counts"].copy(),
                "anomalies": list(self.score["anomalies"]),
            },
        )


class SSHC(BaseEstimator):
    """Cluster separate labelled and unlabelled numerical fitting pools.

    Persistent IDs settle exact nearest-centroid ties. Dominant observed labels
    settle cluster labels, with birth-label preference on vote ties. Unanchored
    clusters retain their birth label and are explicitly diagnosed. They must
    not become SSHR pseudo-label candidates without an observed minority anchor.

    ``tol`` controls centroid convergence only. Objective declines are compared
    literally, without clipping the printed objective or permitting decreases.
    Guards are engineering limits, not claims of convergence.
    """

    def __init__(
        self,
        minority_label: Any = None,
        mu: float = 0.5,
        objective_mode: str = "paper_printed",
        prune_scope: str = "majority_only",
        tol: float = 1e-8,
        max_split_rounds: int = 100,
        max_refine_iter: int = 100,
    ) -> None:
        self.minority_label = minority_label
        self.mu = mu
        self.objective_mode = objective_mode
        self.prune_scope = prune_scope
        self.tol = tol
        self.max_split_rounds = max_split_rounds
        self.max_refine_iter = max_refine_iter

    def _clear_fitted_state(self) -> None:
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)

    def fit(self, X_labeled: Any, y_labeled: Any, X_unlabeled: Any = None) -> "SSHC":
        """Fit using observed labels only; copy inputs and clear failed refits."""
        self._clear_fitted_state()
        try:
            return self._fit(X_labeled, y_labeled, X_unlabeled)
        except Exception:
            self._clear_fitted_state()
            raise

    def _fit(self, X_labeled: Any, y_labeled: Any, X_unlabeled: Any) -> "SSHC":
        X_l = _feature_matrix(X_labeled)
        y, classes = binary_labels(y_labeled, len(X_l))
        if any(not isinstance(_scalar(label), (str, int, float, bool)) for label in classes):
            raise ValueError("Labels must be JSON-compatible real numeric, boolean, or string scalars.")
        X_u = (np.empty((0, X_l.shape[1]), dtype=float) if X_unlabeled is None
               else _feature_matrix(X_unlabeled, allow_empty=True))
        if X_u.shape[1] != X_l.shape[1]:
            raise ValueError("Labelled and unlabelled feature matrices must have the same width.")
        mu = finite_real(self.mu, "mu")
        if not 0 <= mu <= 1:
            raise ValueError("mu must lie in [0, 1].")
        if not isinstance(self.objective_mode, str) or self.objective_mode not in {"paper_printed", "standard_minority_f1"}:
            raise ValueError("objective_mode must be 'paper_printed' or 'standard_minority_f1'.")
        if not isinstance(self.prune_scope, str) or self.prune_scope not in {"majority_only", "all"}:
            raise ValueError("prune_scope must be 'majority_only' or 'all'.")
        tol = finite_real(self.tol, "tol")
        if tol < 0:
            raise ValueError("tol must be nonnegative.")
        split_limit = positive_integer(self.max_split_rounds, "max_split_rounds")
        refine_limit = positive_integer(self.max_refine_iter, "max_refine_iter")
        counts = np.array([np.count_nonzero(y == label) for label in classes])
        if self.minority_label is None:
            if counts[0] == counts[1]:
                raise ValueError("Tied class counts require an explicit minority_label.")
            minority = classes[int(np.argmin(counts))]
        else:
            if np.asarray(self.minority_label).ndim != 0 or not np.any(classes == self.minority_label):
                raise ValueError("minority_label must be one of the observed classes.")
            minority = classes[int(np.flatnonzero(classes == self.minority_label)[0])]
        majority = classes[int(np.flatnonzero(classes != minority)[0])]
        # These identities remain fixed even when cluster predictions change.
        self.classes_ = classes.copy()
        self.minority_label_ = _scalar(minority)
        self.majority_label_ = _scalar(majority)
        X = np.concatenate((X_l, X_u), axis=0)
        initial_centroids = np.stack([self._mean(X_l[y == label]) for label in classes])
        state = _Model(initial_centroids.copy(), np.arange(2, dtype=np.int64),
                       classes.copy(), classes.copy(), np.empty(0, dtype=np.int64))
        state = self._assign(X, y, state)
        state.score = self._evaluate(y, state)
        if not state.score["defined"]:
            raise ValueError("Initial SSHC objective is undefined under the selected objective_mode.")
        initial_assignment = state.assignments.copy()
        history: list[dict[str, Any]] = [self._event("initial", 0, True, "initialized", state)]
        next_id = 2
        split_stop = "max_split_rounds"
        for iteration in range(1, split_limit + 1):
            proposal, proposed_next_id = self._split(X, y, state, next_id)
            if proposal is None:
                split_stop = "no_impure_clusters"
                history.append(self._event("split", iteration, False, split_stop, state))
                break
            proposal.score = self._evaluate(y, proposal)
            reason = self._rejection_reason(state, proposal)
            if reason is None and self._same_partition(state, proposal):
                reason = "no_progress_partition"
            if reason is not None:
                split_stop = reason
                history.append(self._event("split", iteration, False, reason, state, proposal))
                break
            state = proposal
            next_id = proposed_next_id
            history.append(self._event("split", iteration, True, "accepted", state))
        state, prune_details = self._prune(X, y, state)
        history.append(self._event("prune", 1, prune_details["accepted"],
                                   prune_details["reason"], state,
                                   extra={"pruning": prune_details}))
        refine_stop = "max_refine_iter"
        for iteration in range(1, refine_limit + 1):
            proposal = self._refine_step(X, y, state)
            proposal.score = self._evaluate(y, proposal)
            reason = self._rejection_reason(state, proposal)
            if reason is not None:
                refine_stop = reason
                history.append(self._event("refine", iteration, False, reason, state, proposal))
                break
            converged = (
                np.array_equal(state.ids, proposal.ids)
                and np.array_equal(state.labels, proposal.labels)
                and np.array_equal(state.assignments, proposal.assignments)
                and np.max(np.abs(state.centroids - proposal.centroids)) <= tol
            )
            state = proposal
            reason = "centroids_converged" if converged else "accepted"
            history.append(self._event("refine", iteration, True, reason, state))
            if converged:
                refine_stop = reason
                break
        records = self._cluster_records(y, state)
        self.n_features_in_ = X_l.shape[1]
        self.centroids_ = state.centroids.copy()
        self.cluster_ids_ = state.ids.copy()
        self.cluster_labels_ = state.labels.copy()
        self.birth_labels_ = state.births.copy()
        self.clusters_ = tuple(np.flatnonzero(state.assignments == cluster_id)
                               for cluster_id in state.ids)
        self.labeled_cluster_ids_ = state.assignments[:len(y)].copy()
        self.unlabeled_cluster_ids_ = state.assignments[len(y):].copy()
        self.labeled_predictions_ = self._predictions(y, state)
        self.objective_ = float(state.score["objective"])
        self.diagnostics_ = {
            "implementation_status": "paper_rules_with_declared_interpretations",
            "author_code_verified": False,
            "parameters": {
                "minority_label": self.minority_label_, "majority_label": self.majority_label_,
                "mu": mu, "objective_mode": self.objective_mode, "prune_scope": self.prune_scope,
                "tol": tol, "max_split_rounds": split_limit, "max_refine_iter": refine_limit,
            },
            "label_policy": "dominant_observed_birth_label_then_input_class_order",
            "distance_tie_policy": "lowest_persistent_cluster_id",
            "objective_decline_policy": "literal_strict_decline_full_model_rollback",
            "initial_centroids": initial_centroids.tolist(),
            "initial_cluster_ids": [0, 1],
            "initial_birth_labels": [_scalar(label) for label in classes],
            "initial_assignment": initial_assignment.tolist(),
            "n_labeled": len(y), "n_unlabeled": len(X_u),
            "history": history,
            "stop_reasons": {"split": split_stop, "prune": prune_details["reason"], "refine": refine_stop},
            "guard_reached": split_stop == "max_split_rounds" or refine_stop == "max_refine_iter",
            "objective": dict(state.score),
            "clusters": records,
        }
        return self

    @staticmethod
    def _mean(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.mean(rows, axis=0)
        if not np.isfinite(result).all():
            raise ValueError("Centroid calculation overflowed; rescale fitting features.")
        return result

    def _assign(self, X: NDArray, y: NDArray, state: _Model) -> _Model:
        """Reassign all rows, discard empty clusters, and update observed votes."""
        proposal = state.clone()
        order = np.argsort(proposal.ids, kind="stable")
        proposal.centroids = proposal.centroids[order]
        proposal.ids = proposal.ids[order]
        proposal.labels = proposal.labels[order]
        proposal.births = proposal.births[order]
        distances = cdist(X, proposal.centroids, metric="euclidean")
        if not np.isfinite(distances).all():
            raise ValueError("Euclidean distances overflowed; rescale fitting features.")
        proposal.assignments = proposal.ids[np.argmin(distances, axis=1)]
        nonempty = np.array([np.any(proposal.assignments == cluster_id)
                             for cluster_id in proposal.ids])
        proposal.centroids = proposal.centroids[nonempty]
        proposal.ids = proposal.ids[nonempty]
        proposal.labels = proposal.labels[nonempty]
        proposal.births = proposal.births[nonempty]
        for position, cluster_id in enumerate(proposal.ids):
            observed = y[proposal.assignments[:len(y)] == cluster_id]
            if len(observed) == 0:
                proposal.labels[position] = proposal.births[position]
                continue
            votes = np.array([np.count_nonzero(observed == label) for label in self.classes_])
            tied = self.classes_[votes == votes.max()]
            birth = proposal.births[position]
            proposal.labels[position] = birth if np.any(tied == birth) else tied[0]
        return proposal

    @staticmethod
    def _predictions(y: NDArray, state: _Model) -> NDArray:
        positions = np.searchsorted(state.ids, state.assignments[:len(y)])
        return state.labels[positions].copy()

    def _evaluate(self, y: NDArray, state: _Model) -> dict[str, Any]:
        return clustering_objective(y, self._predictions(y, state), self.minority_label_,
                                    self.mu, self.objective_mode)

    def _split(self, X: NDArray, y: NDArray, state: _Model,
               next_id: int) -> tuple[_Model | None, int]:
        """Initialize observed-label children without recentering parents."""
        child_centroids = []
        child_labels = []
        for cluster_id, label in zip(state.ids, state.labels):
            members = np.flatnonzero(state.assignments[:len(y)] == cluster_id)
            observed = np.unique(y[members])
            if len(observed) < 2:
                continue
            for child_label in observed:
                if child_label != label:
                    child_centroids.append(self._mean(X[members[y[members] == child_label]]))
                    child_labels.append(child_label)
        if not child_centroids:
            return None, next_id
        labels = np.array(child_labels, dtype=state.labels.dtype)
        proposal = state.clone()
        proposal.centroids = np.concatenate((proposal.centroids, np.stack(child_centroids)))
        proposal.ids = np.concatenate((proposal.ids, np.arange(next_id, next_id + len(labels))))
        proposal.labels = np.concatenate((proposal.labels, labels))
        proposal.births = np.concatenate((proposal.births, labels.copy()))
        return self._assign(X, y, proposal), next_id + len(labels)

    @staticmethod
    def _same_partition(first: _Model, second: _Model) -> bool:
        # Ignore IDs, but retain row-level labels: a new birth label may resolve
        # tied observed votes differently without changing the member groups.
        def groups(state: _Model) -> set[tuple[int, ...]]:
            return {tuple(np.flatnonzero(state.assignments == cluster_id)) for cluster_id in state.ids}
        first_labels = first.labels[np.searchsorted(first.ids, first.assignments)]
        second_labels = second.labels[np.searchsorted(second.ids, second.assignments)]
        return groups(first) == groups(second) and np.array_equal(first_labels, second_labels)

    @staticmethod
    def _rejection_reason(state: _Model, proposal: _Model) -> str | None:
        if not proposal.score["defined"]:
            return "undefined_candidate_objective"
        if proposal.score["objective"] < state.score["objective"]:
            return "objective_declined"
        return None

    def _prune(self, X: NDArray, y: NDArray, state: _Model) -> tuple[_Model, dict[str, Any]]:
        records = self._cluster_records(y, state)
        defined = [record["accuracy"] for record in records if record["accuracy"] is not None]
        average = float(np.mean(defined)) if defined else None
        removed = [record["id"] for record in records
                   if record["accuracy"] is not None and record["accuracy"] < average
                   and (self.prune_scope == "all" or record["label"] == self.majority_label_)]
        details = {
            "scope": self.prune_scope, "accuracy_basis": "observed_labeled_members",
            "average_basis": "unweighted_defined_cluster_accuracies", "average_accuracy": average,
            "pre_prune_clusters": records, "removed_ids": removed, "accepted": False,
            "reason": "no_clusters_below_average",
        }
        if not removed:
            return state, details
        keep = ~np.isin(state.ids, removed)
        if not np.any(keep):
            details["reason"] = "all_clusters_pruned_rejected"
            return state, details
        proposal = state.clone()
        proposal.centroids = proposal.centroids[keep]
        proposal.ids = proposal.ids[keep]
        proposal.labels = proposal.labels[keep]
        proposal.births = proposal.births[keep]
        proposal = self._assign(X, y, proposal)
        proposal.score = self._evaluate(y, proposal)
        if not proposal.score["defined"]:
            details["reason"] = "undefined_candidate_objective"
            return state, details
        # Algorithm 1 specifies no objective gate for this separate pruning step.
        details["accepted"] = True
        details["reason"] = "clusters_pruned"
        return proposal, details

    def _refine_step(self, X: NDArray, y: NDArray, state: _Model) -> _Model:
        proposal = state.clone()
        proposal.centroids = np.stack([self._mean(X[state.assignments == cluster_id])
                                       for cluster_id in state.ids])
        return self._assign(X, y, proposal)

    def _cluster_records(self, y: NDArray, state: _Model) -> list[dict[str, Any]]:
        records = []
        for position, cluster_id in enumerate(state.ids):
            members = state.assignments == cluster_id
            observed = y[members[:len(y)]]
            counts = [{"label": _scalar(label), "count": int(np.count_nonzero(observed == label))}
                      for label in self.classes_]
            records.append({
                "id": int(cluster_id), "label": _scalar(state.labels[position]),
                "birth_label": _scalar(state.births[position]),
                "centroid": state.centroids[position].tolist(),
                "size": int(np.count_nonzero(members)), "labeled_size": len(observed),
                "unlabeled_size": int(np.count_nonzero(members[len(y):])),
                "observed_counts": counts,
                "observed_minority_count": int(np.count_nonzero(observed == self.minority_label_)),
                "observed_majority_count": int(np.count_nonzero(observed == self.majority_label_)),
                "accuracy": float(np.mean(observed == state.labels[position])) if len(observed) else None,
                "unanchored": len(observed) == 0,
                "impure": len(np.unique(observed)) > 1,
            })
        return records

    @staticmethod
    def _event(phase: str, iteration: int, accepted: bool, reason: str,
               state: _Model, candidate: _Model | None = None,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "phase": phase, "iteration": iteration, "accepted": accepted, "reason": reason,
            "objective": dict(state.score),
            "candidate_objective": None if candidate is None else dict(candidate.score),
            "snapshot": {
                "centroids": state.centroids.tolist(), "cluster_ids": state.ids.tolist(),
                "labels": [_scalar(label) for label in state.labels],
                "birth_labels": [_scalar(label) for label in state.births],
                "assignments": state.assignments.tolist(),
            },
        }
        if extra:
            event.update(extra)
        return event


__all__ = ["SSHC", "clustering_objective"]
