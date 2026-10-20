"""Isolated multiclass GMMSampling conventions and experimental density gates.

Independent adapter; pinned reference and departures documented in frozen protocol.
No binary-project API is imported or changed. Subgroup identities are never inputs.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


def keyed_rng(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, keys)]))


def validate_xy(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError("Expected aligned finite matrix and one-dimensional labels")
    if len(np.unique(y)) < 3:
        raise ValueError("This research interface requires at least three classes")
    return X, y


def safe_levels(X, y, k):
    X, y = validate_xy(X, y)
    if not 1 <= k < len(X):
        raise ValueError("Invalid neighbor count")
    labels, counts = np.unique(y, return_counts=True)
    sizes = dict(zip(labels, counts))
    distance = pairwise_distances(X)
    np.fill_diagonal(distance, np.inf)  # self exclusion by identity, even with duplicates
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
    own = np.array([sizes[c] for c in y])[:, None]
    other = np.array([[sizes[c] for c in row] for row in y[neighbors]])
    return (np.minimum(own, other) / np.maximum(own, other)).mean(axis=1)


def allocation_weights(responsibilities, safety):
    weights = ((1 - np.asarray(safety)[:, None]) * responsibilities).sum(axis=0) + 1e-8
    return weights / weights.sum()


def component_quotas(weights, budget):
    weights = np.asarray(weights, float)
    if budget < 0 or int(budget) != budget or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Invalid component allocation")
    raw = weights / weights.sum() * int(budget)
    quota = np.floor(raw).astype(int)
    order = np.argsort(-(raw - quota), kind="stable")
    quota[order[:int(budget) - quota.sum()]] += 1
    return quota


def density_accept(log_densities, own_index, tau):
    """Return acceptance and numerical-failure masks; use only minority densities."""
    logs = np.asarray(log_densities, float)
    if logs.ndim != 2 or not 0 <= own_index < logs.shape[1] or not 0 <= tau <= 1:
        raise ValueError("Invalid density gate input")
    invalid = np.isnan(logs).any(axis=1) | np.isposinf(logs).any(axis=1) | np.isneginf(logs).all(axis=1)
    if tau == 0:
        return np.ones(len(logs), bool), np.zeros(len(logs), bool)
    score = np.full(len(logs), -np.inf)
    good = ~invalid
    score[good] = logs[good, own_index] - logsumexp(logs[good], axis=1)
    return good & (score >= np.log(tau)), invalid


def fit_class_gmm(X, seed, settings):
    """Last improving development model, capped without unused K+1 trial."""
    train, dev = train_test_split(X, test_size=settings["valid_size"], random_state=seed)
    trace, selected, previous = [], 1, -np.inf
    def build(k):
        return GaussianMixture(n_components=k, covariance_type="full", random_state=seed,
                               **{p: settings[p] for p in ("n_init", "max_iter", "tol", "reg_covar")})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for k in range(1, min(settings["max_components"], len(train)-1) + 1):
            model = build(k).fit(train)
            score = float(model.score(dev))
            trace.append({"components": k, "development_log_likelihood": score})
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite development likelihood")
            if score < previous:
                break
            selected, previous = k, score
        model = build(selected).fit(X)
    return model, trace


@dataclass
class Sampled:
    X: np.ndarray
    y: np.ndarray
    parent: np.ndarray
    component: np.ndarray
    kind: np.ndarray
    counts: list
    proposals: dict


class OverlapSampler:
    """Original-fitting-only state. No validation/test or evaluator-group argument."""
    def __init__(self, X, y, minority_labels, seed, gmm_settings):
        self.X, self.y = validate_xy(X, y)
        self.X, self.y = self.X.copy(), self.y.copy()
        self.labels = tuple(minority_labels)
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels) or any(c not in self.y for c in self.labels):
            raise ValueError("Need distinct competing minority labels present in original data")
        self.seed = seed
        self.models, self.traces, self.members, self.responsibilities = {}, {}, {}, {}
        for c in self.labels:
            ids = np.flatnonzero(self.y == c)
            self.models[c], self.traces[c] = fit_class_gmm(self.X[ids], seed, gmm_settings)
            r = self.models[c].predict_proba(self.X[ids])
            self.responsibilities[c] = r
            assignment = r.argmax(axis=1)
            self.members[c] = {j: ids[assignment == j] for j in range(r.shape[1])}
        self.safety_cache = {}

    def sample(self, method, target, k=7, tau=0., retry_limit=20):
        if method not in ("A", "B", "C", "D") or retry_limit < 1:
            raise ValueError("Unknown arm or invalid retry budget")
        if k not in self.safety_cache:
            self.safety_cache[k] = safe_levels(self.X, self.y, k)
        Xs, ys, parents, components, kinds = [self.X], [self.y], [np.arange(len(self.y))], [np.full(len(self.y), -1)], [np.full(len(self.y), "original")]
        counts, proposals = [], {}
        for ci, c in enumerate(self.labels):
            model = self.models[c]
            quota = component_quotas(allocation_weights(self.responsibilities[c], self.safety_cache[k][self.y == c]), target - int((self.y == c).sum()))
            for j, q in enumerate(quota):
                q = int(q)
                members = self.members[c][j]
                if q and not len(members):
                    raise ValueError(f"Positive quota but no original hard-MAP members: {c}/{j}")
                fallback_rng = keyed_rng(self.seed, 801, ci, j)
                # Generate a deterministic stream once; B uses its first q draws,
                # C consumes only its bounded prefix, all tau values share the stream.
                limit = q * retry_limit if method == "C" and tau > 0 else q
                draw = np.empty((0, self.X.shape[1]))
                accept = np.empty(0, bool)
                invalid = np.empty(0, bool)
                if method != "D" and q:
                    draw = keyed_rng(self.seed, 800, ci, j).multivariate_normal(model.means_[j], model.covariances_[j], size=limit)
                    if method == "A" or tau == 0:
                        accept, invalid = np.ones(limit, bool), np.zeros(limit, bool)
                    else:
                        logs = np.column_stack([self.models[d].score_samples(draw) for d in self.labels])
                        accept, invalid = density_accept(logs, ci, tau)
                    hit = np.flatnonzero(accept)
                    used = int(hit[q-1]+1) if len(hit) >= q else limit
                    draw, accept, invalid = draw[:used], accept[:used], invalid[:used]
                accepted = draw[accept]
                missing = q - len(accepted)
                n_fallback = missing if method in ("C", "D") else 0
                parent = fallback_rng.choice(members, size=n_fallback, replace=True) if n_fallback else np.empty(0, int)
                added = np.vstack([accepted, self.X[parent]])
                Xs.append(added)
                ys.append(np.full(len(added), c, dtype=self.y.dtype))
                parents.append(np.r_[np.full(len(accepted), -1), parent])
                components.append(np.full(len(added), j))
                kinds.append(np.r_[np.full(len(accepted), "gaussian"), np.full(n_fallback, "duplicate")])
                counts.append(dict(label=c.item() if isinstance(c,np.generic) else c, component=j, requested=q, generated=len(draw), accepted=int(accept.sum()), rejected=int((~accept).sum()), numerical_rejections=int(invalid.sum()), fallback=n_fallback, realized=len(added), unfilled=q-len(added)))
                proposals[f"{c}_{j}_X"] = draw
                proposals[f"{c}_{j}_accepted"] = accept
                proposals[f"{c}_{j}_numerical_failure"] = invalid
        return Sampled(np.vstack(Xs), np.concatenate(ys), np.concatenate(parents), np.concatenate(components), np.concatenate(kinds), counts, proposals)
