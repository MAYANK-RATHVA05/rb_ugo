"""Exploratory fixed-UGO-trace allocation diagnostic; no novelty/safety claim.

This deliberately does NOT feed U-informed children back into UGO's anchor
selection. All controls replay one supervised trace and one checked candidate
bank. It is not the proposal's complete adaptive semi-supervised algorithm.
Hidden U labels and evaluation data are absent from this module's interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer
from src.noise_detection import nearest_indices, noise_final_detect_and_relocate
from src.ugo import UGO


@dataclass
class Allocation:
    counts: np.ndarray
    continuous: np.ndarray
    exposure: float
    rounding_loss: float


def allocate(errors: Any, limits: Any, capacity: int, budget: float) -> Allocation:
    """Continuous C5 via SLSQP, then floor: feasible, not integer-optimal.

    Solver failures are errors, never silent successful fallback. No greedy
    completion is used, so rounding loss is explicit and reproducible.
    """
    e = np.asarray(errors, dtype=float)
    upper = np.asarray(limits)
    if (e.ndim != 1 or upper.shape != e.shape or not np.isfinite(e).all()
            or np.any((e < 0) | (e > 1)) or upper.dtype.kind not in 'iu'
            or np.any(upper < 0)):
        raise ValueError('Expected aligned probabilities and nonnegative integer limits')
    if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity < 0:
        raise ValueError('capacity must be a nonnegative integer')
    budget = finite_real(budget, 'budget')
    if budget < 0:
        raise ValueError('budget must be nonnegative')
    if not len(e) or capacity == 0:
        z = np.zeros(len(e)); return Allocation(z.astype(int), z, 0., 0.)
    # Exact reduction avoids a degenerate zero-budget constraint in SLSQP.
    active = (upper > 0) & ((e == 0) if budget == 0 else True)
    x = np.zeros(len(e))
    if active.any():
        a = e[active]; bound = upper[active].astype(float)
        constraints = [{'type': 'ineq', 'fun': lambda m: capacity - m.sum(),
                        'jac': lambda m: -np.ones(len(m))}]
        if budget > 0 and np.any(a):
            constraints.append({'type': 'ineq', 'fun': lambda m: budget - a @ m,
                                'jac': lambda m: -a})
        result = minimize(lambda m: -np.log1p(m).sum(), np.zeros(len(a)),
                          jac=lambda m: -1 / (1 + m), bounds=list(zip(np.zeros(len(a)), bound)),
                          constraints=constraints, method='SLSQP',
                          options={'ftol': 1e-10, 'maxiter': 1000})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f'Allocation solver failed: {result.message}')
        x[active] = np.clip(result.x, 0, bound)
    if x.sum() > capacity + 1e-6 or e @ x > budget + 1e-6:
        raise RuntimeError('Continuous allocation is infeasible')
    # Tiny inward scaling prevents a solver tolerance from crossing an exact
    # integer boundary. Flooring can conservatively lose a nearly integral row.
    if x.sum() > capacity:
        x *= capacity / x.sum()
    if e @ x > budget:
        x *= budget / (e @ x)
    counts = np.floor(x).astype(int)
    if counts.sum() > capacity or e @ counts > budget + 1e-10:
        raise RuntimeError('Integer allocation is infeasible')
    return Allocation(counts, x, float(e @ counts), float(x.sum() - counts.sum()))


def fair_counts(limits: np.ndarray, count: int) -> np.ndarray:
    """Exact count, diminishing-multiplicity allocation; stable endpoint ties."""
    result = np.zeros(len(limits), dtype=int)
    if count < 0 or count > limits.sum():
        raise ValueError('Requested count is infeasible')
    for _ in range(count):
        available = np.flatnonzero(result < limits)
        j = available[np.argmin(result[available])]
        result[j] += 1
    return result


def policy_counts(bank: dict, policy: str) -> tuple[np.ndarray, dict]:
    limits = np.array([len(c) for c in bank['candidates']], dtype=int)
    errors = 1 - bank['probabilities']
    constrained = allocate(errors, limits, bank['capacity'], bank['budget'])
    target = int(constrained.counts.sum())
    if policy == 'budget':
        counts = constrained.counts
    elif policy == 'unconstrained':
        counts = fair_counts(limits, min(bank['capacity'], int(limits.sum())))
    elif policy == 'count_matched':
        counts = fair_counts(limits, target)
    elif policy == 'threshold_matched':
        # Smallest descending-confidence prefix with capacity for target. All
        # endpoint ties at the cutoff are admitted; this is a threshold control.
        order = np.argsort(-bank['probabilities'], kind='stable')
        admitted = np.zeros(len(limits), dtype=bool)
        if target:
            position = np.searchsorted(np.cumsum(limits[order]), target)
            cutoff = bank['probabilities'][order[position]]
            admitted = bank['probabilities'] >= cutoff
        counts = fair_counts(limits * admitted, target)
    else:
        raise ValueError(f'Unknown policy: {policy}')
    return counts, {'count': int(counts.sum()), 'estimated_exposure': float(errors @ counts),
                    'budget': bank['budget'], 'planned_capacity': bank['capacity'],
                    'constrained_count': target, 'rounding_loss': constrained.rounding_loss,
                    'budget_reduces_integer_count': target < min(bank['capacity'], int(limits.sum()))}


def prepare(X: Any, y: Any, U: Any, parent: UGO, *, threshold: float = .9,
            exposure_rate: float = .05, endpoint_cap: int = 5, guide: Any = None) -> dict:
    """Fit one parent/guide and freeze candidate banks using fitting inputs only.

    Distances must be meaningful in X's coordinates. Callers fit preprocessing
    on L only. The default calibrated guide includes its own inner-fitted scaler.
    With fewer than three labels per class, the declared fallback is parent UGO.
    A supplied guide is explicitly uncertified, even if it reports probabilities.
    """
    X = numeric_matrix(X); y, classes = binary_labels(y, len(X))
    U = numeric_matrix(U, allow_empty=True)
    if X.shape[1] != U.shape[1]:
        raise ValueError('L/U feature dimensions differ')
    threshold = finite_real(threshold, 'threshold')
    exposure_rate = finite_real(exposure_rate, 'exposure_rate')
    if not 0 <= threshold <= 1 or not 0 <= exposure_rate <= 1:
        raise ValueError('threshold/exposure_rate must lie in [0,1]')
    endpoint_cap = positive_integer(endpoint_cap, 'endpoint_cap')
    if parent.variant != 'smote':
        raise ValueError('This diagnostic requires an explicit UGO-SMOTE parent')
    model = clone(parent)
    baseline_X, baseline_y = model.fit_resample(X, y)
    result = dict(X=X, y=y, U=U, baseline_X=baseline_X, baseline_y=baseline_y,
                  parent=model, banks=[], guide_status='unused_no_U', probabilities=np.empty(0))
    if not len(U):
        return result
    counts = np.array([(y == c).sum() for c in classes])
    if counts.min() < 3 or counts[0] == counts[1]:
        result['guide_status'] = 'parent_fallback_insufficient_or_tied_class_counts'
        return result
    minority = classes[np.argmin(counts)]
    if guide is None:
        guide = CalibratedClassifierCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=parent.random_state)),
            method='sigmoid', cv=StratifiedKFold(3, shuffle=True, random_state=parent.random_state))
        result['guide_status'] = 'three_fold_sigmoid_calibration_not_pointwise_error_bound'
    else:
        result['guide_status'] = 'custom_guide_uncertified'
    guide = clone(guide).fit(X.copy(), y.copy())
    proba = np.asarray(guide.predict_proba(U.copy()))
    if (proba.shape != (len(U), 2) or not np.isfinite(proba).all()
            or np.any((proba < 0) | (proba > 1)) or not np.allclose(proba.sum(1), 1)
            or set(guide.classes_) != set(classes)):
        raise ValueError('Guide probabilities/classes violate the binary contract')
    q = proba[:, list(guide.classes_).index(minority)]
    result.update(probabilities=q, guide=guide)
    # LOO minority-distance screen uses only observed clean L, never U labels.
    clean = np.flatnonzero((y == minority) & ~model.pre_noise_mask_)
    if len(clean) < 2:
        result['guide_status'] += ';no_clean_anchor_radius'
        return result
    near = nearest_indices(X[clean], X[clean], 1, exclude_self=True)[:, 0]
    radius = float(np.quantile(np.linalg.norm(X[clean] - X[clean][near], axis=1), .95))
    offset = len(X)
    for record in model.diagnostics_['history']:
        size = record['generated_count']
        if size and record['selected_class'] == minority:
            anchors = np.array(record['seed_indices'], dtype=int)
            association = nearest_indices(U, X[anchors], 1)[:, 0]
            distance = np.linalg.norm(U - X[anchors[association]], axis=1)
            ids = np.flatnonzero((q >= threshold) & (distance <= radius) & (distance > 0))
            candidates = []; ledger = []; rejected = 0
            for j in ids:
                anchor = int(anchors[association[j]])
                rng = np.random.default_rng(np.random.SeedSequence([parent.random_state, record['iteration'], int(j)]))
                fractions = rng.uniform(np.nextafter(0., 1.), 1., min(endpoint_cap, size))
                proposed = X[anchor] + fractions[:, None] * (U[j] - X[anchor])
                _, check = noise_final_detect_and_relocate(proposed, minority, X, y, parent.k_after,
                                                           reference_noise_mask=model.pre_noise_mask_)
                # Explicit U-branch rejection, rather than supervised relocation.
                # Remove coincidences with observed L/U: retain actual new vectors.
                keep = check['relocation_reference_indices'] == -1
                for i, row in enumerate(proposed):
                    if np.any(np.all(np.vstack((X, U)) == row, axis=1)):
                        keep[i] = False
                rejected += int((~keep).sum())
                candidates.append(proposed[keep])
                ledger.append([dict(round=record['iteration'], endpoint=int(j), anchor=anchor,
                                    fraction=float(f), guide_probability=float(q[j])) for f in fractions[keep]])
            result['banks'].append(dict(offset=offset, capacity=size, budget=exposure_rate * size,
                                        endpoint_ids=ids, probabilities=q[ids], candidates=candidates,
                                        ledger=ledger, rejected_candidates=rejected, radius=radius))
        offset += size
    return result


def replay(prepared: dict, policy: str = 'budget') -> dict:
    """Replace parent rows with checked U children; total sample count is fixed.

    Count/threshold controls have exactly the budgeted U-child count per round.
    Weighted PL uses the budgeted child multiplicities as sample weights, keeping
    the same supervised fallback rows and total fitting weight.
    """
    allowed = {'parent', 'budget', 'unconstrained', 'count_matched', 'threshold_matched', 'weighted_pl'}
    if policy not in allowed:
        raise ValueError(f'Unknown policy: {policy}')
    X = prepared['baseline_X'].copy(); y = prepared['baseline_y'].copy()
    weights = np.ones(len(X)); ledger = []; rounds = []
    if policy != 'parent':
        for bank in prepared['banks']:
            counts, info = policy_counts(bank, 'budget' if policy == 'weighted_pl' else policy)
            rounds.append(info)
            cursor = bank['offset']
            for j, count in enumerate(counts):
                for k in range(count):
                    child = dict(bank['ledger'][j][k], output_index=cursor)
                    X[cursor] = (prepared['U'][child['endpoint']] if policy == 'weighted_pl'
                                 else bank['candidates'][j][k])
                    ledger.append(child); cursor += 1
    if policy == 'weighted_pl' and ledger:
        # Collapse repeated endpoints into actual sample weights. Supervised
        # fallback rows retain unit weight; endpoint mass equals child count.
        removed = np.zeros(len(X), dtype=bool)
        endpoint_rows = {}
        for child in ledger:
            endpoint = child['endpoint']; index = child['output_index']
            if endpoint in endpoint_rows:
                weights[endpoint_rows[endpoint]] += 1
                removed[index] = True
            else:
                endpoint_rows[endpoint] = index
        old_to_new = np.cumsum(~removed) - 1
        for child in ledger:
            child['output_index'] = int(old_to_new[endpoint_rows[child['endpoint']]])
        X, y, weights = X[~removed], y[~removed], weights[~removed]
    return dict(X=X, y=y, sample_weight=weights, ledger=ledger, rounds=rounds,
                policy=policy, guide_status=prepared['guide_status'],
                parent_generated_count=len(prepared['baseline_X']) - len(prepared['X']),
                u_informed_count=len(ledger),
                new_u_vectors=0 if policy == 'weighted_pl' else len(ledger),
                interpretation='exploratory_fixed_supervised_trace; not_adaptive_SSL_UGO')
