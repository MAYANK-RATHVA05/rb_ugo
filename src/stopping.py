"""Welch stopping and source-explicit UGO generation thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import ttest_ind

from src._validation import finite_real


@dataclass(frozen=True)
class WelchDecision:
    """A valid p>=alpha fails to reject; undefined tests never mean equality."""

    statistic: float | None
    p_value: float | None
    degrees_of_freedom: float | None
    stop: bool
    defined: bool
    reason: str


def welch_stopping(group_a: Any, group_b: Any, alpha: float = 0.05) -> WelchDecision:
    """Two-sided Welch t-test, equivalent to two-group Welch ANOVA (F=t²).

    Undefined too-small or doubly constant groups return stop=False, with no
    p-value. The UGO loop terminates separately with an engineering reason when
    the test is undefined. One constant group is valid if the other varies.
    """
    alpha = finite_real(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one.")
    groups = [np.asarray(group_a), np.asarray(group_b)]
    for group in groups:
        if group.ndim != 1 or group.dtype.kind not in "fiu" or not np.isfinite(group).all():
            raise ValueError("Welch groups must be one-dimensional finite numerical arrays.")
    if any(len(group) < 2 for group in groups):
        return WelchDecision(None, None, None, False, False, "too_small_groups")
    if all(np.ptp(group) == 0 for group in groups):
        return WelchDecision(None, None, None, False, False, "both_groups_constant")
    result = ttest_ind(*groups, equal_var=False, alternative="two-sided")
    if not np.isfinite([result.statistic, result.pvalue, result.df]).all():
        return WelchDecision(None, None, None, False, False, "nonfinite_statistical_result")
    if not 0 <= result.pvalue <= 1 or result.df <= 0:
        return WelchDecision(None, None, None, False, False, "invalid_statistical_result")
    stop = bool(result.pvalue >= alpha)
    return WelchDecision(float(result.statistic), float(result.pvalue), float(result.df),
                         stop, True, "fail_to_reject" if stop else "reject_equal_means")


def generation_cap(class_counts: Any, phi: float = 2.5, *, interpretation: str = "section_3_3") -> int:
    """Minimal integer synthetic count meeting a documented paper threshold.

    section_3_3 (default): ceil(phi*K*N_max-N_original), PDF p6.
    table4: ceil(phi*N_max), PDF p8. The paper gives conflicting rules;
    selecting one does not establish complete paper fidelity. Counts are frozen.
    Batch clipping is a documented finite-budget safeguard, not a table rule.
    """
    counts = np.asarray(class_counts)
    if counts.shape != (2,) or counts.dtype.kind not in "iu" or (counts <= 0).any():
        raise ValueError("Generation cap requires two positive integer original class counts.")
    phi = finite_real(phi, "phi")
    if phi <= 0:
        raise ValueError("phi must be a finite positive number.")
    if interpretation not in {"section_3_3", "table4"}:
        raise ValueError("interpretation must be 'section_3_3' or 'table4'.")
    budget = (phi * len(counts) * int(counts.max()) - sum(int(count) for count in counts)
              if interpretation == "section_3_3" else phi * int(counts.max()))
    if not np.isfinite(budget):
        raise ValueError("Generation budget is not finite.")
    return max(0, int(np.ceil(budget)))
