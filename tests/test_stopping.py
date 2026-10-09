"""Welch decisions are inferential decisions, never proof of equality."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import f, ttest_ind

import src.stopping as stopping
from src.stopping import generation_cap, welch_stopping


def test_welch_rejects_controlled_different_means() -> None:
    a = np.array([-0.9, -0.8, -0.7, -0.6])
    b = np.array([0.5, 0.6, 0.7, 0.8])
    result = welch_stopping(a, b)
    expected = ttest_ind(a, b, equal_var=False)
    assert result.defined and not result.stop
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.p_value == pytest.approx(f.sf(result.statistic ** 2, 1, result.degrees_of_freedom))


def test_welch_fails_to_reject() -> None:
    result = welch_stopping([-0.5, 0, 0.5], [-0.5, 0, 0.5])
    assert result.defined and result.stop
    assert result.p_value == 1.0
    assert result.reason == "fail_to_reject"


@pytest.mark.parametrize("a,b,reason", [
    ([0.1], [0.1, 0.2], "too_small_groups"),
    ([], [0.1, 0.2], "too_small_groups"),
    ([0.2, 0.2], [0.2, 0.2], "both_groups_constant"),
    ([0.2, 0.2], [0.8, 0.8], "both_groups_constant"),
])
def test_undefined_is_not_equality(a, b, reason) -> None:
    result = welch_stopping(a, b)
    assert not result.defined and not result.stop
    assert result.p_value is None and result.statistic is None
    assert result.reason == reason


def test_nonfinite_statistical_result(monkeypatch) -> None:
    monkeypatch.setattr(stopping, "ttest_ind", lambda *a, **k:
                        SimpleNamespace(statistic=np.nan, pvalue=np.nan, df=np.nan))
    result = welch_stopping([0, 1], [0.1, 0.2])
    assert not result.stop and not result.defined
    assert result.reason == "nonfinite_statistical_result"


@pytest.mark.parametrize("a,b,alpha", [([0, np.nan], [0, 1], 0.05),
                                         ([0, 1], [0, 1], 0),
                                         ([0, 1], [0, 1], np.nan)])
def test_invalid_welch_input(a, b, alpha) -> None:
    with pytest.raises(ValueError):
        welch_stopping(a, b, alpha)


def test_provisional_cap_arithmetic_and_rounding() -> None:
    # These validate the DECLARED interpretation, not the inaccessible paper.
    assert generation_cap([90, 10], 2.5) == 350
    assert generation_cap([3, 2], 1.1) == 1
    assert generation_cap([3, 2], 0.1) == 0


@pytest.mark.parametrize("counts,phi", [([1, 0], 2.5), ([1.5, 2], 2.5),
                                      ([1, 2, 3], 2.5), ([1, 2], np.inf)])
def test_invalid_cap(counts, phi) -> None:
    with pytest.raises(ValueError):
        generation_cap(counts, phi)
