"""Descriptive-statistics tests on synthetic software fixtures only."""
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.imbalance_stats import calculate_statistics, format_summary

FIXTURES = Path(__file__).parent / "fixtures"


def test_all_statistics_and_no_mutation() -> None:
    dataset = load_dataset(FIXTURES / "toy.csv", "class")
    X_before, y_before = dataset.X.copy(), dataset.y.copy()
    stats = calculate_statistics(dataset)
    assert stats.to_dict() == {
        "total_samples": 6, "number_of_features": 2,
        "majority_count": 5, "minority_count": 1,
        "majority_percentage": 100 * 5 / 6, "minority_percentage": 100 / 6,
        "imbalance_ratio": 5.0, "missing_feature_values": 2,
        "rows_with_missing_features": 2, "duplicated_feature_rows": 2,
        "duplicated_complete_rows": 1, "numerical_features": 1,
        "categorical_features": 1,
    }
    pd.testing.assert_frame_equal(dataset.X, X_before)
    pd.testing.assert_series_equal(dataset.y, y_before)
    summary = format_summary(dataset, stats)
    assert "Majority 'normal' -> 0: 5 (83.33%)" in summary
    assert "Minority 'rare' -> 1: 1 (16.67%)" in summary
    assert "Duplicate feature rows (beyond first): 2" in summary
    assert "features + target, beyond first): 1" in summary


def test_ninety_ten_ratio(tmp_path: Path) -> None:
    path = tmp_path / "ratio.csv"
    pd.DataFrame({"x": range(100), "class": ["major"] * 90 + ["minor"] * 10}).to_csv(path, index=False)
    stats = calculate_statistics(load_dataset(path, "class"))
    assert stats.imbalance_ratio == 9.0
    assert stats.majority_percentage == 90.0
    assert stats.minority_percentage == 10.0


def test_duplicate_missing_rows(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    path.write_text("x,class\n?,a\n?,a\n?,b\n", encoding="utf-8")
    stats = calculate_statistics(load_dataset(path, "class"))
    assert stats.missing_feature_values == stats.rows_with_missing_features == 3
    assert stats.duplicated_feature_rows == 2
    assert stats.duplicated_complete_rows == 1


def test_equal_counts_with_override(tmp_path: Path) -> None:
    path = tmp_path / "tie.csv"
    path.write_text("x,class\n1,a\n2,b\n", encoding="utf-8")
    stats = calculate_statistics(load_dataset(path, "class", "b"))
    assert stats.imbalance_ratio == 1.0
    assert stats.majority_percentage == stats.minority_percentage == 50.0


def test_csv_and_keel_statistics_match() -> None:
    csv_stats = calculate_statistics(load_dataset(FIXTURES / "toy.csv", "class"))
    keel_stats = calculate_statistics(load_dataset(FIXTURES / "toy.dat"))
    assert csv_stats == keel_stats


def test_mutated_targets_rejected() -> None:
    dataset = load_dataset(FIXTURES / "toy.csv", "class")
    dataset.y[:] = 0
    with pytest.raises(ValueError, match="both encoded classes"):
        calculate_statistics(dataset)
