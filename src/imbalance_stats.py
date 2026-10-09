"""Descriptive statistics only; never remove rows or fit a model."""
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.data_loader import LoadedDataset


@dataclass(frozen=True)
class ImbalanceStats:
    """File-level descriptive counts; percentages use all loaded samples.

    Duplicate counts are additional occurrences beyond the first, following
    pandas duplicated(keep='first'). Feature duplicates compare X alone; full
    duplicates compare X plus the original target. Matching missing values count
    as equal for duplicate detection. No rows are removed.
    """

    total_samples: int
    number_of_features: int
    majority_count: int
    minority_count: int
    majority_percentage: float
    minority_percentage: float
    imbalance_ratio: float
    missing_feature_values: int
    rows_with_missing_features: int
    duplicated_feature_rows: int
    duplicated_complete_rows: int
    numerical_features: int
    categorical_features: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible Python integers and floats."""
        return asdict(self)


def calculate_statistics(dataset: LoadedDataset) -> ImbalanceStats:
    """Summarize a validated LoadedDataset without modifying its contents."""
    X = dataset.X
    total = len(X)
    majority = int((dataset.y == 0).sum())
    minority = int((dataset.y == 1).sum())
    # Guard against accidental mutation of the returned pandas structures.
    if total == 0 or majority == 0 or minority == 0 or majority + minority != total:
        raise ValueError("Statistics require nonempty data with both encoded classes 0 and 1.")
    if not X.index.equals(dataset.original_y.index) or not X.index.equals(dataset.y.index):
        raise ValueError("Features and target indexes must remain aligned.")
    missing = X.isna()
    # A tuple column name cannot collide with loaded string feature names.
    complete = X.copy()
    complete[("original_target",)] = dataset.original_y
    return ImbalanceStats(
        total_samples=total, number_of_features=X.shape[1],
        majority_count=majority, minority_count=minority,
        majority_percentage=100.0 * majority / total,
        minority_percentage=100.0 * minority / total,
        imbalance_ratio=majority / minority,
        missing_feature_values=int(missing.sum().sum()),
        rows_with_missing_features=int(missing.any(axis=1).sum()),
        duplicated_feature_rows=int(X.duplicated().sum()),
        duplicated_complete_rows=int(complete.duplicated().sum()),
        numerical_features=len(dataset.numeric_feature_names),
        categorical_features=len(dataset.categorical_feature_names),
    )


def format_summary(dataset: LoadedDataset, stats: ImbalanceStats) -> str:
    """Render a readable summary with explicit label and duplicate definitions."""
    return "\n".join([
        f"Dataset: {dataset.source_filename} ({dataset.source_format.upper()})",
        f"Target: {dataset.target_name}",
        f"Samples: {stats.total_samples}; features: {stats.number_of_features}",
        f"Majority {dataset.majority_label!r} -> 0: {stats.majority_count} ({stats.majority_percentage:.2f}%)",
        f"Minority {dataset.minority_label!r} -> 1: {stats.minority_count} ({stats.minority_percentage:.2f}%)",
        f"Imbalance ratio (majority/minority): {stats.imbalance_ratio:.4f}",
        f"Missing feature values: {stats.missing_feature_values}",
        f"Rows with missing features: {stats.rows_with_missing_features}",
        f"Duplicate feature rows (beyond first): {stats.duplicated_feature_rows}",
        f"Duplicate complete rows (features + target, beyond first): {stats.duplicated_complete_rows}",
        f"Numerical features: {stats.numerical_features}; categorical features: {stats.categorical_features}",
    ])
