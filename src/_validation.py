"""Shared numerical validation; never encode or preprocess features."""
from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def numeric_matrix(X: Any, *, allow_empty: bool = False) -> NDArray[np.float64]:
    """Copy a finite, dense, real numerical matrix; reject categorical dtypes."""
    if isinstance(X, pd.DataFrame):
        if any(not pd.api.types.is_numeric_dtype(dtype) or
               isinstance(dtype, pd.CategoricalDtype) or
               pd.api.types.is_bool_dtype(dtype) for dtype in X.dtypes):
            raise ValueError("Features must be numerical; categorical input is unsupported.")
        array = X.to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] == 0 or (not allow_empty and len(array) == 0):
        raise ValueError("X must be a nonempty two-dimensional numerical feature matrix.")
    if array.dtype.kind not in "fiu":
        raise ValueError("Features must be numerical; categorical/object/complex input is unsupported.")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Features must contain only finite numerical values (no missing values).")
    return result


def binary_labels(y: Any, n_samples: int) -> tuple[NDArray, NDArray]:
    """Copy aligned, nonmissing labels with exactly two comparable classes."""
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != n_samples:
        raise ValueError("y must be one-dimensional and aligned with X.")
    if pd.isna(labels).any():
        raise ValueError("Labels must not be missing.")
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise ValueError("Labels must be finite.")
    try:
        classes = np.unique(labels)
    except TypeError as exc:
        raise ValueError("Labels must have mutually comparable scalar types.") from exc
    if len(classes) != 2:
        raise ValueError("Exactly two classes are required.")
    return labels.copy(), classes


def positive_integer(value: Any, name: str) -> int:
    """Validate a positive integer parameter without silently truncating it."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def finite_real(value: Any, name: str) -> float:
    """Validate a finite real scalar, excluding booleans and strings."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)
