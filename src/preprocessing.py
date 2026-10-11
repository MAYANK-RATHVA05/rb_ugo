"""Fold-local feature preparation with explicitly declared reproduction choices."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src._validation import binary_labels


MISSING_CATEGORY = "__ugo_missing__"
PREPROCESSING_POLICY = {
    "numeric_missing": "training median; all-missing training columns filled with 0",
    "categorical_missing": MISSING_CATEGORY,
    "categorical_encoding": "training one-hot categories; unknown test categories become all-zero",
    "variance_filter": "after imputation/encoding, drop a feature constant within either training class",
    "scaling": "training StandardScaler after filtering, including encoded columns",
    "fidelity": "declared engineering conventions; exact author ordering/encoding/scaling unknown",
}


def feature_frame(X: Any) -> pd.DataFrame:
    """Copy input without fitting a transformation or inspecting held-out labels."""
    if isinstance(X, pd.DataFrame):
        frame = X.copy(deep=True)
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        frame = pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])
    if frame.empty or not frame.columns.is_unique or any(not isinstance(c, str) for c in frame.columns):
        raise ValueError("X must have rows and unique string feature names.")
    return frame


class FoldPreprocessor(TransformerMixin, BaseEstimator):
    """Fit on training rows and labels only; transform takes no labels."""

    def __init__(self, *, scale: bool = True):
        self.scale = scale

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Feature names/order differ from training input.")
        prepared = frame.copy(deep=True)
        for column in self.numeric_columns_:
            values = pd.to_numeric(prepared[column], errors="raise").to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError("Infinite numerical features are unsupported.")
            prepared[column] = values
        for column in self.categorical_columns_:
            values = prepared[column].astype(object)
            present = ~pd.isna(values)
            if (values[present].astype(str) == MISSING_CATEGORY).any():
                raise ValueError(f"Reserved missing-category token {MISSING_CATEGORY!r} occurs in input.")
            prepared[column] = values.map(lambda v: np.nan if pd.isna(v) else str(v))
        return prepared

    def fit(self, X: Any, y: Any):
        for name in ("encoder_", "scaler_", "keep_mask_", "diagnostics_"):
            self.__dict__.pop(name, None)
        if not isinstance(self.scale, bool):
            raise ValueError("scale must be boolean.")
        frame = feature_frame(X)
        labels, classes = binary_labels(y, len(frame))
        if any(pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError("Complex numerical features are unsupported.")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        self.numeric_columns_ = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])
                                 and not pd.api.types.is_bool_dtype(frame[c])]
        self.categorical_columns_ = [c for c in frame if c not in self.numeric_columns_]
        prepared = self._prepare(frame)
        transformers = []
        if self.numeric_columns_:
            transformers.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
                                 self.numeric_columns_))
        if self.categorical_columns_:
            transformers.append(("categorical", make_pipeline(
                SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY, keep_empty_features=True),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)), self.categorical_columns_))
        encoder = ColumnTransformer(transformers, sparse_threshold=0)
        encoded = encoder.fit_transform(prepared)
        # Exact constancy avoids arbitrary variance thresholds and roundoff from scaling.
        keep = np.ones(encoded.shape[1], dtype=bool)
        for label in classes:
            within = encoded[labels == label]
            keep &= np.any(within != within[0], axis=0)
        if not keep.any():
            raise ValueError("No features remain after the within-either-class variance filter.")
        scaler = StandardScaler() if self.scale else None
        if scaler is not None:
            scaler.fit(encoded[:, keep])
        names = encoder.get_feature_names_out()
        self.encoder_, self.scaler_, self.keep_mask_ = encoder, scaler, keep
        self.diagnostics_ = {
            "policy": {**PREPROCESSING_POLICY, "scale_enabled": self.scale},
            "training_rows": len(frame), "input_features": self.n_features_in_,
            "encoded_features": len(keep), "output_features": int(keep.sum()),
            "retained_features": names[keep].tolist(), "dropped_features": names[~keep].tolist(),
            "numeric_columns": self.numeric_columns_, "categorical_columns": self.categorical_columns_,
            "numeric_imputation_values": (encoder.named_transformers_["numeric"].statistics_.tolist()
                                           if self.numeric_columns_ else []),
            "categorical_levels": ([v.tolist() for v in encoder.named_transformers_["categorical"]
                                     .named_steps["onehotencoder"].categories_]
                                    if self.categorical_columns_ else []),
            "scaler_mean": scaler.mean_.tolist() if scaler is not None else None,
            "scaler_scale": scaler.scale_.tolist() if scaler is not None else None,
        }
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "keep_mask_")
        encoded = self.encoder_.transform(self._prepare(feature_frame(X)))[:, self.keep_mask_]
        result = self.scaler_.transform(encoded) if self.scaler_ is not None else encoded
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced nonfinite values.")
        return np.array(result, dtype=float, copy=True)
