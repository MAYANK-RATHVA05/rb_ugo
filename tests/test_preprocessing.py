import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.preprocessing import FoldPreprocessor, MISSING_CATEGORY


def training():
    return pd.DataFrame({"value": [0., 2., 4., 6., np.nan, 10.],
                         "category": ["a", "b", "a", "b", None, "a"],
                         "class_constant": [0, 0, 0, 1, 2, 3],
                         "empty": [np.nan] * 6}), np.array([0, 0, 0, 1, 1, 1])


def test_training_only_imputation_categories_scaling_and_filter():
    X, y = training()
    original = X.copy(deep=True)
    model = FoldPreprocessor().fit(X, y)
    assert model.diagnostics_["numeric_imputation_values"] == [4., .5, 0.]
    assert "numeric__class_constant" in model.diagnostics_["dropped_features"]
    assert "numeric__empty" in model.diagnostics_["dropped_features"]
    assert "numeric__value" in model.diagnostics_["retained_features"]
    assert "categorical__category_a" in model.diagnostics_["retained_features"]
    before = model.diagnostics_.copy()
    heldout = pd.DataFrame({"value": [10000., np.nan], "category": ["never_seen", "b"],
                            "class_constant": [999, 888], "empty": [42., 20.]})
    transformed = model.transform(heldout)
    # Training median and scaler, even with an extreme held-out value/new level.
    position = model.diagnostics_["retained_features"].index("numeric__value")
    expected = (np.array([10000., 4.]) - model.scaler_.mean_[position]) / model.scaler_.scale_[position]
    np.testing.assert_allclose(transformed[:, position], expected)
    assert model.diagnostics_ == before
    assert "never_seen" not in model.diagnostics_["categorical_levels"][0]
    pd.testing.assert_frame_equal(X, original)


def test_filter_uses_either_training_class_before_scaling():
    X = [[1., 1.], [1., 2.], [1., 3.], [2., 4.], [3., 5.], [4., 6.]]
    model = FoldPreprocessor(scale=False).fit(X, [0, 0, 0, 1, 1, 1])
    assert model.keep_mask_.tolist() == [False, True]
    np.testing.assert_array_equal(model.transform([[999., 100.]]), [[100.]])


def test_nullable_numeric_boolean_and_missing_categories():
    frame = pd.DataFrame({"number": pd.Series([1, 2, None, 4, 5, 6], dtype="Float64"),
                          "boolean": pd.Series([True, False, None, True, False, None], dtype="boolean")})
    result = FoldPreprocessor().fit_transform(frame, [0, 0, 0, 1, 1, 1])
    assert result.shape == (6, 4)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("X,match", [([[1], [1], [1], [2]], "No features"),
                                    ([[1], [np.inf], [3], [4]], "Infinite"),
                                    ([[1j], [2j], [3j], [4j]], "Complex")])
def test_infeasible_features_fail(X, match):
    with pytest.raises(ValueError, match=match):
        FoldPreprocessor().fit(X, [0, 0, 1, 1])


def test_schema_and_reserved_category_fail():
    X, y = training()
    model = FoldPreprocessor().fit(X, y)
    with pytest.raises(ValueError, match="names/order"):
        model.transform(X[X.columns[::-1]])
    X.loc[0, "category"] = MISSING_CATEGORY
    with pytest.raises(ValueError, match="Reserved"):
        FoldPreprocessor().fit(X, y)


def test_failed_refit_cannot_expose_previous_transformation():
    X, y = training()
    model = FoldPreprocessor().fit(X, y)
    with pytest.raises(ValueError):
        model.fit(np.ones((6, 2)), y)
    with pytest.raises(NotFittedError):
        model.transform(X)
