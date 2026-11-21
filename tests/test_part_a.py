# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-19
"""Unit tests for Part A cleaning and validation functions."""

import pandas as pd
import pytest

from src.pipeline_02_validate import (
    QualityGateFailedError,
    run_quality_gate,
    validate_no_nulls,
    validate_numeric_range,
    validate_required_columns,
    validate_uniqueness,
)
from src.pipeline_03_preprocess import (
    fix_types_and_missing,
    handle_outliers_iqr,
    remove_duplicate_rows,
    standardize_categories,
)
from src.pipeline_04_features import add_vectorized_features


def test_remove_duplicate_rows():
    df = pd.DataFrame({"id": [1, 1, 2], "val": [5, 5, 10]})
    out, removed = remove_duplicate_rows(df)
    assert removed == 1
    assert len(out) == 2


def test_standardize_categories():
    df = pd.DataFrame({"city": ["  peshawar ", "N/A", None, "lahore"]})
    out = standardize_categories(df, ["city"])
    assert out["city"].tolist() == ["Peshawar", "Unknown", "Unknown", "Lahore"]


def test_fix_types_and_missing():
    df = pd.DataFrame({"price": ["120.5", None, "bad"], "created_at": ["2026-01-01", "2026-01-02", "bad_date"]})
    out = fix_types_and_missing(df, num_cols=["price"], date_cols=["created_at"])
    assert out["price"].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(out["created_at"])


def test_handle_outliers_iqr():
    df = pd.DataFrame({"amount": [10, 12, 11, 13, 12, 9999]})
    out, clipped = handle_outliers_iqr(df, "amount")
    assert clipped >= 1
    assert out["amount"].max() < 9999


def test_add_vectorized_features():
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 0.0]})
    out = add_vectorized_features(df, "a", "b")
    assert "log1p_a" in out.columns
    assert "ratio_a_to_b" in out.columns
    assert out["ratio_a_to_b"].iloc[1] == 0.0


def test_validations_and_quality_gate(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "score": [10.0, 25.0]})
    validate_required_columns(df, ["id", "score"])
    validate_no_nulls(df, ["id", "score"])
    validate_uniqueness(df, "id")
    mask = validate_numeric_range(df, "score", 0.0, 50.0)
    assert mask.all()

    metrics_file = tmp_path / "metrics.json"
    with pytest.raises(QualityGateFailedError):
        run_quality_gate(pd.Series([True, False]), {"dups": 0}, threshold=0.9, metrics_path=metrics_file)