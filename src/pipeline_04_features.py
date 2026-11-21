# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-20
"""Pipeline 4: Vectorized NumPy feature creation and ML-ready table encoding."""

import numpy as np
import pandas as pd


def add_vectorized_features(
    df: pd.DataFrame, col_a: str | None = None, col_b: str | None = None
) -> pd.DataFrame:
    """Add vectorized log1p and safe ratio columns using NumPy ufuncs."""
    out = df.copy()
    num_cols = [
        c
        for c in out.select_dtypes(include=[np.number]).columns
        if not pd.api.types.is_bool_dtype(out[c])
    ]
    if len(num_cols) < 2:
        return out

    primary = col_a if col_a in num_cols else num_cols[0]
    secondary = (
        col_b
        if col_b in num_cols and col_b != primary
        else next(c for c in num_cols if c != primary)
    )

    arr_a = pd.to_numeric(out[primary], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    arr_b = pd.to_numeric(out[secondary], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    out[f"log1p_{primary}"] = np.log1p(np.maximum(arr_a, 0.0))
    out[f"ratio_{primary}_to_{secondary}"] = np.where(np.abs(arr_b) > 1e-9, arr_a / arr_b, 0.0)
    return out


def build_ml_ready_table(
    df: pd.DataFrame,
    target_col: str | None = None,
    max_onehot_cardinality: int = 15,
) -> pd.DataFrame:
    """Expand datetime fields and one-hot encode low-cardinality categories for model training."""
    ml_df = df.copy()
    candidates = [c for c in ml_df.columns if c != target_col]

    for col in candidates:
        if pd.api.types.is_datetime64_any_dtype(ml_df[col]):
            ml_df[f"{col}_year"] = ml_df[col].dt.year.fillna(0).astype(int)
            ml_df[f"{col}_month"] = ml_df[col].dt.month.fillna(0).astype(int)
            ml_df[f"{col}_dayofweek"] = ml_df[col].dt.dayofweek.fillna(0).astype(int)
            ml_df = ml_df.drop(columns=[col])

    cat_cols = [
        c
        for c in candidates
        if c in ml_df.columns
        and (pd.api.types.is_object_dtype(ml_df[c]) or pd.api.types.is_string_dtype(ml_df[c]))
        and ml_df[c].nunique(dropna=True) <= max_onehot_cardinality
    ]
    if cat_cols:
        ml_df = pd.get_dummies(ml_df, columns=cat_cols, drop_first=False, dtype=int)

    return ml_df