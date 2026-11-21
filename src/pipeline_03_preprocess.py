# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-20
"""Pipeline 3: Reusable dataset cleaning for duplicates, types, missing values, and outliers."""

import numpy as np
import pandas as pd


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows and return (cleaned_df, removed_count)."""
    before = len(df)
    cleaned = df.drop_duplicates().copy()
    return cleaned, before - len(cleaned)


def standardize_categories(df: pd.DataFrame, cat_cols: list[str] | None = None) -> pd.DataFrame:
    """Normalize text casing/whitespace and map missing or placeholder tokens to 'Unknown'."""
    out = df.copy()
    cols = (
        cat_cols
        if cat_cols is not None
        else out.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    )
    placeholders = {"", "nan", "none", "null", "n/a", "na", "undefined", "unknown"}
    for col in cols:
        if col in out.columns:
            s = out[col].astype("object").where(out[col].notna(), "Unknown").astype(str).str.strip()
            s = s.where(~s.str.lower().isin(placeholders), "Unknown")
            out[col] = s.str.title()
    return out


def _is_datetime_like(series: pd.Series) -> bool:
    """Detect whether a column represents dates or timestamps."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= 0.70


def fix_types_and_missing(
    df: pd.DataFrame,
    num_cols: list[str] | None = None,
    date_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Coerce numeric and datetime columns and impute missing numeric values with median."""
    out = df.copy()
    detected_dates = (
        list(date_cols)
        if date_cols is not None
        else [c for c in out.columns if _is_datetime_like(out[c])]
    )
    for col in detected_dates:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if num_cols is not None:
        numeric_cols = [c for c in num_cols if c in out.columns and c not in detected_dates]
    else:
        numeric_cols = []
        for col in out.columns:
            if col in detected_dates:
                continue
            if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
                numeric_cols.append(col)
            elif pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
                non_empty = out[col].dropna().astype(str).str.strip()
                non_empty = non_empty[non_empty != ""]
                if not non_empty.empty:
                    parsed_num = pd.to_numeric(non_empty.head(100), errors="coerce")
                    if float(parsed_num.notna().mean()) >= 0.80:
                        numeric_cols.append(col)

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        med = out[col].median()
        fill_val = 0.0 if pd.isna(med) else float(med)
        out[col] = out[col].fillna(fill_val)

    return out


def handle_outliers_iqr(
    df: pd.DataFrame, col: str, factor: float = 1.5
) -> tuple[pd.DataFrame, int]:
    """Clip numeric values outside [Q1 - factor*IQR, Q3 + factor*IQR] and return clipped count."""
    out = df.copy()
    if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
        return out, 0
    q1 = out[col].quantile(0.25)
    q3 = out[col].quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return out, 0
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    clipped = int(((out[col] < low) | (out[col] > high)).sum())
    out[col] = out[col].clip(lower=low, upper=high)
    return out, clipped


def preprocess_dataset(
    df_dirty: pd.DataFrame,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    date_cols: list[str] | None = None,
    outlier_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run cleaning steps on any tabular dataset and return (cleaned_df, metrics)."""
    rows_before = len(df_dirty)
    cells_before = max(df_dirty.shape[0] * max(df_dirty.shape[1], 1), 1)
    missing_before = int(df_dirty.isna().sum().sum())

    df1, dupes_removed = remove_duplicate_rows(df_dirty)
    df2 = fix_types_and_missing(df1, num_cols=num_cols, date_cols=date_cols)
    cats = (
        cat_cols
        if cat_cols is not None
        else df2.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    )
    df_clean = standardize_categories(df2, cats)

    num_list = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    total_clipped = 0
    tracked_col = outlier_col

    if outlier_col and outlier_col in num_list:
        df_clean, total_clipped = handle_outliers_iqr(df_clean, outlier_col)
    elif num_list:
        candidates = [
            c for c in num_list if not c.lower().endswith(("id", "_code", "zip", "year"))
        ] or num_list
        tracked_col = ", ".join(candidates[:3])
        for col in candidates:
            df_clean, cnt = handle_outliers_iqr(df_clean, col)
            total_clipped += cnt
    else:
        tracked_col = "None"

    missing_after = int(df_clean.isna().sum().sum())
    cells_after = max(df_clean.shape[0] * max(df_clean.shape[1], 1), 1)

    metrics = {
        "rows_before": rows_before,
        "rows_after": len(df_clean),
        "rows_rejected": max(rows_before - len(df_clean), 0),
        "duplicates_removed": dupes_removed,
        "missing_cells_before": missing_before,
        "missing_cells_after": missing_after,
        "missing_pct_before": round((missing_before / cells_before) * 100.0, 2),
        "missing_pct_after": round((missing_after / cells_after) * 100.0, 2),
        "outliers_clipped": total_clipped,
        "outlier_column": tracked_col or "None",
    }
    return df_clean, metrics