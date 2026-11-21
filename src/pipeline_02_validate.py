# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-20
"""Pipeline 2: Schema documentation, loud validation checks, and quality gate."""

import json
from pathlib import Path
from typing import Iterable
import pandas as pd


class QualityGateFailedError(RuntimeError):
    """Raised when dataset validation pass rate falls below threshold."""


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table with safe fallback if tabulate is absent."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = [str(col) for col in df.columns]
        sep = ["---"] * len(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in df.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(val) for val in row) + " |")
        return "\n".join(lines)


def generate_schema_report(
    csv_path: str | Path, schema_output_path: str | Path = "docs/schema.md"
) -> pd.DataFrame:
    """Inspect raw CSV schema and write docs/schema.md without touching raw data."""
    df = pd.read_csv(csv_path)
    rows = []
    for col in df.columns:
        missing_cnt = int(df[col].isna().sum())
        missing_pct = (missing_cnt / max(len(df), 1)) * 100.0
        rows.append(
            {
                "Column": col,
                "Dtype": str(df[col].dtype),
                "Missing Count": missing_cnt,
                "Missing %": f"{missing_pct:.2f}%",
                "Unique Count": int(df[col].nunique(dropna=True)),
            }
        )

    schema_df = pd.DataFrame(rows)
    out_file = Path(schema_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    md_content = (
        "# Dataset Schema\n\n"
        f"- **Rows:** {len(df):,}\n"
        f"- **Columns:** {len(df.columns):,}\n"
        "- **Raw Integrity:** Files in `data/raw/` remain untouched.\n\n"
        f"{df_to_markdown_table(schema_df)}\n"
    )
    out_file.write_text(md_content, encoding="utf-8")
    return schema_df


def validate_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    """Raise ValueError if dataset is empty or required columns are missing."""
    if df.empty:
        raise ValueError("Validation failed: dataset contains zero rows.")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: missing required columns {missing}")


def validate_no_nulls(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Raise ValueError if any specified column has missing values."""
    nulls = {c: int(df[c].isna().sum()) for c in cols if int(df[c].isna().sum()) > 0}
    if nulls:
        raise ValueError(f"Validation failed: null values detected in {nulls}")


def validate_uniqueness(df: pd.DataFrame, col: str) -> None:
    """Raise ValueError if identifier column contains duplicate entries."""
    if col not in df.columns:
        raise ValueError(f"Validation failed: column '{col}' not found.")
    dups = int(df[col].duplicated().sum())
    if dups > 0:
        raise ValueError(f"Validation failed: {dups} duplicate values in '{col}'.")


def validate_numeric_range(
    df: pd.DataFrame, col: str, min_val: float | None = None, max_val: float | None = None
) -> pd.Series:
    """Return boolean mask of rows falling within [min_val, max_val]."""
    if col not in df.columns:
        raise ValueError(f"Validation failed: column '{col}' not found.")
    s = pd.to_numeric(df[col], errors="coerce")
    mask = s.notna()
    if min_val is not None:
        mask &= s >= min_val
    if max_val is not None:
        mask &= s <= max_val
    return mask


def run_quality_gate(
    row_valid_mask: pd.Series,
    metrics: dict,
    threshold: float = 0.90,
    metrics_path: str | Path = "outputs/metrics.json",
) -> dict:
    """Persist quality gate metrics to JSON and halt if pass rate is below threshold."""
    total = max(len(row_valid_mask), 1)
    passed = int(row_valid_mask.sum())
    pass_rate = passed / total

    payload = {
        **metrics,
        "rows_checked": int(len(row_valid_mask)),
        "rows_passed": passed,
        "validation_pass_rate": round(pass_rate, 6),
        "threshold": threshold,
        "gate_passed": bool(pass_rate >= threshold),
    }

    out = Path(metrics_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if pass_rate < threshold:
        raise QualityGateFailedError(
            f"Quality gate failed: {pass_rate:.2%} pass rate is below {threshold:.2%} threshold."
        )
    return payload