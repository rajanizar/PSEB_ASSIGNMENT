# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-20
"""Single-command CLI runner for reusable Part A data preparation."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.pipeline_01_ingest import ingest_raw_csv
from src.pipeline_02_validate import (
    df_to_markdown_table,
    generate_schema_report,
    run_quality_gate,
    validate_numeric_range,
    validate_required_columns,
)
from src.pipeline_03_preprocess import preprocess_dataset
from src.pipeline_04_features import add_vectorized_features, build_ml_ready_table
from src.pipeline_05_analytics import (
    compute_pandas_summaries,
    run_sqlite_query,
    save_visualizations,
)


def run_part_a(
    input_csv: str | Path,
    threshold: float = 0.90,
    units: str = "units",
    target_col: str | None = None,
) -> pd.DataFrame:
    """Execute full Part A pipeline on any dataset and save outputs."""
    raw_path = ingest_raw_csv(input_csv)
    generate_schema_report(raw_path)

    df_dirty = pd.read_csv(raw_path)
    validate_required_columns(df_dirty, list(df_dirty.columns[: min(2, len(df_dirty.columns))]))

    df_clean, clean_metrics = preprocess_dataset(df_dirty)
    df_feat = add_vectorized_features(df_clean)
    df_ml = build_ml_ready_table(df_feat, target_col=target_col)

    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_feat.select_dtypes(include=["object", "string"]).columns.tolist()
    num_col = num_cols[0] if num_cols else df_feat.columns[0]
    cat_col = cat_cols[0] if cat_cols else df_feat.columns[0]

    valid_mask = (
        validate_numeric_range(df_feat, num_col)
        if num_col in num_cols
        else pd.Series(True, index=df_feat.index)
    )
    gate_stats = run_quality_gate(valid_mask, clean_metrics, threshold=threshold)

    out_dir = Path("data/cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_file = out_dir / "cleaned_data.csv"
    ml_file = out_dir / "model_ready_data.csv"
    df_feat.to_csv(cleaned_file, index=False)
    df_ml.to_csv(ml_file, index=False)

    compute_pandas_summaries(df_feat, cat_col, num_col)
    sql_q, sql_res = run_sqlite_query(df_feat, cat_col, num_col)
    save_visualizations(df_feat, cat_col, num_col, units=units)

    report_file = Path("reports/data_quality_report.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        "# Part A Data Quality Report\n\n"
        f"- **Rows Before -> After:** {clean_metrics['rows_before']:,} -> {clean_metrics['rows_after']:,}\n"
        f"- **Columns (Raw -> Cleaned -> ML-Ready):** {df_dirty.shape[1]:,} -> {df_feat.shape[1]:,} -> {df_ml.shape[1]:,}\n"
        f"- **Duplicates Removed:** {clean_metrics['duplicates_removed']:,}\n"
        f"- **Missing Cells Before -> After:** {clean_metrics['missing_cells_before']:,} ({clean_metrics['missing_pct_before']}%) "
        f"-> {clean_metrics['missing_cells_after']:,} ({clean_metrics['missing_pct_after']}%)\n"
        f"- **Outliers Clipped ({clean_metrics['outlier_column']}):** {clean_metrics['outliers_clipped']:,}\n"
        f"- **Validation Pass Rate:** {gate_stats['validation_pass_rate']:.2%}\n\n"
        "## SQLite Query & Summary\n"
        f"```sql\n{sql_q}\n```\n\n"
        f"{df_to_markdown_table(sql_res)}\n",
        encoding="utf-8",
    )
    return df_ml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reusable Part A pipeline.")
    parser.add_argument("--input", required=True, help="Path to raw .csv file")
    parser.add_argument("--threshold", type=float, default=0.90, help="Quality gate threshold")
    parser.add_argument("--units", type=str, default="units", help="Units label for charts")
    parser.add_argument("--target", type=str, default=None, help="Optional target column")
    args = parser.parse_args()
    run_part_a(args.input, threshold=args.threshold, units=args.units, target_col=args.target)