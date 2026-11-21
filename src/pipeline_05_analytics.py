# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-20
"""Pipeline 5: Pandas summaries, SQLite GROUP BY query, and saved visualizations."""

from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd


def compute_pandas_summaries(
    df: pd.DataFrame, cat_col: str, num_col: str
) -> dict[str, pd.DataFrame]:
    """Compute grouped statistics, value counts, and pivot table using Pandas."""
    grouped = (
        df.groupby(cat_col, dropna=False)[num_col]
        .agg(count="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    counts = df[cat_col].value_counts(dropna=False).reset_index()
    counts.columns = [cat_col, "count"]
    pivot = pd.pivot_table(
        df,
        values=num_col,
        index=cat_col,
        aggfunc=["mean", "median"],
        fill_value=0.0,
    )
    return {"grouped": grouped, "value_counts": counts, "pivot": pivot}


def run_sqlite_query(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    db_path: str | Path = "sql/part_a.db",
) -> tuple[str, pd.DataFrame]:
    """Load cleaned records into SQLite and execute a GROUP BY aggregation query."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    safe_cat = f'"{cat_col}"'
    safe_num = f'"{num_col}"'

    query = f"""
        SELECT
            {safe_cat} AS category_group,
            COUNT(*) AS record_count,
            ROUND(AVG({safe_num}), 2) AS avg_value,
            ROUND(SUM({safe_num}), 2) AS total_value
        FROM cleaned_records
        GROUP BY {safe_cat}
        ORDER BY record_count DESC
        LIMIT 10;
    """.strip()

    with sqlite3.connect(db_file) as conn:
        df.to_sql("cleaned_records", conn, if_exists="replace", index=False)
        result_df = pd.read_sql_query(query, conn)

    return query, result_df


def save_visualizations(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    units: str = "units",
    figures_dir: str | Path = "figures",
) -> list[str]:
    """Save 3 charts with clear titles, unit-labeled axes, and high-contrast styling."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Chart 1: Category volume
    p1 = out_dir / "fig1_top_categories.png"
    plt.figure(figsize=(8.5, 4.5))
    df[cat_col].value_counts().head(8).plot(kind="bar", color="#06B6D4", edgecolor="#0F172A")
    plt.title(f"Top {cat_col} Record Counts")
    plt.xlabel(cat_col)
    plt.ylabel("Records (count)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    plt.close()
    saved.append(str(p1))

    # Chart 2: Numeric distribution
    p2 = out_dir / "fig2_numeric_distribution.png"
    plt.figure(figsize=(8.5, 4.5))
    plt.hist(
        pd.to_numeric(df[num_col], errors="coerce").dropna(),
        bins=25,
        color="#0EA5E9",
        edgecolor="#0F172A",
    )
    plt.title(f"Distribution of {num_col}")
    plt.xlabel(f"{num_col} ({units})")
    plt.ylabel("Records (count)")
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    plt.close()
    saved.append(str(p2))

    # Chart 3: Category mean
    p3 = out_dir / "fig3_mean_by_category.png"
    plt.figure(figsize=(8.5, 4.5))
    df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(8).plot(
        kind="barh", color="#10B981", edgecolor="#0F172A"
    )
    plt.title(f"Mean {num_col} by {cat_col}")
    plt.xlabel(f"Mean {num_col} ({units})")
    plt.ylabel(cat_col)
    plt.tight_layout()
    plt.savefig(p3, dpi=200)
    plt.close()
    saved.append(str(p3))

    return saved