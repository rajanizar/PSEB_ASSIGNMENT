# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-21
"""Portable Streamlit interface for reusable ML data preparation, SQLite analytics, and Part B modeling."""

from pathlib import Path
import re
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.pipeline_01_ingest import convert_to_csv, ingest_raw_csv
from src.pipeline_02_validate import generate_schema_report, validate_required_columns
from src.pipeline_03_preprocess import preprocess_dataset
from src.pipeline_04_features import add_vectorized_features, build_ml_ready_table
from src.pipeline_05_analytics import run_sqlite_query, save_visualizations
from src.pipeline_06_ml_baseline import resolve_final_clean_dataset, train_validate_and_test_models

RAW_DIR = ROOT_DIR / "data" / "raw"
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"
FIGURES_DIR = ROOT_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
SQL_DIR = ROOT_DIR / "sql"
DOCS_DIR = ROOT_DIR / "docs"
PSEB_DATA_DIR = ROOT_DIR / "PSEB Assignment 2" / "data"

for folder in (RAW_DIR, CLEANED_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SQL_DIR, DOCS_DIR, PSEB_DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

STAGES = [
    "1. Ingest & Convert",
    "2. Dirty Data & Preprocessing",
    "3. Feature Engineering (ML)",
    "4. Before vs. After (ML)",
    "5. Charts & SQLite",
    "6. Part B: Train 3 ML Models",
]

st.set_page_config(
    page_title="ML Data Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0B0F17;
        color: #F9FAFB;
    }

    header[data-testid="stHeader"] {
        background-color: #0B0F17;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #263244 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #0F172A !important;
        border: 1px solid #1F2937 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin-bottom: 6px !important;
        color: #F3F4F6 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: rgba(6, 182, 212, 0.18) !important;
        border-color: #06B6D4 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #111827 !important;
        border: 1.5px dashed #06B6D4 !important;
        border-radius: 12px !important;
        padding: 18px !important;
    }

    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section div {
        color: #F3F4F6 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #1E293B !important;
        color: #F9FAFB !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    .top-bar {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }

    .top-kicker {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #22D3EE;
        margin-bottom: 4px;
    }

    .top-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F9FAFB;
        margin: 0;
    }

    .top-sub {
        font-size: 0.88rem;
        color: #CBD5E1;
        margin-top: 4px;
    }

    .stat-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 14px 16px;
    }

    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9CA3AF;
    }

    .stat-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F9FAFB;
        margin-top: 4px;
    }

    .stat-note {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    .dataset-row {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 9px 12px;
        margin-bottom: 6px;
        font-size: 0.88rem;
        color: #F3F4F6;
    }

    .badge-raw {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        background-color: rgba(245, 158, 11, 0.16);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .badge-clean {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        background-color: rgba(16, 185, 129, 0.16);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .badge-info {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        background-color: rgba(6, 182, 212, 0.16);
        color: #67E8F9;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }

    .caption-box {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-left: 3px solid #06B6D4;
        border-radius: 8px;
        padding: 11px 14px;
        font-size: 0.88rem;
        color: #E5E7EB;
        margin-top: 8px;
    }

    .stButton > button,
    .stDownloadButton > button {
        background-color: #06B6D4 !important;
        color: #061018 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.52rem 1.1rem !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #22D3EE !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1F2937;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def stat_box(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="stat-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def style_chart(fig, height: int = 330):
    fig.update_layout(
        height=height,
        paper_bgcolor="#111827",
        plot_bgcolor="#0F172A",
        font=dict(color="#E5E7EB", family="Inter, sans-serif", size=12),
        margin=dict(l=18, r=18, t=54, b=18),
    )
    fig.update_xaxes(gridcolor="#1F2937", zerolinecolor="#374151")
    fig.update_yaxes(gridcolor="#1F2937", zerolinecolor="#374151")
    return fig


def save_to_pseb_data_dir(df: pd.DataFrame, filename: str) -> Path:
    """Save a cleaned/processed CSV directly into PSEB Assignment 2/data/ on disk."""
    PSEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"
    dest_path = PSEB_DATA_DIR / safe_name
    df.to_csv(dest_path, index=False)
    return dest_path


def _on_clean_download(df_to_save: pd.DataFrame, out_filename: str, state_key: str) -> None:
    saved = save_to_pseb_data_dir(df_to_save, out_filename)
    st.session_state[state_key] = str(saved.relative_to(ROOT_DIR))


def save_chart_to_figures_dir(
    fig,
    filename: str,
    chart_style: str = "Bar Chart",
    data_df: pd.DataFrame | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
) -> Path:
    """Save a chart into figures/ as PNG across any OS (uses Plotly image export if available, with Matplotlib fallback)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_") or "chart"
    out_path = FIGURES_DIR / f"{stem}.png"

    try:
        fig.write_image(str(out_path), width=1100, height=600, scale=2)
        return out_path
    except Exception:
        fig_mpl, ax = plt.subplots(figsize=(10, 5))
        fig_mpl.patch.set_facecolor("#111827")
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#E5E7EB")
        ax.xaxis.label.set_color("#E5E7EB")
        ax.yaxis.label.set_color("#E5E7EB")
        ax.title.set_color("#F9FAFB")

        if data_df is not None and not data_df.empty and x_col and y_col and x_col in data_df.columns and y_col in data_df.columns:
            x_vals = data_df[x_col].astype(str).tolist()
            y_vals = pd.to_numeric(data_df[y_col], errors="coerce").fillna(0.0).tolist()
            if chart_style == "Line Chart":
                ax.plot(x_vals, y_vals, marker="o", color="#22D3EE", linewidth=2)
            elif chart_style == "Pie / Donut":
                ax.pie(y_vals, labels=x_vals, autopct="%1.1f%%", textprops={"color": "#E5E7EB"})
            else:
                ax.bar(x_vals, y_vals, color="#06B6D4")
            if chart_style != "Pie / Donut":
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                plt.xticks(rotation=30, ha="right")
            ax.set_title(f"{y_col} by {x_col}")
        else:
            ax.text(0.5, 0.5, stem, color="#E5E7EB", ha="center", va="center", fontsize=14)

        fig_mpl.tight_layout()
        fig_mpl.savefig(out_path, dpi=150, facecolor=fig_mpl.get_facecolor())
        plt.close(fig_mpl)
        return out_path


def get_active_raw_csv(raw_files: list[Path]) -> Path | None:
    """Resolve the active CSV consistently across all pipeline stages."""
    if not raw_files:
        return None
    sel_str = st.session_state.get("selected_csv")
    if sel_str:
        for fpath in raw_files:
            if str(fpath) == sel_str:
                return fpath
    st.session_state["selected_csv"] = str(raw_files[0])
    return raw_files[0]


def _pick_best_numeric_pair(df: pd.DataFrame, num_cols: list[str]) -> tuple[str, str] | None:
    """Find the pair of numeric columns with the strongest correlation magnitude."""
    if len(num_cols) < 2:
        return None
    corr_mat = df[num_cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True).abs()
    best_pair = (num_cols[0], num_cols[1])
    best_val = -1.0
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1 :]:
            val = corr_mat.loc[c1, c2] if (c1 in corr_mat.index and c2 in corr_mat.columns) else np.nan
            if pd.notna(val) and float(val) > best_val:
                best_val = float(val)
                best_pair = (c1, c2)
    return best_pair


def _pick_relatable_category(df: pd.DataFrame) -> str | None:
    """Pick a categorical column with readable groups (avoiding unique IDs or long free text)."""
    candidates = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    best_col = None
    best_score = -10.0
    n_rows = max(len(df), 1)
    for col in candidates:
        uniq = df[col].nunique(dropna=True)
        if uniq < 2:
            continue
        uniq_ratio = uniq / n_rows
        avg_len = df[col].dropna().astype(str).str.len().mean() if df[col].notna().any() else 0.0
        score = 0.0
        if 2 <= uniq <= 25:
            score += 5.0
        elif uniq_ratio > 0.85:
            score -= 5.0
        if avg_len > 45:
            score -= 3.0
        if score > best_score:
            best_score = score
            best_col = col
    return best_col if best_col is not None else (candidates[0] if candidates else None)


def _render_dataset_insight_charts(
    df: pd.DataFrame,
    prefix: str,
    accent_color: str,
    secondary_color: str,
    stage_label: str,
) -> None:
    """Render dynamic relationship & correlation charts with plain-language takeaways."""
    cats = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [
        c
        for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if not pd.api.types.is_bool_dtype(df[c])
    ]
    best_cat = _pick_relatable_category(df)
    best_pair = _pick_best_numeric_pair(df, nums)

    g1, g2 = st.columns(2)

    with g1:
        if best_cat and nums:
            c_idx = cats.index(best_cat) if best_cat in cats else 0
            sel_cat = st.selectbox("Group / Category column", cats, index=c_idx, key=f"{prefix}_g1_cat")
            sel_num = st.selectbox("Measured number column", nums, index=0, key=f"{prefix}_g1_num")

            grp_df = (
                df.assign(_grp=df[sel_cat].fillna("Missing").astype(str))
                .groupby("_grp", as_index=False)[sel_num]
                .mean()
                .sort_values(sel_num, ascending=False)
                .head(8)
            )
            fig1 = px.bar(
                grp_df,
                x="_grp",
                y=sel_num,
                title=f"{stage_label}: Avg {sel_num} by {sel_cat}",
                labels={"_grp": sel_cat, sel_num: f"Average {sel_num}"},
                color_discrete_sequence=[accent_color],
            )
            st.plotly_chart(style_chart(fig1, 290), use_container_width=True)
            if not grp_df.empty:
                top_g = grp_df.iloc[0]
                st.markdown(
                    f'<div class="caption-box"><b>What this tells you:</b> In the <b>{stage_label.lower()}</b> data, <b>{top_g["_grp"]}</b> has the highest average <b>{sel_num}</b> ({top_g[sel_num]:,.2f}) across top <b>{sel_cat}</b> groups.</div>',
                    unsafe_allow_html=True,
                )
        elif len(nums) >= 2:
            ia = nums.index(best_pair[0]) if best_pair else 0
            ib = nums.index(best_pair[1]) if best_pair and len(nums) > 1 else 0
            cx = st.selectbox("First numeric column (X)", nums, index=ia, key=f"{prefix}_g1_nx")
            cy = st.selectbox("Second numeric column (Y)", nums, index=ib, key=f"{prefix}_g1_ny")

            pair_df = df[[cx, cy]].dropna()
            r_val = float(pair_df[cx].corr(pair_df[cy])) if len(pair_df) > 1 else 0.0
            fig1 = px.scatter(
                pair_df.head(500),
                x=cx,
                y=cy,
                title=f"{stage_label}: {cx} vs. {cy} (r = {r_val:.2f})",
                color_discrete_sequence=[accent_color],
            )
            st.plotly_chart(style_chart(fig1, 290), use_container_width=True)
            rel_txt = (
                "move upward together (positive correlation)"
                if r_val >= 0.5
                else "move in opposite directions (negative correlation)"
                if r_val <= -0.5
                else "show a weak or scattered relationship"
            )
            st.markdown(
                f'<div class="caption-box"><b>What this tells you:</b> <b>{cx}</b> and <b>{cy}</b> have a correlation of <b>{r_val:.2f}</b>, meaning they {rel_txt}.</div>',
                unsafe_allow_html=True,
            )
        else:
            null_df = df.isna().sum().reset_index()
            null_df.columns = ["Column", "Missing Rows"]
            null_df = null_df.sort_values("Missing Rows", ascending=False).head(8)
            fig1 = px.bar(
                null_df,
                x="Column",
                y="Missing Rows",
                title=f"{stage_label}: Missing Values by Column",
                color_discrete_sequence=[accent_color],
            )
            st.plotly_chart(style_chart(fig1, 290), use_container_width=True)
            worst = null_df.iloc[0]
            st.markdown(
                f'<div class="caption-box"><b>What this tells you:</b> <b>{worst["Column"]}</b> has <b>{int(worst["Missing Rows"]):,}</b> missing rows in this stage.</div>',
                unsafe_allow_html=True,
            )

    with g2:
        if len(nums) >= 2:
            corr_sub = df[nums].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
            fig2 = px.imshow(
                corr_sub,
                text_auto=".2f",
                aspect="auto",
                title=f"{stage_label}: Column Correlation Matrix",
                color_continuous_scale=["#1E293B", secondary_color, accent_color],
                zmin=-1,
                zmax=1,
            )
            st.plotly_chart(style_chart(fig2, 290), use_container_width=True)
            if best_pair:
                ca, cb = best_pair
                score = float(corr_sub.loc[ca, cb]) if ca in corr_sub.index and cb in corr_sub.columns else 0.0
                st.markdown(
                    f'<div class="caption-box"><b>What this tells you:</b> Values near <b>+1</b> or <b>-1</b> show strong links between columns; the strongest link here is between <b>{ca}</b> and <b>{cb}</b> (<b>{score:.2f}</b>).</div>',
                    unsafe_allow_html=True,
                )
        elif cats:
            c_idx2 = cats.index(best_cat) if best_cat in cats else 0
            c_share = st.selectbox("Top groups by column", cats, index=c_idx2, key=f"{prefix}_g2_cat")
            vc_df = (
                df[c_share]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(7)
                .reset_index()
            )
            vc_df.columns = ["Group", "Records"]
            fig2 = px.bar(
                vc_df,
                x="Records",
                y="Group",
                orientation="h",
                title=f"{stage_label}: Top {c_share} Groups",
                color_discrete_sequence=[secondary_color],
            )
            fig2.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_chart(fig2, 290), use_container_width=True)
            top_c = vc_df.iloc[0]
            st.markdown(
                f'<div class="caption-box"><b>What this tells you:</b> <b>{top_c["Group"]}</b> accounts for the largest share ({int(top_c["Records"]):,} rows) in <b>{c_share}</b>.</div>',
                unsafe_allow_html=True,
            )
        else:
            only_n = nums[0]
            fig2 = px.histogram(
                df,
                x=only_n,
                nbins=20,
                title=f"{stage_label}: Distribution of {only_n}",
                color_discrete_sequence=[secondary_color],
            )
            st.plotly_chart(style_chart(fig2, 290), use_container_width=True)
            med_v = float(pd.to_numeric(df[only_n], errors="coerce").median())
            st.markdown(
                f'<div class="caption-box"><b>What this tells you:</b> The typical (median) value for <b>{only_n}</b> is <b>{med_v:,.2f}</b>.</div>',
                unsafe_allow_html=True,
            )


def _detect_type_and_metric_columns(
    df: pd.DataFrame,
    tokens: list[str],
    brand_col: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Detect vehicle/entity type column, units/volume column, and revenue/sales column."""
    cats = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [
        c
        for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if not pd.api.types.is_bool_dtype(df[c])
    ]

    type_keywords = [
        "type",
        "vehicle_type",
        "body",
        "style",
        "segment",
        "model",
        "category",
        "class",
        "trim",
        "variant",
        "product",
    ]
    type_col = None
    best_t_score = -10.0
    for col in cats:
        if col == brand_col:
            continue
        low = col.lower()
        uniq = df[col].nunique(dropna=True)
        if uniq < 1:
            continue
        score = 0.0
        for kw in type_keywords:
            if kw in low:
                score += 6.0
        if 2 <= uniq <= 40:
            score += 3.0
        if score > best_t_score:
            best_t_score = score
            type_col = col
    if type_col is None and cats:
        non_brand = [c for c in cats if c != brand_col]
        type_col = non_brand[0] if non_brand else cats[0]

    units_keywords = ["unit", "units", "sold", "qty", "quantity", "volume", "count"]
    revenue_keywords = ["revenue", "sales", "amount", "total", "price", "income", "value", "turnover"]

    units_col = None
    best_u = 0
    for col in nums:
        low = col.lower()
        sc = sum(2 for kw in units_keywords if kw in low)
        if sc > best_u:
            best_u = sc
            units_col = col

    revenue_col = None
    best_r = 0
    for col in nums:
        if col == units_col:
            continue
        low = col.lower()
        sc = sum(2 for kw in revenue_keywords if kw in low)
        if sc > best_r:
            best_r = sc
            revenue_col = col

    if revenue_col is None:
        for col in nums:
            if col != units_col and any(p in tokens for p in re.findall(r"[a-z0-9]+", col.lower())):
                revenue_col = col
                break
    if revenue_col is None and nums:
        revenue_col = nums[0] if nums[0] != units_col else (nums[1] if len(nums) > 1 else nums[0])

    return type_col, units_col, revenue_col


def _query_cleaned_dataset(
    df: pd.DataFrame,
    query_text: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str | None, str | None, str | None]:
    """Filter cleaned data and build a deduplicated per-vehicle-type worth summary."""
    q = query_text.strip()
    q_low = q.lower()
    tokens = re.findall(r"[a-z0-9_]+", q_low)

    stop_words = {
        "please",
        "tell",
        "me",
        "the",
        "of",
        "all",
        "in",
        "on",
        "at",
        "for",
        "to",
        "a",
        "an",
        "and",
        "or",
        "show",
        "find",
        "get",
        "give",
        "what",
        "which",
        "is",
        "are",
        "with",
        "by",
        "from",
        "total",
        "sum",
        "average",
        "avg",
        "mean",
        "count",
        "max",
        "min",
        "most",
        "best",
        "top",
        "sold",
        "sale",
        "sales",
        "revenue",
        "units",
        "unit",
        "worth",
        "generate",
        "generated",
        "vehicle",
        "vehicles",
        "type",
        "types",
        "company",
        "companies",
        "data",
        "dataset",
        "record",
        "records",
    }

    search_terms = [t for t in tokens if t not in stop_words and (len(t) > 1 or t.isdigit())]

    filtered = df.copy()
    matched_terms: list[str] = []
    brand_col: str | None = None

    for term in search_terms:
        row_hit = pd.Series(False, index=filtered.index)
        hit_cols: list[str] = []
        for col in filtered.columns:
            col_str = filtered[col].astype(str).str.lower()
            if term.isdigit():
                pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
                hit = col_str.str.contains(pattern, regex=True, na=False)
            else:
                hit = col_str.str.contains(re.escape(term), case=False, na=False)
            if hit.any():
                row_hit = row_hit | hit
                hit_cols.append(col)

        if row_hit.any():
            filtered = filtered[row_hit]
            matched_terms.append(term)
            if brand_col is None and hit_cols and not term.isdigit():
                brand_col = hit_cols[0]

    if not matched_terms and search_terms:
        any_hit = pd.Series(False, index=df.index)
        for term in search_terms:
            for col in df.columns:
                any_hit = any_hit | df[col].astype(str).str.lower().str.contains(re.escape(term), case=False, na=False)
        if any_hit.any():
            filtered = df[any_hit]
            matched_terms = search_terms

    type_col, units_col, revenue_col = _detect_type_and_metric_columns(
        filtered if not filtered.empty else df, tokens, brand_col
    )

    if filtered.empty:
        return (
            filtered,
            pd.DataFrame(),
            f'No rows in the cleaned dataset matched **"{q}"**. Try broader keywords or check spelling.',
            type_col,
            units_col,
            revenue_col,
        )

    brand_prefix = f"{matched_terms[0].title()} " if matched_terms and not matched_terms[0].isdigit() else ""
    summary_rows = pd.DataFrame()

    if type_col is not None:
        work = filtered.copy()
        work["_vehicle_type"] = work[type_col].fillna("Unspecified").astype(str).str.strip()
        agg_dict: dict[str, tuple[str, str]] = {"Records": ("_vehicle_type", "count")}

        if units_col is not None:
            work["_units_val"] = pd.to_numeric(work[units_col], errors="coerce").fillna(0.0)
            agg_dict["Units Sold"] = ("_units_val", "sum")
        else:
            agg_dict["Units Sold"] = ("_vehicle_type", "count")

        if revenue_col is not None:
            work["_rev_val"] = pd.to_numeric(work[revenue_col], errors="coerce").fillna(0.0)
            agg_dict["Total Revenue"] = ("_rev_val", "sum")

        grouped = work.groupby("_vehicle_type", as_index=False).agg(**agg_dict)
        sort_col = (
            "Units Sold"
            if "Units Sold" in grouped.columns
            else ("Total Revenue" if "Total Revenue" in grouped.columns else "Records")
        )
        grouped = grouped.sort_values(sort_col, ascending=False).reset_index(drop=True)
        grouped.insert(0, "Vehicle Type", grouped["_vehicle_type"].apply(lambda v: f"{brand_prefix}{v}".strip()))
        grouped = grouped.drop(columns=["_vehicle_type"])
        summary_rows = grouped

    filter_desc = f" matching **{', '.join(matched_terms)}**" if matched_terms else ""
    if not summary_rows.empty:
        top_row = summary_rows.iloc[0]
        top_units = float(top_row["Units Sold"])
        top_rev_txt = f" | Total Revenue = **{float(top_row['Total Revenue']):,.2f}**" if "Total Revenue" in top_row else ""
        summary = (
            f"Top seller{filter_desc}: **{top_row['Vehicle Type']}** with "
            f"**{top_units:,.0f}** units sold{top_rev_txt} (across **{len(summary_rows)}** unique vehicle types)."
        )
    else:
        summary = f"Found **{len(filtered):,}** matching records{filter_desc}."

    return filtered, summary_rows, summary, type_col, units_col, revenue_col


def _extract_year_series_from_df(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Extract clean 4-digit year values from a Year column or Date/Sale_Date column."""
    for col in df.columns:
        parts = re.findall(r"[a-z0-9]+", col.lower())
        if "year" in parts:
            yr = pd.to_numeric(df[col], errors="coerce")
            valid = yr.dropna()
            if not valid.empty and valid.between(1900, 2100).mean() > 0.4:
                s = yr.where(yr.between(1900, 2100)).round().astype("Int64").astype(str)
                s = s.replace({"<NA>": np.nan, "nan": np.nan})
                return col, s

    date_priority = [
        c for c in df.columns if any(k in c.lower() for k in ["sale_date", "order_date", "date", "time"])
    ]
    for col in date_priority:
        dt = pd.to_datetime(df[col], errors="coerce")
        if dt.notna().mean() > 0.3:
            s = dt.dt.year.astype("Int64").astype(str).replace({"<NA>": np.nan, "nan": np.nan})
            if s.notna().any():
                return "Year", s

    return None, None


def _build_dynamic_chart_from_request(df: pd.DataFrame, request_text: str):
    """Build a clean, meaningful grouped table & summary for natural-language chart requests."""
    q = request_text.strip()
    tokens = re.findall(r"[a-z0-9_]+", q.lower())

    cats = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [
        c
        for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if not pd.api.types.is_bool_dtype(df[c])
    ]

    agg_mode = "sum"
    agg_label = "Total"
    if any(w in tokens for w in ["average", "avg", "mean"]):
        agg_mode = "mean"
        agg_label = "Average"
    elif any(w in tokens for w in ["count", "how many", "records"]):
        agg_mode = "count"
        agg_label = "Count of"
    elif any(w in tokens for w in ["max", "maximum", "highest", "peak"]):
        agg_mode = "max"
        agg_label = "Maximum"
    elif any(w in tokens for w in ["min", "minimum", "lowest"]):
        agg_mode = "min"
        agg_label = "Minimum"

    wants_ratio = "ratio" in tokens
    wants_log = "log" in tokens
    wants_year = any(w in tokens for w in ["year", "years", "yearly", "annual", "annually"])

    metric_col = None
    best_m_score = -100.0
    for col in nums:
        low = col.lower()
        parts = re.findall(r"[a-z0-9]+", low)
        score = 0.0

        if low == "year" or low.endswith("_year"):
            score -= 50.0
        if any(k in low for k in ["id", "zip", "code", "index"]):
            score -= 20.0
        if ("ratio" in low or "_ratio" in low) and not wants_ratio:
            score -= 35.0
        if ("_log" in low or "log1p" in low) and not wants_log:
            score -= 25.0

        if any(k in low for k in ["total_sales", "sales", "revenue", "amount", "price", "turnover"]):
            score += 12.0
        if any(k in low for k in ["units_sold", "units", "qty", "quantity", "volume"]):
            score += 6.0

        for p in parts:
            if len(p) > 1 and (p in tokens or p.rstrip("s") in tokens):
                score += 5.0

        if score > best_m_score:
            best_m_score = score
            metric_col = col

    if metric_col is None and nums:
        metric_col = nums[0]

    year_col_name, year_series = _extract_year_series_from_df(df)
    group_col = None
    group_series = None

    if wants_year and year_series is not None:
        group_col = year_col_name
        group_series = year_series
    else:
        best_g_score = 0
        for col in cats:
            parts = [p for p in re.findall(r"[a-z0-9]+", col.lower()) if len(p) > 1]
            sc = sum(5 for p in parts if p in tokens or (p + "s") in tokens or p.rstrip("s") in tokens)
            if sc > best_g_score:
                best_g_score = sc
                group_col = col

        if group_col is None:
            rel_cat = _pick_relatable_category(df)
            if rel_cat:
                group_col = rel_cat
            elif year_series is not None:
                group_col = year_col_name
                group_series = year_series
            elif cats:
                group_col = cats[0]

    stop_words = {
        "please",
        "make",
        "create",
        "show",
        "draw",
        "plot",
        "chart",
        "graph",
        "graphs",
        "charts",
        "which",
        "what",
        "tell",
        "me",
        "the",
        "of",
        "in",
        "all",
        "for",
        "by",
        "to",
        "from",
        "with",
        "on",
        "at",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "total",
        "sum",
        "average",
        "avg",
        "mean",
        "count",
        "max",
        "min",
        "most",
        "across",
        "per",
        "each",
        "year",
        "years",
        "yearly",
        "annual",
        "annually",
        "sale",
        "sales",
        "revenue",
        "data",
        "dataset",
    }
    col_words = {w for c in df.columns for w in re.findall(r"[a-z0-9]+", c.lower())}
    terms = [t for t in tokens if t not in stop_words and (len(t) > 1 or t.isdigit())]

    filtered = df.copy()
    matched_filters: list[str] = []

    for term in terms:
        if term.isdigit() and len(term) == 4 and wants_year:
            continue
        if group_col and term in re.findall(r"[a-z0-9]+", group_col.lower()) and not term.isdigit():
            continue

        hit_any = pd.Series(False, index=filtered.index)
        for col in filtered.columns:
            col_s = filtered[col].astype(str).str.lower()
            if term.isdigit():
                pat = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
                hit_any = hit_any | col_s.str.contains(pat, regex=True, na=False)
            else:
                hit_any = hit_any | col_s.str.contains(re.escape(term), case=False, na=False)

        if hit_any.any() and (term not in col_words or term.isdigit()):
            filtered = filtered[hit_any]
            matched_filters.append(term)

    if filtered.empty:
        return (
            filtered,
            pd.DataFrame(),
            group_col,
            metric_col,
            matched_filters,
            f'No rows matched **"{q}"**.',
        )

    work = filtered.copy()
    if group_col == year_col_name and group_series is not None:
        work["_grp"] = group_series.loc[work.index]
        work = work[work["_grp"].notna()]
    elif group_col is not None:
        work["_grp"] = work[group_col].fillna("").astype(str).str.strip()
        work = work[work["_grp"] != ""]
    else:
        return (
            filtered,
            pd.DataFrame(),
            None,
            metric_col,
            matched_filters,
            f"Found **{len(filtered):,}** matching rows.",
        )

    if work.empty:
        return (
            filtered,
            pd.DataFrame(),
            group_col,
            metric_col,
            matched_filters,
            f"No valid **{group_col}** values found for matching rows.",
        )

    if agg_mode == "count" or metric_col is None:
        grp_df = work.groupby("_grp", as_index=False).size().rename(columns={"size": "Value"})
        y_label = f"Record Count ({group_col})"
    else:
        work["_val"] = pd.to_numeric(work[metric_col], errors="coerce").fillna(0.0)
        grp_df = work.groupby("_grp", as_index=False)["_val"].agg(agg_mode).rename(columns={"_val": "Value"})
        y_label = f"{agg_label} {metric_col}"

    if group_col == year_col_name:
        grp_df = grp_df.sort_values("_grp", ascending=True).reset_index(drop=True)
    else:
        grp_df = grp_df.sort_values("Value", ascending=False).head(20).reset_index(drop=True)

    grp_df.columns = [group_col, y_label]

    filt_txt = f" for **{', '.join(matched_filters).title()}**" if matched_filters else ""
    top_row = grp_df.loc[grp_df[y_label].idxmax()]
    if len(grp_df) > 1:
        bot_row = grp_df.loc[grp_df[y_label].idxmin()]
        explanation = (
            f"This chart shows **{y_label}** by **{group_col}**{filt_txt}. "
            f"**{top_row[group_col]}** is highest (**{float(top_row[y_label]):,.2f}**), "
            f"and **{bot_row[group_col]}** is lowest (**{float(bot_row[y_label]):,.2f}**)."
        )
    else:
        explanation = (
            f"This chart shows **{y_label}** for **{top_row[group_col]}**{filt_txt}: "
            f"**{float(top_row[y_label]):,.2f}**."
        )

    return filtered, grp_df, group_col, y_label, matched_filters, explanation


def run_all_pipelines(
    raw_csv_path: Path,
    col_a: str | None = None,
    col_b: str | None = None,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Execute schema, cleaning, vectorized features, and ML-ready table generation."""
    df_raw = pd.read_csv(raw_csv_path)
    generate_schema_report(raw_csv_path, DOCS_DIR / "schema.md")
    df_clean, stats = preprocess_dataset(df_raw)
    df_feat = add_vectorized_features(df_clean, col_a=col_a, col_b=col_b)
    target_clean = None if target_col in (None, "(None)") else target_col
    df_ml = build_ml_ready_table(df_feat, target_col=target_clean)

    df_feat.to_csv(CLEANED_DIR / "cleaned_data.csv", index=False)
    df_ml.to_csv(CLEANED_DIR / "model_ready_data.csv", index=False)

    st.session_state["raw_snapshot"] = df_raw
    st.session_state["cleaned_snapshot"] = df_feat
    st.session_state["ml_snapshot"] = df_ml
    st.session_state["clean_stats"] = stats
    st.session_state["processed_for_file"] = str(raw_csv_path)
    return df_raw, df_feat, df_ml, stats


if "_pending_stage" in st.session_state:
    st.session_state["active_stage"] = st.session_state.pop("_pending_stage")

if "active_stage" not in st.session_state:
    st.session_state["active_stage"] = STAGES[0]

st.markdown(
    """
    <div class="top-bar">
        <div>
            <div class="top-kicker">Reusable ML Preparation & Baseline Pipeline</div>
            <h1 class="top-title">Dataset Quality & ML Readiness Studio</h1>
            <div class="top-sub">Clean raw data, inspect summaries & SQLite queries, and train/validate/test 3 ML models without leakage.</div>
        </div>
        <span class="badge-info">● Pipeline Ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### PIPELINE STAGES")
stage = st.sidebar.radio("Select pipeline stage", STAGES, key="active_stage", label_visibility="collapsed")

uploader_key = f"upload_widget_{st.session_state.get('upload_nonce', 0)}"
raw_files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
active_csv = get_active_raw_csv(raw_files)
cleaned_csv = CLEANED_DIR / "cleaned_data.csv"
ml_csv = CLEANED_DIR / "model_ready_data.csv"

if active_csv is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ACTIVE DATASET**")
    st.sidebar.caption(active_csv.name)

# --- STAGE 1: INGEST & CONVERT ---
if stage == "1. Ingest & Convert":
    st.markdown("### 1. Dataset Ingestion & Format Gate")
    st.caption("Upload a `.csv` file for untouched storage in `data/raw/`, or convert/replace non-CSV files.")

    if st.session_state.pop("just_uploaded", False):
        st.success("Dataset Uploaded successfully")

    if st.session_state.pop("just_deleted", None):
        st.info("Dataset deleted from `data/raw/`.")

    uploaded = st.file_uploader(
        "Upload your dataset (.csv required; other formats can be converted to .csv)",
        type=["csv", "xlsx", "xls", "json", "parquet", "txt", "tsv"],
        key=uploader_key,
    )

    if uploaded is not None:
        ext = Path(uploaded.name).suffix.lower()
        if ext == ".csv":
            tmp_in = CLEANED_DIR / f"_tmp_{uploaded.name}"
            tmp_in.write_bytes(uploaded.getbuffer())
            try:
                saved = ingest_raw_csv(tmp_in, RAW_DIR)
                final_dest = RAW_DIR / uploaded.name
                if saved != final_dest:
                    final_dest.write_bytes(tmp_in.read_bytes())
                    saved.unlink(missing_ok=True)
                st.session_state["selected_csv"] = str(final_dest)
                st.session_state["just_uploaded"] = True
                st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
            finally:
                tmp_in.unlink(missing_ok=True)
        else:
            st.warning(
                f"Uploaded file `{uploaded.name}` is not a `.csv` file (detected `{ext}`). "
                "Choose whether to convert this dataset to `.csv` or upload a new dataset."
            )
            c_left, c_right = st.columns(2)
            with c_left:
                if st.button("Convert Dataset to .CSV"):
                    tmp_src = CLEANED_DIR / f"_conv_{uploaded.name}"
                    tmp_src.write_bytes(uploaded.getbuffer())
                    csv_name = f"{Path(uploaded.name).stem}.csv"
                    out_csv = RAW_DIR / csv_name
                    try:
                        convert_to_csv(tmp_src, out_csv)
                        st.session_state["selected_csv"] = str(out_csv)
                        st.session_state["just_uploaded"] = True
                        st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Conversion failed: {exc}")
                    finally:
                        tmp_src.unlink(missing_ok=True)
            with c_right:
                if st.button("Upload a New Dataset"):
                    st.session_state.pop("selected_csv", None)
                    st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1
                    st.rerun()

    if raw_files:
        st.markdown("#### Uploaded Datasets in `data/raw/`")
        for idx, rfile in enumerate(raw_files):
            is_selected = active_csv is not None and rfile == active_csv
            row_c1, row_c2, row_c3 = st.columns([6, 1.5, 1.3])
            with row_c1:
                status_tag = " *(Active)*" if is_selected else ""
                st.markdown(
                    f'<div class="dataset-row"><b>{rfile.name}</b>{status_tag}</div>',
                    unsafe_allow_html=True,
                )
            with row_c2:
                if st.button("Select", key=f"select_ds_{idx}_{rfile.name}"):
                    st.session_state["selected_csv"] = str(rfile)
                    st.rerun()
            with row_c3:
                if st.button("Delete", key=f"delete_ds_{idx}_{rfile.name}"):
                    rfile.unlink(missing_ok=True)
                    remaining = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if is_selected:
                        st.session_state["selected_csv"] = str(remaining[0]) if remaining else None
                    st.session_state["just_deleted"] = True
                    st.rerun()

        raw_files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        chosen = get_active_raw_csv(raw_files)

    if raw_files and chosen is not None:
        df_raw = pd.read_csv(chosen)
        st.session_state["raw_snapshot"] = df_raw

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            stat_box("Raw Rows", f"{len(df_raw):,}", "Before cleaning")
        with s2:
            stat_box("Columns", f"{df_raw.shape[1]:,}", "Input fields")
        with s3:
            stat_box("Missing Cells", f"{int(df_raw.isna().sum().sum()):,}", "Null entries")
        with s4:
            stat_box("Duplicate Rows", f"{int(df_raw.duplicated().sum()):,}", "Exact repeats")

        st.markdown(f"#### Raw Dataset Preview — `{chosen.name}`")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.markdown(
            '<div class="caption-box"><b>Raw State:</b> Unmodified input records saved in <code>data/raw/</code> before schema validation, cleaning, feature engineering, or model encoding.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Move to Next Step: Dirty Data & Preprocessing →"):
                st.session_state["selected_csv"] = str(chosen)
                st.session_state["_pending_stage"] = STAGES[1]
                st.rerun()
        with btn2:
            if st.button("Run All Pipelines & View Before/After →"):
                st.session_state["selected_csv"] = str(chosen)
                run_all_pipelines(chosen)
                st.session_state["_pending_stage"] = STAGES[3]
                st.rerun()
    else:
        st.info("No dataset in `data/raw/` yet. Upload a `.csv` file or convert another format above.")

# --- STAGE 2: DIRTY DATA & PREPROCESSING ---
elif stage == "2. Dirty Data & Preprocessing":
    st.markdown("### 2. Dirty Dataset Inspection & Preprocessing")
    st.caption(
        "Inspect the raw dirty dataset (`head(5)`) and its relationship/correlation graphs before cleaning, "
        "then clean the dataset, explore cleaned graphs, and search anything in the cleaned dataset."
    )

    chosen = get_active_raw_csv(raw_files)
    if chosen is None:
        st.warning("Upload a dataset in Stage 1 first.")
    else:
        df_dirty = pd.read_csv(chosen)
        st.session_state["raw_snapshot"] = df_dirty
        st.info(f"Active Dataset from Pipeline 1: **`{chosen.name}`**")

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            stat_box("Dirty Rows", f"{len(df_dirty):,}", "Before preprocessing")
        with s2:
            stat_box("Columns", f"{df_dirty.shape[1]:,}", "Raw fields")
        with s3:
            stat_box("Missing Cells", f"{int(df_dirty.isna().sum().sum()):,}", "Before cleaning")
        with s4:
            stat_box("Duplicate Rows", f"{int(df_dirty.duplicated().sum()):,}", "Exact duplicates")

        st.markdown(f"#### Before Preprocessing — Dirty Data (`{chosen.name}` · `head(5)`)")
        st.markdown('<span class="badge-raw">BEFORE PREPROCESSING (DIRTY)</span>', unsafe_allow_html=True)
        st.dataframe(df_dirty.head(5), use_container_width=True)

        st.markdown("#### Dirty Dataset Graphs — Patterns & Correlations (Before Cleaning)")
        _render_dataset_insight_charts(
            df=df_dirty,
            prefix=f"dirty_{chosen.name}",
            accent_color="#F59E0B",
            secondary_color="#FB923C",
            stage_label="Dirty Data",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Preprocessing Pipeline"):
            validate_required_columns(df_dirty, df_dirty.columns)
            generate_schema_report(chosen, DOCS_DIR / "schema.md")
            df_clean, stats = preprocess_dataset(df_dirty)
            df_clean.to_csv(cleaned_csv, index=False)
            st.session_state["cleaned_snapshot"] = df_clean
            st.session_state["clean_stats"] = stats
            st.session_state["preprocessed_for_file"] = str(chosen)

        if st.session_state.get("preprocessed_for_file") == str(chosen) and "cleaned_snapshot" in st.session_state:
            df_cleaned_now = st.session_state["cleaned_snapshot"]
            st.success("Dataset Preprocessed successfully")

            clean_dl_name = f"cleaned_{chosen.stem}.csv"
            st.download_button(
                label="Download Your clean dataset",
                data=df_cleaned_now.to_csv(index=False).encode("utf-8"),
                file_name=clean_dl_name,
                mime="text/csv",
                key=f"dl_clean_{chosen.name}",
                on_click=_on_clean_download,
                args=(df_cleaned_now, clean_dl_name, "s2_pseb_saved_path"),
            )
            if st.session_state.get("s2_pseb_saved_path"):
                st.caption(f"Saved copy on disk: `{st.session_state['s2_pseb_saved_path']}`")

            stats = st.session_state.get("clean_stats")
            if stats:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    stat_box(
                        "Rows After Clean",
                        f"{stats['rows_before']:,} → {stats['rows_after']:,}",
                        f"{stats['duplicates_removed']:,} dupes removed",
                    )
                with m2:
                    stat_box(
                        "Missing Cells",
                        f"{stats['missing_cells_before']:,} → {stats['missing_cells_after']:,}",
                        f"{stats['missing_pct_before']}% → {stats['missing_pct_after']}%",
                    )
                with m3:
                    stat_box("Outliers Clipped", f"{stats['outliers_clipped']:,}", "IQR rule")
                with m4:
                    stat_box("Cleaned Output", "Saved", "data/cleaned/cleaned_data.csv")

            st.markdown("#### After Preprocessing — Cleaned Data (`head(5)`)")
            st.markdown('<span class="badge-clean">AFTER PREPROCESSING (CLEAN)</span>', unsafe_allow_html=True)
            st.dataframe(df_cleaned_now.head(5), use_container_width=True)

            st.markdown("#### Cleaned Dataset Graphs — Patterns & Correlations (After Cleaning)")
            _render_dataset_insight_charts(
                df=df_cleaned_now,
                prefix=f"clean_{chosen.name}",
                accent_color="#10B981",
                secondary_color="#06B6D4",
                stage_label="Cleaned Data",
            )

            st.markdown("---")
            st.markdown("#### Search & Ask About Your Cleaned Dataset")
            st.caption(
                "Ask a question or enter keywords (e.g., *'Which type of ford vehicles sold the most In 2020'*) "
                "to see unique vehicle types with units sold & total revenue, matching rows, and charts."
            )
            search_query = st.text_input(
                "Search cleaned dataset",
                value="",
                placeholder="e.g., Which type of ford vehicles sold the most In 2020",
                key=f"clean_search_{chosen.name}",
            )

            if search_query.strip():
                matched_df, summary_tbl, answer_msg, type_col, units_col, revenue_col = _query_cleaned_dataset(
                    df_cleaned_now, search_query
                )
                st.markdown(
                    f'<div class="caption-box"><b>Search Result:</b> {answer_msg}</div>',
                    unsafe_allow_html=True,
                )

                if not matched_df.empty:
                    if not summary_tbl.empty:
                        st.markdown("##### Unique Vehicle Type Summary (No Duplicates)")
                        lines = []
                        for _, row in summary_tbl.iterrows():
                            u_str = f"Units Sold = **{float(row['Units Sold']):,.0f}**"
                            r_str = (
                                f" | Total Revenue = **{float(row['Total Revenue']):,.2f}**"
                                if "Total Revenue" in row
                                else ""
                            )
                            lines.append(f"* **{row['Vehicle Type']}**: {u_str}{r_str}")
                        st.markdown("\n".join(lines))
                        st.dataframe(summary_tbl, use_container_width=True)

                    tot_u_val = (
                        float(summary_tbl["Units Sold"].sum())
                        if not summary_tbl.empty and "Units Sold" in summary_tbl.columns
                        else (
                            float(pd.to_numeric(matched_df[units_col], errors="coerce").sum())
                            if units_col
                            else float(len(matched_df))
                        )
                    )
                    tot_r_val = (
                        float(summary_tbl["Total Revenue"].sum())
                        if not summary_tbl.empty and "Total Revenue" in summary_tbl.columns
                        else (
                            float(pd.to_numeric(matched_df[revenue_col], errors="coerce").sum())
                            if revenue_col
                            else None
                        )
                    )

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        stat_box(
                            "Unique Types",
                            f"{len(summary_tbl):,}" if not summary_tbl.empty else "1",
                            f"{len(matched_df):,} raw rows matched",
                        )
                    with r2:
                        stat_box("Total Units Sold", f"{tot_u_val:,.0f}", units_col or "Matched rows")
                    with r3:
                        if tot_r_val is not None:
                            stat_box("Total Revenue Generated", f"{tot_r_val:,.2f}", revenue_col or "Sum")
                        else:
                            stat_box("Matching Rows", f"{len(matched_df):,}", "Filtered records")

                    st.markdown("##### Filtered Raw Matching Rows")
                    st.dataframe(matched_df.head(50), use_container_width=True)

                    st.download_button(
                        label="Download Filtered Search CSV",
                        data=matched_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"search_{chosen.stem}.csv",
                        mime="text/csv",
                        key=f"dl_search_{chosen.name}",
                    )

                    st.markdown("##### Graphs for Searched Data")
                    sg1, sg2 = st.columns(2)
                    with sg1:
                        if not summary_tbl.empty and "Units Sold" in summary_tbl.columns:
                            fig_s1 = px.bar(
                                summary_tbl.head(10),
                                x="Vehicle Type",
                                y="Units Sold",
                                title="Searched Results: Units Sold by Type",
                                color_discrete_sequence=["#22D3EE"],
                            )
                            st.plotly_chart(style_chart(fig_s1, 290), use_container_width=True)
                        else:
                            m_nums = [
                                c
                                for c in matched_df.select_dtypes(include=[np.number]).columns.tolist()
                                if not pd.api.types.is_bool_dtype(matched_df[c])
                            ]
                            if m_nums:
                                fig_s1 = px.histogram(
                                    matched_df,
                                    x=m_nums[0],
                                    nbins=15,
                                    title=f"Searched Results: {m_nums[0]} Distribution",
                                    color_discrete_sequence=["#22D3EE"],
                                )
                                st.plotly_chart(style_chart(fig_s1, 290), use_container_width=True)

                    with sg2:
                        if not summary_tbl.empty and "Total Revenue" in summary_tbl.columns:
                            fig_s2 = px.bar(
                                summary_tbl.head(10),
                                x="Vehicle Type",
                                y="Total Revenue",
                                title="Searched Results: Total Revenue by Type",
                                color_discrete_sequence=["#10B981"],
                            )
                            st.plotly_chart(style_chart(fig_s2, 290), use_container_width=True)
                        elif not summary_tbl.empty and "Units Sold" in summary_tbl.columns:
                            fig_s2 = px.pie(
                                summary_tbl.head(8),
                                names="Vehicle Type",
                                values="Units Sold",
                                hole=0.5,
                                title="Searched Results: Share of Units Sold",
                                color_discrete_sequence=[
                                    "#06B6D4",
                                    "#10B981",
                                    "#F59E0B",
                                    "#3B82F6",
                                    "#8B5CF6",
                                    "#EC4899",
                                ],
                            )
                            st.plotly_chart(style_chart(fig_s2, 290), use_container_width=True)

            st.markdown("#### Before vs. After Data Quality Impact")
            impact_df = pd.DataFrame(
                {
                    "Metric": ["Missing Cells", "Duplicate Rows", "Row Count"],
                    "Before Cleaning (Dirty)": [
                        int(df_dirty.isna().sum().sum()),
                        int(df_dirty.duplicated().sum()),
                        len(df_dirty),
                    ],
                    "After Cleaning (Clean)": [
                        int(df_cleaned_now.isna().sum().sum()),
                        int(df_cleaned_now.duplicated().sum()),
                        len(df_cleaned_now),
                    ],
                }
            )
            impact_long = impact_df.melt(
                id_vars="Metric",
                value_vars=["Before Cleaning (Dirty)", "After Cleaning (Clean)"],
                var_name="Stage",
                value_name="Count",
            )
            fig_cmp = px.bar(
                impact_long,
                x="Metric",
                y="Count",
                color="Stage",
                barmode="group",
                title="How Preprocessing Improved Dataset Reliability",
                color_discrete_map={
                    "Before Cleaning (Dirty)": "#F59E0B",
                    "After Cleaning (Clean)": "#10B981",
                },
            )
            st.plotly_chart(style_chart(fig_cmp, 280), use_container_width=True)
            st.markdown(
                '<div class="caption-box"><b>What this tells you:</b> Comparing raw vs. cleaned counts shows how many missing gaps and repeated rows were resolved so downstream analysis and ML models train on trustworthy records.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Move to Next Step: Feature Engineering (ML) →"):
            st.session_state["_pending_stage"] = STAGES[2]
            st.rerun()

# --- STAGE 3: FEATURE ENGINEERING (ML) ---
elif stage == "3. Feature Engineering (ML)":
    st.markdown("### 3. Vectorized Feature Engineering & ML Formatting")
    st.caption("Perform feature transformations, ratio generation, and encoding to export model-ready datasets.")

    chosen = get_active_raw_csv(raw_files)
    if chosen is None:
        st.warning("Upload a dataset in Stage 1 first.")
    else:
        source_df = (
            st.session_state.get("cleaned_snapshot")
            if "cleaned_snapshot" in st.session_state
            else (pd.read_csv(cleaned_csv) if cleaned_csv.exists() else pd.read_csv(chosen))
        )
        st.info(f"Active Dataset: **`{chosen.name}`** ({len(source_df):,} rows × {source_df.shape[1]:,} columns)")

        num_cols = source_df.select_dtypes(include=[np.number]).columns.tolist()

        suitable_nums = [
            c for c in num_cols if not any(kw in c.lower() for kw in ["year", "id", "zip", "code", "index"])
        ]
        if not suitable_nums:
            suitable_nums = num_cols or source_df.columns.tolist()

        default_num_idx = 0
        default_den_idx = 1 if len(suitable_nums) > 1 else 0
        target_opts = ["(None)"] + source_df.columns.tolist()

        with st.expander("Feature Engineering Configuration", expanded=True):
            p1, p2, p3 = st.columns(3)
            with p1:
                col_a = st.selectbox(
                    "Numerator Column (log1p & Ratio)",
                    suitable_nums,
                    index=default_num_idx,
                    key=f"num_a_{chosen.name}",
                )
            with p2:
                col_b = st.selectbox(
                    "Denominator Column (Ratio)",
                    suitable_nums,
                    index=default_den_idx,
                    key=f"num_b_{chosen.name}",
                )
            with p3:
                target_col = st.selectbox(
                    "Target Column (Preserved for Supervised ML)",
                    target_opts,
                    index=0,
                    key=f"target_{chosen.name}",
                )

            st.markdown("##### Transformation Options")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                opt_log = st.checkbox("Log Transform Numerator (log1p)", value=True)
            with tc2:
                opt_ratio = st.checkbox("Create Ratio Feature (Num / Den)", value=True)
            with tc3:
                opt_ohe = st.checkbox("One-Hot Encode Categorical Features", value=True)

        if st.button("Execute Feature Engineering & Build ML Table"):
            _, df_feat, df_ml, stats = run_all_pipelines(
                chosen,
                col_a=col_a if opt_ratio or opt_log else None,
                col_b=col_b if opt_ratio else None,
                target_col=target_col,
            )
            st.success(
                f"Successfully generated `model_ready_data.csv` with {df_ml.shape[1]:,} numeric features!"
            )
            st.session_state["ml_snapshot"] = df_ml
            st.session_state["cleaned_snapshot"] = df_feat

        if "ml_snapshot" in st.session_state:
            df_ml = st.session_state["ml_snapshot"]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                stat_box("ML Sample Count", f"{len(df_ml):,}", "Ready for modeling")
            with m2:
                stat_box("Total Features", f"{df_ml.shape[1]:,}", "Encoded & numeric")
            with m3:
                eng_cols = [c for c in df_ml.columns if "_log" in c or "_ratio" in c or "ratio" in c]
                stat_box(
                    "Engineered Features",
                    f"{len(eng_cols):,}",
                    ", ".join(eng_cols[:2]) if eng_cols else "Vectorized",
                )
            with m4:
                stat_box("Target Status", "Selected" if target_col != "(None)" else "None", target_col)

            st.markdown("#### Engineered ML Dataset Preview (`head(10)`)")
            st.dataframe(df_ml.head(10), use_container_width=True)

            st.download_button(
                label="Download Model-Ready CSV",
                data=df_ml.to_csv(index=False).encode("utf-8"),
                file_name=f"model_ready_{chosen.stem}.csv",
                mime="text/csv",
                key=f"dl_ml_{chosen.name}",
            )

            if target_col != "(None)":
                corr_num = df_ml.corr(numeric_only=True)
                if target_col in corr_num.columns:
                    st.markdown("#### Feature Correlation with Target Column")
                    corr_target = (
                        corr_num[[target_col]]
                        .drop(index=[target_col], errors="ignore")
                        .sort_values(by=target_col, ascending=False)
                        .reset_index()
                    )
                    corr_target.columns = ["Feature", "Correlation"]
                    fig_target = px.bar(
                        corr_target,
                        x="Feature",
                        y="Correlation",
                        title=f"Feature Correlation Strength relative to '{target_col}'",
                        color_discrete_sequence=["#06B6D4"],
                    )
                    st.plotly_chart(style_chart(fig_target, 300), use_container_width=True)
                else:
                    st.info(
                        f"`{target_col}` is non-numeric in the ML table, so numeric correlation isn't applicable. "
                        "Select a numeric target column (like `Units_Sold` or `Revenue`) to plot feature correlation."
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Move to Next Step: Before vs. After (ML) →"):
            st.session_state["_pending_stage"] = STAGES[3]
            st.rerun()

# --- STAGE 4: BEFORE VS. AFTER (ML-READY) ---
elif stage == "4. Before vs. After (ML)":
    st.markdown("### 4. Before Cleaning vs. After Full Pipeline (Model-Ready)")
    st.caption("Side-by-side comparison of raw input data against the cleaned, feature-engineered ML dataset.")

    chosen = get_active_raw_csv(raw_files)
    before_df = pd.read_csv(chosen) if chosen is not None else st.session_state.get("raw_snapshot")

    after_df = st.session_state.get("ml_snapshot")
    if after_df is None and ml_csv.exists():
        after_df = pd.read_csv(ml_csv)
    elif after_df is None and cleaned_csv.exists():
        after_df = pd.read_csv(cleaned_csv)

    if before_df is None:
        st.warning("Upload a dataset in Stage 1 first.")
    elif after_df is None:
        st.warning("Run the pipelines in Stage 2/3 (or click 'Run All Pipelines' in Stage 1) first.")
    else:
        if chosen is not None and st.session_state.get("processed_for_file") not in (None, str(chosen)):
            st.warning(
                f"Active dataset is now `{chosen.name}`, but the saved output was built from another file. "
                "Run the pipelines in Stage 2/3 to refresh for this dataset."
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat_box("Rows", f"{len(before_df):,} → {len(after_df):,}", "Before → After")
        with c2:
            stat_box("Columns", f"{before_df.shape[1]:,} → {after_df.shape[1]:,}", "Raw → ML-Ready")
        with c3:
            stat_box(
                "Null Cells",
                f"{int(before_df.isna().sum().sum()):,} → {int(after_df.isna().sum().sum()):,}",
                "Before → After",
            )
        with c4:
            rem_non_num = int(after_df.select_dtypes(exclude=[np.number]).shape[1])
            stat_box("Non-Numeric Cols", f"{rem_non_num:,}", "Remaining text/other")

        col_b_view, col_a_view = st.columns(2)
        with col_b_view:
            st.markdown('<span class="badge-raw">BEFORE CLEANING (RAW)</span>', unsafe_allow_html=True)
            st.caption(f"{before_df.shape[0]:,} rows × {before_df.shape[1]:,} columns")
            st.dataframe(before_df.head(15), use_container_width=True)

        with col_a_view:
            st.markdown('<span class="badge-clean">AFTER PIPELINES (ML-READY)</span>', unsafe_allow_html=True)
            st.caption(f"{after_df.shape[0]:,} rows × {after_df.shape[1]:,} columns")
            st.dataframe(after_df.head(15), use_container_width=True)

        st.markdown(
            '<div class="caption-box"><b>Summary:</b> Left shows the raw dataset before cleaning; right shows the processed dataset after deduplication, type coercion, missing-value imputation, IQR outlier capping, vectorized features, and encoding.</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            label="Download Model-Ready CSV",
            data=after_df.to_csv(index=False).encode("utf-8"),
            file_name="model_ready_data.csv",
            mime="text/csv",
            on_click=_on_clean_download,
            args=(after_df, "model_ready_data.csv", "s4_pseb_saved_path"),
        )
        if st.session_state.get("s4_pseb_saved_path"):
            st.caption(f"Saved copy on disk: `{st.session_state['s4_pseb_saved_path']}`")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Move to Next Step: Charts & SQLite →"):
            st.session_state["_pending_stage"] = STAGES[4]
            st.rerun()

# --- STAGE 5: CHARTS & SQLITE ---
elif stage == "5. Charts & SQLite":
    st.markdown("### 5. Visual Summary, Custom Chart Search & SQLite Aggregation")
    st.caption(
        "Search to generate custom relationship charts dynamically, save charts directly to `figures/`, "
        "explore category shares and numeric spread, export PNG charts, and run SQLite GROUP BY queries."
    )

    source_file = cleaned_csv if cleaned_csv.exists() else (ml_csv if ml_csv.exists() else None)
    if source_file is None:
        st.warning("Run Stage 2 or Stage 3 first to generate cleaned data.")
    else:
        df_vis = pd.read_csv(source_file)
        num_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_vis.select_dtypes(include=["object", "string"]).columns.tolist()

        st.markdown("#### Search to Build Custom Charts (Column Relationships)")
        st.caption(
            "Type what chart you want—e.g. *'make a chart for the total sales of toyota for all the years'*—"
            "and add multiple chart requests dynamically."
        )

        if "p5_saved_charts" not in st.session_state:
            st.session_state["p5_saved_charts"] = []

        sb_c1, sb_c2, sb_c3 = st.columns([6, 1.6, 1.4])
        with sb_c1:
            chart_req_input = st.text_input(
                "Search for a chart",
                value="",
                placeholder="e.g., make a chart for the total sales of toyota for all the years",
                key="p5_chart_search",
            )
        with sb_c2:
            chart_style = st.selectbox(
                "Chart style",
                ["Bar Chart", "Line Chart", "Pie / Donut"],
                key="p5_chart_style",
            )
        with sb_c3:
            st.markdown("<br>", unsafe_allow_html=True)
            add_clicked = st.button("Add Chart", key="p5_add_chart")

        if add_clicked and chart_req_input.strip():
            st.session_state["p5_saved_charts"].append(
                {"query": chart_req_input.strip(), "style": chart_style}
            )

        if st.session_state["p5_saved_charts"]:
            clr_col, _ = st.columns([2, 8])
            with clr_col:
                if st.button("Clear Saved Charts", key="p5_clear_charts"):
                    st.session_state["p5_saved_charts"] = []
                    st.rerun()

        active_requests: list[dict] = []
        if chart_req_input.strip():
            active_requests.append({"query": chart_req_input.strip(), "style": chart_style})
        for item in st.session_state["p5_saved_charts"]:
            if not any(r["query"] == item["query"] and r["style"] == item["style"] for r in active_requests):
                active_requests.append(item)

        for r_idx, req_obj in enumerate(active_requests):
            _, grp_tbl, g_col, y_lbl, _, exp_txt = _build_dynamic_chart_from_request(
                df_vis, req_obj["query"]
            )
            st.markdown(f"##### Chart #{r_idx + 1}: *\"{req_obj['query']}\"*")
            if not grp_tbl.empty and g_col and y_lbl:
                c_left, c_right = st.columns([1.6, 1])
                with c_left:
                    if req_obj["style"] == "Line Chart":
                        fig_dyn = px.line(
                            grp_tbl,
                            x=g_col,
                            y=y_lbl,
                            markers=True,
                            title=f"{y_lbl} by {g_col}",
                            color_discrete_sequence=["#22D3EE"],
                        )
                    elif req_obj["style"] == "Pie / Donut":
                        fig_dyn = px.pie(
                            grp_tbl,
                            names=g_col,
                            values=y_lbl,
                            hole=0.45,
                            title=f"{y_lbl} Share by {g_col}",
                            color_discrete_sequence=[
                                "#06B6D4",
                                "#10B981",
                                "#F59E0B",
                                "#3B82F6",
                                "#8B5CF6",
                                "#EC4899",
                            ],
                        )
                    else:
                        fig_dyn = px.bar(
                            grp_tbl,
                            x=g_col,
                            y=y_lbl,
                            title=f"{y_lbl} by {g_col}",
                            color_discrete_sequence=["#06B6D4"],
                        )
                    st.plotly_chart(style_chart(fig_dyn, 330), use_container_width=True)
                    if st.button("Save Chart to figures/", key=f"save_dyn_fig_{r_idx}"):
                        saved_fig = save_chart_to_figures_dir(
                            fig=fig_dyn,
                            filename=f"chart_{r_idx + 1}_{req_obj['query'][:30]}",
                            chart_style=req_obj["style"],
                            data_df=grp_tbl,
                            x_col=g_col,
                            y_col=y_lbl,
                        )
                        st.success(f"Saved chart to `{saved_fig.relative_to(ROOT_DIR)}`")
                with c_right:
                    st.markdown(
                        f'<div class="caption-box"><b>What this chart tells you:</b> {exp_txt}</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(grp_tbl, use_container_width=True)
            else:
                st.markdown(
                    f'<div class="caption-box"><b>What this chart tells you:</b> {exp_txt}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### Category Share")
            if cat_cols:
                c_sel = st.selectbox("Category column", cat_cols, key="vis_cat")
                vc = df_vis[c_sel].value_counts().head(8).reset_index()
                vc.columns = ["Category", "Count"]
                fig1 = px.pie(
                    vc,
                    names="Category",
                    values="Count",
                    hole=0.55,
                    color_discrete_sequence=[
                        "#06B6D4",
                        "#0EA5E9",
                        "#10B981",
                        "#3B82F6",
                        "#6366F1",
                        "#14B8A6",
                        "#8B5CF6",
                        "#64748B",
                    ],
                )
                st.plotly_chart(style_chart(fig1), use_container_width=True)
                if st.button("Save Category Share to figures/", key="save_fig_cat_share"):
                    saved_p = save_chart_to_figures_dir(
                        fig=fig1,
                        filename=f"category_share_{c_sel}",
                        chart_style="Pie / Donut",
                        data_df=vc,
                        x_col="Category",
                        y_col="Count",
                    )
                    st.success(f"Saved chart to `{saved_p.relative_to(ROOT_DIR)}`")
                top_row = vc.iloc[0]
                st.markdown(
                    f'<div class="caption-box"><b>Finding:</b> <b>{top_row["Category"]}</b> leads top categories with <b>{int(top_row["Count"]):,}</b> records.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No categorical columns available.")

        with g2:
            st.markdown("#### Numeric Distribution")
            if num_cols:
                n_sel = st.selectbox("Numeric column", num_cols, key="vis_num")
                unit_lbl = st.text_input("Units", value="units", key="vis_units")
                fig2 = px.histogram(
                    df_vis,
                    x=n_sel,
                    nbins=25,
                    labels={n_sel: f"{n_sel} ({unit_lbl})"},
                    color_discrete_sequence=["#06B6D4"],
                )
                st.plotly_chart(style_chart(fig2), use_container_width=True)
                if st.button("Save Numeric Distribution to figures/", key="save_fig_num_dist"):
                    hist_counts, hist_edges = np.histogram(
                        pd.to_numeric(df_vis[n_sel], errors="coerce").dropna(), bins=20
                    )
                    hist_tbl = pd.DataFrame(
                        {
                            "Bin": [f"{hist_edges[i]:.1f}–{hist_edges[i+1]:.1f}" for i in range(len(hist_counts))],
                            "Count": hist_counts,
                        }
                    )
                    saved_p = save_chart_to_figures_dir(
                        fig=fig2,
                        filename=f"numeric_dist_{n_sel}",
                        chart_style="Bar Chart",
                        data_df=hist_tbl,
                        x_col="Bin",
                        y_col="Count",
                    )
                    st.success(f"Saved chart to `{saved_p.relative_to(ROOT_DIR)}`")
                med_val = float(df_vis[n_sel].median())
                st.markdown(
                    f'<div class="caption-box"><b>Finding:</b> Median value for <b>{n_sel}</b> is <b>{med_val:,.2f} {unit_lbl}</b>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No numeric columns available.")

        if cat_cols and num_cols:
            st.markdown("---")
            st.markdown("#### SQLite GROUP BY Query & Exported PNGs")
            if st.button("Export 3 PNG Charts"):
                save_visualizations(
                    df_vis,
                    cat_cols[0],
                    num_cols[0],
                    units=st.session_state.get("vis_units", "units"),
                    figures_dir=FIGURES_DIR,
                )
                st.success(
                    "Saved `fig1_top_categories.png`, `fig2_numeric_distribution.png`, and "
                    "`fig3_mean_by_category.png` in `figures/`."
                )

            sql_stmt, sql_tbl = run_sqlite_query(
                df_vis, cat_cols[0], num_cols[0], SQL_DIR / "part_a.db"
            )
            st.code(sql_stmt, language="sql")
            st.dataframe(sql_tbl, use_container_width=True)
            if not sql_tbl.empty:
                st.markdown(
                    f'<div class="caption-box"><b>SQLite Query Explanation:</b> Grouping the cleaned table in <code>sql/part_a.db</code> by <b>{cat_cols[0]}</b> summarizes record counts and <b>{num_cols[0]}</b> totals/averages so top-performing groups are easy to compare.</div>',
                    unsafe_allow_html=True,
                )

        # --- AFTER PIPELINE 5 ACTION BUTTONS ---
        st.markdown("---")
        st.markdown("#### Next Steps With Your Cleaned Dataset")
        st.caption("Download your clean dataset into `PSEB Assignment 2/data/`, open Part B to train/validate/test 3 ML models, or view a Deep Learning starter.")

        p5_clean_name = f"clean_{source_file.name}"
        act_c1, act_c2, act_c3 = st.columns(3)
        with act_c1:
            st.download_button(
                label="Download your Clean dataset",
                data=df_vis.to_csv(index=False).encode("utf-8"),
                file_name=p5_clean_name,
                mime="text/csv",
                key="p5_dl_clean_dataset",
                on_click=_on_clean_download,
                args=(df_vis, p5_clean_name, "p5_pseb_saved_path"),
            )
        with act_c2:
            if st.button("Use your data for further Machine learning Models", key="p5_btn_ml"):
                st.session_state["_pending_stage"] = STAGES[5]
                st.rerun()
        with act_c3:
            if st.button("Use You This clean data set for Deep learning models", key="p5_btn_dl"):
                st.session_state["p5_next_workflow"] = "dl"

        if st.session_state.get("p5_pseb_saved_path"):
            st.caption(f"Saved copy on disk: `{st.session_state['p5_pseb_saved_path']}`")

        if st.session_state.get("p5_next_workflow") == "dl":
            st.markdown("##### Deep Learning Starter (PyTorch / Keras)")
            st.markdown(
                f'<div class="caption-box"><b>Dataset Loaded:</b> <code>{source_file.name}</code> '
                f'({len(df_vis):,} rows × {df_vis.shape[1]:,} columns). Scale numeric features with '
                f'<code>StandardScaler</code> before training a tabular neural network (MLP).</div>',
                unsafe_allow_html=True,
            )
            dl_target = st.selectbox("Select target column for Deep Learning", df_vis.columns.tolist(), key="p5_dl_target")
            st.code(
                f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

df = pd.read_csv("{source_file.as_posix()}")
X = pd.get_dummies(df.drop(columns=["{dl_target}"]), drop_first=True).fillna(0)
y = df["{dl_target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Input(shape=(X_train_s.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_train_s, y_train, validation_split=0.2, epochs=25, batch_size=32, verbose=0)
print("Test MAE:", model.evaluate(X_test_s, y_test, verbose=0)[1])
""",
                language="python",
            )

# --- STAGE 6: PART B — TRAIN, VALIDATE & TEST 3 ML MODELS ---
elif stage == "6. Part B: Train 3 ML Models":
    st.markdown("### 6. Part B: Train, Validate & Test 3 Machine Learning Models")
    st.caption(
        "Uses the final cleaned dataset from Part A (`data/cleaned/cleaned_data.csv`), splits train/test first "
        "to prevent data leakage, runs cross-validation, trains 3 distinct models (1 baseline + 2 ML models), "
        "and evaluates on unseen test data."
    )

    try:
        clean_path = resolve_final_clean_dataset(CLEANED_DIR)
        df_b = pd.read_csv(clean_path)

        st.info(f"Final Clean Dataset from Part A: **`{clean_path.name}`** ({len(df_b):,} rows × {df_b.shape[1]:,} columns)")

        num_b_cols = df_b.select_dtypes(include=[np.number]).columns.tolist()
        default_b_target = next(
            (c for c in num_b_cols if any(k in c.lower() for k in ["sales", "revenue", "price", "amount", "units"])),
            num_b_cols[-1] if num_b_cols else df_b.columns[-1],
        )
        default_b_idx = df_b.columns.tolist().index(default_b_target)

        b1, b2, b3 = st.columns(3)
        with b1:
            b_target = st.selectbox("Target Column to Predict", df_b.columns.tolist(), index=default_b_idx, key="b_target_col")
        with b2:
            b_test_size = st.slider("Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="b_test_split")
        with b3:
            b_cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5, step=1, key="b_cv_folds")

        extra_drop = st.multiselect(
            "Optional columns to exclude (e.g., IDs or potential leakage fields)",
            [c for c in df_b.columns if c != b_target],
            default=[],
            key="b_extra_drop",
        )

        if st.button("Train, Validate & Test 3 ML Models", key="b_run_3_models"):
            with st.spinner("Training, validating, and testing 3 leakage-free models..."):
                b_summary = train_validate_and_test_models(
                    target_col=b_target,
                    data_path=clean_path,
                    drop_cols=extra_drop,
                    test_size=b_test_size,
                    cv_splits=b_cv_folds,
                    random_state=42,
                    artifacts_dir=MODELS_DIR,
                    reports_dir=REPORTS_DIR,
                )
                st.session_state["part_b_summary"] = b_summary

        b_res = st.session_state.get("part_b_summary")
        if b_res:
            st.success(
                f"Completed training, validation, and testing for **3 models** ({b_res['task_type']})! "
                f"Best model saved to `{b_res['saved_model_path']}`."
            )

            if b_res.get("leakage_warnings"):
                st.warning(
                    f"Potential ID or leakage-prone columns in features: `{', '.join(b_res['leakage_warnings'])}`. "
                    "Exclude any post-outcome or identifier columns if needed."
                )

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                stat_box("Task Type", b_res["task_type"].title(), f"Target: {b_res['target_col']}")
            with k2:
                stat_box("Train / Test Rows", f"{b_res['train_rows']:,} / {b_res['test_rows']:,}", f"{b_res['cv_folds']}-Fold CV")
            with k3:
                stat_box("Models Trained", "3 Models", "1 Baseline + 2 ML")
            with k4:
                stat_box("Best Model", b_res["best_model"], "Selected on CV score")

            comp_df = pd.DataFrame(b_res["model_comparison"])
            st.markdown("#### 3-Model Validation & Test Comparison")
            st.dataframe(comp_df, use_container_width=True)

            metric_candidates = [c for c in comp_df.columns if c != "model"]
            if metric_candidates and "model" in comp_df.columns:
                sel_metric = st.selectbox("Compare models on metric", metric_candidates, index=0, key="b_cmp_metric")
                fig_m = px.bar(
                    comp_df,
                    x="model",
                    y=sel_metric,
                    title=f"3-Model Comparison — {sel_metric}",
                    color="model",
                    color_discrete_sequence=["#64748B", "#06B6D4", "#10B981"],
                )
                st.plotly_chart(style_chart(fig_m, 300), use_container_width=True)

            st.markdown("#### Sample Predictions on Unseen Test Rows")
            st.dataframe(pd.DataFrame(b_res["sample_unseen_predictions"]), use_container_width=True)

            if b_res.get("top_errors"):
                st.markdown("#### Error Inspection (Largest / Misclassified Test Cases)")
                st.dataframe(pd.DataFrame(b_res["top_errors"]), use_container_width=True)

            if b_res.get("subgroup_breakdown"):
                st.markdown("#### Subgroup Error / Accuracy Check")
                st.dataframe(pd.DataFrame(b_res["subgroup_breakdown"]), use_container_width=True)

            with st.expander("Model Card (Assumptions, Intended Use, Limitations & Risks)"):
                st.markdown(f"**Split Strategy:** `{b_res.get('split_strategy', 'train_test_split')}` (split before fitting any transformer)")
                st.markdown(f"**Metric Justification:** {b_res.get('metric_justification', '')}")
                st.markdown(
                    "- **Assumptions:** Test split is isolated before preprocessing; unseen categories use `handle_unknown='ignore'`.\n"
                    "- **Intended Use:** Reproducible tabular ML baseline benchmarking and scoring new records via `src/score_new_records.py`.\n"
                    "- **Limitations & Risks:** Sensitive to class imbalance, rare categories, and data drift; verify subgroup errors before practical use."
                )

            st.markdown(
                '<div class="caption-box"><b>Leakage & Methodology Note:</b> Train/test splitting happens before any preprocessing; <code>SimpleImputer</code>, <code>StandardScaler</code>, and <code>OneHotEncoder</code> are fit strictly inside each training fold/split via a scikit-learn <code>Pipeline</code>.</div>',
                unsafe_allow_html=True,
            )

    except FileNotFoundError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Part B model training error: {exc}")