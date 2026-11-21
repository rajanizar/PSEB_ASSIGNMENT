# Reproducible Data Pipelines, Tabular ML & Systems Exploration

**Name:** Nizar Bahadur  
**Roll Number:** DS-GRAD  
**Date:** 2026-11-21  
**Location:** Peshawar, Pakistan  

---

## Project Overview

This repository implements a portable, reproducible data engineering and machine learning workflow covering:
- **Part A:** Reproducible Data Quality & Insights Pipeline (ingestion, validation, cleaning, vectorized feature engineering, PNG visualizations, SQLite `GROUP BY` analytics, and quality gate metrics).
- **Part B:** End-to-End Tabular ML Baseline (leakage-free `train_test_split` before fitting transformers, cross-validation, 3-model comparison, error & subgroup inspection, saved `.joblib` pipeline, and standalone unseen inference).
- **Part C:** Systems & Engineering Practice Exploration (4 topics).

All file paths in the codebase are resolved relative to the project root (`Path(__file__).resolve().parent`) and required output directories are created automatically so the project runs cleanly across Windows, macOS, and Linux without machine-specific paths.

---

## Dataset Details

- **Raw Data Folder:** `data/raw/` (raw files are kept untouched and never overwritten)
- **Cleaned & ML Data Folders:** `data/cleaned/` and `PSEB Assignment 2/data/` (cleaned CSV downloads save directly to `PSEB Assignment 2/data/`)
- **Chart Output Folder:** `figures/` (PNG charts and saved custom/distribution charts are written here)
- **Schema Documentation:** [`docs/schema.md`](docs/schema.md)
- **Source, Licence & Access Date:** `[Insert dataset source URL, open/educational licence, and access date]`

---

## Repository Layout

```text
.
├── README.md
├── app.py
├── requirements.txt
├── PSEB Assignment 2/
│   └── data/                 # Cleaned CSV copies saved on download
├── data/
│   ├── raw/                  # Untouched raw CSV inputs
│   └── cleaned/              # cleaned_data.csv & model_ready_data.csv
├── docs/
│   ├── schema.md
│   ├── model_card.md
│   └── part_c_exploration.md
├── figures/                  # Exported PNG charts
├── models/
│   └── best_ml_pipeline.joblib
├── reports/
│   ├── metrics.json
│   └── part_b_results.json
├── sql/
│   └── part_a.db
└── src/
    ├── pipeline_01_ingest.py
    ├── pipeline_02_validate.py
    ├── pipeline_03_preprocess.py
    ├── pipeline_04_features.py
    ├── pipeline_05_analytics.py
    ├── pipeline_06_ml_baseline.py
    ├── quality_gate.py
    └── score_new_records.py