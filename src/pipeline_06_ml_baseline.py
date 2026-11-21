# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-21
"""End-to-end leakage-free Part B ML baseline, validation, tuning, error audit, and persistence."""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_task_type(y: pd.Series) -> str:
    """Infer whether target column is classification or regression."""
    if is_bool_dtype(y) or is_string_dtype(y) or str(y.dtype) in ("object", "category"):
        return "classification"
    if not is_numeric_dtype(y):
        return "classification"
    if is_integer_dtype(y) and y.nunique(dropna=True) <= 25:
        return "classification"
    return "regression"


def find_potential_leakage_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Flag columns that may leak target information or act as row identifiers."""
    suspicious = []
    t_low = target_col.lower()
    for col in df.columns:
        if col == target_col:
            continue
        c_low = col.lower()
        if c_low in {"id", "row_id", "index"} or c_low.endswith("_id"):
            suspicious.append(col)
        elif t_low in c_low or any(tok in c_low for tok in ["target", "label", "actual", "residual", "error"]):
            suspicious.append(col)
    return suspicious


def _build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a leakage-free preprocessor fit strictly inside each training split/fold."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def resolve_final_clean_dataset(
    cleaned_dir: Path = Path("data/cleaned"),
    preferred_path: Path | None = None,
) -> Path:
    """Resolve the final cleaned dataset produced by Part A."""
    if preferred_path is not None and preferred_path.exists():
        return preferred_path
    final_clean = cleaned_dir / "cleaned_data.csv"
    if final_clean.exists():
        return final_clean
    fallback_ml = cleaned_dir / "model_ready_data.csv"
    if fallback_ml.exists():
        return fallback_ml
    raise FileNotFoundError(
        "Final cleaned dataset not found in data/cleaned/. Complete Part A preprocessing first."
    )


def _select_best_hgb_params_on_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    cv,
    random_state: int,
) -> dict:
    """Lightweight hyperparameter check on training folds only (no test leakage)."""
    candidates = [
        {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
        {"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 63},
    ]
    best_params = candidates[0]
    best_score = -np.inf

    for params in candidates:
        est = (
            HistGradientBoostingRegressor(random_state=random_state, **params)
            if task == "regression"
            else HistGradientBoostingClassifier(random_state=random_state, **params)
        )
        pipe = Pipeline([("preprocess", _build_preprocessor(X_train)), ("model", est)])
        scoring = "r2" if task == "regression" else "f1_weighted"
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        mean_val = float(np.mean(scores["test_score"]))
        if np.isfinite(mean_val) and mean_val > best_score:
            best_score = mean_val
            best_params = params

    return best_params


def train_validate_and_test_models(
    target_col: str,
    data_path: Path | None = None,
    drop_cols: list[str] | None = None,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = 42,
    artifacts_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
) -> dict:
    """Split first, cross-validate 3 models, tune candidate on train, test once on hold-out, and inspect errors."""
    resolved_path = resolve_final_clean_dataset(preferred_path=data_path)
    df = pd.read_csv(resolved_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in '{resolved_path.name}'.")

    leakage_warnings = find_potential_leakage_columns(df, target_col)
    cols_to_drop = [c for c in (drop_cols or []) if c in df.columns and c != target_col]

    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 10:
        raise ValueError("Dataset has too few rows after dropping missing target values.")

    X = df.drop(columns=[target_col] + cols_to_drop)
    y = df[target_col]
    task = detect_task_type(y)

    if task == "classification":
        y = y.astype(str).str.strip()
        split_strategy = "stratified_train_test" if (y.nunique() > 1 and y.value_counts().min() >= 2) else "random_train_test"
        stratify_arg = y if split_strategy == "stratified_train_test" else None
    else:
        split_strategy = "random_train_test"
        stratify_arg = None

    # 1. Split BEFORE fitting any transformation or estimator
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    # 2. Cross-validation setup on training set only
    if task == "regression":
        cv = KFold(n_splits=min(cv_splits, len(X_train)), shuffle=True, random_state=random_state)
    else:
        min_class_count = int(y_train.value_counts().min()) if y_train.nunique() > 1 else 0
        use_strat_cv = y_train.nunique() > 1 and min_class_count >= cv_splits
        cv = (
            StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            if use_strat_cv
            else KFold(n_splits=min(cv_splits, len(X_train)), shuffle=True, random_state=random_state)
        )

    tuned_hgb_params = _select_best_hgb_params_on_train(X_train, y_train, task, cv, random_state)

    # 3. Baseline + 2 distinct candidate models
    if task == "regression":
        models = {
            "1_baseline_dummy_mean": DummyRegressor(strategy="mean"),
            "2_linear_ridge": Ridge(alpha=1.0, random_state=random_state),
            "3_hist_gradient_boosting": HistGradientBoostingRegressor(
                random_state=random_state, **tuned_hgb_params
            ),
        }
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
        metric_justification = (
            "MAE (interpretable error in target units), RMSE (penalizes large errors), and R² "
            "(variance explained) are used rather than accuracy because the target is continuous."
        )
    else:
        models = {
            "1_baseline_dummy_majority": DummyClassifier(strategy="most_frequent"),
            "2_linear_logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "3_hist_gradient_boosting": HistGradientBoostingClassifier(
                random_state=random_state, **tuned_hgb_params
            ),
        }
        scoring = {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
        }
        metric_justification = (
            "Weighted F1, Precision, and Recall are prioritized alongside Accuracy to account for "
            "class imbalance across categories rather than relying on raw accuracy alone."
        )

    comparison_rows = []
    fitted_pipelines: dict[str, Pipeline] = {}
    best_model_name = None
    best_val_score = -np.inf

    for model_name, estimator in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", _build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        pipe.fit(X_train, y_train)
        test_preds = pipe.predict(X_test)
        fitted_pipelines[model_name] = pipe

        if task == "regression":
            val_mae = float(-np.mean(cv_scores["test_mae"]))
            val_rmse = float(-np.mean(cv_scores["test_rmse"]))
            val_r2 = float(np.mean(cv_scores["test_r2"]))

            test_mae = float(mean_absolute_error(y_test, test_preds))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
            test_r2 = float(r2_score(y_test, test_preds))

            selection_score = val_r2
            comparison_rows.append(
                {
                    "model": model_name,
                    "val_mae": round(val_mae, 4),
                    "val_rmse": round(val_rmse, 4),
                    "val_r2": round(val_r2, 4),
                    "test_mae": round(test_mae, 4),
                    "test_rmse": round(test_rmse, 4),
                    "test_r2": round(test_r2, 4),
                }
            )
        else:
            val_acc = float(np.mean(cv_scores["test_accuracy"]))
            val_f1 = float(np.mean(cv_scores["test_f1_weighted"]))

            test_acc = float(accuracy_score(y_test, test_preds))
            test_prec = float(precision_score(y_test, test_preds, average="weighted", zero_division=0))
            test_rec = float(recall_score(y_test, test_preds, average="weighted", zero_division=0))
            test_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))
            test_roc_auc = None
            if hasattr(pipe, "predict_proba") and len(np.unique(y_test)) == 2:
                probs = pipe.predict_proba(X_test)[:, 1]
                test_roc_auc = round(float(roc_auc_score(y_test, probs)), 4)

            selection_score = val_f1
            comparison_rows.append(
                {
                    "model": model_name,
                    "val_accuracy": round(val_acc, 4),
                    "val_f1_weighted": round(val_f1, 4),
                    "test_accuracy": round(test_acc, 4),
                    "test_precision_weighted": round(test_prec, 4),
                    "test_recall_weighted": round(test_rec, 4),
                    "test_f1_weighted": round(test_f1, 4),
                    "test_roc_auc": test_roc_auc,
                }
            )

        if np.isfinite(selection_score) and selection_score > best_val_score:
            best_val_score = selection_score
            best_model_name = model_name

    if best_model_name is None:
        best_model_name = list(models.keys())[-1]

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_pipe = fitted_pipelines[best_model_name]
    saved_model_path = artifacts_dir / "best_ml_pipeline.joblib"
    joblib.dump(best_pipe, saved_model_path)

    best_test_preds = best_pipe.predict(X_test)

    # Error analysis, subgroup performance, and diagnostics
    error_examples = []
    subgroup_breakdown = []
    diagnostics: dict = {}

    cat_candidates = [c for c in X_test.select_dtypes(include=["object", "string", "category"]).columns if X_test[c].nunique() <= 15]
    subgroup_col = cat_candidates[0] if cat_candidates else None

    if task == "regression":
        y_test_num = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
        preds_num = np.asarray(best_test_preds, dtype=float)
        residuals = y_test_num - preds_num
        abs_errors = np.abs(residuals)

        worst_positions = np.argsort(abs_errors)[::-1][:5]
        for pos in worst_positions:
            raw_idx = X_test.index[pos]
            error_examples.append(
                {
                    "row_index": int(raw_idx) if isinstance(raw_idx, (int, np.integer)) else str(raw_idx),
                    "actual": float(y_test_num[pos]),
                    "predicted": float(preds_num[pos]),
                    "residual": float(residuals[pos]),
                    "abs_error": float(abs_errors[pos]),
                }
            )

        if subgroup_col:
            sub_df = pd.DataFrame({"group": X_test[subgroup_col].astype(str), "abs_err": abs_errors})
            grp = sub_df.groupby("group", as_index=False)["abs_err"].agg(count="count", mean_abs_error="mean")
            subgroup_breakdown = grp.sort_values("mean_abs_error", ascending=False).head(8).to_dict(orient="records")

        diagnostics = {
            "subgroup_col": subgroup_col,
            "residuals_sample": residuals[:200].tolist(),
            "predicted_sample": preds_num[:200].tolist(),
        }
    else:
        labels_sorted = sorted(list(set(y_test.unique()).union(set(np.asarray(best_test_preds)))))
        cm = confusion_matrix(y_test, best_test_preds, labels=labels_sorted)

        wrong_positions = np.where(np.asarray(y_test) != np.asarray(best_test_preds))[0][:5]
        for pos in wrong_positions:
            raw_idx = X_test.index[pos]
            error_examples.append(
                {
                    "row_index": int(raw_idx) if isinstance(raw_idx, (int, np.integer)) else str(raw_idx),
                    "actual": str(y_test.iloc[pos]),
                    "predicted": str(best_test_preds[pos]),
                }
            )

        if subgroup_col:
            sub_df = pd.DataFrame({
                "group": X_test[subgroup_col].astype(str),
                "correct": (np.asarray(y_test) == np.asarray(best_test_preds)).astype(float),
            })
            grp = sub_df.groupby("group", as_index=False)["correct"].agg(count="count", accuracy="mean")
            subgroup_breakdown = grp.sort_values("accuracy", ascending=True).head(8).to_dict(orient="records")

        diagnostics = {
            "subgroup_col": subgroup_col,
            "confusion_matrix_labels": [str(x) for x in labels_sorted[:20]],
            "confusion_matrix": cm[:20, :20].tolist(),
        }

    unseen_preview = X_test.head(5).copy()
    unseen_preview["predicted"] = best_test_preds[:5]
    unseen_preview["actual"] = y_test.iloc[:5].values

    summary = {
        "source_clean_dataset": str(resolved_path),
        "target_col": target_col,
        "task_type": task,
        "split_strategy": split_strategy,
        "dropped_cols": cols_to_drop,
        "leakage_warnings": leakage_warnings,
        "metric_justification": metric_justification,
        "tuned_hgb_params": tuned_hgb_params,
        "total_rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_folds": int(cv_splits),
        "best_model": best_model_name,
        "saved_model_path": str(saved_model_path),
        "model_comparison": comparison_rows,
        "subgroup_breakdown": subgroup_breakdown,
        "top_errors": error_examples,
        "diagnostics": diagnostics,
        "sample_unseen_predictions": unseen_preview.to_dict(orient="records"),
    }

    (reports_dir / "part_b_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    clean_file = resolve_final_clean_dataset()
    preview = pd.read_csv(clean_file)
    num_cols = preview.select_dtypes(include=[np.number]).columns.tolist()
    default_target = next(
        (c for c in num_cols if any(k in c.lower() for k in ["sales", "revenue", "price", "amount", "units"])),
        num_cols[-1] if num_cols else preview.columns[-1],
    )
    res = train_validate_and_test_models(target_col=default_target, data_path=clean_file)
    print(f"Dataset: {res['source_clean_dataset']} | Target: {res['target_col']} | Task: {res['task_type']}")
    print(f"Best Model: {res['best_model']} -> {res['saved_model_path']}")
    print(json.dumps(res["model_comparison"], indent=2))