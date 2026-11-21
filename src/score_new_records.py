# Name: Nizar Bahadur | Roll Number: DS-GRAD | Date: 2026-11-21
"""Load saved Part B pipeline in a fresh session and score unseen records."""

from pathlib import Path
import joblib
import pandas as pd


def score_records(
    model_path: Path = Path("models/best_ml_pipeline.joblib"),
    input_csv: Path = Path("data/cleaned/cleaned_data.csv"),
    target_col: str | None = None,
    n_rows: int = 10,
    output_csv: Path = Path("reports/unseen_scored_predictions.csv"),
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at '{model_path}'. Run Part B training first.")

    pipeline = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    sample = df.tail(n_rows).copy()

    if target_col and target_col in sample.columns:
        features = sample.drop(columns=[target_col])
    else:
        features = sample

    preds = pipeline.predict(features)
    out = sample.copy()
    out["predicted_target"] = preds

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


if __name__ == "__main__":
    scored = score_records()
    print(scored[["predicted_target"]].head())