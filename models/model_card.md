# Model Card — Part B Tabular ML Baseline

- **Owner:** Nizar Bahadur
- **Input Data:** Part A cleaned dataset (`data/cleaned/cleaned_data.csv`)
- **Split & Leakage Controls:** Train/test split performed before fitting any imputer, scaler, or one-hot encoder; cross-validation run strictly on training folds; test set evaluated once at the end.
- **Models Compared:** Dummy baseline (`most_frequent` / `mean`), Linear model (`LogisticRegression` / `Ridge`), and Gradient Boosted Trees (`HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` with lightweight training-only hyperparameter check).
- **Intended Use:** Offline benchmarking, error inspection, and batch scoring on tabular records matching the training schema.
- **Limitations & Risks:**
  - Performance degrades on unseen categories with few training examples or out-of-distribution numeric ranges.
  - Class imbalance or noisy labels can skew precision/recall or residuals across subgroups.
  - Not suitable for high-stakes autonomous decisions without domain review and drift monitoring.