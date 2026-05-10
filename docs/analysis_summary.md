# Analysis Summary

## Objective

Predict daily visitors for Seoul tourism points of interest using 2020-2024 data.

## Evaluation Design

- Train period: 2020-2023
- Test period: 2024
- Target transformation: `log1p(daily_visitors)` for model fitting, then `expm1` for evaluation on the original visitor scale
- Metrics: MSE, MAE, R2

## Modeling Flow

1. Load cleaned tourism dataset.
2. Add date-derived features such as year, month, and day of year.
3. One-hot encode categorical columns except identifier/name columns.
4. Remove leakage-prone fields that are likely unavailable at prediction time: `foreign_visitors`, `foreign_share`, and `crowd_level`.
5. Impute numeric missing values and standardize numeric features.
6. Train baseline and regression models.
7. Compare Random Forest parameter candidates.
8. Analyze residuals and feature importance for the best Random Forest model.

## Main Artifacts

After running `python src/train_regression_models.py`, review:

- `outputs/tables/model_comparison_with_rf_sweep.csv`
- `outputs/tables/rf_parameter_sweep.csv`
- `outputs/tables/top_feature_importance.csv`
- `outputs/tables/top_residual_cases.csv`
- `outputs/figures/`

## Current Result

With leakage-prone visitor-derived fields excluded, `Random Forest (default)` performs best in the current run:

| Model | Test MSE | Test MAE | Test R2 |
|---|---:|---:|---:|
| Random Forest (default) | 5,015,768.8 | 1,267.1 | 0.9273 |
| Random Forest (sweep_best) | 5,557,737.4 | 1,292.7 | 0.9194 |
| Polynomial Regression | 16,781,533.0 | 2,335.7 | 0.7567 |
| Baseline (Train mean) | 80,177,671.9 | 5,687.6 | -0.1623 |

## Limitations

- The test split is year-based, so results reflect 2024 generalization rather than random holdout performance.
- Feature importance is model-based and should be interpreted as an exploratory signal, not a causal explanation.
- Hyperparameter search is a small manual sweep, not an exhaustive optimization.
- Excluding leakage-prone visitor-derived fields gives a more conservative but more realistic forecasting setup.
