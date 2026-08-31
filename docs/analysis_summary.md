# Analysis Summary

## Evaluation Protocol

- Train: 2020–2022
- Validation: 2023
- Test: 2024
- Selection metric: 2023 validation MSE on the original visitor scale
- Final fit: selected configuration retrained on 2020–2023
- Final evaluation: 2024 holdout after selection

Preprocessing is contained in sklearn pipelines. `poi_id` is explicitly cast to a string and routed to the categorical transformer, where `OneHotEncoder(handle_unknown="ignore")` is applied; it is never included in the numeric transformer. Imputation, scaling, and one-hot category discovery are fitted on the current training period and reused without refitting on validation or test data.

## Model Selection

Ridge (`alpha=1`) had the lowest 2023 validation MSE (4,079,333.3) among the evaluated regressors and baselines. The strongest Random Forest candidate had validation MSE 8,290,972.1.

## Final Holdout

After retraining Ridge on 2020–2023, the 2024 holdout result was:

| MSE | MAE | R² |
|---:|---:|---:|
| 12,643,390.6 | 2,309.2 | 0.8167 |

The past-only POI-month-weekday baseline reached MSE 24,887,899.6 and R² 0.6392 on the same holdout. The selected model therefore improved on a seasonal baseline, not only on a global mean baseline.

## Interpretation

The corrected R² is lower than the repository's earlier 0.9273 claim because the earlier workflow selected a Random Forest using 2024 test error. That result is no longer treated as a final holdout estimate. The current result keeps 2024 out of model and hyperparameter selection.

## Artifacts

- `outputs/tables/model_selection_validation.csv`
- `outputs/tables/rf_parameter_sweep_validation.csv`
- `outputs/tables/final_test_metrics.csv`
- `outputs/tables/top_residual_cases.csv`
- `outputs/tables/evaluation_protocol.json`

The selected model is linear, so an impurity-based feature-importance artifact is not produced. Coefficients require care because they follow scaling and one-hot encoding and are not causal effects.
