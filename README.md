# Seoul Tourism Visitor Forecast

서울 주요 관광지의 일별 방문객 수를 2020-2024년 데이터로 예측하는 회귀 모델 프로젝트입니다. 2020-2023년 데이터를 학습에 사용하고, 2024년 데이터를 테스트셋으로 분리해 시계열 누수를 줄였습니다.

## Project Structure

```text
tourist-forecast/
├─ data/
│  └─ seoul_tourism_2020_2024_clean_common.csv
├─ docs/
│  └─ analysis_summary.md
├─ outputs/
│  ├─ figures/
│  └─ tables/
├─ src/
│  └─ train_regression_models.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## Data

- Period: 2020-2024
- Target: `daily_visitors`
- Main features: date features, POI attributes, weather, air quality, tourism comfort index, accessibility, event/holiday variables
- Leakage-prone fields such as `foreign_visitors`, `foreign_share`, and `crowd_level` are excluded from modeling because they are likely unavailable before the visit outcome is observed.

## Modeling

The script compares several regression models on the original visitor-count scale after training with `log1p(daily_visitors)`.

- Baseline: train-set mean prediction
- Linear Regression
- Ridge
- LASSO
- Decision Tree
- Random Forest
- Polynomial Regression
- Random Forest parameter sweep

The final residual and feature-importance analysis uses the best Random Forest model by 2024 test MSE.

## How to Run

```bash
pip install -r requirements.txt
python src/train_regression_models.py
```

Running the script creates result files under:

- `outputs/figures/`
- `outputs/tables/`

## Key Outputs

- `outputs/tables/model_comparison_with_rf_sweep.csv`
- `outputs/tables/rf_parameter_sweep.csv`
- `outputs/tables/top_feature_importance.csv`
- `outputs/tables/top_residual_cases.csv`
- `outputs/figures/model_comparison_with_rf_sweep.png`
- `outputs/figures/top_feature_importance.png`
- `outputs/figures/actual_vs_predicted_and_residuals.png`

## Current Result

With leakage-prone visitor-derived fields excluded, the best model in the current run is `Random Forest (default)`.

| Model | Test MSE | Test MAE | Test R2 |
|---|---:|---:|---:|
| Random Forest (default) | 5,015,768.8 | 1,267.1 | 0.9273 |
| Random Forest (sweep_best) | 5,557,737.4 | 1,292.7 | 0.9194 |
| Polynomial Regression | 16,781,533.0 | 2,335.7 | 0.7567 |
| Baseline (Train mean) | 80,177,671.9 | 5,687.6 | -0.1623 |

## Notes

The included CSV is about 14 MB. If the data source or license changes, replace the file in `data/` and keep the same filename, or update `DATA_PATH` in `src/train_regression_models.py`.
