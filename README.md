# Seoul Tourism Visitor Forecast

## Overview

서울 주요 관광지의 일별 방문객 수를 회귀 모델로 예측하고, 미래 연도에 대한 성능을 시간 순서대로 평가하는 프로젝트입니다. 모델은 방문객 수의 `log1p` 값을 학습하고, 원래 방문객 수 단위로 복원해 MSE, MAE, R²를 계산합니다.

## Data

- Period: 2020-01-01 through 2024-12-31
- Unit: day × tourism point of interest (POI)
- Target: `daily_visitors`
- Feature groups: calendar, POI attributes, weather, air quality, events, accessibility

`foreign_visitors`, `foreign_share`, `crowd_level`은 목표값이 관측되기 전에 알 수 있다고 보기 어려워 모델에서 제외합니다. 범주형 인코딩과 결측치 처리는 sklearn `Pipeline`과 `ColumnTransformer` 안에서 학습 기간에만 fit됩니다.

데이터 제공기관, 원본 URL, 라이선스, 다운로드 날짜, 재배포 가능 여부는 현재 저장소만으로 확인할 수 없습니다. 자세한 TODO는 [`data/README.md`](data/README.md)에 있습니다. 재배포 권한을 확인하기 전에는 포함된 CSV를 공개 배포 가능하다고 가정하면 안 됩니다.

## Evaluation Design

```text
2020-2022  train and fit preprocessing
2023       validation for model and hyperparameter selection
2020-2023  retrain the selected configuration
2024       final holdout evaluation
```

2024 결과는 모델 종류나 하이퍼파라미터 선택에 사용하지 않습니다. Linear Regression, Ridge, Decision Tree와 네 가지 Random Forest 설정을 2023 validation MSE로 비교한 뒤 한 모델을 선택합니다. 선택 후 train+validation 기간으로 다시 학습하고 2024를 최종 평가합니다.

## Models

- Linear Regression
- Ridge (`alpha=1`, `alpha=10`)
- Decision Tree
- Random Forest parameter candidates

모든 학습 모델은 `log1p(daily_visitors)`를 예측합니다. 평가 전 `expm1`으로 복원하고 음수 예측을 0으로 제한합니다.

## Baselines

- Train mean: 과거 학습 기간의 전체 평균을 예측
- POI-month-weekday seasonal mean: 과거 데이터의 `poi_id × month × day_of_week` 평균을 사용하고, 조합이 없으면 POI-month, POI, 전체 평균 순으로 대체

Validation baseline은 2020–2022 타깃만 사용하고, final test baseline은 2020–2023 타깃만 사용합니다. 미래 타깃은 평균 계산에 포함되지 않습니다.

## Results

2023 validation에서 Ridge (`alpha=1`)가 가장 낮은 MSE를 기록해 최종 모델로 선택되었습니다.

| Validation model | MSE | MAE | R² |
|---|---:|---:|---:|
| Ridge (`alpha=1`) | 4,079,333.3 | 1,204.1 | 0.9289 |
| Ridge (`alpha=10`) | 4,152,509.9 | 1,207.8 | 0.9276 |
| Linear Regression | 4,590,579.1 | 1,304.4 | 0.9199 |
| Best Random Forest candidate (`RF_C`) | 8,290,972.1 | 1,847.9 | 0.8554 |
| Seasonal baseline | 24,193,735.1 | 3,332.3 | 0.5781 |

선택된 Ridge를 2020–2023 데이터로 다시 학습한 뒤 얻은 2024 holdout 결과입니다.

| Final 2024 evaluation | MSE | MAE | R² |
|---|---:|---:|---:|
| Selected Ridge (`alpha=1`) | 12,643,390.6 | 2,309.2 | 0.8167 |
| Seasonal baseline | 24,887,899.6 | 3,358.6 | 0.6392 |
| Train mean baseline | 80,177,671.9 | 5,687.6 | -0.1623 |

이 수치는 수정된 evaluation pipeline을 2026-08-31에 실행해 재생성했습니다. 이전의 Random Forest R² 0.9273 결과는 2024 test set으로 모델을 선택한 평가에서 나온 값이므로 최종 성능으로 유지하지 않습니다.

## Run

```bash
pip install -r requirements.txt
python src/train_regression_models.py
```

주요 출력:

- `outputs/tables/model_selection_validation.csv`
- `outputs/tables/rf_parameter_sweep_validation.csv`
- `outputs/tables/final_test_metrics.csv`
- `outputs/tables/top_residual_cases.csv`
- `outputs/tables/evaluation_protocol.json`
- `outputs/figures/model_selection_validation.png`
- `outputs/figures/final_test_metrics.png`

## Data Source

현재 CSV의 provenance와 재배포 권한은 확인되지 않았습니다. 공개 저장소를 유지하기 전에 원본 수집 기록과 이용 조건을 수동으로 확인해야 합니다. 확인 결과 재배포가 불가능하면 데이터 파일을 별도 변경에서 제거하고 공식 다운로드 및 전처리 절차를 문서화해야 합니다.

## Limitations

- 단일 연도 validation과 test이므로 여러 시점에 대한 안정성을 충분히 평가하지 못합니다.
- 실제 미래 예측에서는 날씨 관련 입력을 예보 시점에 이용 가능한 값으로 바꿔야 합니다.
- POI와 시계열의 구조가 고정되어 있어 신규 관광지에 대한 성능은 별도로 검증되지 않았습니다.
- 수요 충격과 휴업·정책 변화 같은 외생 변수를 모두 반영하지 못합니다.
- 모델 계수와 Random Forest importance는 인과효과가 아닙니다.
- 데이터 출처와 라이선스가 문서화되지 않아 공개 배포 적합성을 확인해야 합니다.
