# Seoul Tourism Visitor Forecast

## Overview

서울 주요 관광지의 일별 방문객 수를 회귀 모델로 예측하고, 미래 연도에 대한 성능을 시간 순서대로 평가하는 프로젝트입니다. 모델은 방문객 수의 `log1p` 값을 학습하고, 원래 방문객 수 단위로 복원해 MSE, MAE, R²를 계산합니다.

Primary question: 과거 기간에서 학습한 POI·달력·환경 정보를 이용한 회귀 모델이 historical/seasonal baseline보다 미래 기간의 일별 방문객 수를 더 정확하게 예측할 수 있는가?

Secondary question: POI와 달력 정보에 weather, air quality, TCI 등의 environmental feature를 추가했을 때 미래 예측 오차가 추가로 줄어드는가? 이 프로젝트는 인과추론이 아니며 환경 feature의 효과를 주장하지 않습니다.

## Data

- Period: 2020-01-01 through 2024-12-31
- Unit: day × tourism point of interest (POI)
- Target: `daily_visitors`
- Feature groups: calendar, POI attributes, weather, air quality, events, accessibility

이 프로젝트의 현재 분석 입력은 사용자가 첨부해 제공한 `seoul_tourism_2020_2024_clean_common.csv`입니다. 첨부 파일과 저장소의 동명 CSV는 직렬화 방식에는 차이가 있지만, 파싱한 91,350행 × 32열의 모든 셀 값이 동일함을 확인했습니다. 이 설명은 이번 작업에서 사용한 파일의 전달 경로를 뜻하며, 최초 제공기관이나 수집 원출처를 뜻하지 않습니다.

테이블 감사 결과는 [`outputs/tables/data_audit.json`](outputs/tables/data_audit.json)에 저장됩니다. 현재 파일은 50개 POI이며 결측치·완전 중복·`date × poi_id` 중복은 각각 0건입니다. `daily_visitors`의 평균은 6,227.8, 중앙값은 3,824, 최댓값은 73,322이고 왜도는 2.39입니다. 이 오른쪽 꼬리와 음수 예측 방지를 고려해 `log1p` 타깃 학습을 사용합니다.

`foreign_visitors`, `foreign_share`, `crowd_level`은 목표값이 관측되기 전에 알 수 있다고 보기 어려워 모델에서 제외합니다. 관광지 식별자인 `poi_id`는 연속형 수치가 아니라 명시적인 categorical feature로 고정해 `OneHotEncoder(handle_unknown="ignore")`로 처리하며, numeric transformer에서는 제외합니다. 범주형 인코딩과 결측치 처리는 sklearn `Pipeline`과 `ColumnTransformer` 안에서 학습 기간에만 fit됩니다.

현재 분석 파일의 직접 제공 경로는 사용자 첨부 파일로 확인했습니다. 다만 최초 데이터 제공기관, 원본 URL, 라이선스, 최초 다운로드 날짜, 재배포 가능 여부는 확인되지 않았습니다. 자세한 TODO는 [`data/README.md`](data/README.md)에 있습니다. 재배포 권한을 확인하기 전에는 포함된 CSV를 공개 배포 가능하다고 가정하면 안 됩니다.

## Evaluation Design

```text
2020-2022  train and fit preprocessing
2023       validation for model and hyperparameter selection
2020-2023  retrain the selected configuration
2024       final holdout evaluation
```

2024 결과는 모델 종류나 하이퍼파라미터, feature group 선택에 사용하지 않습니다. Linear Regression, Ridge, Decision Tree와 네 가지 Random Forest 설정을 2023 validation MSE로 비교한 뒤 모델 family를 정하고, 같은 validation에서 B1–B3 feature configuration을 정합니다. 선택 후 train+validation 기간으로 다시 학습하고 2024를 한 번 최종 평가합니다.

Primary selection metric은 MSE입니다. 큰 방문객 수 오차에 더 큰 비용을 부여해 급증일의 실패를 모델 선택에 반영하기 위해 선택했으며, 해석 편의를 위해 RMSE·MAE·R²도 원래 방문객 수 단위로 함께 저장합니다.

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

| Validation model | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Ridge (`alpha=1`) | 4,079,253.9 | 2,019.7 | 1,204.1 | 0.9289 |
| Ridge (`alpha=10`) | 4,146,529.6 | 2,036.3 | 1,207.6 | 0.9277 |
| Linear Regression | 4,590,562.3 | 2,142.6 | 1,304.4 | 0.9199 |
| Best Random Forest candidate (`RF_C`) | 8,297,459.2 | 2,880.5 | 1,848.0 | 0.8553 |
| Seasonal baseline | 24,193,735.1 | 4,918.7 | 3,332.3 | 0.5781 |

### Feature ablation on 2023 validation

모델 family를 Ridge (`alpha=1`)로 정한 뒤, 2024를 보지 않고 다음 cumulative feature group을 비교했습니다.

| Group | Included information | MSE | RMSE | MAE | R² |
|---|---|---:|---:|---:|---:|
| B0 | past-only POI × month × weekday mean | 24,193,735.1 | 4,918.7 | 3,332.3 | 0.5781 |
| B1 | POI identifier + calendar/event | 7,602,321.2 | 2,757.2 | 1,648.2 | 0.8674 |
| B2 | B1 + weather/air quality/TCI | 4,080,262.2 | 2,020.0 | 1,204.2 | 0.9288 |
| B3 | B2 + available POI attributes | 4,079,253.9 | 2,019.7 | 1,204.1 | 0.9289 |

B2는 B1보다 validation MSE가 약 46.3% 낮았습니다. 이는 해당 환경 변수가 이 데이터에서 추가 predictive signal을 제공했다는 뜻이며 관광객 수를 변화시킨 인과효과를 뜻하지 않습니다. B3의 B2 대비 개선은 매우 작습니다.

선택된 Ridge를 2020–2023 데이터로 다시 학습한 뒤 얻은 2024 holdout 결과입니다.

| Final 2024 evaluation | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Selected Ridge (`alpha=1`, B3) | 12,642,553.2 | 3,555.6 | 2,309.2 | 0.8167 |
| Seasonal baseline | 24,887,899.6 | 4,988.8 | 3,358.6 | 0.6392 |
| Train mean baseline | 80,177,671.9 | 8,954.2 | 5,687.6 | -0.1623 |

이 수치는 수정된 evaluation pipeline을 2026-09-02에 실행해 재생성했습니다. 이전의 Random Forest R² 0.9273 결과는 2024 test set으로 모델을 선택한 평가에서 나온 값이므로 최종 성능으로 유지하지 않습니다.

## Error Analysis

2024에서 가장 큰 절대오차는 2024-05-05 코엑스몰(`POI046`)의 실제 36,884명 대비 예측 약 82,614명인 과대예측이었습니다. POI별 MAE는 `POI046`이 8,167.2로 가장 컸고, 이어 `POI021` 6,269.2, `POI006` 5,898.4였습니다. 월별 MAE는 9월 3,456.5, 5월 3,416.5가 가장 컸습니다. 방문량 사분위가 높아질수록 MAE가 커지고 모든 구간의 평균 residual이 음수여서 전반적인 과대예측 패턴이 보입니다. 데이터만으로 그 원인을 단정하지 않습니다.

세부 파일은 `final_predictions.csv`, `error_by_month.csv`, `error_by_poi.csv`, `error_by_visitor_volume.csv`, `residual_summary.json`입니다.

## Run

```bash
pip install -r requirements.txt
python src/train_regression_models.py
```

주요 출력:

- `outputs/tables/model_selection_validation.csv`
- `outputs/tables/rf_parameter_sweep_validation.csv`
- `outputs/tables/final_test_metrics.csv`
- `outputs/tables/feature_ablation_validation.csv`
- `outputs/tables/data_audit.json`
- `outputs/tables/top_residual_cases.csv`
- `outputs/tables/evaluation_protocol.json`
- `outputs/figures/model_selection_validation.png`
- `outputs/figures/final_test_metrics.png`

## Data Source

과거 기록에는 관광/POI 출처가 “Seoul Open Data Portal – Tourism POI & visitor statistics”, 날씨 출처가 “Seoul Historical Weather Data”, Kaggle 식별자가 `alfredkondoro/seoul-historical-weather-data-2024`, 중간 파일이 `seoul_poi_weather_tci_merged.csv`로 남아 있습니다. 그러나 정확한 서울 열린데이터광장 dataset 이름·URL, 전체 preprocessing lineage, feature engineering 정의 일부, `daily_visitors`가 공식 관측치인지 여부, 라이선스와 재배포 권한은 복원되지 않았습니다. publication이나 외부 연구 전에 원자료와 이용 조건을 재확인해야 합니다.

## Limitations

- 단일 연도 validation과 test이므로 여러 시점에 대한 안정성을 충분히 평가하지 못합니다.
- 실제 미래 예측에서는 날씨 관련 입력을 예보 시점에 이용 가능한 값으로 바꿔야 합니다.
- POI와 시계열의 구조가 고정되어 있어 신규 관광지에 대한 성능은 별도로 검증되지 않았습니다.
- 수요 충격과 휴업·정책 변화 같은 외생 변수를 모두 반영하지 못합니다.
- 모델 계수와 Random Forest importance는 인과효과가 아닙니다.
- 데이터 출처와 라이선스가 문서화되지 않아 공개 배포 적합성을 확인해야 합니다.
