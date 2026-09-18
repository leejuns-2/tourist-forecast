# Dataset Notes

## Dataset Used in This Project

이 프로젝트에서는 다음 정제 데이터셋을 분석 입력으로 사용했습니다.

`seoul_tourism_2020_2024_clean_common.csv`

현재 확인된 데이터 특성:

- Period: 2020-01-01 through 2024-12-31
- Unit: daily record by tourism point of interest (POI)
- Rows: 91,350
- Columns: 32
- POIs: 50
- Target: `daily_visitors`

현재 분석 파일에서는 결측치, 완전 중복,
`date × poi_id` 중복이 각각 0건입니다.

---

## Field Groups

| Category | Examples |
|---|---|
| Time | `date`, `year`, `month`, `day`, `day_of_week`, holiday/weekend indicators |
| POI | `poi_id`, `poi_name`, `category`, access and facility attributes |
| Weather and air quality | temperature, humidity, precipitation, PM2.5, UV index, TCI |
| Calendar and events | `is_holiday`, `special_event`, season encoding |
| Target and visitor-derived fields | `daily_visitors`, `foreign_visitors`, `foreign_share`, `crowd_level` |

The training pipeline parses the date and derives additional temporal features.

`foreign_visitors`, `foreign_share`, and `crowd_level` are excluded from the
prediction features because they are treated as unavailable or target-derived
at the time `daily_visitors` would be predicted.

Missing-value processing and categorical encoding are fitted inside
scikit-learn pipelines using the training period only.

---

## Data Provenance

현재 저장소에 남아 있는 과거 기록을 통해
다음과 같은 데이터 출처 관련 단서를 확인했습니다.

- Tourism / POI:
  `Seoul Open Data Portal – Tourism POI & visitor statistics`
- Weather:
  `Seoul Historical Weather Data`
- Recorded Kaggle identifier:
  `alfredkondoro/seoul-historical-weather-data-2024`
- Historical intermediate file:
  `seoul_poi_weather_tci_merged.csv`
- Current analysis file:
  `seoul_tourism_2020_2024_clean_common.csv`

다만 위 기록만으로는 원 데이터의 정확한 provenance를
완전히 검증할 수 없습니다.

현재 확인되지 않은 항목:

- Original provider: **not fully verified**
- Original dataset URL: **not fully verified**
- License / terms of use: **not verified**
- Original download date: **not verified**
- Redistribution rights: **not verified**
- Full preprocessing lineage: **not fully recovered**
- Definitions of some engineered features: **not fully recovered**
- Whether `daily_visitors` represents an official observed visitor count:
  **not fully verified**

---

## Redistribution Note

원 데이터의 라이선스와 재배포 조건이 완전히 확인되지 않았기 때문에,
이 데이터셋을 공개적으로 재배포할 수 있다고 가정하지 않습니다.

외부 연구, publication 또는 추가 배포 전에
원 데이터의 제공기관과 이용 조건을 다시 확인해야 합니다.

재배포가 허용되지 않는 것으로 확인될 경우,
공개 저장소에서는 원 데이터 파일을 제외하고
데이터를 얻고 준비하는 방법만 문서화하는 방식으로 변경해야 합니다.
