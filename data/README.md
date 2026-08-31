# Dataset Notes

## Included File

`seoul_tourism_2020_2024_clean_common.csv`

- Period: 2020-01-01 through 2024-12-31
- Unit: daily record by tourism point of interest (POI)
- Target: `daily_visitors`
- Size in the current repository: approximately 14 MB

## Field Groups

| Category | Examples |
|---|---|
| Time | `date`, `year`, `month`, `day`, `day_of_week`, holiday/weekend indicators |
| POI | `poi_id`, `poi_name`, `category`, access and facility attributes |
| Weather and air quality | temperature, humidity, precipitation, PM2.5, UV index, TCI |
| Calendar and events | `is_holiday`, `special_event`, season encoding |
| Target and visitor-derived fields | `daily_visitors`, `foreign_visitors`, `foreign_share`, `crowd_level` |

The training script parses the date, derives year/month/day-of-year fields, and excludes `foreign_visitors`, `foreign_share`, and `crowd_level` because they are not treated as available before the target is observed. Missing values and categorical encoding are fitted inside sklearn pipelines using the training period only.

## Source and Redistribution

The repository does not currently contain enough documentation to verify the dataset's provenance or redistribution permission. These fields must be completed from the original collection records:

- Provider: **TODO — not verified**
- Original URL: **TODO — not verified**
- License or terms of use: **TODO — not verified**
- Download date: **TODO — not verified**
- Redistribution rights for the included CSV: **TODO — not verified**

Until those fields are verified, the included CSV should not be assumed to be redistributable. Before merging or presenting the repository publicly, confirm the original source and terms. If redistribution is not allowed, remove the CSV from Git history in a separate, reviewed change and provide documented download/preparation instructions instead.
