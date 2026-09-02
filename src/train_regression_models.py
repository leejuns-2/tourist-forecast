"""Train and evaluate tourism visitor regressors with a time-aware holdout."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "seoul_tourism_2020_2024_clean_common.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

TRAIN_END_YEAR = 2022
VALIDATION_YEAR = 2023
TEST_YEAR = 2024
TARGET = "daily_visitors"
LEAKAGE_COLUMNS = ["foreign_visitors", "foreign_share", "crowd_level"]
EXPLICIT_CATEGORICAL_FEATURES = ["poi_id"]
FEATURE_GROUP_CANDIDATES = {
    "B1_structural_calendar": [
        "poi_id",
        "year",
        "month",
        "day",
        "day_of_year",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "special_event",
        "season_encoded",
    ],
    "environment": [
        "temp",
        "temp_feel",
        "humidity",
        "is_rainy",
        "precip",
        "pm25",
        "uv_index",
        "TCI",
    ],
    "poi_attributes": [
        "category",
        "outdoor_ratio",
        "is_free",
        "is_heritage",
        "night_open",
        "has_parking",
        "subway_access",
        "foreign_friendly",
        "accessibility_score",
    ],
}

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(filename: str) -> None:
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.relative_to(PROJECT_ROOT)}")


def regression_metrics(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype=float), 0, None)
    mse = mean_squared_error(y_true, prediction)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, prediction),
        "R2": r2_score(y_true, prediction),
    }


class SeasonalMeanBaseline:
    """Past-only POI/month/weekday mean with progressively broader fallbacks."""

    keys = ["poi_id", "month", "day_of_week"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalMeanBaseline":
        missing = [column for column in self.keys if column not in X.columns]
        if missing:
            raise ValueError(f"Seasonal baseline requires columns: {missing}")

        frame = X[self.keys].copy()
        frame["target"] = np.asarray(y)
        self.poi_month_weekday_ = frame.groupby(self.keys)["target"].mean().to_dict()
        self.poi_month_ = frame.groupby(["poi_id", "month"])["target"].mean().to_dict()
        self.poi_ = frame.groupby("poi_id")["target"].mean().to_dict()
        self.global_mean_ = float(frame["target"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions: list[float] = []
        for poi_id, month, day_of_week in X[self.keys].itertuples(index=False, name=None):
            value = self.poi_month_weekday_.get((poi_id, month, day_of_week))
            if value is None:
                value = self.poi_month_.get((poi_id, month))
            if value is None:
                value = self.poi_.get(poi_id, self.global_mean_)
            predictions.append(float(value))
        return np.asarray(predictions)


def write_data_audit(df: pd.DataFrame) -> None:
    """Write a machine-readable audit of the unmodified input table."""
    parsed_dates = pd.to_datetime(df["date"], errors="raise")
    feature_frame = df.drop(columns=[TARGET], errors="ignore").copy()
    feature_frame["poi_id"] = feature_frame["poi_id"].astype("string")
    numeric_columns, categorical_columns = split_feature_columns(feature_frame)

    unique_counts = feature_frame.nunique(dropna=False)
    constant_features = unique_counts[unique_counts <= 1].index.tolist()
    near_constant_features: dict[str, float] = {}
    for column in feature_frame.columns:
        counts = feature_frame[column].value_counts(normalize=True, dropna=False)
        if len(counts) > 1 and float(counts.iloc[0]) >= 0.99:
            near_constant_features[column] = float(counts.iloc[0])

    target_description = df[TARGET].describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    audit = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "date_range": {
            "min": str(parsed_dates.min().date()),
            "max": str(parsed_dates.max().date()),
        },
        "poi_count": int(df["poi_id"].nunique()),
        "rows_by_year": {
            str(year): int(count)
            for year, count in parsed_dates.dt.year.value_counts().sort_index().items()
        },
        "rows_by_poi": {
            str(poi): int(count)
            for poi, count in df["poi_id"].value_counts().sort_index().items()
        },
        "missing_by_column": {
            column: int(count) for column, count in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_date_poi_rows": int(df.duplicated(["date", "poi_id"]).sum()),
        "target_distribution": {
            key: float(value) for key, value in target_description.items()
        },
        "target_skewness": float(df[TARGET].skew()),
        "target_transform": "log1p for fitting; expm1 and non-negative clipping for evaluation",
        "constant_features": constant_features,
        "near_constant_features_dominant_share_at_least_0_99": near_constant_features,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "prediction_time_exclusions": {
            "foreign_visitors": "target-derived or unavailable before daily_visitors is observed",
            "foreign_share": "derived from daily and foreign visitor counts",
            "crowd_level": "treated as a post-observation target proxy",
        },
    }
    (TABLE_DIR / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    write_data_audit(df)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["poi_id"] = df["poi_id"].astype("string")
    df = df.dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    present_leakage = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    if present_leakage:
        print(f"Excluding fields unavailable at prediction time: {present_leakage}")
        df = df.drop(columns=present_leakage)
    return df


def split_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Route identifiers to categorical preprocessing regardless of source dtype."""
    explicit_categorical = [
        column for column in EXPLICIT_CATEGORICAL_FEATURES if column in X.columns
    ]
    numeric_columns = [
        column
        for column in X.select_dtypes(include=[np.number, "bool"]).columns
        if column not in explicit_categorical
    ]
    categorical_columns = explicit_categorical + [
        column
        for column in X.columns
        if column not in numeric_columns and column not in explicit_categorical
    ]
    return numeric_columns, categorical_columns


def build_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Build cumulative, prediction-time feature groups from columns that exist."""
    structural = [
        column for column in FEATURE_GROUP_CANDIDATES["B1_structural_calendar"] if column in df
    ]
    environment = [
        column for column in FEATURE_GROUP_CANDIDATES["environment"] if column in df
    ]
    poi_attributes = [
        column for column in FEATURE_GROUP_CANDIDATES["poi_attributes"] if column in df
    ]
    groups = {
        "B1_structural_calendar": structural,
        "B2_plus_environment": structural + environment,
        "B3_full_pre_observation": structural + environment + poi_attributes,
    }
    forbidden = set(LEAKAGE_COLUMNS + [TARGET, "poi_name", "date"])
    for name, columns in groups.items():
        overlap = forbidden.intersection(columns)
        if overlap:
            raise AssertionError(f"Forbidden fields in {name}: {sorted(overlap)}")
        if not columns:
            raise ValueError(f"No usable columns found for {name}")
    return groups


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns, categorical_columns = split_feature_columns(X)

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def make_pipeline(X: pd.DataFrame, estimator: object, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(X, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def candidate_models(X_train: pd.DataFrame) -> list[tuple[str, Pipeline, dict[str, object]]]:
    candidates: list[tuple[str, object, bool, dict[str, object]]] = [
        ("Linear Regression", LinearRegression(), True, {}),
        ("Ridge (alpha=1)", Ridge(alpha=1.0, solver="lsqr"), True, {"alpha": 1.0}),
        ("Ridge (alpha=10)", Ridge(alpha=10.0, solver="lsqr"), True, {"alpha": 10.0}),
        (
            "Decision Tree (depth=8)",
            DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42),
            False,
            {"max_depth": 8, "min_samples_leaf": 10},
        ),
        (
            "RF_A",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            False,
            {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
            },
        ),
        (
            "RF_B",
            RandomForestRegressor(
                n_estimators=250,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=0.7,
                random_state=42,
                n_jobs=-1,
            ),
            False,
            {
                "n_estimators": 250,
                "max_depth": 20,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": 0.7,
            },
        ),
        (
            "RF_C",
            RandomForestRegressor(
                n_estimators=250,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.7,
                random_state=42,
                n_jobs=-1,
            ),
            False,
            {
                "n_estimators": 250,
                "max_depth": None,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": 0.7,
            },
        ),
        (
            "RF_D",
            RandomForestRegressor(
                n_estimators=250,
                max_depth=None,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.5,
                random_state=42,
                n_jobs=-1,
            ),
            False,
            {
                "n_estimators": 250,
                "max_depth": None,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": 0.5,
            },
        ),
    ]
    return [
        (name, make_pipeline(X_train, estimator, scale_numeric), params)
        for name, estimator, scale_numeric, params in candidates
    ]


def candidate_pipeline(X_train: pd.DataFrame, selected_name: str) -> Pipeline:
    for name, pipeline, _ in candidate_models(X_train):
        if name == selected_name:
            return pipeline
    raise KeyError(f"Unknown candidate model: {selected_name}")


def grouped_error_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(group_column, observed=True, sort=True):
        metrics = regression_metrics(group["actual"], group["predicted"].to_numpy())
        rows.append(
            {
                group_column: value,
                "n": int(len(group)),
                "mean_actual": float(group["actual"].mean()),
                "mean_prediction": float(group["predicted"].mean()),
                "mean_residual": float(group["residual"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def plot_eda(df: pd.DataFrame) -> None:
    target_log = np.log1p(df[TARGET])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(df[TARGET], bins=50, edgecolor="black")
    axes[0].set_title("Visitor count distribution")
    axes[1].hist(target_log, bins=50, edgecolor="black")
    axes[1].set_title("log1p visitor count")
    axes[2].boxplot(df[TARGET])
    axes[2].set_title("Visitor count boxplot")
    save_figure("target_distribution.png")

    daily_total = df.groupby("date")[TARGET].sum()
    plt.figure(figsize=(14, 4))
    plt.plot(daily_total.index, daily_total.values, linewidth=0.7)
    plt.title("Daily total visitors over time")
    plt.xlabel("Date")
    plt.ylabel("Visitors")
    plt.grid(alpha=0.3)
    save_figure("daily_total_visitors.png")

    correlation_columns = [
        column
        for column in [
            "temp",
            "humidity",
            "precip",
            "pm25",
            "TCI",
            "outdoor_ratio",
            "accessibility_score",
            "year",
            "month",
            "day_of_year",
            TARGET,
        ]
        if column in df.columns
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[correlation_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    save_figure("correlation_matrix.png")


def main() -> None:
    df = load_dataset()
    plot_eda(df)

    feature_groups = build_feature_groups(df)
    full_feature_columns = feature_groups["B3_full_pre_observation"]
    X = df[full_feature_columns].copy()
    y = df[TARGET].copy()

    train_mask = df["year"] <= TRAIN_END_YEAR
    validation_mask = df["year"] == VALIDATION_YEAR
    test_mask = df["year"] == TEST_YEAR

    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Expected data in train (<=2022), validation (2023), and test (2024).")

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_validation, y_validation = X.loc[validation_mask], y.loc[validation_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    print(f"Train: {df.loc[train_mask, 'date'].min().date()} to {df.loc[train_mask, 'date'].max().date()} ({len(X_train):,} rows)")
    print(f"Validation: {df.loc[validation_mask, 'date'].min().date()} to {df.loc[validation_mask, 'date'].max().date()} ({len(X_validation):,} rows)")
    print(f"Test: {df.loc[test_mask, 'date'].min().date()} to {df.loc[test_mask, 'date'].max().date()} ({len(X_test):,} rows)")

    validation_rows: list[dict[str, object]] = []
    candidate_params: dict[str, dict[str, object]] = {}

    train_mean_prediction = np.full(len(y_validation), float(y_train.mean()))
    validation_rows.append(
        {"Model": "Train mean baseline", **regression_metrics(y_validation, train_mean_prediction)}
    )

    seasonal = SeasonalMeanBaseline().fit(X_train, y_train)
    seasonal_prediction = seasonal.predict(X_validation)
    validation_rows.append(
        {"Model": "POI-month-weekday seasonal baseline", **regression_metrics(y_validation, seasonal_prediction)}
    )

    y_train_log = np.log1p(y_train)
    for name, pipeline, params in candidate_models(X_train):
        print(f"Fitting {name} on 2020-2022; evaluating on 2023")
        pipeline.fit(X_train, y_train_log)
        prediction = np.clip(np.expm1(pipeline.predict(X_validation)), 0, None)
        validation_rows.append({"Model": name, **regression_metrics(y_validation, prediction)})
        candidate_params[name] = params

    validation_results = (
        pd.DataFrame(validation_rows).sort_values("MSE").reset_index(drop=True)
    )
    validation_results.to_csv(TABLE_DIR / "model_selection_validation.csv", index=False)
    print("\nModel selection results (2023 validation only):")
    print(validation_results.to_string(index=False))

    rf_validation_rows = []
    for row in validation_rows:
        name = str(row["Model"])
        if name.startswith("RF_"):
            rf_validation_rows.append({**row, **candidate_params[name]})
    pd.DataFrame(rf_validation_rows).sort_values("MSE").to_csv(
        TABLE_DIR / "rf_parameter_sweep_validation.csv", index=False
    )

    learned_results = validation_results[
        ~validation_results["Model"].isin(
            ["Train mean baseline", "POI-month-weekday seasonal baseline"]
        )
    ]
    selected_model_name = str(learned_results.iloc[0]["Model"])
    print(f"\nSelected model family from 2023 validation: {selected_model_name}")

    ablation_rows = [
        {
            "Feature_Group": "B0_seasonal_baseline",
            "Model": "POI-month-weekday seasonal baseline",
            "Features": "poi_id × month × day_of_week historical target means",
            **regression_metrics(y_validation, seasonal_prediction),
        }
    ]
    for group_name, group_columns in feature_groups.items():
        group_train = X_train[group_columns]
        group_validation = X_validation[group_columns]
        pipeline = candidate_pipeline(group_train, selected_model_name)
        pipeline.fit(group_train, y_train_log)
        prediction = np.clip(np.expm1(pipeline.predict(group_validation)), 0, None)
        ablation_rows.append(
            {
                "Feature_Group": group_name,
                "Model": selected_model_name,
                "Features": " | ".join(group_columns),
                **regression_metrics(y_validation, prediction),
            }
        )
    ablation_results = (
        pd.DataFrame(ablation_rows).sort_values("MSE").reset_index(drop=True)
    )
    ablation_results.to_csv(TABLE_DIR / "feature_ablation_validation.csv", index=False)
    print("\nFeature ablation results (2023 validation only):")
    print(ablation_results.to_string(index=False))

    selected_feature_group = str(ablation_results.iloc[0]["Feature_Group"])
    final_model_name = (
        "POI-month-weekday seasonal baseline"
        if selected_feature_group == "B0_seasonal_baseline"
        else selected_model_name
    )
    print(
        "\nSelected configuration from 2023 validation: "
        f"{selected_model_name} with {selected_feature_group}"
    )

    train_validation_mask = train_mask | validation_mask
    X_train_validation = X.loc[train_validation_mask]
    y_train_validation = y.loc[train_validation_mask]

    if selected_feature_group == "B0_seasonal_baseline":
        final_model = SeasonalMeanBaseline().fit(X_train_validation, y_train_validation)
        final_prediction = final_model.predict(X_test)
    else:
        selected_columns = feature_groups[selected_feature_group]
        selected_pipeline = candidate_pipeline(
            X_train_validation[selected_columns], selected_model_name
        )
        selected_pipeline.fit(
            X_train_validation[selected_columns], np.log1p(y_train_validation)
        )
        final_model = selected_pipeline
        final_prediction = np.clip(
            np.expm1(selected_pipeline.predict(X_test[selected_columns])), 0, None
        )

    test_rows = [
        {
            "Model": f"Selected: {final_model_name} [{selected_feature_group}]",
            **regression_metrics(y_test, final_prediction),
        },
    ]
    test_train_mean = np.full(len(y_test), float(y_train_validation.mean()))
    test_rows.append(
        {"Model": "Train mean baseline", **regression_metrics(y_test, test_train_mean)}
    )
    test_seasonal = SeasonalMeanBaseline().fit(X_train_validation, y_train_validation).predict(X_test)
    test_rows.append(
        {
            "Model": "POI-month-weekday seasonal baseline",
            **regression_metrics(y_test, test_seasonal),
        }
    )
    final_test_results = pd.DataFrame(test_rows).sort_values("MSE").reset_index(drop=True)
    final_test_results.to_csv(TABLE_DIR / "final_test_metrics.csv", index=False)
    print("\nFinal 2024 test results (evaluated after selection):")
    print(final_test_results.to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.barh(validation_results["Model"][::-1], validation_results["MSE"][::-1])
    plt.xlabel("2023 validation MSE")
    plt.title("Model selection on validation period")
    save_figure("model_selection_validation.png")

    plt.figure(figsize=(9, 4))
    plt.bar(final_test_results["Model"], final_test_results["MSE"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("2024 test MSE")
    plt.title("Final holdout evaluation")
    save_figure("final_test_metrics.png")

    residuals = y_test.to_numpy() - final_prediction
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, final_prediction, alpha=0.4, edgecolor="white")
    lower = min(float(y_test.min()), float(final_prediction.min()))
    upper = max(float(y_test.max()), float(final_prediction.max()))
    axes[0].plot([lower, upper], [lower, upper], "--", linewidth=2)
    axes[0].set_xlabel("Actual visitors")
    axes[0].set_ylabel("Predicted visitors")
    axes[0].set_title(f"Actual vs predicted ({selected_model_name})")
    axes[1].scatter(final_prediction, residuals, alpha=0.4, edgecolor="white")
    axes[1].axhline(0, linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted visitors")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual plot")
    save_figure("actual_vs_predicted_and_residuals.png")

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, edgecolor="black")
    plt.title("2024 test residuals")
    plt.xlabel("Actual - predicted")
    save_figure("residuals_histogram.png")

    test_metadata = df.loc[
        test_mask, ["date", "month", "poi_id", "poi_name"]
    ].reset_index(drop=True)
    residual_table = test_metadata.assign(
        actual=y_test.reset_index(drop=True),
        predicted=final_prediction,
        residual=residuals,
        absolute_residual=np.abs(residuals),
    )
    residual_table.nlargest(10, "absolute_residual").to_csv(
        TABLE_DIR / "top_residual_cases.csv", index=False
    )
    residual_table.to_csv(TABLE_DIR / "final_predictions.csv", index=False)
    grouped_error_table(residual_table, "month").to_csv(
        TABLE_DIR / "error_by_month.csv", index=False
    )
    grouped_error_table(residual_table, "poi_id").sort_values(
        "MAE", ascending=False
    ).to_csv(TABLE_DIR / "error_by_poi.csv", index=False)
    residual_table["visitor_volume_range"] = pd.qcut(
        residual_table["actual"],
        q=4,
        labels=["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"],
    )
    grouped_error_table(residual_table, "visitor_volume_range").to_csv(
        TABLE_DIR / "error_by_visitor_volume.csv", index=False
    )
    residual_summary = {
        "definition": "residual = actual - predicted",
        "mean": float(residual_table["residual"].mean()),
        "median": float(residual_table["residual"].median()),
        "std": float(residual_table["residual"].std()),
        "quantiles": {
            str(key): float(value)
            for key, value in residual_table["residual"].quantile(
                [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            ).items()
        },
    }
    (TABLE_DIR / "residual_summary.json").write_text(
        json.dumps(residual_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    importance_path = TABLE_DIR / "top_feature_importance.csv"
    if isinstance(final_model, Pipeline) and hasattr(final_model.named_steps["model"], "feature_importances_"):
        names = final_model.named_steps["preprocess"].get_feature_names_out()
        importances = final_model.named_steps["model"].feature_importances_
        importance = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        importance.to_csv(importance_path, index=False)
        plt.figure(figsize=(9, 6))
        plt.barh(importance["feature"][::-1], importance["importance"][::-1])
        plt.xlabel("Impurity-based importance")
        plt.title(f"Top features ({selected_model_name})")
        save_figure("top_feature_importance.png")
    else:
        if importance_path.exists():
            importance_path.unlink()
        importance_figure = FIGURE_DIR / "top_feature_importance.png"
        if importance_figure.exists():
            importance_figure.unlink()
        print("Selected model has no impurity-based feature importance; no feature-importance artifact was written.")

    summary = {
        "train_period": f"2020-{TRAIN_END_YEAR}",
        "validation_period": str(VALIDATION_YEAR),
        "test_period": str(TEST_YEAR),
        "selected_model": final_model_name,
        "selected_learned_model_family": selected_model_name,
        "selected_feature_group": selected_feature_group,
        "feature_groups": feature_groups,
        "selection_metric": "validation MSE on original visitor scale",
        "test_evaluated_after_selection": True,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "explicit_categorical_features": EXPLICIT_CATEGORICAL_FEATURES,
    }
    (TABLE_DIR / "evaluation_protocol.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
