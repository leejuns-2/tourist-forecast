import unittest

import numpy as np
import pandas as pd

from src.train_regression_models import (
    build_feature_groups,
    make_preprocessor,
    regression_metrics,
    split_feature_columns,
)


class PreprocessingTest(unittest.TestCase):
    def test_poi_id_is_one_hot_categorical_even_when_input_is_numeric(self) -> None:
        features = pd.DataFrame(
            {
                "poi_id": [101, 202],
                "temperature": [20.0, 21.5],
                "district": ["A", "B"],
            }
        )

        numeric_columns, categorical_columns = split_feature_columns(features)
        self.assertNotIn("poi_id", numeric_columns)
        self.assertIn("poi_id", categorical_columns)

        preprocessor = make_preprocessor(features, scale_numeric=True)
        transformers = {
            name: (transformer, columns)
            for name, transformer, columns in preprocessor.transformers
        }
        categorical_pipeline, routed_columns = transformers["categorical"]

        self.assertIn("poi_id", routed_columns)
        self.assertNotIn("poi_id", transformers["numeric"][1])
        self.assertEqual(
            categorical_pipeline.named_steps["onehot"].handle_unknown,
            "ignore",
        )

        preprocessor.fit_transform(features)
        transformed_names = preprocessor.get_feature_names_out().tolist()
        self.assertTrue(
            any(name.startswith("categorical__poi_id_") for name in transformed_names)
        )
        self.assertFalse(any(name == "numeric__poi_id" for name in transformed_names))

    def test_feature_groups_exclude_target_derived_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "poi_id": ["POI001"],
                "month": [1],
                "day_of_week": [2],
                "temp": [10.0],
                "category": ["museum"],
                "daily_visitors": [100],
                "foreign_visitors": [10],
                "foreign_share": [0.1],
                "crowd_level": ["low"],
            }
        )
        groups = build_feature_groups(frame)
        for columns in groups.values():
            self.assertNotIn("daily_visitors", columns)
            self.assertNotIn("foreign_visitors", columns)
            self.assertNotIn("foreign_share", columns)
            self.assertNotIn("crowd_level", columns)

    def test_metrics_include_rmse_on_original_scale(self) -> None:
        metrics = regression_metrics(pd.Series([1.0, 5.0]), np.array([1.0, 3.0]))
        self.assertAlmostEqual(metrics["MSE"], 2.0)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(2.0))


if __name__ == "__main__":
    unittest.main()
