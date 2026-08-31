import unittest

import pandas as pd

from src.train_regression_models import make_preprocessor, split_feature_columns


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


if __name__ == "__main__":
    unittest.main()
