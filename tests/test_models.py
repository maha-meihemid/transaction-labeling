import unittest

import numpy as np
import pandas as pd

from src.transaction_labeling.models import build_model, predict_minor


class FixedModel:
    def predict(self, rows):
        return np.array([1, 0])


class PredictionDecodingTests(unittest.TestCase):
    def test_decodes_integer_predictions_back_to_category_labels(self):
        rows = pd.DataFrame(index=[5, 9])
        bundle = {
            "model": FixedModel(),
            "label_encoding": "integer",
            "classes": ["atm", "refund"],
        }

        predictions = predict_minor(bundle, rows)

        self.assertEqual(predictions.tolist(), ["refund", "atm"])
        self.assertEqual(predictions.index.tolist(), [5, 9])

    def test_character_candidate_uses_character_tfidf_and_selected_c(self):
        model = build_model("logistic_word_char_unweighted_c4")
        feature_pipeline = model.named_steps["features"]
        classifier = model.named_steps["classifier"]

        self.assertIn("title_char", [name for name, _, _ in feature_pipeline.transformers])
        self.assertEqual(classifier.C, 4.0)


if __name__ == "__main__":
    unittest.main()
