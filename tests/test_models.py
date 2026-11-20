import unittest

import numpy as np
import pandas as pd

from src.transaction_labeling.models import predict_minor


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


if __name__ == "__main__":
    unittest.main()
