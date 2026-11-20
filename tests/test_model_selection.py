import unittest

import pandas as pd

from src.transaction_labeling.train import select_variant


class ModelSelectionTests(unittest.TestCase):
    def test_selects_xgboost_when_it_has_the_best_cross_validation_score(self):
        summary = pd.DataFrame({
            "model": ["logistic_unweighted", "logistic_balanced", "xgboost_unweighted", "xgboost_balanced"],
            "macro_f1_mean": [0.80, 0.81, 0.86, 0.82],
        })

        self.assertEqual(select_variant(summary), "xgboost_unweighted")

    def test_exact_tie_prefers_unweighted_logistic(self):
        summary = pd.DataFrame({
            "model": ["xgboost_balanced", "logistic_balanced", "xgboost_unweighted", "logistic_unweighted"],
            "macro_f1_mean": [0.80, 0.80, 0.80, 0.80],
        })

        self.assertEqual(select_variant(summary), "logistic_unweighted")


if __name__ == "__main__":
    unittest.main()
