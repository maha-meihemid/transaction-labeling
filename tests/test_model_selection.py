import unittest

import pandas as pd

from src.transaction_labeling.models import MODEL_CANDIDATES
from src.transaction_labeling.train import select_variant


class ModelSelectionTests(unittest.TestCase):
    def test_selects_xgboost_when_it_has_the_best_cross_validation_score(self):
        names = [candidate["name"] for candidate in MODEL_CANDIDATES]
        scores = [0.80] * len(names)
        selected_name = "xgboost_word_balanced"
        scores[names.index(selected_name)] = 0.86
        summary = pd.DataFrame({
            "model": [*names, "title_history_rule"],
            "macro_f1_mean": [*scores, 0.99],
        })

        self.assertEqual(select_variant(summary), selected_name)

    def test_exact_tie_prefers_unweighted_logistic(self):
        names = [candidate["name"] for candidate in MODEL_CANDIDATES]
        summary = pd.DataFrame({
            "model": [*reversed(names), "title_history_rule"],
            "macro_f1_mean": [*([0.80] * len(names)), 0.99],
        })

        self.assertEqual(select_variant(summary), names[0])

    def test_requires_every_model_candidate_but_not_a_baseline_score(self):
        summary = pd.DataFrame({
            "model": ["title_history_rule"],
            "macro_f1_mean": [0.99],
        })

        with self.assertRaises(ValueError):
            select_variant(summary)


if __name__ == "__main__":
    unittest.main()
