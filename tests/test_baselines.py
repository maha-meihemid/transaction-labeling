import unittest

import pandas as pd

from src.transaction_labeling.baselines import (
    fit_title_history_rule,
    predict_title_history_rule,
)


class TitleHistoryBaselineTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "title": ["Cafe", "CAFE", "Cafe", "Market", "Market", "Refund"],
            "amount_value": [-8, -12, -5, -10, -20, 15],
            "minor_category": [
                "restaurants", "restaurants", "supermarkets",
                "supermarkets", "supermarkets", "refund",
            ],
        })

    def test_uses_past_title_mode_refund_signal_and_majority_fallback(self):
        rule = fit_title_history_rule(self.train)
        rows = pd.DataFrame({
            "title": [" cafe ", "New shop", "New refund"],
            "amount_value": [-4, -7, 20],
        }, index=[4, 8, 12])

        predictions = predict_title_history_rule(rule, rows)

        self.assertEqual(predictions.tolist(), [
            "restaurants", "supermarkets", "refund",
        ])
        self.assertEqual(predictions.index.tolist(), [4, 8, 12])

    def test_does_not_apply_refund_sign_rule_when_train_is_not_perfect(self):
        train = pd.concat([
            self.train,
            pd.DataFrame({
                "title": ["Positive purchase"],
                "amount_value": [5],
                "minor_category": ["shopping"],
            }),
        ], ignore_index=True)

        rule = fit_title_history_rule(train)

        self.assertFalse(rule["refund_sign_rule"])


if __name__ == "__main__":
    unittest.main()
