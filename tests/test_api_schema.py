import unittest

from pydantic import ValidationError

from src.transaction_labeling.schemas import Transaction


class TransactionSchemaTests(unittest.TestCase):
    def test_accepts_timezone_aware_transaction_and_strips_title(self):
        transaction = Transaction.model_validate({
            "title": "  CARREFOUR  ",
            "amount_value": -12.5,
            "created_at": "2026-06-11T12:30:00Z",
        })

        self.assertEqual(transaction.title, "CARREFOUR")
        self.assertEqual(transaction.fees, 0)

    def test_rejects_invalid_time_fee_and_non_finite_amount(self):
        base = {
            "title": "CARREFOUR",
            "amount_value": -12.5,
            "created_at": "2026-06-11T12:30:00Z",
        }
        invalid_cases = [
            {**base, "created_at": "2026-06-11T12:30:00"},
            {**base, "fees": -1},
            {**base, "amount_value": float("inf")},
            {**base, "title": "   "},
        ]

        for payload in invalid_cases:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                Transaction.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
