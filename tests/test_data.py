import unittest

import pandas as pd

from src.transaction_labeling.data import remove_exact_ref_duplicates


class RemoveExactRefDuplicatesTests(unittest.TestCase):
    def test_removes_one_identical_record_per_repeated_ref_id(self):
        rows = pd.DataFrame([
            {"transaction_id": "tx-1", "ref_id": "ref-1", "created_at": "2026-01-01", "minor_category": "refund", "major_category": None},
            {"transaction_id": "tx-2", "ref_id": "ref-1", "created_at": "2026-01-01", "minor_category": "refund", "major_category": None},
        ])

        cleaned, audit = remove_exact_ref_duplicates(rows)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(audit), 1)
        self.assertEqual(cleaned.iloc[0]["transaction_id"], "tx-1")

    def test_keeps_data_unchanged_when_repeated_ref_id_has_conflicting_labels(self):
        rows = pd.DataFrame([
            {"transaction_id": "tx-1", "ref_id": "ref-1", "created_at": "2026-01-01", "minor_category": "refund", "major_category": None},
            {"transaction_id": "tx-2", "ref_id": "ref-1", "created_at": "2026-01-01", "minor_category": "supermarkets", "major_category": "groceries"},
        ])

        with self.assertRaises(ValueError):
            remove_exact_ref_duplicates(rows)


if __name__ == "__main__":
    unittest.main()
