import unittest

import pandas as pd

from src.transaction_labeling.targets import build_major_category_mapping, map_minor_to_major


class MajorCategoryMappingTests(unittest.TestCase):
    def test_builds_unambiguous_mapping_and_preserves_refund_as_null(self):
        rows = pd.DataFrame({
            "minor_category": ["restaurants", "restaurants", "refund"],
            "major_category": ["restaurants_bars_cafes", "restaurants_bars_cafes", None],
        })

        mapping = build_major_category_mapping(rows)
        result = map_minor_to_major(pd.Series(["restaurants", "refund"]), mapping)

        self.assertEqual(mapping["restaurants"], "restaurants_bars_cafes")
        self.assertEqual(result.iloc[0], "restaurants_bars_cafes")
        self.assertIsNone(result.iloc[1])

    def test_rejects_a_minor_category_assigned_to_multiple_major_categories(self):
        rows = pd.DataFrame({
            "minor_category": ["restaurants", "restaurants"],
            "major_category": ["restaurants_bars_cafes", "shopping"],
        })

        with self.assertRaises(ValueError):
            build_major_category_mapping(rows)


if __name__ == "__main__":
    unittest.main()
