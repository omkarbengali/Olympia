"""
Unit tests for ETL pipeline modules.
"""

import unittest
import pandas as pd
from etl.extract import extract_data, get_default_csv_path
from etl.transform import transform_data, MEDAL_POINTS_MAP


class TestETLPipeline(unittest.TestCase):

    def test_extract_data(self):
        """Verify extraction returns DataFrame with expected columns and non-empty rows."""
        df, meta = extract_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 20000)
        expected_cols = {
            "season", "year", "medal", "country_code", "country",
            "athletes", "games", "sport", "event_gender", "event_name"
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        self.assertEqual(meta["column_count"], 10)

    def test_transform_data(self):
        """Verify transformation cleaning rules, medal points, and null handling."""
        raw_df, _ = extract_data()
        clean_df, metrics = transform_data(raw_df)

        # Duplicates check
        self.assertEqual(clean_df.duplicated().sum(), 0)
        self.assertEqual(metrics["duplicates_removed"], 3)

        # Non-awarded events check (null medals)
        self.assertEqual(metrics["non_awarded_events_removed"], 4)
        self.assertEqual(clean_df["medal"].isna().sum(), 0)

        # Missing athlete imputation check
        self.assertEqual(clean_df["athletes"].isna().sum(), 0)
        self.assertIn("Team / Not Listed", clean_df["athletes"].values)

        # Medal points validation
        self.assertIn("medal_points", clean_df.columns)
        self.assertTrue(set(clean_df["medal_points"].unique()).issubset({1, 2, 3}))
        for medal_name, pts in MEDAL_POINTS_MAP.items():
            subset = clean_df[clean_df["medal"] == medal_name]
            self.assertTrue((subset["medal_points"] == pts).all())

        # Total clean records
        self.assertEqual(len(clean_df), 20240)


if __name__ == "__main__":
    unittest.main()
