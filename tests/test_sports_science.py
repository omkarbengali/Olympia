"""
Unit tests for analytics/sports_science.py modules:
1. Performance Benchmarking & Peak Years
2. Consistency Scoring
3. Sport Specialization Index
4. Talent Distribution Pipeline
5. Competitive Parity Analyzer
"""

import unittest
import pandas as pd
from analytics.sports_science import (
    compute_peak_performance_window,
    compute_consistency_scores,
    compute_sport_specialization_index,
    compute_talent_pipeline,
    compute_competitive_parity
)


class TestSportsScience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load processed dataset
        cls.df = pd.read_csv("data/processed/cleaned_olympic_medalists.csv")

    def test_peak_performance_window(self):
        res = compute_peak_performance_window(self.df, sport="Swimming", medal_filter="Gold")
        self.assertIn("mean_age", res)
        self.assertIn("std_age", res)
        self.assertIn("peak_window_min", res)
        self.assertIn("peak_window_max", res)
        self.assertGreater(res["mean_age"], 15.0)
        self.assertLess(res["mean_age"], 35.0)
        self.assertGreaterEqual(res["sample_size"], 10)

    def test_consistency_scoring_nation(self):
        nation_scores = compute_consistency_scores(self.df, mode="Nation", top_n=10)
        self.assertFalse(nation_scores.empty)
        self.assertIn("consistency_score", nation_scores.columns)
        self.assertIn("mean_per_cycle", nation_scores.columns)
        self.assertIn("std_per_cycle", nation_scores.columns)

    def test_consistency_scoring_athlete(self):
        athlete_scores = compute_consistency_scores(self.df, mode="Athlete", top_n=10)
        self.assertFalse(athlete_scores.empty)
        self.assertIn("consistency_score", athlete_scores.columns)
        self.assertIn("editions_count", athlete_scores.columns)
        self.assertGreaterEqual(athlete_scores.iloc[0]["editions_count"], 2)

    def test_specialization_index(self):
        spec_df = compute_sport_specialization_index(self.df, min_medals=25)
        self.assertFalse(spec_df.empty)
        self.assertIn("specialization_index", spec_df.columns)
        self.assertIn("diversification_index", spec_df.columns)
        self.assertIn("top_sport_share_pct", spec_df.columns)
        self.assertIn("archetype", spec_df.columns)

    def test_talent_pipeline(self):
        pipe = compute_talent_pipeline(self.df)
        self.assertIn("breakthrough_df", pipe)
        self.assertIn("era_summary", pipe)
        self.assertIn("latest_breakthrough_rate", pipe)
        self.assertFalse(pipe["breakthrough_df"].empty)
        self.assertFalse(pipe["era_summary"].empty)

    def test_competitive_parity(self):
        # Table Tennis should show higher concentration than Athletics
        tt_parity = compute_competitive_parity(self.df, sport="Table Tennis")
        ath_parity = compute_competitive_parity(self.df, sport="Athletics")

        self.assertGreater(tt_parity["hhi"], ath_parity["hhi"])
        self.assertIn("status", tt_parity)
        self.assertIn("gini", tt_parity)
        self.assertIn("lorenz_x", tt_parity)
        self.assertIn("lorenz_y", tt_parity)
        self.assertIn("hhi_trend", tt_parity)


if __name__ == "__main__":
    unittest.main()
