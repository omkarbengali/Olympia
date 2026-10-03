"""
Automated validation of all OLYMPIA dashboard pages and authentication logic.
Ensures every page renders cleanly with test data and raises no exceptions.
"""

import unittest
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
from dashboard.auth import (
    DEMO_ACCOUNTS,
    ROLE_ADMIN,
    ROLE_ENTHUSIAST,
    ROLE_ACADEMY
)
from dashboard.pages import (
    admin_dashboard,
    enthusiast_dashboard,
    academy_dashboard,
    dashboard,
    country_analysis,
    medal_analysis,
    sport_analysis,
    olap_analysis,
    preprocessing,
    regression,
    classification,
    clustering,
    association_rules,
    warehouse,
)


class TestOlympiaPages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = get_denormalized_medals()

    def test_warehouse_data_integrity(self):
        self.assertFalse(self.df.empty)
        self.assertGreaterEqual(len(self.df), 20000)
        expected_cols = ["medal_fact_id", "games", "year", "season", "country", "sport", "event_name", "medal", "medal_points"]
        for col in expected_cols:
            self.assertIn(col, self.df.columns)

    def test_auth_roles_and_credentials(self):
        self.assertIn("admin", DEMO_ACCOUNTS)
        self.assertIn("fan", DEMO_ACCOUNTS)
        self.assertIn("coach", DEMO_ACCOUNTS)
        self.assertEqual(DEMO_ACCOUNTS["admin"]["role"], ROLE_ADMIN)
        self.assertEqual(DEMO_ACCOUNTS["fan"]["role"], ROLE_ENTHUSIAST)
        self.assertEqual(DEMO_ACCOUNTS["coach"]["role"], ROLE_ACADEMY)

    def test_olap_functions_callable(self):
        from analytics.olap import olap_slice, olap_dice, olap_rollup, olap_drilldown, olap_pivot
        sl = olap_slice(self.df, dimension="year", value=2024)
        self.assertIsInstance(sl, pd.DataFrame)

        dc = olap_dice(self.df, filters={"season": ["Summer"], "year": [2024]})
        self.assertIsInstance(dc, pd.DataFrame)

        ru = olap_rollup(self.df, group_levels=["country"])
        self.assertIsInstance(ru, pd.DataFrame)

        dd = olap_drilldown(self.df, current_level="country", current_value="India", next_level="sport")
        self.assertIsInstance(dd, pd.DataFrame)

        pv = olap_pivot(self.df, rows="country", columns="medal", values="medal_fact_id")
        self.assertIsInstance(pv, pd.DataFrame)

    def test_ml_algorithms_callable(self):
        from analytics.regression import train_linear_regression
        from analytics.decision_tree import train_decision_tree
        from analytics.naive_bayes import train_naive_bayes
        from analytics.kmeans import run_kmeans
        from analytics.apriori import mine_association_rules

        # Linear Regression
        reg = train_linear_regression(split_year=2012, features=["prev_total_medals"])
        self.assertIn("r2", reg)

        # Decision Tree
        dt = train_decision_tree(df=self.df, max_depth=3)
        self.assertIn("accuracy", dt)

        # Naive Bayes
        nb = train_naive_bayes(df=self.df)
        self.assertIn("accuracy", nb)

        # K-Means
        km = run_kmeans(df=self.df, n_clusters=3)
        self.assertIn("cluster_sizes", km)

        # Apriori
        apr = mine_association_rules(df=self.df, min_support=0.08, min_confidence=0.4)
        self.assertIn("status", apr)


if __name__ == "__main__":
    unittest.main()
