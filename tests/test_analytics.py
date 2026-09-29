"""
Unit tests for Analytics modules:
- OLAP operations (Slice, Dice, Roll-up, Drill-down, Pivot)
- Linear Regression
- Decision Tree & Naive Bayes Classification
- K-Means & Hierarchical Clustering
- Apriori Association Rule Mining
"""

import unittest
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
from analytics.olap import (
    olap_slice,
    olap_dice,
    olap_rollup,
    olap_drilldown,
    olap_pivot
)
from analytics.regression import train_linear_regression
from analytics.decision_tree import train_decision_tree
from analytics.naive_bayes import train_naive_bayes
from analytics.kmeans import run_kmeans, compute_elbow_curve
from analytics.hierarchical import run_hierarchical_clustering
from analytics.apriori import mine_association_rules


class TestAnalyticsSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = get_denormalized_medals()

    # --- OLAP Tests ---
    def test_olap_slice_year(self):
        sliced = olap_slice(self.df, dimension="year", value=2024)
        self.assertFalse(sliced.empty)
        self.assertTrue((sliced["year"] == 2024).all())
        self.assertEqual(len(sliced), 1044)

    def test_olap_slice_country(self):
        sliced = olap_slice(self.df, dimension="country", value="India")
        self.assertEqual(len(sliced), 34)
        self.assertTrue((sliced["country"] == "India").all())

    def test_olap_dice_multidimensional(self):
        filters = {"year": [2024], "country": ["India"], "season": ["Summer"]}
        diced = olap_dice(self.df, filters=filters)
        self.assertFalse(diced.empty)
        self.assertTrue((diced["year"] == 2024).all())

    def test_olap_rollup_hierarchy(self):
        rollup_df = olap_rollup(self.df, group_levels=["country"])
        self.assertFalse(rollup_df.empty)
        self.assertEqual(rollup_df["total_medals"].sum(), len(self.df))

    def test_olap_drilldown(self):
        drill_df = olap_drilldown(self.df, current_level="country", current_value="India", next_level="sport")
        self.assertFalse(drill_df.empty)
        self.assertEqual(drill_df.iloc[0]["sport"], "Hockey")

    def test_olap_pivot(self):
        pivot_df = olap_pivot(self.df, rows="country", columns="medal")
        self.assertEqual(pivot_df.iloc[0]["country"], "United States")
        self.assertEqual(pivot_df.iloc[0]["Total"], 2975)

    # --- Machine Learning Tests ---
    def test_linear_regression(self):
        res = train_linear_regression(split_year=2012)
        self.assertIn("r2", res)
        self.assertIn("mae", res)
        self.assertGreater(res["r2"], 0.75)
        self.assertGreater(len(res["results_df"]), 0)

    def test_decision_tree_classification(self):
        res = train_decision_tree(max_depth=4)
        self.assertIn("accuracy", res)
        self.assertIn("confusion_matrix", res)
        self.assertEqual(len(res["classes"]), 3)
        self.assertGreater(res["accuracy"], 0.30)

    def test_naive_bayes_classification(self):
        res = train_naive_bayes()
        self.assertIn("accuracy", res)
        self.assertIn("f1_score", res)
        self.assertGreater(res["accuracy"], 0.30)

    # --- Clustering Tests ---
    def test_kmeans_clustering(self):
        res = run_kmeans(n_clusters=3)
        self.assertEqual(len(res["cluster_sizes"]), 3)
        elbow = compute_elbow_curve(max_k=4)
        self.assertEqual(len(elbow), 4)

    def test_hierarchical_clustering(self):
        res = run_hierarchical_clustering(n_clusters=3)
        self.assertEqual(len(res["cluster_sizes"]), 3)

    # --- Association Rules Tests ---
    def test_apriori_rules(self):
        res = mine_association_rules(min_support=0.08, min_confidence=0.4)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(len(res["rules"]), 0)
        self.assertIn("lift", res["rules"].columns)


if __name__ == "__main__":
    unittest.main()
