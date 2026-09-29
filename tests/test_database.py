"""
Unit tests for Database and Star Schema warehouse modules.
"""

import unittest
import sqlite3
from warehouse.warehouse import (
    get_db_connection,
    get_table_counts,
    get_schema_info,
    get_denormalized_medals,
    execute_query
)


class TestDatabaseWarehouse(unittest.TestCase):

    def setUp(self):
        self.conn = get_db_connection()

    def tearDown(self):
        self.conn.close()

    def test_database_tables_exist(self):
        """Verify all star schema dimension and fact tables exist."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = set(r[0] for r in cursor.fetchall())
        expected_tables = {"dim_game", "dim_country", "dim_sport", "dim_event", "dim_medal", "fact_medal"}
        self.assertTrue(expected_tables.issubset(tables))

    def test_table_row_counts(self):
        """Verify exact row counts for dimensions and facts."""
        counts = get_table_counts()
        self.assertEqual(counts["dim_game"], 53)
        self.assertEqual(counts["dim_country"], 168)
        self.assertEqual(counts["dim_sport"], 77)
        self.assertEqual(counts["dim_event"], 1006)
        self.assertEqual(counts["dim_medal"], 3)
        self.assertEqual(counts["fact_medal"], 20240)

    def test_foreign_key_integrity(self):
        """Verify no orphaned foreign keys in fact_medal."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_key_check;")
        violations = cursor.fetchall()
        self.assertEqual(len(violations), 0, f"Foreign key violations found: {violations}")

    def test_star_join_denormalization(self):
        """Verify star join returns all 20,240 records with all dimension attributes."""
        df = get_denormalized_medals()
        self.assertEqual(len(df), 20240)
        expected_columns = {
            "medal_fact_id", "year", "season", "games",
            "country_code", "country", "sport", "event_name",
            "event_gender", "medal", "medal_points", "athletes"
        }
        self.assertTrue(expected_columns.issubset(set(df.columns)))
        # Check India records
        india_medals = df[df["country"] == "India"]
        self.assertEqual(len(india_medals), 34)


if __name__ == "__main__":
    unittest.main()
