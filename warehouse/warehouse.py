"""
Warehouse access and querying module for OLYMPIA.
Interfaces directly with SQLite database/olympics.db.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional


DEFAULT_DB_PATH = os.path.join("database", "olympics.db")


def get_db_path() -> str:
    """Finds the database file path."""
    candidates = [
        DEFAULT_DB_PATH,
        os.path.join("..", "database", "olympics.db"),
        os.path.join(".", "olympics.db")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return DEFAULT_DB_PATH


def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Creates a connection to SQLite database with foreign keys enabled."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_table_counts(db_path: str = None) -> Dict[str, int]:
    """Retrieves row counts for all warehouse tables."""
    conn = get_db_connection(db_path)
    try:
        tables = ["dim_game", "dim_country", "dim_sport", "dim_event", "dim_medal", "fact_medal"]
        counts = {}
        for tbl in tables:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
            counts[tbl] = cursor.fetchone()[0]
        return counts
    finally:
        conn.close()


def get_schema_info(db_path: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieves column metadata and primary/foreign keys for all warehouse tables."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cursor.fetchall()]
        schema_info = {}
        for tbl in tables:
            cursor.execute(f"PRAGMA table_info({tbl});")
            cols = [
                {
                    "cid": r[0],
                    "name": r[1],
                    "type": r[2],
                    "notnull": bool(r[3]),
                    "default_value": r[4],
                    "is_pk": bool(r[5])
                }
                for r in cursor.fetchall()
            ]
            cursor.execute(f"PRAGMA foreign_key_list({tbl});")
            fks = [
                {
                    "id": r[0],
                    "seq": r[1],
                    "table": r[2],
                    "from": r[3],
                    "to": r[4]
                }
                for r in cursor.fetchall()
            ]
            schema_info[tbl] = {"columns": cols, "foreign_keys": fks}
        return schema_info
    finally:
        conn.close()


def execute_query(query: str, params: tuple = (), db_path: str = None) -> pd.DataFrame:
    """Executes a custom SELECT SQL query and returns results as a pandas DataFrame."""
    conn = get_db_connection(db_path)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def get_denormalized_medals(
    years: Optional[List[int]] = None,
    seasons: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
    sports: Optional[List[str]] = None,
    medals: Optional[List[str]] = None,
    genders: Optional[List[str]] = None,
    db_path: str = None
) -> pd.DataFrame:
    """
    Executes a high-performance Star Join query retrieving denormalized medal facts
    with optional filtering. Used as the unified data source for OLAP, visualizations,
    and downstream machine learning modules.
    """
    conn = get_db_connection(db_path)
    try:
        base_query = """
        SELECT
            f.medal_fact_id,
            g.year,
            g.season,
            g.games,
            c.country_code,
            c.country_name AS country,
            s.sport_name AS sport,
            e.event_name,
            e.event_gender,
            m.medal_name AS medal,
            m.medal_points,
            f.athletes
        FROM fact_medal f
        JOIN dim_game g ON f.game_id = g.game_id
        JOIN dim_country c ON f.country_id = c.country_id
        JOIN dim_sport s ON f.sport_id = s.sport_id
        JOIN dim_event e ON f.event_id = e.event_id
        JOIN dim_medal m ON f.medal_id = m.medal_id
        WHERE 1=1
        """
        filters = []
        params = []

        if years:
            placeholders = ",".join("?" for _ in years)
            filters.append(f"g.year IN ({placeholders})")
            params.extend(years)

        if seasons:
            placeholders = ",".join("?" for _ in seasons)
            filters.append(f"g.season IN ({placeholders})")
            params.extend(seasons)

        if countries:
            placeholders = ",".join("?" for _ in countries)
            filters.append(f"c.country_name IN ({placeholders})")
            params.extend(countries)

        if sports:
            placeholders = ",".join("?" for _ in sports)
            filters.append(f"s.sport_name IN ({placeholders})")
            params.extend(sports)

        if medals:
            placeholders = ",".join("?" for _ in medals)
            filters.append(f"m.medal_name IN ({placeholders})")
            params.extend(medals)

        if genders:
            placeholders = ",".join("?" for _ in genders)
            filters.append(f"e.event_gender IN ({placeholders})")
            params.extend(genders)

        if filters:
            base_query += " AND " + " AND ".join(filters)

        base_query += " ORDER BY g.year ASC, c.country_name ASC"

        return pd.read_sql_query(base_query, conn, params=params)
    finally:
        conn.close()


if __name__ == "__main__":
    counts = get_table_counts()
    print("Warehouse Table Row Counts:")
    for tbl, count in counts.items():
        print(f"  {tbl}: {count:,}")
