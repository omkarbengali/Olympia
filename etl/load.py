"""
Load module for OLYMPIA ETL Pipeline.
Loads cleaned data into the SQLite Star Schema database.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, Any, Tuple


def get_db_path() -> str:
    """Returns the default SQLite database path."""
    db_dir = "database"
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "olympics.db")


def get_schema_path() -> str:
    """Returns the default schema.sql path."""
    schema_path = os.path.join("database", "schema.sql")
    if not os.path.exists(schema_path):
        schema_path = os.path.join("..", "database", "schema.sql")
    return schema_path


def init_database(conn: sqlite3.Connection, schema_path: str = None) -> None:
    """Initializes tables and indexes using schema.sql."""
    if schema_path is None:
        schema_path = get_schema_path()
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()


def load_data(cleaned_df: pd.DataFrame, db_path: str = None, schema_path: str = None) -> Dict[str, Any]:
    """
    Loads transformed DataFrame into SQLite Star Schema.
    This operation is repeatable and idempotent.

    Parameters:
        cleaned_df (pd.DataFrame): Transformed and cleaned Olympic data.
        db_path (str, optional): Target database file path.
        schema_path (str, optional): DDL schema file path.

    Returns:
        Dict[str, Any]: Verification summary and row counts of all warehouse tables.
    """
    if db_path is None:
        db_path = get_db_path()

    print(f"[LOAD] Initializing database at: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        # Step 1: Re-initialize Schema (Idempotency guarantee)
        init_database(conn, schema_path)
        cur = conn.cursor()

        # Step 2: Populate dim_game
        games_df = (
            cleaned_df[["year", "season", "games"]]
            .drop_duplicates()
            .sort_values(by=["year", "season"])
        )
        cur.executemany(
            "INSERT INTO dim_game (year, season, games) VALUES (?, ?, ?);",
            games_df.itertuples(index=False, name=None)
        )
        cur.execute("SELECT games, game_id FROM dim_game;")
        game_map = dict(cur.fetchall())
        print(f"[LOAD] Inserted {len(game_map)} records into dim_game.")

        # Step 3: Populate dim_country
        countries_df = (
            cleaned_df[["country_code", "country"]]
            .drop_duplicates()
            .sort_values(by=["country"])
        )
        cur.executemany(
            "INSERT INTO dim_country (country_code, country_name) VALUES (?, ?);",
            countries_df.itertuples(index=False, name=None)
        )
        cur.execute("SELECT country_name, country_id FROM dim_country;")
        country_map = dict(cur.fetchall())
        print(f"[LOAD] Inserted {len(country_map)} records into dim_country.")

        # Step 4: Populate dim_sport
        sports = sorted(cleaned_df["sport"].unique())
        cur.executemany(
            "INSERT INTO dim_sport (sport_name) VALUES (?);",
            [(s,) for s in sports]
        )
        cur.execute("SELECT sport_name, sport_id FROM dim_sport;")
        sport_map = dict(cur.fetchall())
        print(f"[LOAD] Inserted {len(sport_map)} records into dim_sport.")

        # Step 5: Populate dim_event
        # Unique combination: sport_id, event_name, event_gender
        events_df = (
            cleaned_df[["sport", "event_name", "event_gender"]]
            .drop_duplicates()
            .sort_values(by=["sport", "event_name", "event_gender"])
        )
        events_data = []
        for row in events_df.itertuples(index=False):
            sport_name, event_name, event_gender = row
            s_id = sport_map[sport_name]
            events_data.append((s_id, event_name, event_gender))

        cur.executemany(
            "INSERT INTO dim_event (sport_id, event_name, event_gender) VALUES (?, ?, ?);",
            events_data
        )
        cur.execute("SELECT sport_id, event_name, event_gender, event_id FROM dim_event;")
        event_map = {(r[0], r[1], r[2]): r[3] for r in cur.fetchall()}
        print(f"[LOAD] Inserted {len(event_map)} records into dim_event.")

        # Step 6: Populate dim_medal
        medals_data = [
            ("Gold", 3),
            ("Silver", 2),
            ("Bronze", 1)
        ]
        cur.executemany(
            "INSERT INTO dim_medal (medal_name, medal_points) VALUES (?, ?);",
            medals_data
        )
        cur.execute("SELECT medal_name, medal_id FROM dim_medal;")
        medal_map = dict(cur.fetchall())
        print(f"[LOAD] Inserted {len(medal_map)} records into dim_medal.")

        # Step 7: Populate fact_medal
        print("[LOAD] Preparing fact_medal records...")
        fact_rows = []
        for row in cleaned_df.itertuples(index=False):
            g_id = game_map[row.games]
            c_id = country_map[row.country]
            s_id = sport_map[row.sport]
            e_id = event_map[(s_id, row.event_name, row.event_gender)]
            m_id = medal_map[row.medal]
            athletes_val = str(row.athletes) if pd.notna(row.athletes) else "Team / Not Listed"

            fact_rows.append((g_id, c_id, s_id, e_id, m_id, athletes_val))

        cur.executemany(
            """
            INSERT INTO fact_medal (game_id, country_id, sport_id, event_id, medal_id, athletes)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            fact_rows
        )
        print(f"[LOAD] Inserted {len(fact_rows)} records into fact_medal.")

        # Step 8: Verify Foreign Key Constraints and Row Integrity
        cur.execute("PRAGMA foreign_key_check;")
        fk_violations = cur.fetchall()
        if fk_violations:
            raise RuntimeError(f"Foreign key violations detected: {fk_violations}")

        conn.commit()

        # Step 9: Collect Table Counts
        table_counts = {}
        for tbl in ["dim_game", "dim_country", "dim_sport", "dim_event", "dim_medal", "fact_medal"]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl};")
            table_counts[tbl] = cur.fetchone()[0]

        summary = {
            "database_path": db_path,
            "status": "SUCCESS",
            "table_counts": table_counts,
            "fact_rows_inserted": len(fact_rows),
            "integrity_check": "PASSED"
        }
        print("\n[LOAD] Load successfully completed!")
        print(f"       Fact Medal Rows: {table_counts['fact_medal']}")
        print(f"       Integrity Check: PASSED")
        return summary

    finally:
        conn.close()


def run_etl() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Executes the entire end-to-end ETL pipeline:
    1. Extract
    2. Transform
    3. Load
    """
    from etl.extract import extract_data
    from etl.transform import transform_data

    print("=" * 60)
    print("OLYMPIA ETL PIPELINE EXECUTION")
    print("=" * 60)

    # 1. Extract
    raw_df, extract_meta = extract_data()

    # 2. Transform
    clean_df, transform_meta = transform_data(raw_df)

    # 3. Load
    load_meta = load_data(clean_df)

    print("=" * 60)
    print("ETL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return extract_meta, transform_meta, load_meta


if __name__ == "__main__":
    run_etl()
