"""
Extract module for OLYMPIA ETL Pipeline.
Extracts raw Olympic medalists dataset from CSV source.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any


def get_default_csv_path() -> str:
    """Finds the raw CSV file in standard locations."""
    candidates = [
        os.path.join("data", "raw", "all_olympic_medalists.csv"),
        "all_olympic_medalists.csv",
        os.path.join("..", "data", "raw", "all_olympic_medalists.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find 'all_olympic_medalists.csv' in expected locations: "
        + ", ".join(candidates)
    )


def extract_data(csv_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts raw Olympic medalists CSV data into a pandas DataFrame.

    Parameters:
        csv_path (str, optional): Custom path to CSV file.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Extracted DataFrame and extraction metadata.
    """
    if csv_path is None:
        csv_path = get_default_csv_path()

    print(f"[EXTRACT] Reading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")

    meta = {
        "source_file": csv_path,
        "raw_rows": len(df),
        "raw_columns": list(df.columns),
        "column_count": len(df.columns),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }

    print(f"[EXTRACT] Successfully extracted {meta['raw_rows']} rows and {meta['column_count']} columns.")
    return df, meta


if __name__ == "__main__":
    df, meta = extract_data()
    print("Sample Data:")
    print(df.head(3))
