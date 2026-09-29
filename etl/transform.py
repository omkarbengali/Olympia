"""
Transform module for OLYMPIA ETL Pipeline.
Cleans, normalizes, validates, and enhances the Olympic dataset.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any


MEDAL_POINTS_MAP = {
    "Gold": 3,
    "Silver": 2,
    "Bronze": 1,
}


def transform_data(df: pd.DataFrame, output_csv: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transforms the raw Olympic medalists DataFrame:
    1. Records data quality metrics before transformation.
    2. Drops exact duplicate rows.
    3. Handles missing values (drops non-awarded medal events; imputes missing athlete names).
    4. Normalizes string values (stripping whitespace, standardizing case).
    5. Casts appropriate data types.
    6. Adds analytical 'medal_points' (Gold=3, Silver=2, Bronze=1).
    7. Validates cleaned data constraints.
    8. Saves processed dataset to data/processed/.

    Parameters:
        df (pd.DataFrame): Raw DataFrame from extract step.
        output_csv (str, optional): Target path to save cleaned CSV.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and transformation report.
    """
    print("[TRANSFORM] Starting data transformation and cleaning...")
    
    # 1. Capture Initial State
    initial_rows = len(df)
    initial_dups = int(df.duplicated().sum())
    initial_missing = df.isnull().sum().to_dict()

    cleaned = df.copy()

    # 2. Duplicate Removal
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(cleaned)
    print(f"[TRANSFORM] Removed {duplicates_removed} duplicate row(s).")

    # 3. Handle Missing Medals (non-awarded events)
    # The 4 records with NaN medal, NaN country, NaN country_code are historical
    # voided/canceled races where no medal was awarded.
    missing_medal_mask = cleaned["medal"].isna()
    non_awarded_count = int(missing_medal_mask.sum())
    cleaned = cleaned[~missing_medal_mask].reset_index(drop=True)
    print(f"[TRANSFORM] Removed {non_awarded_count} non-awarded / voided event row(s).")

    # 4. Handle Missing Athletes
    # Team sports or historical summaries often omit athlete names.
    missing_athletes_count = int(cleaned["athletes"].isna().sum())
    cleaned["athletes"] = cleaned["athletes"].fillna("Team / Not Listed")
    print(f"[TRANSFORM] Imputed {missing_athletes_count} missing athlete entries with 'Team / Not Listed'.")

    # 5. Whitespace and String Normalization
    string_cols = ["season", "medal", "country_code", "country", "athletes", "games", "sport", "event_gender", "event_name"]
    for col in string_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Standardize medal capitalization
    cleaned["medal"] = cleaned["medal"].str.capitalize()

    # Ensure valid medal categories
    valid_medals = {"Gold", "Silver", "Bronze"}
    invalid_medals = set(cleaned["medal"].unique()) - valid_medals
    if invalid_medals:
        raise ValueError(f"Encountered unexpected medal values: {invalid_medals}")

    # Standardize seasons
    valid_seasons = {"Summer", "Winter"}
    invalid_seasons = set(cleaned["season"].unique()) - valid_seasons
    if invalid_seasons:
        raise ValueError(f"Encountered unexpected season values: {invalid_seasons}")

    # 6. Type Conversions
    cleaned["year"] = cleaned["year"].astype(int)

    # 7. Add Medal Points (Internal analytical convention: Gold=3, Silver=2, Bronze=1)
    cleaned["medal_points"] = cleaned["medal"].map(MEDAL_POINTS_MAP).astype(int)

    # 8. Final Validation Checks
    assert cleaned["medal"].isna().sum() == 0, "Cleaned dataset contains NaN medals!"
    assert cleaned["country"].isna().sum() == 0, "Cleaned dataset contains NaN country!"
    assert cleaned["country_code"].isna().sum() == 0, "Cleaned dataset contains NaN country_code!"
    assert cleaned["year"].isna().sum() == 0, "Cleaned dataset contains NaN year!"
    assert cleaned["medal_points"].isna().sum() == 0, "Cleaned dataset contains NaN medal_points!"

    # 9. Save Processed CSV
    if output_csv is None:
        output_dir = os.path.join("data", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "cleaned_olympic_medalists.csv")

    cleaned.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[TRANSFORM] Saved processed data ({len(cleaned)} rows) to: {output_csv}")

    metrics = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned),
        "duplicates_removed": duplicates_removed,
        "non_awarded_events_removed": non_awarded_count,
        "athletes_imputed": missing_athletes_count,
        "initial_missing_by_column": initial_missing,
        "final_missing_by_column": cleaned.isnull().sum().to_dict(),
        "unique_games": int(cleaned["games"].nunique()),
        "unique_countries": int(cleaned["country"].nunique()),
        "unique_country_codes": int(cleaned["country_code"].nunique()),
        "unique_sports": int(cleaned["sport"].nunique()),
        "unique_events": int(cleaned["event_name"].nunique()),
        "gold_count": int((cleaned["medal"] == "Gold").sum()),
        "silver_count": int((cleaned["medal"] == "Silver").sum()),
        "bronze_count": int((cleaned["medal"] == "Bronze").sum()),
        "total_medal_points": int(cleaned["medal_points"].sum()),
        "output_file": output_csv,
    }

    print(f"[TRANSFORM] Transformation complete: {metrics['initial_rows']} -> {metrics['final_rows']} records.")
    return cleaned, metrics


if __name__ == "__main__":
    from etl.extract import extract_data
    df, _ = extract_data()
    clean_df, metrics = transform_data(df)
    print("\nTransformation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
