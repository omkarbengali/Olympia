"""
OLAP (Online Analytical Processing) Module for OLYMPIA.
Implements the 5 core OLAP cube operations:
1. SLICE: Filtering on a single dimension.
2. DICE: Filtering on multiple dimensions simultaneously to extract a sub-cube.
3. ROLL-UP: Aggregating measures up the hierarchy (less detail, higher summary).
4. DRILL-DOWN: Navigating down the hierarchy (more detail, lower summary).
5. PIVOT: Rotating axes of the multidimensional dataset for cross-tabulation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from typing import List, Dict, Any, Optional, Union
from warehouse.warehouse import get_denormalized_medals


def olap_slice(
    df: Optional[pd.DataFrame] = None,
    dimension: str = "year",
    value: Any = 2024
) -> pd.DataFrame:
    """
    OLAP SLICE operation:
    Fixes ONE dimension to a specific single value, producing a 2D/3D slice of the cube.

    Example:
        Slice by Year = 2024
        Slice by Country = 'India'
        Slice by Season = 'Winter'
    """
    if df is None:
        df = get_denormalized_medals()

    if dimension not in df.columns:
        raise ValueError(f"Dimension '{dimension}' not found in dataset columns: {list(df.columns)}")

    sliced = df[df[dimension] == value].copy().reset_index(drop=True)
    return sliced


def olap_dice(
    df: Optional[pd.DataFrame] = None,
    filters: Optional[Dict[str, Union[List[Any], Any]]] = None
) -> pd.DataFrame:
    """
    OLAP DICE operation:
    Defines a sub-cube by specifying filtering criteria across MULTIPLE dimensions.

    Example:
        filters = {
            'year': [2000, 2004, 2008, 2012, 2016, 2020, 2024],
            'season': ['Summer'],
            'country': ['United States', 'China', 'Great Britain', 'India'],
            'medal': ['Gold', 'Silver']
        }
    """
    if df is None:
        df = get_denormalized_medals()

    if not filters:
        return df.copy()

    diced = df.copy()
    for dim, condition in filters.items():
        if dim not in diced.columns:
            continue
        if condition is None:
            continue

        if isinstance(condition, (list, tuple, set)):
            if len(condition) > 0:
                diced = diced[diced[dim].isin(condition)]
        else:
            diced = diced[diced[dim] == condition]

    return diced.reset_index(drop=True)


def olap_rollup(
    df: Optional[pd.DataFrame] = None,
    group_levels: Optional[List[str]] = None,
    include_points: bool = True
) -> pd.DataFrame:
    """
    OLAP ROLL-UP operation:
    Aggregates facts from a detailed level to a higher conceptual level.

    Hierarchy Examples:
        Event -> Sport -> Country -> Year
    """
    if df is None:
        df = get_denormalized_medals()

    if group_levels is None or len(group_levels) == 0:
        group_levels = ["country"]

    valid_levels = [lvl for lvl in group_levels if lvl in df.columns]
    if not valid_levels:
        raise ValueError(f"None of {group_levels} exist in dataset columns.")

    if df.empty:
        cols = valid_levels + ["total_medals", "gold_medals", "silver_medals", "bronze_medals"]
        if include_points:
            cols.append("medal_points")
        return pd.DataFrame(columns=cols)

    grouped = df.groupby(valid_levels)

    res = grouped.agg(
        total_medals=("medal_fact_id", "count"),
        gold_medals=("medal", lambda s: (s == "Gold").sum()),
        silver_medals=("medal", lambda s: (s == "Silver").sum()),
        bronze_medals=("medal", lambda s: (s == "Bronze").sum()),
        medal_points=("medal_points", "sum") if include_points else ("medal_fact_id", "count")
    ).reset_index()

    if not include_points:
        res = res.drop(columns=["medal_points"])

    res = res.sort_values(by="total_medals", ascending=False).reset_index(drop=True)
    return res


def olap_drilldown(
    df: Optional[pd.DataFrame] = None,
    current_level: str = "country",
    current_value: Any = "India",
    next_level: str = "sport"
) -> pd.DataFrame:
    """
    OLAP DRILL-DOWN operation:
    Steps down a hierarchy from a summarized summary to a more detailed breakdown.

    Example:
        country ('India') -> sport -> event -> athlete
    """
    if df is None:
        df = get_denormalized_medals()

    if current_level not in df.columns:
        raise ValueError(f"Level '{current_level}' not in dataset.")
    if next_level not in df.columns:
        raise ValueError(f"Next level '{next_level}' not in dataset.")

    # 1. Filter to current level's entity
    subset = df[df[current_level] == current_value].copy()
    if subset.empty:
        return pd.DataFrame(columns=[next_level, "total_medals", "gold_medals", "silver_medals", "bronze_medals", "medal_points"])

    # 2. Group by next level
    drill_df = subset.groupby(next_level).agg(
        total_medals=("medal_fact_id", "count"),
        gold_medals=("medal", lambda s: (s == "Gold").sum()),
        silver_medals=("medal", lambda s: (s == "Silver").sum()),
        bronze_medals=("medal", lambda s: (s == "Bronze").sum()),
        medal_points=("medal_points", "sum")
    ).reset_index()

    drill_df = drill_df.sort_values(by=["total_medals", "medal_points"], ascending=False).reset_index(drop=True)
    return drill_df


def olap_pivot(
    df: Optional[pd.DataFrame] = None,
    rows: str = "country",
    columns: str = "medal",
    values: str = "medal_fact_id",
    aggfunc: str = "count",
    add_total: bool = True
) -> pd.DataFrame:
    """
    OLAP PIVOT operation:
    Rotates the multidimensional view to present cross-tabulated data.

    Example:
        Rows = country
        Columns = medal ('Gold', 'Silver', 'Bronze')
        Values = count
    """
    if df is None:
        df = get_denormalized_medals()

    if df.empty:
        return pd.DataFrame()

    if rows not in df.columns or columns not in df.columns:
        raise ValueError(f"Both '{rows}' and '{columns}' must exist in DataFrame columns.")

    pivoted = pd.pivot_table(
        df,
        index=rows,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=0
    )

    # Reorder standard medal columns if pivoting on medal
    if columns == "medal":
        standard_cols = [c for c in ["Gold", "Silver", "Bronze"] if c in pivoted.columns]
        other_cols = [c for c in pivoted.columns if c not in standard_cols]
        pivoted = pivoted[standard_cols + other_cols]

    if add_total:
        pivoted["Total"] = pivoted.sum(axis=1)
        pivoted = pivoted.sort_values(by="Total", ascending=False)
    else:
        pivoted = pivoted.sort_values(by=pivoted.columns[0], ascending=False)

    return pivoted.reset_index()


if __name__ == "__main__":
    print("=== OLAP SLICE (Year = 2024) ===")
    slice_2024 = olap_slice(dimension="year", value=2024)
    print(f"Slice 2024 medals count: {len(slice_2024)}")
    print(slice_2024[["games", "country", "sport", "event_name", "medal"]].head(3))

    print("\n=== OLAP DICE (Year >= 2020, Summer, India & USA) ===")
    dice_res = olap_dice(filters={
        "year": [2020, 2024],
        "season": ["Summer"],
        "country": ["India", "United States"]
    })
    print(f"Diced records: {len(dice_res)}")
    print(dice_res[["year", "country", "sport", "medal"]].head(5))

    print("\n=== OLAP ROLL-UP (Country -> Sport) ===")
    rollup_res = olap_rollup(group_levels=["country", "sport"])
    print(rollup_res.head(5))

    print("\n=== OLAP DRILL-DOWN (Country='India' -> Sport) ===")
    drill_res = olap_drilldown(current_level="country", current_value="India", next_level="sport")
    print(drill_res.head(5))

    print("\n=== OLAP PIVOT (Rows=country, Columns=medal) ===")
    pivot_res = olap_pivot(rows="country", columns="medal")
    print(pivot_res.head(5))
