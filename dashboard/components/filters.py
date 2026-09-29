"""
Interactive Filtering Component for OLYMPIA.
Allows dynamic filtering by Year, Season, Country, Sport, Medal, and Gender.
"""

import streamlit as st
import pandas as pd
from typing import Tuple


def render_sidebar_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Renders filter controls in the sidebar and returns the filtered DataFrame.
    """
    st.sidebar.markdown("### 🔍 Global Filters")

    # Season Filter
    all_seasons = sorted(df["season"].unique().tolist())
    selected_seasons = st.sidebar.multiselect(
        "Olympic Season",
        options=all_seasons,
        default=all_seasons
    )

    # Year Range Filter
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    selected_years = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=4
    )

    # Country Filter (Optional search multiselect)
    all_countries = sorted(df["country"].unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Filter Countries (Leave blank for all)",
        options=all_countries,
        default=[]
    )

    # Sport Filter
    all_sports = sorted(df["sport"].unique().tolist())
    selected_sports = st.sidebar.multiselect(
        "Filter Sports (Leave blank for all)",
        options=all_sports,
        default=[]
    )

    # Apply filters
    filtered_df = df.copy()

    if selected_seasons:
        filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]

    filtered_df = filtered_df[
        (filtered_df["year"] >= selected_years[0]) & (filtered_df["year"] <= selected_years[1])
    ]

    if selected_countries:
        filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]

    if selected_sports:
        filtered_df = filtered_df[filtered_df["sport"].isin(selected_sports)]

    st.sidebar.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records")

    filter_state = {
        "seasons": selected_seasons,
        "year_range": selected_years,
        "countries": selected_countries,
        "sports": selected_sports
    }

    return filtered_df, filter_state
