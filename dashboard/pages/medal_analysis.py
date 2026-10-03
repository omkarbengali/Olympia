"""
Medal Analysis & Visualization Page for OLYMPIA.
Multi-dimensional filtering across Year, Season, Country, Sport, Gender, and Medal type,
with comprehensive distribution charts, statistical archetypes, and gender comparisons.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.visualization import (
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_histogram,
    create_box_plot,
    plot_gender_breakdown,
    MEDAL_COLORS
)


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("🏅 Medal Analytics & Distribution Laboratory")
    st.caption("Investigate global podium distributions, seasonal splits, gender evolution, and statistical patterns across Olympic history.")

    if df.empty:
        st.warning("No Olympic records available in the active dataset.")
        return

    # 2. In-Page Filter Controls
    with st.expander("🔍 Filter Controls (Year, Season, Country, Sport, Gender, Medal)", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            all_seasons = sorted(df["season"].unique().tolist())
            sel_season = st.multiselect("Filter Season", options=all_seasons, default=all_seasons, key="medal_sel_season")

            min_yr, max_yr = int(df["year"].min()), int(df["year"].max())
            sel_years = st.slider("Filter Year Range", min_yr, max_yr, (min_yr, max_yr), step=4, key="medal_sel_years")

        with col_f2:
            all_countries = sorted(df["country"].unique().tolist())
            sel_countries = st.multiselect("Filter Countries (Optional)", options=all_countries, default=[], key="medal_sel_countries")

            all_sports = sorted(df["sport"].unique().tolist())
            sel_sports = st.multiselect("Filter Sports (Optional)", options=all_sports, default=[], key="medal_sel_sports")

        with col_f3:
            all_genders = sorted(df["event_gender"].unique().tolist())
            sel_gender = st.multiselect("Filter Gender Category", options=all_genders, default=all_genders, key="medal_sel_gender")

            all_medals = ["Gold", "Silver", "Bronze"]
            sel_medals = st.multiselect("Filter Medal Type", options=all_medals, default=all_medals, key="medal_sel_medals")

    # Apply filters
    filtered = df.copy()
    if sel_season:
        filtered = filtered[filtered["season"].isin(sel_season)]
    filtered = filtered[(filtered["year"] >= sel_years[0]) & (filtered["year"] <= sel_years[1])]
    if sel_countries:
        filtered = filtered[filtered["country"].isin(sel_countries)]
    if sel_sports:
        filtered = filtered[filtered["sport"].isin(sel_sports)]
    if sel_gender:
        filtered = filtered[filtered["event_gender"].isin(sel_gender)]
    if sel_medals:
        filtered = filtered[filtered["medal"].isin(sel_medals)]

    # Handle Empty State
    if filtered.empty:
        st.info("No medal records match the selected filter criteria. Try expanding the year range or removing filter tags.")
        return

    # 3. Key Metrics Row
    total = len(filtered)
    gold = int((filtered["medal"] == "Gold").sum())
    silver = int((filtered["medal"] == "Silver").sum())
    bronze = int((filtered["medal"] == "Bronze").sum())
    pts = int(filtered["medal_points"].sum())

    render_kpi_row([
        {"title": "Total Filtered Medals", "value": f"{total:,}", "icon": "🏅"},
        {"title": "Gold Medals", "value": f"{gold:,}", "subtitle": f"{gold/max(total,1)*100:.1f}% share", "icon": "🥇"},
        {"title": "Silver Medals", "value": f"{silver:,}", "subtitle": f"{silver/max(total,1)*100:.1f}% share", "icon": "🥈"},
        {"title": "Bronze Medals", "value": f"{bronze:,}", "subtitle": f"{bronze/max(total,1)*100:.1f}% share", "icon": "🥉"},
        {"title": "Total Medal Points", "value": f"{pts:,}", "subtitle": "Gold=3, Silver=2, Bronze=1", "icon": "⭐"},
    ])

    st.markdown("---")

    # 4. Main Visualizations: Top Countries & Trend Over Time
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🌍 Top Countries in Selection")
        c_top = filtered.groupby(["country", "medal"]).size().reset_index(name="count")
        top_c_names = c_top.groupby("country")["count"].sum().nlargest(10).index.tolist()
        c_top_filtered = c_top[c_top["country"].isin(top_c_names)]
        fig_top_c = create_bar_chart(
            c_top_filtered,
            x="country",
            y="count",
            color="medal",
            title="Top 10 Nations by Medal Type",
            barmode="stack"
        )
        display_chart(fig_top_c, key="medal_top_countries_chart")

    with col2:
        st.subheader("📈 Medal Trajectory Over Time")
        yearly_m = filtered.groupby(["year", "season"]).size().reset_index(name="count")
        fig_trend = create_line_chart(
            yearly_m,
            x="year",
            y="count",
            color="season" if len(yearly_m["season"].unique()) > 1 else None,
            title="Medal Trend Across Olympic Editions"
        )
        display_chart(fig_trend, key="medal_trend_chart")

    # 5. Secondary Insights: Season, Gender & Sport Distributions
    st.markdown("---")
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("☀️ Summer vs ❄️ Winter Comparison")
        season_summary = filtered.groupby(["season", "medal"]).size().reset_index(name="count")
        fig_season = create_bar_chart(
            season_summary,
            x="season",
            y="count",
            color="medal",
            title="Medal Class Split by Olympic Season",
            barmode="group"
        )
        display_chart(fig_season, key="medal_season_comparison")

    with col4:
        st.subheader("⚖️ Men's vs Women's vs Mixed Events")
        fig_gender = plot_gender_breakdown(filtered)
        display_chart(fig_gender, key="medal_gender_breakdown")

    # Top Sports Breakdown
    st.markdown("---")
    st.subheader("🏃 Medal Distribution Across Olympic Sports")
    sport_dist = filtered.groupby("sport").size().reset_index(name="medals").sort_values(by="medals", ascending=False).head(12)
    fig_sport = create_bar_chart(
        sport_dist,
        x="sport",
        y="medals",
        title="Top 12 Sports by Awarded Medals in Selection",
        orientation="v"
    )
    display_chart(fig_sport, key="medal_sport_chart")

    # 6. Optional Statistical Archetypes & Technical Details
    with st.expander("🛠️ Advanced Statistical Charts (Scatter, Histogram, Box Plot)"):
        st.markdown("Explore statistical distributions across countries and editions:")
        tab_scat, tab_hist, tab_box = st.tabs(["🔵 Correlation Scatter", "📉 Distribution Histogram", "📦 Dispersion Box Plot"])

        with tab_scat:
            country_pts = filtered.groupby("country").agg(
                Gold=("medal", lambda m: (m == "Gold").sum()),
                Points=("medal_points", "sum")
            ).reset_index()
            fig_scatter = create_scatter_plot(country_pts, x="Gold", y="Points", title="Correlation: Gold Medals vs Total Points")
            display_chart(fig_scatter, key="medal_stat_scatter")

        with tab_hist:
            country_totals = filtered.groupby("country").size().reset_index(name="total_medals")
            fig_hist = create_histogram(country_totals, x="total_medals", title="Histogram: National Medal Count Frequency")
            display_chart(fig_hist, key="medal_stat_hist")

        with tab_box:
            sport_editions = filtered.groupby(["sport", "year"]).size().reset_index(name="medals_per_edition")
            top_sports_box = filtered["sport"].value_counts().nlargest(8).index.tolist()
            sub_box = sport_editions[sport_editions["sport"].isin(top_sports_box)]
            fig_box = create_box_plot(sub_box, x="sport", y="medals_per_edition", title="Box Plot: Medals Awarded per Edition (Dispersion)")
            display_chart(fig_box, key="medal_stat_box")

        render_viva_note(
            "Visual Encodings & Aggregations",
            "This laboratory computes dynamic SQL group-by aggregations across dimensional hierarchies (dim_game, dim_country, dim_sport, dim_event).",
            "Box plots highlight statistical outliers and interquartile range (IQR), while histograms measure empirical probability density."
        )


if __name__ == "__main__":
    render_page()
