"""
Main Dashboard Page for OLYMPIA.
Displays system KPI cards, global Olympic distributions, and India spotlight.
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.visualization import (
    plot_top_countries,
    plot_medals_over_time,
    plot_gender_breakdown,
    create_bar_chart,
    create_line_chart,
    MEDAL_COLORS
)


def render_page(df: pd.DataFrame):
    st.title("🏅 Olympic Performance Intelligence Dashboard")
    st.caption("Comprehensive overview of modern Olympic Games medal history (1896 – 2024)")

    # 1. Top KPI Summary Cards
    total_medals = len(df)
    unique_games = df["games"].nunique()
    unique_countries = df["country"].nunique()
    unique_sports = df["sport"].nunique()
    unique_events = df["event_name"].nunique()

    kpis = [
        {"title": "Olympic Editions", "value": f"{unique_games}", "subtitle": "1896 Athens to 2024 Paris", "icon": "🏛️"},
        {"title": "NOC Countries", "value": f"{unique_countries}", "subtitle": "Participating nations", "icon": "🌍"},
        {"title": "Sports / Disciplines", "value": f"{unique_sports}", "subtitle": "Distinct Olympic sports", "icon": "🏃"},
        {"title": "Competitive Events", "value": f"{unique_events}", "subtitle": "Individual & team events", "icon": "🎯"},
        {"title": "Total Medal Facts", "value": f"{total_medals:,}", "subtitle": "Verified podium finishes", "icon": "🥇"},
    ]
    render_kpi_row(kpis)

    render_viva_note(
        "Data Warehouse Star Schema",
        "The metrics and charts on this dashboard are dynamically computed via SQL joins from the SQLite Star Schema warehouse (fact_medal joined with dim_game, dim_country, dim_sport, dim_event, and dim_medal).",
        "Star Schema optimizes read-heavy analytical aggregations through centralized foreign-key indexed fact records."
    )

    st.markdown("---")

    # 2. Charts Row 1: Top Countries & Medal Trends
    col1, col2 = st.columns([1, 1])

    with col1:
        top_n = st.slider("Select Top N Countries", min_value=5, max_value=20, value=10, key="top_countries_slider")
        fig_top = plot_top_countries(df, top_n=top_n)
        display_chart(fig_top, key="chart_top_countries")

    with col2:
        yearly_medals = df.groupby(["year", "season"]).size().reset_index(name="medals_awarded")
        fig_trend = create_line_chart(
            yearly_medals,
            x="year",
            y="medals_awarded",
            color="season",
            title="Total Medals Awarded per Olympic Edition"
        )
        display_chart(fig_trend, key="chart_yearly_trend")

    # 3. Charts Row 2: Medal Category Distribution & Gender Distribution
    col3, col4 = st.columns([1, 1])

    with col3:
        medal_dist = df["medal"].value_counts().reset_index()
        medal_dist.columns = ["medal", "count"]
        fig_medal = create_bar_chart(
            medal_dist,
            x="medal",
            y="count",
            color="medal",
            title="Medal Distribution (Gold / Silver / Bronze)"
        )
        display_chart(fig_medal, key="chart_medal_distribution")

    with col4:
        fig_gender = plot_gender_breakdown(df)
        display_chart(fig_gender, key="chart_gender_breakdown")

    # 4. Charts Row 3: Top Sports
    st.markdown("### 🏆 Top Olympic Sports by Total Medals")
    top_sports = (
        df.groupby("sport")
        .size()
        .reset_index(name="medal_count")
        .sort_values(by="medal_count", ascending=False)
        .head(15)
    )
    fig_sports = create_bar_chart(
        top_sports,
        x="sport",
        y="medal_count",
        title="Top 15 Most Awarded Olympic Sports",
        orientation="v"
    )
    display_chart(fig_sports, key="chart_top_sports")

    # 5. India Focus Section
    st.markdown("---")
    st.subheader("🇮🇳 Spotlight: India at the Olympics")
    india_df = df[df["country"] == "India"]

    if not india_df.empty:
        i_gold = int((india_df["medal"] == "Gold").sum())
        i_silver = int((india_df["medal"] == "Silver").sum())
        i_bronze = int((india_df["medal"] == "Bronze").sum())
        i_total = len(india_df)
        i_points = int(india_df["medal_points"].sum())

        i_kpis = [
            {"title": "Total Medals", "value": str(i_total), "icon": "🏅"},
            {"title": "Gold Medals", "value": str(i_gold), "icon": "🥇"},
            {"title": "Silver Medals", "value": str(i_silver), "icon": "🥈"},
            {"title": "Bronze Medals", "value": str(i_bronze), "icon": "🥉"},
            {"title": "Medal Points", "value": str(i_points), "subtitle": "Gold=3, Silver=2, Bronze=1", "icon": "⭐"},
        ]
        render_kpi_row(i_kpis)

        icol1, icol2 = st.columns([1, 1])
        with icol1:
            india_sports = (
                india_df.groupby(["sport", "medal"])
                .size()
                .reset_index(name="count")
            )
            fig_i_sport = create_bar_chart(
                india_sports,
                x="sport",
                y="count",
                color="medal",
                title="India: Medals Won by Sport Discipline",
                barmode="stack"
            )
            display_chart(fig_i_sport, key="chart_india_sports")

        with icol2:
            india_trend = (
                india_df.groupby("year")
                .size()
                .reset_index(name="medals")
            )
            fig_i_trend = create_line_chart(
                india_trend,
                x="year",
                y="medals",
                title="India: Medal Trajectory (1900 – 2024)"
            )
            display_chart(fig_i_trend, key="chart_india_trend")
    else:
        st.info("No records found for India with the currently applied global filters.")
