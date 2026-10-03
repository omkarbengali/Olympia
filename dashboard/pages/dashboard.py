"""
Main Olympic Intelligence Dashboard Page for OLYMPIA.
High-level overview of modern Olympic Games history (1896 – 2024),
featuring global KPIs, Summer vs Winter comparisons, dominant nations, and India spotlight.
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
    plot_top_countries,
    plot_gender_breakdown,
    create_bar_chart,
    create_line_chart,
    MEDAL_COLORS
)


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("🏅 Olympic Performance Intelligence Dashboard")
    st.caption("A multi-edition executive overview of modern Olympic history, medal distributions, national powerhouses, and seasonal trends.")

    # 2. Key Metrics Row
    total_medals = len(df)
    if total_medals == 0:
        st.warning("No Olympic records match the currently selected filter combination.")
        return

    unique_games = df["games"].nunique()
    unique_countries = df["country"].nunique()
    unique_sports = df["sport"].nunique()
    unique_events = df["event_name"].nunique()

    kpis = [
        {"title": "Olympic Editions", "value": f"{unique_games}", "subtitle": "Athens 1896 to Paris 2024", "icon": "🏛️"},
        {"title": "NOC Countries", "value": f"{unique_countries}", "subtitle": "Podium-winning nations", "icon": "🌍"},
        {"title": "Sports / Disciplines", "value": f"{unique_sports}", "subtitle": "Track, Aquatics, Winter, etc.", "icon": "🏃"},
        {"title": "Competitive Events", "value": f"{unique_events}", "subtitle": "Individual & team events", "icon": "🎯"},
        {"title": "Total Medal Facts", "value": f"{total_medals:,}", "subtitle": "Verified podium finishes", "icon": "🥇"},
    ]
    render_kpi_row(kpis)

    # 3. Main Visualization: Top Countries & Medal Growth
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🌍 All-Time National Medal Leaders")
        top_n = st.slider("Select Number of Top Nations", min_value=5, max_value=25, value=10, key="dash_top_n_slider")
        fig_top = plot_top_countries(df, top_n=top_n)
        display_chart(fig_top, key="dash_chart_top_countries")

    with col2:
        st.subheader("📈 Medal Trajectory by Season")
        yearly_medals = df.groupby(["year", "season"]).size().reset_index(name="medals_awarded")
        fig_trend = create_line_chart(
            yearly_medals,
            x="year",
            y="medals_awarded",
            color="season",
            title="Medals Awarded per Edition (Summer vs Winter)"
        )
        display_chart(fig_trend, key="dash_chart_yearly_trend")

    # 4. Secondary Insights: Medal Classes, Gender Evolution & Top Sports
    st.markdown("---")
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("🥇 Podium Distribution")
        medal_dist = df["medal"].value_counts().reset_index()
        medal_dist.columns = ["medal", "count"]
        fig_medal = create_bar_chart(
            medal_dist,
            x="medal",
            y="count",
            color="medal",
            title="Medal Distribution (Gold / Silver / Bronze)"
        )
        display_chart(fig_medal, key="dash_chart_medal_distribution")

    with col4:
        st.subheader("⚖️ Gender Participation Evolution")
        fig_gender = plot_gender_breakdown(df)
        display_chart(fig_gender, key="dash_chart_gender_breakdown")

    # Top Sports Breakdown
    st.markdown("---")
    st.subheader("🏆 Most Contested Olympic Disciplines")
    top_sports = (
        df.groupby("sport")
        .size()
        .reset_index(name="medal_count")
        .sort_values(by="medal_count", ascending=False)
        .head(12)
    )
    fig_sports = create_bar_chart(
        top_sports,
        x="sport",
        y="medal_count",
        title="Top 12 Sports by Total Podium Finishes Awarded",
        orientation="v"
    )
    display_chart(fig_sports, key="dash_chart_top_sports")

    # 5. India Spotlight Section
    st.markdown("---")
    st.subheader("🇮🇳 Spotlight: India at the Olympic Games")
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
            display_chart(fig_i_sport, key="dash_chart_india_sports")

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
            display_chart(fig_i_trend, key="dash_chart_india_trend")
    else:
        st.info("No records found for India with the currently applied global filters.")

    # 6. Optional Technical Details (Expandable)
    with st.expander("🛠️ Technical Details & Data Warehouse Star Schema"):
        render_viva_note(
            "Star Schema Dynamic Query",
            "The metrics and charts on this dashboard are dynamically computed via SQL joins from the SQLite Star Schema warehouse (fact_medal joined with dim_game, dim_country, dim_sport, dim_event, and dim_medal).",
            "Star Schema optimizes analytical throughput via foreign-key indexed fact records and denormalized dimensional tables."
        )


if __name__ == "__main__":
    render_page()
