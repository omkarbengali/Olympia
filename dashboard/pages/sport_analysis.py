"""
Sport Analysis Page for OLYMPIA.
Examines individual sports, discipline history, country dominance, gender evolution,
and event breakdowns for any modern Olympic sport.
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
    plot_gender_breakdown
)


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("🏃 Sport Olympic Intelligence")
    st.caption("Investigate participation trends, dominant countries, gender evolution, and event breakdowns for any Olympic sport.")

    if df.empty:
        st.warning("No Olympic records available in the active dataset.")
        return

    all_sports = sorted(df["sport"].unique().tolist())
    if not all_sports:
        st.warning("No sports available in the active dataset.")
        return

    # Sport Selector
    default_idx = all_sports.index("Athletics") if "Athletics" in all_sports else 0
    selected_sport = st.selectbox("Select Olympic Sport", options=all_sports, index=default_idx)

    s_df = df[df["sport"] == selected_sport]

    if s_df.empty:
        st.info(f"No records found for sport **'{selected_sport}'** under the currently applied filters.")
        return

    total_medals = len(s_df)
    unique_countries = s_df["country"].nunique()
    unique_events = s_df["event_name"].nunique()
    golds = int((s_df["medal"] == "Gold").sum())
    silvers = int((s_df["medal"] == "Silver").sum())
    bronzes = int((s_df["medal"] == "Bronze").sum())

    # 2. Key Metrics Row
    render_kpi_row([
        {"title": "Total Medals Awarded", "value": f"{total_medals:,}", "icon": "🏅"},
        {"title": "Winning Nations", "value": f"{unique_countries}", "subtitle": "Countries on podium", "icon": "🌍"},
        {"title": "Distinct Events", "value": f"{unique_events}", "subtitle": "Event categories", "icon": "🎯"},
        {"title": "Golds", "value": f"{golds:,}", "icon": "🥇"},
        {"title": "Silvers", "value": f"{silvers:,}", "icon": "🥈"},
        {"title": "Bronzes", "value": f"{bronzes:,}", "icon": "🥉"},
    ])

    st.markdown("---")

    # 3. Main Visualizations: Top Countries & Historical Trend
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"🌍 Dominant Nations in {selected_sport}")
        top_countries = (
            s_df.groupby(["country", "medal"])
            .size()
            .reset_index(name="count")
        )
        top_c_names = (
            top_countries.groupby("country")["count"]
            .sum()
            .nlargest(10)
            .index.tolist()
        )
        bar_df = top_countries[top_countries["country"].isin(top_c_names)]
        fig_top_c = create_bar_chart(
            bar_df,
            x="country",
            y="count",
            color="medal",
            title=f"Top 10 Nations in {selected_sport}",
            barmode="stack"
        )
        display_chart(fig_top_c, key="sport_top_c_chart")

    with col2:
        st.subheader(f"📈 {selected_sport}: Medals Awarded Across Years")
        yearly_sport = s_df.groupby("year").size().reset_index(name="medals_awarded")
        fig_trend = create_line_chart(
            yearly_sport,
            x="year",
            y="medals_awarded",
            title=f"{selected_sport}: Historical Trend Across Olympic Editions"
        )
        display_chart(fig_trend, key="sport_yearly_trend_chart")

    # 4. Secondary Insights: Gender Breakdown & Events Table
    st.markdown("---")
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader(f"⚖️ {selected_sport}: Gender Distribution")
        fig_gender = plot_gender_breakdown(s_df)
        display_chart(fig_gender, key="sport_gender_chart")

    with col4:
        st.subheader("🎯 Most Contested Event Categories")
        top_events = (
            s_df.groupby("event_name")
            .agg(
                Medals=("medal", "count"),
                Genders=("event_gender", lambda g: ", ".join(sorted(g.unique())))
            )
            .reset_index()
            .sort_values(by="Medals", ascending=False)
            .head(10)
        )
        st.dataframe(top_events, use_container_width=True, hide_index=True)

    # Historical Podium Table
    with st.expander(f"📋 View All {total_medals} Podium Records for {selected_sport}"):
        display_cols = ["year", "games", "country", "event_name", "event_gender", "medal", "athletes"]
        st.dataframe(s_df[display_cols].sort_values(by="year", ascending=False), use_container_width=True, hide_index=True)

    # 5. Optional Technical Details
    with st.expander("🛠️ Technical Details & Dimensional Modeling"):
        render_viva_note(
            f"{selected_sport} Discipline Profile Query",
            f"Filtered on dim_sport.sport_name = '{selected_sport}'. Relates to dim_event (event hierarchy) and dim_game (temporal editions).",
            "Demonstrates 1:N dimensional relationships where one sport spans multiple distinct competitive events."
        )


if __name__ == "__main__":
    render_page()
