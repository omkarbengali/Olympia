"""
Sport & Gender Analysis Page for OLYMPIA.
Examines individual sports, discipline history, country dominance, and gender evolution.
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.visualization import create_bar_chart, create_line_chart, plot_gender_breakdown


def render_page(df: pd.DataFrame):
    st.title("🏃 Sport & Gender Olympic Analysis")
    st.caption("Investigate participation trends, dominant countries, and event breakdowns for any Olympic sport.")

    all_sports = sorted(df["sport"].unique().tolist())
    default_idx = all_sports.index("Athletics") if "Athletics" in all_sports else 0
    selected_sport = st.selectbox("Select Olympic Sport", options=all_sports, index=default_idx)

    s_df = df[df["sport"] == selected_sport]

    if s_df.empty:
        st.warning(f"No records found for sport '{selected_sport}' under current filters.")
        return

    total_medals = len(s_df)
    unique_countries = s_df["country"].nunique()
    unique_events = s_df["event_name"].nunique()
    golds = int((s_df["medal"] == "Gold").sum())
    silvers = int((s_df["medal"] == "Silver").sum())
    bronzes = int((s_df["medal"] == "Bronze").sum())

    render_kpi_row([
        {"title": "Total Medals", "value": f"{total_medals:,}", "icon": "🏅"},
        {"title": "Winning Nations", "value": f"{unique_countries}", "subtitle": "Nations on podium", "icon": "🌍"},
        {"title": "Events Contested", "value": f"{unique_events}", "subtitle": "Distinct event categories", "icon": "🎯"},
        {"title": "Golds Awarded", "value": f"{golds:,}", "icon": "🥇"},
        {"title": "Silvers Awarded", "value": f"{silvers:,}", "icon": "🥈"},
        {"title": "Bronzes Awarded", "value": f"{bronzes:,}", "icon": "🥉"},
    ])

    render_viva_note(
        f"{selected_sport} Discipline Profile",
        f"A total of {total_medals} medal finishes have been awarded in {selected_sport} across {unique_events} unique events, distributed among {unique_countries} different countries.",
        "Sport-level analytics reveal geographic clusters of excellence and illustrate the steady expansion of women's and mixed events across Olympic history."
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Top Countries in this Sport
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
            title=f"Top 10 Dominant Nations in {selected_sport}",
            barmode="stack"
        )
        display_chart(fig_top_c, key="sport_top_c")

    with col2:
        fig_gender = plot_gender_breakdown(s_df)
        display_chart(fig_gender, key="sport_gender")

    col3, col4 = st.columns([1, 1])

    with col3:
        yearly_sport = s_df.groupby("year").size().reset_index(name="medals_awarded")
        fig_trend = create_line_chart(
            yearly_sport,
            x="year",
            y="medals_awarded",
            title=f"{selected_sport}: Medals Awarded Across Olympic Years"
        )
        display_chart(fig_trend, key="sport_yearly_trend")

    with col4:
        top_events = (
            s_df.groupby("event_name")
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
            .head(10)
        )
        fig_events = create_bar_chart(
            top_events,
            x="event_name",
            y="count",
            title=f"Top Events Contested in {selected_sport}",
            orientation="v"
        )
        display_chart(fig_events, key="sport_events_bar")
