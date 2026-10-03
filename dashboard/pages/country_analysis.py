"""
Country Analysis Page for OLYMPIA.
Detailed drilldown into any Olympic nation, with side-by-side comparative analytics,
sport discipline dominance, and historical medal trends.
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
    st.title("🌍 Country Olympic Intelligence")
    st.caption("Deep-dive into any nation's historical Olympic performance, sport dominance, recent trends, and bilateral comparisons.")

    if df.empty:
        st.warning("No Olympic records match the currently selected filter combination.")
        return

    all_countries = sorted(df["country"].unique().tolist())
    if not all_countries:
        st.warning("No countries available in the active dataset.")
        return

    # Mode Selector
    mode = st.radio("Select Analysis Mode", ["Single Country Deep Dive", "Compare Two Countries"], horizontal=True)

    if mode == "Single Country Deep Dive":
        # Country Selector Filter
        default_idx = all_countries.index("India") if "India" in all_countries else 0
        selected_country = st.selectbox("Select Country", options=all_countries, index=default_idx)

        c_df = df[df["country"] == selected_country]

        if c_df.empty:
            st.info(f"No medal records found for **{selected_country}** under the currently applied filters.")
            return

        total_medals = len(c_df)
        golds = int((c_df["medal"] == "Gold").sum())
        silvers = int((c_df["medal"] == "Silver").sum())
        bronzes = int((c_df["medal"] == "Bronze").sum())
        points = int(c_df["medal_points"].sum())
        editions = c_df["games"].nunique()
        sports_count = c_df["sport"].nunique()

        # 2. Key Metrics Row
        render_kpi_row([
            {"title": "Total Medals", "value": str(total_medals), "icon": "🏅"},
            {"title": "Gold Medals", "value": str(golds), "icon": "🥇"},
            {"title": "Silver Medals", "value": str(silvers), "icon": "🥈"},
            {"title": "Bronze Medals", "value": str(bronzes), "icon": "🥉"},
            {"title": "Medal Points", "value": str(points), "subtitle": "Gold=3, Silver=2, Bronze=1", "icon": "⭐"},
            {"title": "Editions Won", "value": str(editions), "subtitle": f"Across {sports_count} sports", "icon": "🏛️"}
        ])

        st.markdown("---")

        # 3. Main Visualization: Medal Performance Over Time & Gender Evolution
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(f"📈 {selected_country}: Medal Performance Over Time")
            yearly = c_df.groupby("year").size().reset_index(name="medals_won")
            fig_trend = create_line_chart(
                yearly,
                x="year",
                y="medals_won",
                title=f"{selected_country}: Medals Won per Olympic Year"
            )
            display_chart(fig_trend, key="country_trend_chart")

        with col2:
            st.subheader(f"⚖️ {selected_country}: Gender Distribution")
            fig_gender = plot_gender_breakdown(c_df)
            display_chart(fig_gender, key="country_gender_chart")

        # 4. Secondary Insights: Medals By Sport
        st.markdown("---")
        st.subheader(f"🎯 {selected_country}: Medals by Sport Discipline")
        sport_medals = (
            c_df.groupby(["sport", "medal"])
            .size()
            .reset_index(name="count")
        )
        fig_sport = create_bar_chart(
            sport_medals,
            x="sport",
            y="count",
            color="medal",
            title=f"{selected_country}: Medal Distribution by Sport",
            barmode="stack"
        )
        display_chart(fig_sport, key="country_sport_chart")

        # Sport Breakdown Table & Recent Performance
        col_t1, col_t2 = st.columns([1, 1])

        with col_t1:
            st.subheader("🏆 Top Performing Sports")
            top_sports_table = (
                c_df.groupby("sport")
                .agg(
                    Total=("medal", "count"),
                    Gold=("medal", lambda m: (m == "Gold").sum()),
                    Silver=("medal", lambda m: (m == "Silver").sum()),
                    Bronze=("medal", lambda m: (m == "Bronze").sum()),
                    Points=("medal_points", "sum")
                )
                .reset_index()
                .sort_values(by="Total", ascending=False)
            )
            st.dataframe(top_sports_table, use_container_width=True, hide_index=True)

        with col_t2:
            st.subheader("⚡ Recent Olympic Editions")
            recent_table = (
                c_df[c_df["year"] >= 2000]
                .groupby(["year", "games"])
                .agg(
                    Total=("medal", "count"),
                    Gold=("medal", lambda m: (m == "Gold").sum()),
                    Silver=("medal", lambda m: (m == "Silver").sum()),
                    Bronze=("medal", lambda m: (m == "Bronze").sum())
                )
                .reset_index()
                .sort_values(by="year", ascending=False)
            )
            if not recent_table.empty:
                st.dataframe(recent_table, use_container_width=True, hide_index=True)
            else:
                st.info(f"No 2000–2024 Olympic medal finishes on record for {selected_country}.")

        # Expandable Records Explorer
        with st.expander(f"📋 View All {total_medals} Individual Podium Records for {selected_country}"):
            display_cols = ["year", "season", "games", "sport", "event_name", "event_gender", "medal", "athletes"]
            st.dataframe(c_df[display_cols].sort_values(by="year", ascending=False), use_container_width=True, hide_index=True)

        # 5. Optional Technical Details
        with st.expander("🛠️ Technical Details & Dimensional Context"):
            render_viva_note(
                f"{selected_country} Historical Profile Query",
                f"Aggregated from fact_medal filtered on country_id for '{selected_country}'. Demonstrates 1-to-many dimensional relationships across dim_country, dim_sport, and dim_game.",
                "OLAP slice on dim_country.country_name produces deterministic aggregate statistics."
            )

    else:
        # Comparison Mode
        st.subheader("⚔️ Head-to-Head Country Comparison")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            def1 = all_countries.index("India") if "India" in all_countries else 0
            country_a = st.selectbox("Select Country A", options=all_countries, index=def1, key="comp_a_select")
        with col_c2:
            def2 = all_countries.index("United States") if "United States" in all_countries else min(1, len(all_countries) - 1)
            country_b = st.selectbox("Select Country B", options=all_countries, index=def2, key="comp_b_select")

        df_a = df[df["country"] == country_a]
        df_b = df[df["country"] == country_b]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"### {country_a}")
            render_kpi_row([
                {"title": "Total Medals", "value": str(len(df_a)), "icon": "🏅"},
                {"title": "Gold Medals", "value": str(int((df_a["medal"] == "Gold").sum())), "icon": "🥇"},
                {"title": "Points", "value": str(int(df_a["medal_points"].sum())), "icon": "⭐"},
            ])
        with col_m2:
            st.markdown(f"### {country_b}")
            render_kpi_row([
                {"title": "Total Medals", "value": str(len(df_b)), "icon": "🏅"},
                {"title": "Gold Medals", "value": str(int((df_b["medal"] == "Gold").sum())), "icon": "🥇"},
                {"title": "Points", "value": str(int(df_b["medal_points"].sum())), "icon": "⭐"},
            ])

        # Comparative Trend Line Chart
        comp_df = df[df["country"].isin([country_a, country_b])]
        if not comp_df.empty:
            yearly_comp = comp_df.groupby(["year", "country"]).size().reset_index(name="medals_won")
            fig_comp = create_line_chart(
                yearly_comp,
                x="year",
                y="medals_won",
                color="country",
                title=f"Head-to-Head Medal Trajectory: {country_a} vs {country_b}"
            )
            display_chart(fig_comp, key="country_comp_chart")


def render_comparison_page(df: pd.DataFrame = None):
    """Dedicated Head-to-Head Comparison Page."""
    if df is None:
        df = get_denormalized_medals()

    st.title("⚔️ Head-to-Head Country Olympic Showdown")
    st.caption("Compare the historical Olympic performance, gold medal conversion, and medal trajectories between any two nations.")

    if df.empty:
        st.warning("No Olympic records available in the active dataset.")
        return

    all_countries = sorted(df["country"].unique().tolist())
    if len(all_countries) < 2:
        st.warning("At least two countries are required for comparison.")
        return

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        def1 = all_countries.index("India") if "India" in all_countries else 0
        country_a = st.selectbox("Select First Country", options=all_countries, index=def1, key="h2h_a_select")
    with col_c2:
        def2 = all_countries.index("United States") if "United States" in all_countries else min(1, len(all_countries) - 1)
        country_b = st.selectbox("Select Second Country", options=all_countries, index=def2, key="h2h_b_select")

    df_a = df[df["country"] == country_a]
    df_b = df[df["country"] == country_b]

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"### {country_a}")
        render_kpi_row([
            {"title": "Total Medals", "value": str(len(df_a)), "icon": "🏅"},
            {"title": "Gold Medals", "value": str(int((df_a["medal"] == "Gold").sum())), "icon": "🥇"},
            {"title": "Silver Medals", "value": str(int((df_a["medal"] == "Silver").sum())), "icon": "🥈"},
            {"title": "Bronze Medals", "value": str(int((df_a["medal"] == "Bronze").sum())), "icon": "🥉"},
            {"title": "Points", "value": str(int(df_a["medal_points"].sum())), "icon": "⭐"},
        ])
    with col_m2:
        st.markdown(f"### {country_b}")
        render_kpi_row([
            {"title": "Total Medals", "value": str(len(df_b)), "icon": "🏅"},
            {"title": "Gold Medals", "value": str(int((df_b["medal"] == "Gold").sum())), "icon": "🥇"},
            {"title": "Silver Medals", "value": str(int((df_b["medal"] == "Silver").sum())), "icon": "🥈"},
            {"title": "Bronze Medals", "value": str(int((df_b["medal"] == "Bronze").sum())), "icon": "🥉"},
            {"title": "Points", "value": str(int(df_b["medal_points"].sum())), "icon": "⭐"},
        ])

    st.markdown("---")

    # Comparative Trend Line Chart
    comp_df = df[df["country"].isin([country_a, country_b])]
    if not comp_df.empty:
        yearly_comp = comp_df.groupby(["year", "country"]).size().reset_index(name="medals_won")
        fig_comp = create_line_chart(
            yearly_comp,
            x="year",
            y="medals_won",
            color="country",
            title=f"Head-to-Head Medal Trajectory: {country_a} vs {country_b}"
        )
        display_chart(fig_comp, key="h2h_comp_chart")

        # Sport Breakdown Comparison
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader(f"Top Sports for {country_a}")
            sp_a = df_a.groupby("sport").size().reset_index(name="medals").sort_values(by="medals", ascending=False).head(8)
            st.dataframe(sp_a, use_container_width=True, hide_index=True)
        with col_s2:
            st.subheader(f"Top Sports for {country_b}")
            sp_b = df_b.groupby("sport").size().reset_index(name="medals").sort_values(by="medals", ascending=False).head(8)
            st.dataframe(sp_b, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_page()

