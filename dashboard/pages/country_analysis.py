"""
Country Analysis Page for OLYMPIA.
Detailed drilldown into any Olympic nation, with side-by-side comparative analytics.
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.visualization import create_bar_chart, create_line_chart, plot_gender_breakdown


def render_page(df: pd.DataFrame):
    st.title("🌍 Country Olympic Intelligence")
    st.caption("Investigate historical Olympic performance, sport dominance, and gender distribution for any nation.")

    all_countries = sorted(df["country"].unique().tolist())

    mode = st.radio("Analysis Mode", ["Single Country Deep Dive", "Compare Two Countries"], horizontal=True)

    if mode == "Single Country Deep Dive":
        default_idx = all_countries.index("India") if "India" in all_countries else 0
        selected_country = st.selectbox("Select Country", options=all_countries, index=default_idx)

        c_df = df[df["country"] == selected_country]

        if c_df.empty:
            st.warning(f"No records found for {selected_country} under currently applied global filters.")
            return

        total_medals = len(c_df)
        golds = int((c_df["medal"] == "Gold").sum())
        silvers = int((c_df["medal"] == "Silver").sum())
        bronzes = int((c_df["medal"] == "Bronze").sum())
        points = int(c_df["medal_points"].sum())
        editions = c_df["games"].nunique()
        sports_count = c_df["sport"].nunique()

        render_kpi_row([
            {"title": "Total Medals", "value": str(total_medals), "icon": "🏅"},
            {"title": "Gold Medals", "value": str(golds), "icon": "🥇"},
            {"title": "Silver Medals", "value": str(silvers), "icon": "🥈"},
            {"title": "Bronze Medals", "value": str(bronzes), "icon": "🥉"},
            {"title": "Medal Points", "value": str(points), "subtitle": "Gold=3, Silver=2, Bronze=1", "icon": "⭐"},
            {"title": "Editions Won", "value": str(editions), "subtitle": f"Across {sports_count} sports", "icon": "🏛️"}
        ])

        render_viva_note(
            f"{selected_country} Historical Profile",
            f"{selected_country} has won {total_medals} medals across {editions} Olympic Games editions, spanning {sports_count} distinct sports disciplines.",
            "Analyzing nation-level performance over time reveals long-term sporting investment and specialized dominance in specific athletic disciplines."
        )

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            yearly = c_df.groupby("year").size().reset_index(name="medals_won")
            fig_trend = create_line_chart(
                yearly,
                x="year",
                y="medals_won",
                title=f"{selected_country}: Medal Trajectory Across Years"
            )
            display_chart(fig_trend, key="c_trend_chart")

        with col2:
            fig_gender = plot_gender_breakdown(c_df)
            display_chart(fig_gender, key="c_gender_chart")

        st.markdown("### 🎯 Sport Breakdown")
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
            title=f"{selected_country}: Medals Won by Sport Discipline",
            barmode="stack"
        )
        display_chart(fig_sport, key="c_sport_chart")

        # Athlete / Event Table
        with st.expander(f"📋 View All {total_medals} Medal Records for {selected_country}"):
            display_cols = ["year", "season", "games", "sport", "event_name", "event_gender", "medal", "athletes"]
            st.dataframe(c_df[display_cols].sort_values(by="year", ascending=False), use_container_width=True)

    else:
        # Comparison Mode
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            def1 = all_countries.index("India") if "India" in all_countries else 0
            country_a = st.selectbox("Select Country A", options=all_countries, index=def1, key="comp_a")
        with col_c2:
            def2 = all_countries.index("United States") if "United States" in all_countries else 1
            country_b = st.selectbox("Select Country B", options=all_countries, index=def2, key="comp_b")

        df_a = df[df["country"] == country_a]
        df_b = df[df["country"] == country_b]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"#### 🚩 {country_a}")
            render_kpi_row([
                {"title": "Total", "value": str(len(df_a)), "icon": "🏅"},
                {"title": "Golds", "value": str((df_a['medal']=='Gold').sum()), "icon": "🥇"},
                {"title": "Points", "value": str(df_a['medal_points'].sum()), "icon": "⭐"}
            ])
        with col_m2:
            st.markdown(f"#### 🚩 {country_b}")
            render_kpi_row([
                {"title": "Total", "value": str(len(df_b)), "icon": "🏅"},
                {"title": "Golds", "value": str((df_b['medal']=='Gold').sum()), "icon": "🥇"},
                {"title": "Points", "value": str(df_b['medal_points'].sum()), "icon": "⭐"}
            ])

        comp_df = df[df["country"].isin([country_a, country_b])]
        yearly_comp = comp_df.groupby(["year", "country"]).size().reset_index(name="medals_won")

        fig_comp = create_line_chart(
            yearly_comp,
            x="year",
            y="medals_won",
            color="country",
            title=f"Head-to-Head Medal Trend: {country_a} vs {country_b}"
        )
        display_chart(fig_comp, key="comp_line_chart")
