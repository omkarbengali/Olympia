"""
Medal Analysis & Data Visualization Experiment Page for OLYMPIA.
Demonstrates the 5 core statistical visualization archetypes:
1. BAR CHART
2. LINE CHART
3. SCATTER PLOT
4. HISTOGRAM
5. BOX PLOT
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.visualization import (
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_histogram,
    create_box_plot,
    MEDAL_COLORS
)


def render_page(df: pd.DataFrame):
    st.title("🥇 Medal Analytics & Visualization Laboratory")
    st.caption("Statistical medal distributions, outlier detection, and comparative charts across Olympic history.")

    render_viva_note(
        "Data Visualization Experiment",
        "Data visualization maps multi-attribute warehouse data into visual encodings (position, length, color, size) to discover trends, correlations, skewness, and outliers.",
        "Categorical data suits Bar Charts; temporal series require Line Charts; bivariate numerical relations use Scatter Plots; univariate density uses Histograms; and statistical dispersion uses Box Plots."
    )

    # 1. Summary Metrics
    total = len(df)
    gold = int((df["medal"] == "Gold").sum())
    silver = int((df["medal"] == "Silver").sum())
    bronze = int((df["medal"] == "Bronze").sum())
    pts = int(df["medal_points"].sum())

    render_kpi_row([
        {"title": "Total Medals", "value": f"{total:,}", "icon": "🏅"},
        {"title": "Gold Medals", "value": f"{gold:,}", "subtitle": f"{gold/max(total,1)*100:.1f}%", "icon": "🥇"},
        {"title": "Silver Medals", "value": f"{silver:,}", "subtitle": f"{silver/max(total,1)*100:.1f}%", "icon": "🥈"},
        {"title": "Bronze Medals", "value": f"{bronze:,}", "subtitle": f"{bronze/max(total,1)*100:.1f}%", "icon": "🥉"},
        {"title": "Total Medal Points", "value": f"{pts:,}", "subtitle": "Weight: Gold=3, Silver=2, Bronze=1", "icon": "⭐"},
    ])

    st.markdown("---")

    # Tabs for the 5 Core Visualization Types
    tab_bar, tab_line, tab_scatter, tab_hist, tab_box = st.tabs([
        "📊 1. Bar Chart",
        "📈 2. Line Chart",
        "🔵 3. Scatter Plot",
        "📉 4. Histogram",
        "📦 5. Box Plot"
    ])

    # 1. Bar Chart Tab
    with tab_bar:
        st.subheader("1. Bar Chart: Country Medal Standings")
        col_b1, col_b2 = st.columns([1, 3])
        with col_b1:
            top_k = st.slider("Number of Countries", 5, 25, 10, key="bar_top_k")
            barmode = st.selectbox("Bar Stacking Mode", ["stack", "group"], index=0, key="bar_mode")
        with col_b2:
            country_medals = (
                df.groupby(["country", "medal"])
                .size()
                .reset_index(name="count")
            )
            top_countries = (
                country_medals.groupby("country")["count"]
                .sum()
                .nlargest(top_k)
                .index.tolist()
            )
            bar_df = country_medals[country_medals["country"].isin(top_countries)]
            fig_bar = create_bar_chart(
                bar_df,
                x="country",
                y="count",
                color="medal",
                title=f"Top {top_k} Nations by Medal Class",
                barmode=barmode
            )
            display_chart(fig_bar, key="fig_medal_bar")

    # 2. Line Chart Tab
    with tab_line:
        st.subheader("2. Line Chart: Olympic Growth & Nation Trajectories")
        all_countries = sorted(df["country"].unique().tolist())
        default_selected = [c for c in ["United States", "China", "Great Britain", "India"] if c in all_countries]
        selected_nations = st.multiselect(
            "Select Countries to Compare Over Time",
            options=all_countries,
            default=default_selected,
            key="line_countries"
        )
        if selected_nations:
            line_df = (
                df[df["country"].isin(selected_nations)]
                .groupby(["year", "country"])
                .size()
                .reset_index(name="medals_won")
            )
            fig_line = create_line_chart(
                line_df,
                x="year",
                y="medals_won",
                color="country",
                title="Medal Trajectories Across Olympic Editions"
            )
            display_chart(fig_line, key="fig_medal_line")
        else:
            st.warning("Please select at least one country to display the line trend.")

    # 3. Scatter Plot Tab
    with tab_scatter:
        st.subheader("3. Scatter Plot: Correlation of Gold Medals vs Total Points")
        country_agg = df.groupby("country").agg(
            gold=("medal", lambda s: (s == "Gold").sum()),
            silver=("medal", lambda s: (s == "Silver").sum()),
            bronze=("medal", lambda s: (s == "Bronze").sum()),
            total_medals=("medal_fact_id", "count"),
            medal_points=("medal_points", "sum")
        ).reset_index()

        fig_scatter = create_scatter_plot(
            country_agg,
            x="total_medals",
            y="medal_points",
            size="gold",
            color="medal_points",
            hover_name="country",
            title="Total Medals vs Medal Points (Bubble Size = Gold Medals)"
        )
        display_chart(fig_scatter, key="fig_medal_scatter")

    # 4. Histogram Tab
    with tab_hist:
        st.subheader("4. Histogram: Frequency Distribution of Country Medals")
        col_h1, col_h2 = st.columns([1, 3])
        with col_h1:
            nbins = st.slider("Histogram Bins", 10, 60, 30, key="hist_bins")
            hist_var = st.selectbox("Distribution Variable", ["total_medals", "medal_points", "gold"], key="hist_var")
        with col_h2:
            fig_hist = create_histogram(
                country_agg,
                x=hist_var,
                nbins=nbins,
                title=f"Country Distribution of {hist_var.replace('_', ' ').title()} (Log-tailed power law)"
            )
            display_chart(fig_hist, key="fig_medal_hist")

    # 5. Box Plot Tab
    with tab_box:
        st.subheader("5. Box Plot: Statistical Quartiles & Outlier Analysis")
        # Medals won per Olympic games edition across top sports
        top_sports = df["sport"].value_counts().nlargest(8).index.tolist()
        box_data = (
            df[df["sport"].isin(top_sports)]
            .groupby(["year", "sport"])
            .size()
            .reset_index(name="medals_per_edition")
        )
        fig_box = create_box_plot(
            box_data,
            x="sport",
            y="medals_per_edition",
            color="sport",
            title="Medal Distribution Spread & Outliers per Edition Across Top Sports"
        )
        display_chart(fig_box, key="fig_medal_box")
