"""
OLAP Analysis Page for OLYMPIA.
Interactive execution and visualization of the 5 core Online Analytical Processing operations:
1. ROLL-UP
2. DRILL-DOWN
3. SLICE
4. DICE
5. PIVOT
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.olap import (
    olap_slice,
    olap_dice,
    olap_rollup,
    olap_drilldown,
    olap_pivot
)
from analytics.visualization import create_bar_chart


def render_page(df: pd.DataFrame):
    st.title("🧊 OLAP Cube Multi-Dimensional Analysis")
    st.caption("Perform multi-dimensional cube aggregations: Roll-up, Drill-down, Slice, Dice, and Pivot.")

    render_viva_note(
        "Online Analytical Processing (OLAP)",
        "OLAP enables analysts to quickly view data from different viewpoints by navigating a multidimensional data cube. Dimensions form the axes, while facts (medals, points) form the numerical cells.",
        "ROLL-UP summarizes data up a hierarchy (less detail); DRILL-DOWN navigates down to granular data (more detail); SLICE selects a single dimension value; DICE extracts a sub-cube with conditions on multiple dimensions; and PIVOT rotates axes for cross-tabulation."
    )

    st.markdown("---")

    tab_slice, tab_dice, tab_rollup, tab_drill, tab_pivot = st.tabs([
        "🔪 1. SLICE",
        "🎲 2. DICE",
        "⬆️ 3. ROLL-UP",
        "⬇️ 4. DRILL-DOWN",
        "🔄 5. PIVOT"
    ])

    # 1. SLICE TAB
    with tab_slice:
        st.subheader("1. SLICE Operation (Single Dimension Filter)")
        st.info("Fixes exactly ONE dimension to extract a 2D/3D slice of the cube.")

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            slice_dim = st.selectbox("Select Dimension to Slice", ["year", "country", "season", "sport", "event_gender"], index=0)
            dim_values = sorted(df[slice_dim].unique().tolist())
            default_val = 2024 if slice_dim == "year" and 2024 in dim_values else dim_values[0]
            slice_val = st.selectbox(f"Select Value for '{slice_dim}'", options=dim_values, index=dim_values.index(default_val))

        with col_s2:
            sliced_df = olap_slice(df, dimension=slice_dim, value=slice_val)
            render_kpi_row([
                {"title": "Slice Records", "value": f"{len(sliced_df):,}", "icon": "🔪"},
                {"title": "Golds in Slice", "value": f"{(sliced_df['medal']=='Gold').sum():,}", "icon": "🥇"},
                {"title": "Points in Slice", "value": f"{sliced_df['medal_points'].sum():,}", "icon": "⭐"}
            ])

        st.markdown(f"**Resulting Slice for `{slice_dim} = {slice_val}`:**")
        st.dataframe(sliced_df.head(100), use_container_width=True)

    # 2. DICE TAB
    with tab_dice:
        st.subheader("2. DICE Operation (Multi-Dimensional Sub-Cube)")
        st.info("Defines a sub-cube by specifying conditions simultaneously on MULTIPLE dimensions.")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            all_years = sorted(df["year"].unique().tolist())
            d_years = st.multiselect("Filter Years", options=all_years, default=[y for y in [2016, 2020, 2024] if y in all_years])
        with col_d2:
            all_seasons = sorted(df["season"].unique().tolist())
            d_seasons = st.multiselect("Filter Season", options=all_seasons, default=["Summer"])
        with col_d3:
            all_c = sorted(df["country"].unique().tolist())
            d_countries = st.multiselect("Filter Countries", options=all_c, default=[c for c in ["India", "United States", "China", "Australia"] if c in all_c])

        diced_df = olap_dice(df, filters={
            "year": d_years,
            "season": d_seasons,
            "country": d_countries
        })

        st.markdown(f"**Diced Sub-Cube: {len(diced_df):,} fact records found**")
        if not diced_df.empty:
            d_summary = diced_df.groupby(["country", "medal"]).size().reset_index(name="count")
            fig_dice = create_bar_chart(d_summary, x="country", y="count", color="medal", title="Diced Sub-Cube: Medals by Country & Type", barmode="stack")
            display_chart(fig_dice, key="fig_olap_dice")
            st.dataframe(diced_df.head(100), use_container_width=True)
        else:
            st.warning("No records found matching the specified DICE criteria.")

    # 3. ROLL-UP TAB
    with tab_rollup:
        st.subheader("3. ROLL-UP Operation (Hierarchy Aggregation)")
        st.info("Aggregates facts up a hierarchy: Event → Sport → Country → Olympic Edition.")

        levels = st.multiselect(
            "Select Grouping Hierarchy Levels for Roll-Up",
            options=["country", "sport", "event_gender", "season", "year"],
            default=["country", "sport"]
        )

        if levels:
            rollup_df = olap_rollup(df, group_levels=levels)
            st.markdown(f"**Rolled-Up Aggregate Data ({len(rollup_df):,} grouped rows):**")
            st.dataframe(rollup_df.head(50), use_container_width=True)
        else:
            st.warning("Please select at least one hierarchy level for Roll-Up.")

    # 4. DRILL-DOWN TAB
    with tab_drill:
        st.subheader("4. DRILL-DOWN Operation (Detail Breakdown)")
        st.info("Steps down a hierarchy from a summarized entity to detailed components.")

        col_dr1, col_dr2, col_dr3 = st.columns(3)
        with col_dr1:
            dr_level = st.selectbox("Current Hierarchy Level", ["country", "sport", "games"], index=0)
        with col_dr2:
            dr_vals = sorted(df[dr_level].unique().tolist())
            def_idx = dr_vals.index("India") if dr_level == "country" and "India" in dr_vals else 0
            dr_val = st.selectbox(f"Select {dr_level.title()}", options=dr_vals, index=def_idx)
        with col_dr3:
            next_options = [c for c in ["sport", "event_name", "athletes", "medal"] if c != dr_level]
            next_lvl = st.selectbox("Drill-Down Next Level", options=next_options, index=0)

        drill_df = olap_drilldown(df, current_level=dr_level, current_value=dr_val, next_level=next_lvl)
        st.markdown(f"**Drill-down results for {dr_level} = '{dr_val}' → Breakdown by {next_lvl}:**")

        if not drill_df.empty:
            fig_drill = create_bar_chart(
                drill_df.head(15),
                x=next_lvl,
                y="total_medals",
                title=f"Drill-Down: Top {next_lvl.replace('_', ' ').title()} in {dr_val}"
            )
            display_chart(fig_drill, key="fig_olap_drill")
            st.dataframe(drill_df, use_container_width=True)
        else:
            st.info("No data available for this drill-down selection.")

    # 5. PIVOT TAB
    with tab_pivot:
        st.subheader("5. PIVOT Operation (Cross-Tabulation Matrix)")
        st.info("Rotates axes of the multidimensional dataset for cross-tabulated reporting.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_rows = st.selectbox("Pivot Rows (Index)", ["country", "sport", "year", "season"], index=0)
        with col_p2:
            p_cols = st.selectbox("Pivot Columns", ["medal", "event_gender", "season"], index=0)

        pivot_df = olap_pivot(df, rows=p_rows, columns=p_cols)
        st.markdown(f"**Pivoted Matrix (Rows = {p_rows.title()}, Columns = {p_cols.title()}):**")
        st.dataframe(pivot_df.head(50), use_container_width=True)
