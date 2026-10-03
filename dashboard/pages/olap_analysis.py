"""
OLAP Analysis Page for OLYMPIA.
Interactive execution and visualization of the 5 core Online Analytical Processing operations:
1. SLICE: Filtering on a single dimension.
2. DICE: Multi-dimensional sub-cube extraction.
3. ROLL-UP: Summarizing measures up the hierarchy.
4. DRILL-DOWN: Navigating down to granular detail.
5. PIVOT: Rotating axes for multi-dimensional cross-tabulation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
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


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("🧊 OLAP Cube Multi-Dimensional Laboratory")
    st.caption("Perform live Online Analytical Processing operations (Slice, Dice, Roll-up, Drill-down, and Pivot) across the Olympic Star Schema.")

    if df.empty:
        st.warning("No Olympic records available in the active dataset.")
        return

    # 2. Key High-Level Metrics
    render_kpi_row([
        {"title": "Cube Fact Records", "value": f"{len(df):,}", "subtitle": "Total podium cells", "icon": "🧊"},
        {"title": "Available Dimensions", "value": "5", "subtitle": "Year, Season, Country, Sport, Gender", "icon": "📐"},
        {"title": "Numerical Measures", "value": "2", "subtitle": "Medal Count, Medal Points", "icon": "⭐"},
        {"title": "OLAP Operations", "value": "5", "subtitle": "Slice, Dice, Roll-up, Drill, Pivot", "icon": "🔄"},
    ])

    st.markdown("---")

    # 3. Interactive OLAP Operation Tabs
    tab_slice, tab_dice, tab_rollup, tab_drill, tab_pivot = st.tabs([
        "🔪 1. SLICE",
        "🎲 2. DICE",
        "⬆️ 3. ROLL-UP",
        "⬇️ 4. DRILL-DOWN",
        "🔄 5. PIVOT"
    ])

    # ----------------------------------------------------
    # 1. SLICE TAB
    # ----------------------------------------------------
    with tab_slice:
        st.subheader("1. SLICE: Single-Dimension Filter")
        st.markdown("""
        > **What is a Slice?**
        > A Slice selects exactly **one dimension** and fixes it to a specific value (like taking a single 2D slice from a 3D bread loaf).
        """)

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            slice_dim = st.selectbox("Select Dimension to Slice", ["year", "country", "season", "sport", "event_gender"], index=0, key="olap_slice_dim")
            dim_values = sorted(df[slice_dim].unique().tolist())
            default_val = 2024 if slice_dim == "year" and 2024 in dim_values else ("India" if slice_dim == "country" and "India" in dim_values else dim_values[0])
            slice_val = st.selectbox(f"Select Value for '{slice_dim}'", options=dim_values, index=dim_values.index(default_val), key="olap_slice_val")

        with col_s2:
            sliced_df = olap_slice(df, dimension=slice_dim, value=slice_val)
            render_kpi_row([
                {"title": "Slice Records", "value": f"{len(sliced_df):,}", "icon": "🔪"},
                {"title": "Golds in Slice", "value": f"{(sliced_df['medal']=='Gold').sum():,}", "icon": "🥇"},
                {"title": "Points in Slice", "value": f"{sliced_df['medal_points'].sum():,}", "icon": "⭐"}
            ])

        st.markdown(f"**Resulting Slice for `{slice_dim} = {slice_val}`:**")
        if not sliced_df.empty:
            slice_chart_df = (
                sliced_df.groupby("country" if slice_dim != "country" else "sport")
                .size()
                .reset_index(name="medals")
                .sort_values(by="medals", ascending=False)
                .head(10)
            )
            fig_slice = create_bar_chart(
                slice_chart_df,
                x="country" if slice_dim != "country" else "sport",
                y="medals",
                title=f"Top Categories in Slice: {slice_dim} = {slice_val}"
            )
            display_chart(fig_slice, key="olap_slice_chart")

            with st.expander(f"📋 View Records in Slice ({len(sliced_df):,} rows)"):
                st.dataframe(sliced_df[["year", "season", "country", "sport", "event_name", "medal"]].head(100), use_container_width=True)
        else:
            st.info("No records found in this slice.")

    # ----------------------------------------------------
    # 2. DICE TAB
    # ----------------------------------------------------
    with tab_dice:
        st.subheader("2. DICE: Multi-Dimensional Sub-Cube")
        st.markdown("""
        > **What is a Dice?**
        > A Dice defines a smaller sub-cube by specifying conditions on **multiple dimensions simultaneously** (e.g. Years 2016–2024 AND Season Summer AND Selected Countries).
        """)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            all_years = sorted(df["year"].unique().tolist())
            d_years = st.multiselect("Filter Years", options=all_years, default=[y for y in [2016, 2020, 2024] if y in all_years], key="dice_years")
        with col_d2:
            all_seasons = sorted(df["season"].unique().tolist())
            d_seasons = st.multiselect("Filter Season", options=all_seasons, default=["Summer"], key="dice_seasons")
        with col_d3:
            all_c = sorted(df["country"].unique().tolist())
            d_countries = st.multiselect("Filter Countries", options=all_c, default=[c for c in ["India", "United States", "China", "Japan"] if c in all_c], key="dice_countries")

        diced_df = olap_dice(df, filters={
            "year": d_years,
            "season": d_seasons,
            "country": d_countries
        })

        st.markdown(f"**Diced Sub-Cube Result: `{len(diced_df):,}` fact records found**")
        if not diced_df.empty:
            d_summary = diced_df.groupby(["country", "medal"]).size().reset_index(name="count")
            fig_dice = create_bar_chart(d_summary, x="country", y="count", color="medal", title="Diced Sub-Cube: Medals by Country & Medal Type", barmode="stack")
            display_chart(fig_dice, key="olap_dice_chart")

            with st.expander("📋 View Records in Diced Sub-Cube"):
                st.dataframe(diced_df[["year", "season", "country", "sport", "event_name", "medal"]].head(100), use_container_width=True)
        else:
            st.info("No records found matching the specified DICE criteria.")

    # ----------------------------------------------------
    # 3. ROLL-UP TAB
    # ----------------------------------------------------
    with tab_rollup:
        st.subheader("3. ROLL-UP: Summarize Up the Hierarchy")
        st.markdown("""
        > **What is a Roll-Up?**
        > Roll-Up aggregates detailed facts to a **higher, coarser level of granularity** (e.g. rolling up individual events into entire Sports or Countries).
        """)

        rollup_level = st.selectbox(
            "Select Dimension Level to Roll Up To",
            options=["country", "sport", "year", "season", "event_gender"],
            index=0,
            key="rollup_level"
        )
        rollup_df = olap_rollup(df, group_levels=[rollup_level], include_points=True)

        if not rollup_df.empty:
            fig_rollup = create_bar_chart(
                rollup_df.head(10),
                x=rollup_level,
                y="total_medals",
                title=f"Roll-Up Summary Aggregated by '{rollup_level}' (Top 10)"
            )
            display_chart(fig_rollup, key="olap_rollup_chart")
            st.dataframe(rollup_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No data available for roll-up aggregation.")

    # ----------------------------------------------------
    # 4. DRILL-DOWN TAB
    # ----------------------------------------------------
    with tab_drill:
        st.subheader("4. DRILL-DOWN: Navigate Down to Granular Detail")
        st.markdown("""
        > **What is a Drill-Down?**
        > Drill-Down does the opposite of roll-up: it navigates **from a high-level summary down to more detailed lower levels** (e.g. Nation → Sports → Events).
        """)

        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            all_c = sorted(df["country"].unique().tolist())
            dr_country = st.selectbox("1. Select Nation to Drill Into", options=all_c, index=all_c.index("India") if "India" in all_c else 0, key="drill_country")
        with col_dr2:
            dr_level = st.selectbox("2. Drill Down By", options=["sport", "year", "event_gender"], index=0, key="drill_level")

        drill_df = olap_drilldown(df, current_level="country", current_value=dr_country, next_level=dr_level)

        if not drill_df.empty:
            fig_drill = create_bar_chart(
                drill_df.head(10),
                x=dr_level,
                y="total_medals",
                title=f"Drill-Down: {dr_country} Medals Broken Down by '{dr_level}'"
            )
            display_chart(fig_drill, key="olap_drill_chart")
            st.dataframe(drill_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info(f"No records found for drill-down on {dr_country}.")

    # ----------------------------------------------------
    # 5. PIVOT TAB
    # ----------------------------------------------------
    with tab_pivot:
        st.subheader("5. PIVOT: Rotate Axes for Cross-Tabulation")
        st.markdown("""
        > **What is a Pivot?**
        > Pivot rotates the dimensions of a data cube to present an intuitive **two-dimensional cross-tabulation table** (e.g. Countries as Rows, Medal Types as Columns).
        """)

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            p_rows = st.selectbox("Row Dimension", ["country", "sport", "year", "season"], index=0, key="pivot_rows")
        with col_p2:
            p_cols = st.selectbox("Column Dimension", ["medal", "season", "event_gender"], index=0, key="pivot_cols")
        with col_p3:
            p_meas = st.selectbox("Measure", ["medal_count", "medal_points"], index=0, key="pivot_meas")

        if p_rows == p_cols:
            st.warning("Row dimension and Column dimension must be different.")
        else:
            try:
                val_col = "medal_points" if p_meas == "medal_points" else "medal_fact_id"
                agg_fn = "sum" if p_meas == "medal_points" else "count"
                pivot_df = olap_pivot(df, rows=p_rows, columns=p_cols, values=val_col, aggfunc=agg_fn)
                st.markdown(f"**Cross-Tabulation: `{p_rows}` × `{p_cols}` (Measure: `{p_meas}`):**")
                st.dataframe(pivot_df.head(25), use_container_width=True)
            except Exception as e:
                st.error(f"Error computing pivot: {e}")

    # 6. Optional Technical Details
    with st.expander("🛠️ Viva Explanations: OLAP Operations & Star Schema"):
        render_viva_note(
            "OLAP Multi-Dimensional Operations",
            "OLAP operations manipulate the logical hypercube defined by fact_medal and dimensions. In a Star Schema, SLICE and DICE map to SQL WHERE clauses, ROLL-UP and DRILL-DOWN modify GROUP BY granularity, and PIVOT executes matrix transpositions.",
            "OLAP provides sub-second multi-dimensional analytical throughput, serving as the foundation of executive Business Intelligence."
        )


if __name__ == "__main__":
    render_page()
