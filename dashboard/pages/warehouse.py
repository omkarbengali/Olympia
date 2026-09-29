"""
Data Warehouse Architecture Page for OLYMPIA.
Displays Star Schema design, table definitions, row counts, primary/foreign keys,
and viva-oriented architectural explanations.
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from warehouse.warehouse import get_table_counts, get_schema_info, execute_query


def render_page(df: pd.DataFrame):
    st.title("🏛️ Olympic Data Warehouse & Star Schema")
    st.caption("Inspect the dimensional modeling, physical SQLite schema, foreign key relations, and table row counts.")

    render_viva_note(
        "Star Schema Architecture",
        "A Star Schema organizes data into a centralized Fact Table containing quantitative measurements, surrounded by radial Dimension Tables providing contextual attributes.",
        "Viva Distinction: Star Schema maintains denormalized dimensions with 1-hop joins to facts, maximizing analytical OLAP query throughput, unlike normalized 3NF (Snowflake/OLTP) which requires complex multi-table joins."
    )

    st.markdown("---")

    counts = get_table_counts()

    render_kpi_row([
        {"title": "fact_medal (Fact)", "value": f"{counts.get('fact_medal', 0):,}", "subtitle": "Podium medal facts", "icon": "⭐"},
        {"title": "dim_game (Dimension)", "value": f"{counts.get('dim_game', 0)}", "subtitle": "Olympic editions", "icon": "🏛️"},
        {"title": "dim_country (Dimension)", "value": f"{counts.get('dim_country', 0)}", "subtitle": "NOC nations", "icon": "🌍"},
        {"title": "dim_sport (Dimension)", "value": f"{counts.get('dim_sport', 0)}", "subtitle": "Disciplines", "icon": "🏃"},
        {"title": "dim_event (Dimension)", "value": f"{counts.get('dim_event', 0):,}", "subtitle": "Gendered events", "icon": "🎯"},
        {"title": "dim_medal (Dimension)", "value": f"{counts.get('dim_medal', 0)}", "subtitle": "Medal scoring", "icon": "🥇"},
    ])

    st.markdown("---")

    # Star Schema Visual Representation
    st.subheader("📐 Star Schema Relational Diagram")
    st.markdown("""
```
                         +-----------------------+
                         |       dim_game        |
                         +-----------------------+
                         | PK: game_id           |
                         |     year              |
                         |     season            |
                         |     games (UNIQUE)    |
                         +-----------+-----------+
                                     |
                                     | 1:N
+-----------------------+            |            +-----------------------+
|      dim_country      |            |            |       dim_sport       |
+-----------------------+            |            +-----------------------+
| PK: country_id        |            |            | PK: sport_id          |
|     country_code      |            v            |     sport_name (UQ)   |
|     country_name (UQ) |---+ +-------------+ +---|                       |
+-----------------------+   | |             | |   +-----------+-----------+
                            | |             | |               |
                            v v             v v               | 1:N
                         +-----------------------+            v
                         |      fact_medal       |   +-----------------------+
                         +-----------------------+   |       dim_event       |
                         | PK: medal_fact_id     |   +-----------------------+
                         | FK: game_id           |   | PK: event_id          |
                         | FK: country_id        |-->| FK: sport_id          |
                         | FK: sport_id          |   |     event_name        |
                         | FK: event_id          |   |     event_gender      |
                         | FK: medal_id          |<--+-----------------------+
                         |     athletes          |
                         +-----------+-----------+
                                     ^
                                     | 1:N
                         +-----------+-----------+
                         |       dim_medal       |
                         +-----------------------+
                         | PK: medal_id          |
                         |     medal_name (UQ)   |
                         |     medal_points      |
                         +-----------------------+
```
    """)

    st.markdown("---")

    # Table Schema Inspector
    st.subheader("🔍 Physical Table Inspector")
    schema_info = get_schema_info()
    table_names = list(schema_info.keys())
    selected_tbl = st.selectbox("Inspect Table Structure & Sample Records", options=table_names)

    if selected_tbl:
        tbl_meta = schema_info[selected_tbl]
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown(f"**Columns for `{selected_tbl}`:**")
            cols_df = pd.DataFrame(tbl_meta["columns"])[["name", "type", "is_pk", "notnull"]]
            st.dataframe(cols_df, use_container_width=True)

        with col_t2:
            st.markdown(f"**Foreign Keys for `{selected_tbl}`:**")
            if tbl_meta["foreign_keys"]:
                fks_df = pd.DataFrame(tbl_meta["foreign_keys"])[["from", "table", "to"]]
                st.dataframe(fks_df, use_container_width=True)
            else:
                st.info("No foreign keys (Primary Dimension Table)")

        st.markdown(f"**Sample Top 10 Records in `{selected_tbl}`:**")
        sample_df = execute_query(f"SELECT * FROM {selected_tbl} LIMIT 10;")
        st.dataframe(sample_df, use_container_width=True)
