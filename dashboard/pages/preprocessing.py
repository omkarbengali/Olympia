"""
Data Preprocessing & Quality Analysis Page for OLYMPIA.
Demonstrates:
1. Data Quality Assessment (Raw vs Cleaned Audit, Duplicates, Missingness, Types).
2. Categorical Encoding (Label Encoding & One-Hot Encoding).
3. Feature Scaling (Min-Max Normalization & Z-score Standardization).
4. Dimensionality Reduction (Principal Component Analysis - PCA).
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.preprocessing import (
    get_data_quality_report,
    demonstrate_label_encoding,
    demonstrate_one_hot_encoding,
    demonstrate_feature_scaling,
    demonstrate_pca_reduction
)
from analytics.visualization import create_bar_chart, create_line_chart


def render_page(df: pd.DataFrame):
    st.title("🧹 Data Preprocessing & Quality Laboratory")
    st.caption("Inspect data quality audits, missing value treatment, feature encoding, scaling, and PCA dimensionality reduction.")

    render_viva_note(
        "Data Preprocessing in Data Mining",
        "Raw real-world datasets invariably contain noise, missing values, duplicates, and inconsistent representations. Preprocessing prepares raw data for data warehousing and mathematical learning algorithms.",
        "Crucial Principle: In enterprise systems, warehouse tables maintain clean, human-readable categorical strings. Machine Learning preprocessing (such as One-Hot Encoding and Z-score scaling) is applied downstream in memory to preserve data warehouse semantic integrity."
    )

    st.markdown("---")

    report = get_data_quality_report()

    # 1. Quality KPI Row
    render_kpi_row([
        {"title": "Raw CSV Rows", "value": f"{report['raw_rows']:,}", "subtitle": "10 original columns", "icon": "📄"},
        {"title": "Cleaned Fact Rows", "value": f"{report['cleaned_rows']:,}", "subtitle": "Star schema loaded", "icon": "✅"},
        {"title": "Exact Duplicates", "value": f"{report['raw_duplicates']}", "subtitle": "Dropped in ETL", "icon": "✂️"},
        {"title": "Missing Values in Warehouse", "value": f"{report['cleaned_missing_total']}", "subtitle": "100% complete", "icon": "🛡️"},
    ])

    st.markdown("### 📋 Missing Values & Data Types Audit (Raw Dataset)")
    st.dataframe(report["missing_summary_df"], use_container_width=True)

    st.markdown("---")

    # Tabs for ML Preprocessing Experiments
    tab_enc, tab_scale, tab_pca = st.tabs([
        "🏷️ 1. Categorical Encoding",
        "📏 2. Feature Scaling",
        "📉 3. PCA Dimensionality Reduction"
    ])

    # 1. Categorical Encoding Tab
    with tab_enc:
        st.subheader("1. Label Encoding vs One-Hot Encoding")
        st.markdown(
            "Machine learning algorithms require numeric inputs. **Label Encoding** assigns integer ranks (0, 1, 2) "
            "suitable for ordinal/tree-based models. **One-Hot Encoding** creates binary indicators to avoid false ordinal hierarchy."
        )

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("#### A. Label Encoding")
            enc_df, mappings = demonstrate_label_encoding(df.head(10), columns=["event_gender", "medal", "season"])
            st.write("**Integer Mappings:**", mappings)
            st.dataframe(enc_df[["medal", "medal_encoded", "event_gender", "event_gender_encoded"]], use_container_width=True)

        with col_e2:
            st.markdown("#### B. One-Hot Encoding (Dummy Variables)")
            ohe_df, new_cols = demonstrate_one_hot_encoding(df.head(10), columns=["medal", "event_gender"])
            st.write(f"**Generated {len(new_cols)} binary columns:** {new_cols}")
            st.dataframe(ohe_df[new_cols], use_container_width=True)

    # 2. Feature Scaling Tab
    with tab_scale:
        st.subheader("2. Min-Max Normalization vs Standardization (Z-Score)")
        st.markdown(
            "- **Min-Max Normalization:** Rescales values linearly into $[0, 1]$: $X' = \\frac{X - X_{min}}{X_{max} - X_{min}}$.\n"
            "- **Z-Score Standardization:** Rescales distribution to mean = 0, standard deviation = 1: $Z = \\frac{X - \\mu}{\\sigma}$."
        )

        country_totals = df.groupby("country")["medal_points"].sum().reset_index()
        norm_df, std_df = demonstrate_feature_scaling(country_totals, features=["medal_points"])

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Min-Max Normalization $[0, 1]$")
            st.dataframe(norm_df.head(10), use_container_width=True)
        with col_s2:
            st.markdown("#### Z-Score Standardization $(\\mu=0, \\sigma=1)$")
            st.dataframe(std_df.head(10), use_container_width=True)

    # 3. PCA Tab
    with tab_pca:
        st.subheader("3. Principal Component Analysis (PCA)")
        st.markdown(
            "PCA projects correlated high-dimensional features onto lower orthogonal dimensions (principal components) "
            "that maximize explained variance."
        )

        country_perf = df.groupby("country").agg(
            gold=("medal", lambda s: (s == "Gold").sum()),
            silver=("medal", lambda s: (s == "Silver").sum()),
            bronze=("medal", lambda s: (s == "Bronze").sum()),
            total_medals=("medal_fact_id", "count"),
            medal_points=("medal_points", "sum")
        ).reset_index()

        pca_df, explained_var, components = demonstrate_pca_reduction(
            country_perf,
            features=["gold", "silver", "bronze", "total_medals", "medal_points"],
            n_components=2
        )

        pca_df["country"] = country_perf["country"]

        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.write(f"**PC1 Explained Variance:** {explained_var[0] * 100:.2f}%")
            st.write(f"**PC2 Explained Variance:** {explained_var[1] * 100:.2f}%")
            st.write(f"**Cumulative Variance Retained:** {sum(explained_var) * 100:.2f}%")
            st.dataframe(pca_df.head(10), use_container_width=True)

        with col_p2:
            var_df = pd.DataFrame({
                "Component": ["PC1", "PC2"],
                "Explained_Variance": explained_var
            })
            fig_pca = create_bar_chart(
                var_df,
                x="Component",
                y="Explained_Variance",
                title="PCA Scree Plot (Explained Variance Ratio)"
            )
            display_chart(fig_pca, key="pca_scree_chart")
