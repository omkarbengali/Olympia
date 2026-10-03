"""
Data Preprocessing & Data Quality Laboratory Page for OLYMPIA.
Demonstrates:
1. Data Quality Assessment (Raw vs Cleaned Audit, Duplicates, Missingness, Types).
2. Categorical Encoding (Label Encoding & One-Hot Encoding).
3. Feature Scaling (Min-Max Normalization & Z-score Standardization).
4. Dimensionality Reduction (Principal Component Analysis - PCA).
5. Before / After Preprocessing Comparisons.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.preprocessing import (
    get_data_quality_report,
    demonstrate_label_encoding,
    demonstrate_one_hot_encoding,
    demonstrate_feature_scaling,
    demonstrate_pca_reduction
)
from analytics.visualization import create_bar_chart, create_line_chart, create_scatter_plot


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("🧹 Data Preprocessing & Quality Laboratory")
    st.caption("Inspect data quality audits, missing value treatment, feature encoding, scaling, and PCA dimensionality reduction.")

    report = get_data_quality_report()

    # 2. Key Metrics Row
    render_kpi_row([
        {"title": "Raw Ingested Rows", "value": f"{report['raw_rows']:,}", "subtitle": "Original CSV source", "icon": "📄"},
        {"title": "Cleaned Fact Rows", "value": f"{report['cleaned_rows']:,}", "subtitle": "Loaded into warehouse", "icon": "✅"},
        {"title": "Exact Duplicates Dropped", "value": f"{report['raw_duplicates']}", "subtitle": "Removed during ETL", "icon": "✂️"},
        {"title": "Missing Values in Warehouse", "value": f"{report['cleaned_missing_total']}", "subtitle": "100% complete facts", "icon": "🛡️"},
    ])

    st.markdown("---")

    # 3. Before vs After Comparison & Missingness Audit
    st.subheader("🔍 Before vs After Preprocessing Audit")
    c_bfa1, c_bfa2 = st.columns(2)

    with c_bfa1:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem 1.25rem;">
            <div style="font-weight: 700; color: #f87171; margin-bottom: 0.5rem;">❌ Raw Dataset (Before Preprocessing)</div>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                <li><strong>20,247 records</strong> with 10 raw textual columns</li>
                <li><strong>3 exact duplicate rows</strong> detected and flagged</li>
                <li><strong>4 voided / non-awarded</strong> historical event records</li>
                <li><strong>423 missing athlete names</strong> in multi-athlete team events</li>
                <li>Compound strings (e.g. "1896 Athens") requiring normalization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_bfa2:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem 1.25rem;">
            <div style="font-weight: 700; color: #34d399; margin-bottom: 0.5rem;">✅ Cleaned Star Schema (After Preprocessing)</div>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                <li><strong>20,240 verified facts</strong> across 5 normalized dimension tables</li>
                <li><strong>Zero duplicate records</strong> (primary key integrity verified)</li>
                <li><strong>Zero orphan foreign keys</strong> across all dimensions</li>
                <li><strong>Missing athlete names imputed</strong> with 'Team / Not Listed'</li>
                <li>Standardized numerical columns and temporal year/season split</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Missing Values & Data Types Audit (Raw Dataset)")
    st.dataframe(report["missing_summary_df"], use_container_width=True)

    st.markdown("---")

    # 4. Interactive Preprocessing Experiments
    st.subheader("🧪 Interactive Preprocessing Experiments")
    tab_enc, tab_scale, tab_pca = st.tabs([
        "🏷️ 1. Categorical Encoding",
        "📏 2. Feature Scaling",
        "📉 3. PCA Dimensionality Reduction"
    ])

    # TAB 1: CATEGORICAL ENCODING
    with tab_enc:
        st.subheader("1. Categorical Encoding (Label vs One-Hot)")
        st.markdown("""
        > **What is Categorical Encoding?**
        > Machine learning models calculate mathematical equations and cannot process text directly.
        > **Label Encoding** converts categories into integer IDs (0, 1, 2...).
        > **One-Hot Encoding** creates binary columns (1 or 0) for each unique category.
        """)

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

    # TAB 2: FEATURE SCALING
    with tab_scale:
        st.subheader("2. Feature Scaling (Min-Max Normalization vs Standardization)")
        st.markdown("""
        > **What is Feature Scaling?**
        > Features with large numeric scales (e.g. 1000s of medals) can bias machine learning models over smaller features.
        > **Min-Max Normalization** squashes values into a [0, 1] range.
        > **Z-Score Standardization** centers values so mean = 0 and standard deviation = 1.
        """)

        country_totals = df.groupby("country")["medal_points"].sum().reset_index()
        norm_df, std_df = demonstrate_feature_scaling(country_totals, features=["medal_points"])

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Min-Max Normalization `[0, 1]`")
            st.dataframe(norm_df.head(10), use_container_width=True)
        with col_s2:
            st.markdown("#### Z-Score Standardization `(μ=0, σ=1)`")
            st.dataframe(std_df.head(10), use_container_width=True)

    # TAB 3: PCA DIMENSIONALITY REDUCTION
    with tab_pca:
        st.subheader("3. Principal Component Analysis (PCA)")
        st.markdown("""
        > **What is PCA?**
        > PCA reduces high-dimensional data (e.g. 5+ correlated metrics like Gold, Silver, Bronze, Points, Medals) into **2 principal directions (PC1 & PC2)** while retaining maximum variance.
        """)

        try:
            pca_res = demonstrate_pca_reduction(df=df, n_components=2)
            col_p1, col_p2 = st.columns([1, 1])

            with col_p1:
                st.markdown(f"**Variance Explained:** PC1 = `{pca_res['explained_variance_ratio'][0]*100:.1f}%`, PC2 = `{pca_res['explained_variance_ratio'][1]*100:.1f}%` (Total = `{pca_res['total_variance_explained']*100:.1f}%`)")
                fig_pca = create_scatter_plot(
                    pca_res["pca_df"].head(50),
                    x="PC1",
                    y="PC2",
                    title="PCA 2D Projection of Nations (Top 50 by Medal Volume)"
                )
                display_chart(fig_pca, key="pca_scatter_chart")

            with col_p2:
                st.markdown("#### Principal Component Data (PC1 & PC2)")
                st.dataframe(pca_res["pca_df"].head(15), use_container_width=True)
        except Exception as e:
            st.info(f"PCA demonstration notice: {e}")

    # 5. Optional Technical Details
    with st.expander("🛠️ Viva Technical Context: Preprocessing Protocols"):
        render_viva_note(
            "ETL Pipeline vs Downstream ML Transformations",
            "In data warehousing, physical tables store human-readable strings to support business intelligence and ad-hoc SQL reporting. Mathematical transformations (scaling, one-hot encoding, PCA) are executed dynamically in memory for downstream ML pipelines.",
            "This separation guarantees semantic transparency while satisfying algorithmic input constraints."
        )


if __name__ == "__main__":
    render_page()
