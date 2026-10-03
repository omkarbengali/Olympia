"""
Clustering Analysis Page for OLYMPIA.
Implements Unsupervised Country Performance Segmentation:
1. K-Means Clustering (with Elbow Curve & Dynamic Tier Descriptions)
2. Agglomerative Hierarchical Clustering (with SciPy Dendrogram)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.kmeans import (
    run_kmeans,
    compute_elbow_curve,
    plot_elbow_curve,
    plot_kmeans_clusters
)
from analytics.hierarchical import (
    run_hierarchical_clustering,
    generate_dendrogram_figure,
    plot_hierarchical_clusters
)


def get_contextual_cluster_labels(summary_df: pd.DataFrame) -> dict:
    """
    Dynamically generates descriptive performance tier names based on actual calculated
    cluster characteristics (e.g. mean medals, gold counts) rather than hardcoded assignments.
    """
    if summary_df.empty:
        return {}

    # Rank clusters by average medal points or total medals
    sort_col = "medal_points_mean" if "medal_points_mean" in summary_df.columns else summary_df.columns[1]
    sorted_clusters = summary_df.sort_values(by=sort_col, ascending=False).index.tolist()

    n = len(sorted_clusters)
    descriptions = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        c_str = f"Cluster {cluster_id}"
        if rank == 0:
            descriptions[c_str] = f"{c_str} (Tier 1: Global Olympic Powerhouses - High Medal & Gold Density)"
        elif rank == n - 1:
            descriptions[c_str] = f"{c_str} (Tier {rank+1}: Emerging Olympic Nations - Occasional Podium Finishes)"
        else:
            descriptions[c_str] = f"{c_str} (Tier {rank+1}: Competitive & Specialized Contenders)"
    return descriptions


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("🔍 Unsupervised Clustering Laboratory")
    st.caption("Segment Olympic nations into multi-dimensional performance tiers using K-Means and Agglomerative Hierarchical Clustering on standardized medal vectors.")

    render_viva_note(
        "Unsupervised Performance Segmentation",
        "Clustering partitions unlabeled nation vectors (Gold, Silver, Bronze, Total, Points) into clusters maximizing intra-cluster similarity and inter-cluster separation.",
        "Crucial Note: Clusters are algorithmic groupings derived strictly from standardized medal profiles, not subjective or political designations."
    )

    st.markdown("---")

    tab_km, tab_hc = st.tabs([
        "🎯 1. K-Means Clustering",
        "🌳 2. Hierarchical Clustering & Dendrogram"
    ])

    # ----------------------------------------------------
    # 1. K-MEANS TAB
    # ----------------------------------------------------
    with tab_km:
        st.subheader("1. K-Means Clustering")
        col_k1, col_k2 = st.columns([1, 2])

        with col_k1:
            k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3, key="km_k_slider")
            show_elbow = st.checkbox("Show Elbow Curve Analysis", value=True, key="km_show_elbow")

        with st.spinner(f"Computing K-Means for K={k}..."):
            km_res = run_kmeans(df=df, n_clusters=k)

        with col_k2:
            st.markdown(f"**Inertia (Within-Cluster Sum of Squares):** `{km_res['inertia']:.2f}`")
            sizes_text = ", ".join([f"**{c}:** {cnt} countries" for c, cnt in km_res["cluster_sizes"].items()])
            st.markdown(f"**Cluster Distribution:** {sizes_text}")

        # Dynamic contextual tier titles
        tier_names = get_contextual_cluster_labels(km_res["summary"])

        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_km = plot_kmeans_clusters(km_res["result_df"])
            display_chart(fig_km, key="km_scatter_chart")

        with col_c2:
            if show_elbow:
                elbow_data = compute_elbow_curve(df=df, max_k=6)
                fig_elbow = plot_elbow_curve(elbow_data)
                display_chart(fig_elbow, key="km_elbow_chart")
            else:
                st.markdown("### Cluster Characteristics")
                st.dataframe(km_res["summary"], use_container_width=True)

        st.markdown("### 📋 Nations Grouped by Calculated Performance Cluster")
        for c_label in sorted(km_res["result_df"]["cluster"].unique()):
            desc = tier_names.get(c_label, c_label)
            sub_c = km_res["result_df"][km_res["result_df"]["cluster"] == c_label]
            with st.expander(f"📍 {desc} — ({len(sub_c)} Nations)"):
                st.dataframe(
                    sub_c[["country", "total_medals", "gold_medals", "silver_medals", "bronze_medals", "medal_points"]]
                    .sort_values(by="total_medals", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

    # ----------------------------------------------------
    # 2. HIERARCHICAL TAB
    # ----------------------------------------------------
    with tab_hc:
        st.subheader("2. Agglomerative Hierarchical Clustering")
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            h_k = st.slider("Select Hierarchical Clusters", min_value=2, max_value=6, value=3, key="hc_k_slider")
            linkage_m = st.selectbox("Linkage Method", ["ward", "complete", "average"], index=0, key="hc_linkage")

        with st.spinner("Executing Hierarchical Agglomeration..."):
            hc_res = run_hierarchical_clustering(df=df, n_clusters=h_k, linkage_method=linkage_m)

        with col_h2:
            h_sizes_text = ", ".join([f"**{c}:** {cnt} countries" for c, cnt in hc_res["cluster_sizes"].items()])
            st.markdown(f"**Cluster Distribution:** {h_sizes_text}")

        col_hc1, col_hc2 = st.columns([1, 1])
        with col_hc1:
            fig_hc = plot_hierarchical_clusters(hc_res["result_df"])
            display_chart(fig_hc, key="hc_scatter_chart")

        with col_hc2:
            st.markdown("#### 🌳 Dendrogram Tree (Top 25 Nations)")
            dendro_fig = generate_dendrogram_figure(df=df, top_n=25, linkage_method=linkage_m)
            st.pyplot(dendro_fig)

    # 5. Optional Technical Details
    with st.expander("🛠️ Viva Technical Context: Distance Metrics & Standardization"):
        render_viva_note(
            "Feature Standardization in Distance-Based Learning",
            "Because medal volumes span multiple orders of magnitude (e.g. USA > 2,000 vs emerging nations = 1), features are Z-score standardized prior to computing Euclidean distances: d(p, q) = sqrt(sum((p_i - q_i)^2)).",
            "Without standardization, total_medals would dominate distance calculations, distorting qualitative medal mix insights."
        )


if __name__ == "__main__":
    render_page()
