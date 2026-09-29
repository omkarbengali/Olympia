"""
Clustering Analysis Page for OLYMPIA.
Implements Unsupervised Country Performance Segmentation:
1. K-Means Clustering (with Elbow Method)
2. Agglomerative Hierarchical Clustering (with Dendrogram)
"""

import streamlit as st
import pandas as pd
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


def render_page(df: pd.DataFrame):
    st.title("🔍 Unsupervised Clustering Laboratory")
    st.caption("Group nations into performance tiers using K-Means and Agglomerative Hierarchical Clustering on standardized medal vectors.")

    render_viva_note(
        "Unsupervised Clustering in Data Mining",
        "Clustering partitions unlabeled data points into clusters where intra-cluster similarity is maximized and inter-cluster similarity is minimized.",
        "Crucial Note: Clusters are algorithmic groupings based strictly on standardized medal profiles (Gold, Silver, Bronze, Total, Points) and are not official IOC designations."
    )

    st.markdown("---")

    tab_km, tab_hc = st.tabs([
        "🎯 1. K-Means Clustering",
        "🌳 2. Hierarchical Clustering & Dendrogram"
    ])

    # 1. K-MEANS TAB
    with tab_km:
        st.subheader("1. K-Means Clustering")
        col_k1, col_k2 = st.columns([1, 2])

        with col_k1:
            k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3, key="k_slider")
            show_elbow = st.checkbox("Show Elbow Curve Analysis", value=True)

        km_res = run_kmeans(df=df, n_clusters=k)

        with col_k2:
            st.markdown(f"**Inertia (Within-Cluster Sum of Squares):** `{km_res['inertia']:.2f}`")
            sizes_text = ", ".join([f"**{c}:** {cnt} countries" for c, cnt in km_res["cluster_sizes"].items()])
            st.markdown(f"**Cluster Distribution:** {sizes_text}")

        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_km = plot_kmeans_clusters(km_res["result_df"])
            display_chart(fig_km, key="km_scatter")

        with col_c2:
            if show_elbow:
                elbow_data = compute_elbow_curve(df=df, max_k=6)
                fig_elbow = plot_elbow_curve(elbow_data)
                display_chart(fig_elbow, key="km_elbow")
            else:
                st.markdown("### Cluster Characteristics")
                st.dataframe(km_res["summary"], use_container_width=True)

        st.markdown("### 📋 Nations Grouped by K-Means Cluster")
        for c_label in sorted(km_res["result_df"]["cluster"].unique()):
            with st.expander(f"📍 {c_label} ({km_res['cluster_sizes'][c_label]} Nations)"):
                sub_c = km_res["result_df"][km_res["result_df"]["cluster"] == c_label]
                st.dataframe(
                    sub_c[["country", "total_medals", "gold_medals", "silver_medals", "bronze_medals", "medal_points"]]
                    .sort_values(by="total_medals", ascending=False),
                    use_container_width=True
                )

    # 2. HIERARCHICAL TAB
    with tab_hc:
        st.subheader("2. Agglomerative Hierarchical Clustering")
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            h_k = st.slider("Select Hierarchical Clusters", min_value=2, max_value=6, value=3, key="hk_slider")
            linkage_m = st.selectbox("Linkage Method", ["ward", "complete", "average"], index=0)

        hc_res = run_hierarchical_clustering(df=df, n_clusters=h_k, linkage_method=linkage_m)

        with col_h2:
            h_sizes_text = ", ".join([f"**{c}:** {cnt} countries" for c, cnt in hc_res["cluster_sizes"].items()])
            st.markdown(f"**Cluster Distribution:** {h_sizes_text}")

        col_hc1, col_hc2 = st.columns([1, 1])
        with col_hc1:
            fig_hc = plot_hierarchical_clusters(hc_res["result_df"])
            display_chart(fig_hc, key="hc_scatter")

        with col_hc2:
            st.markdown("#### 🌳 Dendrogram Tree (Top 25 Nations)")
            dendro_fig = generate_dendrogram_figure(df=df, top_n=25, linkage_method=linkage_m)
            st.pyplot(dendro_fig)
