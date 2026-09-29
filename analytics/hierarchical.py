"""
Hierarchical Clustering Module for OLYMPIA.
Implements Agglomerative Hierarchical Clustering on Olympic country profiles.
Generates dendrogram tree visualizations and cluster assignments using Scipy and Scikit-learn.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from analytics.kmeans import prepare_clustering_dataset, CLUSTERING_FEATURES
from analytics.visualization import apply_dark_theme, OLYMPIC_COLORS


def run_hierarchical_clustering(
    df: pd.DataFrame = None,
    n_clusters: int = 3,
    linkage_method: str = "ward"
) -> Dict[str, Any]:
    """
    Performs Agglomerative Hierarchical Clustering on standardized country performance metrics.
    """
    country_df, X_scaled, _ = prepare_clustering_dataset(df)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )
    labels = model.fit_predict(X_scaled)

    result_df = country_df.copy()
    result_df["hierarchical_cluster"] = [f"Cluster {i+1}" for i in labels]

    return {
        "model": model,
        "n_clusters": n_clusters,
        "linkage_method": linkage_method,
        "result_df": result_df,
        "cluster_sizes": result_df["hierarchical_cluster"].value_counts().to_dict(),
        "X_scaled": X_scaled,
        "countries": country_df["country"].tolist()
    }


def generate_dendrogram_figure(
    df: pd.DataFrame = None,
    top_n: int = 30,
    linkage_method: str = "ward"
) -> plt.Figure:
    """
    Generates a Matplotlib dendrogram figure for top N countries by medal points.
    Using dark analytics color aesthetics.
    """
    country_df, X_scaled, _ = prepare_clustering_dataset(df)

    # Focus dendrogram on top N countries for clean visualization readability
    top_indices = country_df.nlargest(top_n, "medal_points").index
    X_subset = X_scaled[top_indices]
    labels_subset = country_df.loc[top_indices, "country"].tolist()

    # Compute linkage matrix
    Z = linkage(X_subset, method=linkage_method)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#1e293b")
    ax.set_facecolor("#0f172a")

    dendrogram(
        Z,
        labels=labels_subset,
        leaf_rotation=90,
        leaf_font_size=9,
        ax=ax,
        color_threshold=0.7 * max(Z[:, 2]),
        above_threshold_color="#94a3b8"
    )

    ax.set_title(
        f"Hierarchical Clustering Dendrogram (Top {top_n} Nations, Linkage='{linkage_method}')",
        fontsize=13,
        color="#f8fafc",
        fontweight="bold",
        pad=15
    )
    ax.set_ylabel("Distance (Dissimilarity)", fontsize=11, color="#94a3b8")
    ax.tick_params(axis="x", colors="#f8fafc")
    ax.tick_params(axis="y", colors="#94a3b8")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    return fig


def plot_hierarchical_clusters(
    result_df: pd.DataFrame,
    x_col: str = "total_medals",
    y_col: str = "medal_points"
) -> go.Figure:
    """Generates interactive 2D scatter plot of hierarchical clusters."""
    fig = px.scatter(
        result_df,
        x=x_col,
        y=y_col,
        color="hierarchical_cluster",
        hover_name="country",
        hover_data=["gold_medals", "silver_medals", "bronze_medals", "total_medals"],
        size="total_medals",
        size_max=35,
        color_discrete_sequence=OLYMPIC_COLORS,
        template="plotly_dark"
    )
    return apply_dark_theme(
        fig,
        title=f"Agglomerative Clustering: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
        x_title=x_col.replace("_", " ").title(),
        y_title=y_col.replace("_", " ").title()
    )


if __name__ == "__main__":
    print("Testing Hierarchical Clustering Module...")
    res = run_hierarchical_clustering(n_clusters=3)
    print("Cluster sizes:", res["cluster_sizes"])
    fig = generate_dendrogram_figure(top_n=15)
    print("Dendrogram successfully created!")
