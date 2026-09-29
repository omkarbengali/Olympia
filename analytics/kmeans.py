"""
K-Means Clustering Module for OLYMPIA.
Segments countries based on their historical Olympic medal profiles:
Gold, Silver, Bronze, Total Medals, and Medal Points.
Includes Elbow method analysis and interactive cluster visualizations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from warehouse.warehouse import get_denormalized_medals
from analytics.visualization import apply_dark_theme, OLYMPIC_COLORS


CLUSTERING_FEATURES = [
    "gold_medals",
    "silver_medals",
    "bronze_medals",
    "total_medals",
    "medal_points"
]


def prepare_clustering_dataset(df: pd.DataFrame = None) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Aggregates medals at country level and scales numerical features.
    """
    if df is None:
        df = get_denormalized_medals()

    country_df = df.groupby(["country", "country_code"]).agg(
        gold_medals=("medal", lambda s: (s == "Gold").sum()),
        silver_medals=("medal", lambda s: (s == "Silver").sum()),
        bronze_medals=("medal", lambda s: (s == "Bronze").sum()),
        total_medals=("medal_fact_id", "count"),
        medal_points=("medal_points", "sum")
    ).reset_index()

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(country_df[CLUSTERING_FEATURES])

    return country_df, scaled_matrix, scaler


def run_kmeans(
    df: pd.DataFrame = None,
    n_clusters: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes K-Means algorithm on scaled Olympic country performance profiles.
    """
    country_df, X_scaled, scaler = prepare_clustering_dataset(df)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=random_state
    )
    labels = kmeans.fit_predict(X_scaled)

    result_df = country_df.copy()
    result_df["cluster"] = [f"Cluster {i+1}" for i in labels]

    # Cluster summary
    summary = result_df.groupby("cluster")[CLUSTERING_FEATURES].agg(["count", "mean", "min", "max"])

    return {
        "model": kmeans,
        "n_clusters": n_clusters,
        "inertia": float(kmeans.inertia_),
        "result_df": result_df,
        "summary": summary,
        "cluster_sizes": result_df["cluster"].value_counts().to_dict()
    }


def compute_elbow_curve(
    df: pd.DataFrame = None,
    max_k: int = 8,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Computes within-cluster sum of squares (inertia) for K=1 to max_k to aid optimal cluster selection.
    """
    _, X_scaled, _ = prepare_clustering_dataset(df)
    elbow_data = []

    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km.fit(X_scaled)
        elbow_data.append({"k": k, "inertia": round(float(km.inertia_), 2)})

    return elbow_data


def plot_elbow_curve(elbow_data: List[Dict[str, Any]]) -> go.Figure:
    """Generates an interactive line chart of the Elbow curve."""
    df = pd.DataFrame(elbow_data)
    fig = px.line(
        df,
        x="k",
        y="inertia",
        markers=True,
        template="plotly_dark",
        color_discrete_sequence=["#38bdf8"]
    )
    fig.update_traces(line={"width": 3}, marker={"size": 9, "color": "#fbbf24"})
    return apply_dark_theme(fig, title="K-Means Elbow Method (Inertia vs Number of Clusters)", x_title="Number of Clusters (K)", y_title="Inertia (Sum of Squared Distances)")


def plot_kmeans_clusters(
    result_df: pd.DataFrame,
    x_col: str = "total_medals",
    y_col: str = "medal_points"
) -> go.Figure:
    """Generates interactive 2D scatter plot of clustered nations."""
    fig = px.scatter(
        result_df,
        x=x_col,
        y=y_col,
        color="cluster",
        hover_name="country",
        hover_data=["gold_medals", "silver_medals", "bronze_medals", "total_medals"],
        size="total_medals",
        size_max=35,
        color_discrete_sequence=OLYMPIC_COLORS,
        template="plotly_dark"
    )
    return apply_dark_theme(
        fig,
        title=f"K-Means Country Clusters: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
        x_title=x_col.replace("_", " ").title(),
        y_title=y_col.replace("_", " ").title()
    )


if __name__ == "__main__":
    print("Testing K-Means Module...")
    km_res = run_kmeans(n_clusters=3)
    print("Cluster sizes:", km_res["cluster_sizes"])
    elbow = compute_elbow_curve(max_k=6)
    print("Elbow data:", elbow)
