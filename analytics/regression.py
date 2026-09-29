"""
Linear Regression Module for OLYMPIA.
Predicts country medal totals for an Olympic edition based strictly on historical
lag features, avoiding data leakage and utilizing a chronological train/test split.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from warehouse.warehouse import get_denormalized_medals
from analytics.visualization import apply_dark_theme, THEME_BG, PAPER_BG, TEXT_COLOR


def prepare_regression_dataset(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Constructs the Country-Year Olympic performance panel dataset with lagged features.
    Strictly uses historical (t-1) performance to predict edition (t) medals.
    """
    if df is None:
        df = get_denormalized_medals()

    # Aggregate by country, year, season
    panel = df.groupby(["country", "year", "season"]).agg(
        total_medals=("medal_fact_id", "count"),
        gold_medals=("medal", lambda s: (s == "Gold").sum()),
        silver_medals=("medal", lambda s: (s == "Silver").sum()),
        bronze_medals=("medal", lambda s: (s == "Bronze").sum()),
        medal_points=("medal_points", "sum")
    ).reset_index()

    panel = panel.sort_values(by=["country", "season", "year"]).reset_index(drop=True)

    # Compute lagged features per country & season
    panel["prev_total_medals"] = panel.groupby(["country", "season"])["total_medals"].shift(1)
    panel["prev_gold_medals"] = panel.groupby(["country", "season"])["gold_medals"].shift(1)
    panel["prev_silver_medals"] = panel.groupby(["country", "season"])["silver_medals"].shift(1)
    panel["prev_bronze_medals"] = panel.groupby(["country", "season"])["bronze_medals"].shift(1)
    panel["prev_medal_points"] = panel.groupby(["country", "season"])["medal_points"].shift(1)

    # Drop the first appearance per country where no prior historical data exists
    reg_df = panel.dropna(subset=["prev_total_medals"]).copy().reset_index(drop=True)
    return reg_df


def train_linear_regression(
    df: pd.DataFrame = None,
    split_year: int = 2012,
    features: list = None
) -> Dict[str, Any]:
    """
    Trains a Linear Regression model using a chronological train/test split.

    Parameters:
        df: Input panel DataFrame. If None, generated automatically.
        split_year: Year cutoff. Train <= split_year, Test > split_year.
        features: Feature column names to include.

    Returns:
        Dictionary containing model, metrics (MAE, RMSE, R2), predictions, and coefficients.
    """
    if df is None:
        df = prepare_regression_dataset()

    if features is None:
        features = [
            "prev_total_medals",
            "prev_gold_medals",
            "prev_silver_medals",
            "prev_bronze_medals",
            "prev_medal_points"
        ]

    target = "total_medals"

    # Chronological Split
    train_mask = df["year"] <= split_year
    test_mask = df["year"] > split_year

    train_df = df[train_mask]
    test_df = df[test_mask]

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Split year {split_year} resulted in empty train ({len(train_df)}) or test ({len(test_df)}) split."
        )

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_clipped = np.clip(y_test_pred, 0, None)  # Medals cannot be negative

    # Evaluate
    mae = float(mean_absolute_error(y_test, y_test_pred_clipped))
    rmse = float(root_mean_squared_error(y_test, y_test_pred_clipped))
    r2 = float(r2_score(y_test, y_test_pred_clipped))

    # Compile results table
    results_df = test_df[["country", "year", "season", "total_medals"]].copy()
    results_df["predicted_medals"] = np.round(y_test_pred_clipped, 1)
    results_df["error"] = np.round(results_df["predicted_medals"] - results_df["total_medals"], 1)

    # Feature Coefficients
    coef_dict = dict(zip(features, [round(float(c), 4) for c in model.coef_]))

    return {
        "model": model,
        "features": features,
        "split_year": split_year,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "intercept": round(float(model.intercept_), 4),
        "coefficients": coef_dict,
        "results_df": results_df,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred_clipped
    }


def plot_regression_actual_vs_predicted(reg_results: Dict[str, Any]) -> go.Figure:
    """Generates an interactive scatter plot of Actual vs Predicted Medals with 45-degree reference line."""
    results_df = reg_results["results_df"]
    max_val = max(results_df["total_medals"].max(), results_df["predicted_medals"].max()) + 5

    fig = px.scatter(
        results_df,
        x="total_medals",
        y="predicted_medals",
        hover_name="country",
        hover_data=["year", "season"],
        color="total_medals",
        color_continuous_scale="Blues",
        template="plotly_dark"
    )

    # Add 45-degree ideal prediction line
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Perfect Prediction (y=x)",
            line={"color": "#ef4444", "dash": "dash", "width": 2}
        )
    )

    return apply_dark_theme(
        fig,
        title=f"Actual vs Predicted Medals (Test Split > {reg_results['split_year']} | R² = {reg_results['r2']})",
        x_title="Actual Total Medals Won",
        y_title="Model Predicted Medals"
    )


if __name__ == "__main__":
    print("Testing Linear Regression Module...")
    res = train_linear_regression()
    print(f"Train Samples: {res['train_samples']} | Test Samples: {res['test_samples']}")
    print(f"MAE: {res['mae']} | RMSE: {res['rmse']} | R²: {res['r2']}")
    print("Coefficients:", res["coefficients"])
    print("\nSample Predictions:")
    print(res["results_df"].head(10))
