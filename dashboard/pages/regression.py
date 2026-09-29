"""
Linear Regression Page for OLYMPIA.
Predicts country medal totals using historical Olympic lag features.
Adheres strictly to chronological splitting to prevent data leakage.
"""

import streamlit as st
import pandas as pd
import numpy as np
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.regression import (
    train_linear_regression,
    plot_regression_actual_vs_predicted,
    prepare_regression_dataset
)


def render_page(df: pd.DataFrame):
    st.title("📈 Linear Regression Olympic Medal Prediction")
    st.caption("Supervised regression predicting national Olympic medal counts strictly from historical (t-1) performance.")

    render_viva_note(
        "Linear Regression in Olympic Analytics",
        "Linear Regression estimates the linear relationship between continuous target variable Y (Medals won at edition t) and explanatory features X (Historical medals won at edition t-1): Y = β₀ + β₁X₁ + ... + βₖXₖ + ε.",
        "Zero Data Leakage: Training is split chronologically (e.g. Train on past games ≤ 2012, Test on unseen future games > 2012). Future medal outcomes are never used to predict historical results."
    )

    st.markdown("---")

    col_ctrl1, col_ctrl2 = st.columns([1, 1])

    with col_ctrl1:
        split_year = st.slider(
            "Select Chronological Train/Test Split Cutoff Year",
            min_value=1996,
            max_value=2020,
            value=2012,
            step=4,
            help="Games up to this year are used for training; games after this year serve as unseen test data."
        )

    with col_ctrl2:
        feature_options = [
            "prev_total_medals",
            "prev_gold_medals",
            "prev_silver_medals",
            "prev_bronze_medals",
            "prev_medal_points"
        ]
        selected_features = st.multiselect(
            "Model Predictor Features",
            options=feature_options,
            default=feature_options
        )

    if not selected_features:
        st.warning("Please select at least one predictor feature.")
        return

    # Train model
    with st.spinner("Training Linear Regression model..."):
        try:
            reg_res = train_linear_regression(
                split_year=split_year,
                features=selected_features
            )
        except Exception as e:
            st.error(f"Error training model: {e}")
            return

    # KPI Evaluation Metrics
    render_kpi_row([
        {"title": "R² Score (Goodness of Fit)", "value": f"{reg_res['r2']:.3f}", "subtitle": "Proportion of variance explained", "icon": "🎯"},
        {"title": "Mean Absolute Error (MAE)", "value": f"{reg_res['mae']:.2f}", "subtitle": "Average medal prediction error", "icon": "📏"},
        {"title": "Root Mean Squared Error (RMSE)", "value": f"{reg_res['rmse']:.2f}", "subtitle": "Penalizes large errors", "icon": "📐"},
        {"title": "Training Instances", "value": f"{reg_res['train_samples']:,}", "subtitle": f"Editions ≤ {split_year}", "icon": "🏋️"},
        {"title": "Testing Instances", "value": f"{reg_res['test_samples']:,}", "subtitle": f"Editions > {split_year}", "icon": "🧪"},
    ])

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        fig_scatter = plot_regression_actual_vs_predicted(reg_res)
        display_chart(fig_scatter, key="reg_actual_vs_pred")

    with col2:
        st.markdown("### ⚖️ Learned Feature Coefficients")
        st.markdown(f"**Intercept (β₀):** `{reg_res['intercept']}`")
        coef_df = pd.DataFrame({
            "Feature": list(reg_res["coefficients"].keys()),
            "Coefficient (Weight)": list(reg_res["coefficients"].values())
        })
        st.dataframe(coef_df, use_container_width=True)

        st.markdown("### 🔮 Interactive Live Predictor")
        st.markdown("Test the trained model by entering prior Olympic performance:")
        input_total = st.number_input("Previous Total Medals", min_value=0, max_value=150, value=7)
        input_gold = st.number_input("Previous Gold Medals", min_value=0, max_value=60, value=1)
        input_silver = st.number_input("Previous Silver Medals", min_value=0, max_value=60, value=2)
        input_bronze = st.number_input("Previous Bronze Medals", min_value=0, max_value=60, value=4)
        input_pts = input_gold * 3 + input_silver * 2 + input_bronze * 1

        input_dict = {
            "prev_total_medals": input_total,
            "prev_gold_medals": input_gold,
            "prev_silver_medals": input_silver,
            "prev_bronze_medals": input_bronze,
            "prev_medal_points": input_pts
        }

        features_vector = [input_dict[f] for f in selected_features]
        predicted_val = max(0.0, float(reg_res["model"].predict([features_vector])[0]))
        st.success(f"**Predicted Next Total Medals:** `{predicted_val:.1f}` medals")

    st.markdown("---")
    with st.expander("📋 Inspect Test Set Prediction Results"):
        st.dataframe(reg_res["results_df"].sort_values(by="total_medals", ascending=False), use_container_width=True)
