"""
Linear Regression Page for OLYMPIA.
Supervised linear regression predicting national Olympic medal counts strictly from historical (t-1) performance.
Adheres strictly to chronological splitting to guarantee zero data leakage.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import numpy as np
from warehouse.warehouse import get_denormalized_medals
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.regression import (
    train_linear_regression,
    plot_regression_actual_vs_predicted,
    prepare_regression_dataset
)


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("📈 Linear Regression Olympic Medal Prediction")
    st.caption("Supervised machine learning model predicting national Olympic medal counts from historical (t-1) performance.")

    # Ethical Methodology Disclaimer Banner
    st.warning("⚠️ **Historical-data-based model estimate:** This model estimates medal potential based strictly on past Olympic edition performance. It is an analytical benchmark, not an authoritative guarantee of actual athletic outcomes.")

    # 2. Controls & Model Configuration
    col_ctrl1, col_ctrl2 = st.columns([1, 1])

    with col_ctrl1:
        split_year = st.slider(
            "Select Chronological Train/Test Split Cutoff Year",
            min_value=1996,
            max_value=2020,
            value=2012,
            step=4,
            help="Games up to this year train the model; games after this year serve as strictly unseen test data."
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
        st.info("Please select at least one predictor feature to train the model.")
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

    # 3. Key Metrics Row
    render_kpi_row([
        {"title": "R² Score (Goodness of Fit)", "value": f"{reg_res['r2']:.3f}", "subtitle": "Variance explained", "icon": "🎯"},
        {"title": "Mean Absolute Error (MAE)", "value": f"{reg_res['mae']:.2f}", "subtitle": "Avg medal error", "icon": "📏"},
        {"title": "Root Mean Squared Error (RMSE)", "value": f"{reg_res['rmse']:.2f}", "subtitle": "Penalizes large errors", "icon": "📐"},
        {"title": "Training Period", "value": f"≤ {split_year}", "subtitle": f"{reg_res['train_samples']} samples", "icon": "🏋️"},
        {"title": "Testing Period", "value": f"> {split_year}", "subtitle": f"{reg_res['test_samples']} samples", "icon": "🧪"},
    ])

    st.markdown("---")

    # 4. Main Visualization: Actual vs Predicted Scatter
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📊 Actual vs. Predicted Medals (Test Set)")
        fig_scatter = plot_regression_actual_vs_predicted(reg_res)
        display_chart(fig_scatter, key="reg_actual_vs_pred_chart")
        st.caption("Points near the dashed 45° diagonal line indicate accurate predictions. Points above the line represent nations exceeding historical expectations.")

    with col2:
        st.subheader("⚖️ Learned Feature Weights")
        st.markdown(f"**Intercept (β₀):** `{reg_res['intercept']:.3f}`")
        coef_df = pd.DataFrame({
            "Feature": list(reg_res["coefficients"].keys()),
            "Learned Coefficient (Weight)": [f"{v:.4f}" for v in reg_res["coefficients"].values()]
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.subheader("🔮 Interactive Live Medal Estimator")
        st.caption("Test the trained model by inputting prior Olympic results:")
        test_prev_medals = st.number_input("Previous Edition Total Medals", min_value=0, max_value=150, value=7, key="live_prev_medals")
        test_prev_golds = st.number_input("Previous Edition Gold Medals", min_value=0, max_value=50, value=1, key="live_prev_golds")

        # Compute simple estimation using learned weights
        est_val = reg_res["intercept"]
        if "prev_total_medals" in reg_res["coefficients"]:
            est_val += reg_res["coefficients"]["prev_total_medals"] * test_prev_medals
        if "prev_gold_medals" in reg_res["coefficients"]:
            est_val += reg_res["coefficients"]["prev_gold_medals"] * test_prev_golds
        est_val = max(0.0, round(float(est_val), 1))

        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; border-radius: 8px; padding: 0.8rem; text-align: center; margin-top: 0.5rem;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Estimated Next Edition Medals</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #38bdf8;">~ {est_val} medals</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Optional Technical Details
    with st.expander("🛠️ Methodology & Zero Data Leakage Viva Note"):
        render_viva_note(
            "Zero Data Leakage Chronological Splitting",
            "In temporal forecasting, random k-fold cross-validation causes fatal data leakage by training on future editions to predict past editions. OLYMPIA enforces strict chronological splitting: Games ≤ Split Year train the regression model, and Games > Split Year evaluate out-of-sample accuracy.",
            "Linear regression assumes linear relationships and homoscedasticity. Outliers (e.g. nation boycotts or host nation surges) present real-world variance accounted for by RMSE."
        )


if __name__ == "__main__":
    render_page()
