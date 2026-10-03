"""
Classification Page for OLYMPIA.
Implements and compares Decision Tree Classification and Naive Bayes Classification
for predicting Olympic Medal outcome (Gold, Silver, Bronze) from event contexts.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.decision_tree import (
    train_decision_tree,
    plot_confusion_matrix,
    plot_feature_importance
)
from analytics.naive_bayes import train_naive_bayes, compare_classifiers


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("🌲 Classification: Decision Tree vs Naive Bayes")
    st.caption("Supervised multi-class classification predicting medal podium category (Gold, Silver, Bronze) from event contextual features.")

    # Honest Athletic Disclaimer
    st.warning("⚠️ **Methodological Transparency:** Athletic medals are decided by fractions of a second or single points. Contextual features alone produce realistic ~36–39% baseline accuracy (random guessing = 33.3%). We strictly exclude `medal_points` to prevent target leakage.")

    # 2. Controls & Model Hyperparameters
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        max_depth = st.slider("Decision Tree Max Depth", min_value=2, max_value=12, value=5, key="dt_max_depth")
    with col_ctrl2:
        criterion = st.selectbox("Decision Tree Split Criterion", ["gini", "entropy"], index=0, key="dt_criterion")
    with col_ctrl3:
        test_size = st.slider("Test Set Split Ratio", min_value=0.15, max_value=0.40, value=0.25, step=0.05, key="dt_test_size")

    with st.spinner("Training Decision Tree and Naive Bayes Classifiers..."):
        try:
            dt_res = train_decision_tree(df=df, max_depth=max_depth, criterion=criterion, test_size=test_size)
            nb_res = train_naive_bayes(df=df, test_size=test_size)
        except Exception as e:
            st.error(f"Error training classifiers: {e}")
            return

    # 3. Model Comparison Table & Metrics
    st.subheader("⚖️ Algorithm Performance Comparison")
    comp_df = compare_classifiers(dt_res, nb_res)
    st.table(comp_df)

    st.markdown("---")

    # 4. Side-by-Side Algorithm Deep Dive
    col_dt, col_nb = st.columns(2)

    with col_dt:
        st.subheader("🌲 Decision Tree Classifier")
        render_kpi_row([
            {"title": "Accuracy", "value": f"{dt_res['accuracy']*100:.2f}%", "icon": "🎯"},
            {"title": "Precision", "value": f"{dt_res['precision']*100:.2f}%", "icon": "🔍"},
            {"title": "F1 Score", "value": f"{dt_res['f1_score']*100:.2f}%", "icon": "⚖️"},
        ])
        fig_dt_cm = plot_confusion_matrix(dt_res["confusion_matrix"], dt_res["classes"], title="Decision Tree Confusion Matrix")
        display_chart(fig_dt_cm, key="cm_dt_chart")

        fig_fi = plot_feature_importance(dt_res["feature_importances"])
        display_chart(fig_fi, key="fi_dt_chart")

    with col_nb:
        st.subheader("🎲 Naive Bayes Classifier")
        render_kpi_row([
            {"title": "Accuracy", "value": f"{nb_res['accuracy']*100:.2f}%", "icon": "🎯"},
            {"title": "Precision", "value": f"{nb_res['precision']*100:.2f}%", "icon": "🔍"},
            {"title": "F1 Score", "value": f"{nb_res['f1_score']*100:.2f}%", "icon": "⚖️"},
        ])
        fig_nb_cm = plot_confusion_matrix(nb_res["confusion_matrix"], nb_res["classes"], title="Naive Bayes Confusion Matrix")
        display_chart(fig_nb_cm, key="cm_nb_chart")

        st.markdown("""
        **Algorithm Mechanics:**
        - **Decision Tree:** Partitions the feature space into hierarchical rule splits (e.g. `country_encoded <= 42`). Captures non-linear thresholds but prone to overfitting at high tree depths.
        - **Naive Bayes:** Applies Bayes' Theorem computing conditional likelihoods $P(X_i|C)$. Assumes feature independence; fast, stable, and immune to feature scale.
        """)

    # 5. Optional Technical Details
    with st.expander("🛠️ Viva Technical Context: Classification Theory"):
        render_viva_note(
            "Classification & Target Leakage Prevention",
            "Target variable is multi-class categorical: Gold, Silver, Bronze. The dataset maintains an equal distribution (~33% per class) since every podium event awards all three medals.",
            "Any classifier claiming >80% accuracy on this task has almost certainly leaked medal_points (3, 2, 1) into training features. OLYMPIA enforces strict feature hygiene."
        )


if __name__ == "__main__":
    render_page()
