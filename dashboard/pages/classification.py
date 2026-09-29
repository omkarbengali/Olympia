"""
Classification Page for OLYMPIA.
Implements and compares Decision Tree Classification and Naive Bayes Classification
for predicting Olympic Medal outcome (Gold, Silver, Bronze).
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from dashboard.components.charts import display_chart
from analytics.decision_tree import (
    train_decision_tree,
    plot_confusion_matrix,
    plot_feature_importance
)
from analytics.naive_bayes import train_naive_bayes, compare_classifiers


def render_page(df: pd.DataFrame):
    st.title("🌲 Classification: Decision Tree vs Naive Bayes")
    st.caption("Supervised multi-class classification predicting medal podium category (Gold, Silver, Bronze) from event contexts.")

    render_viva_note(
        "Classification & Evaluation Metrics",
        "Decision Trees recursively partition feature space into pure subsets using Gini Impurity or Information Gain (Entropy). Naive Bayes applies Bayes' Theorem with the assumption of conditional feature independence.",
        "Methodological Honesty: With ~33% Gold, ~33% Silver, and ~34% Bronze in any podium event, random guessing yields 33.3% accuracy. Olympic medals are determined by fractions of a second or point, so contextual features alone produce realistic ~36-39% accuracy. We strictly exclude medal_points to prevent target leakage."
    )

    st.markdown("---")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        max_depth = st.slider("Decision Tree Max Depth", min_value=2, max_value=12, value=5)
    with col_ctrl2:
        criterion = st.selectbox("Split Criterion", ["gini", "entropy"], index=0)
    with col_ctrl3:
        test_size = st.slider("Test Set Split Ratio", min_value=0.15, max_value=0.40, value=0.25, step=0.05)

    with st.spinner("Training Classifiers..."):
        dt_res = train_decision_tree(df=df, max_depth=max_depth, criterion=criterion, test_size=test_size)
        nb_res = train_naive_bayes(df=df, test_size=test_size)

    # Comparison Table
    st.subheader("⚖️ Algorithm Performance Comparison")
    comp_df = compare_classifiers(dt_res, nb_res)
    st.table(comp_df)

    st.markdown("---")

    col_dt, col_nb = st.columns(2)

    with col_dt:
        st.markdown("### 🌲 Decision Tree Classifier")
        render_kpi_row([
            {"title": "Accuracy", "value": f"{dt_res['accuracy']*100:.2f}%", "icon": "🎯"},
            {"title": "Precision", "value": f"{dt_res['precision']*100:.2f}%", "icon": "🔍"},
            {"title": "F1 Score", "value": f"{dt_res['f1_score']*100:.2f}%", "icon": "⚖️"},
        ])
        fig_dt_cm = plot_confusion_matrix(dt_res["confusion_matrix"], dt_res["classes"], title="Decision Tree Confusion Matrix")
        display_chart(fig_dt_cm, key="cm_dt")

        fig_fi = plot_feature_importance(dt_res["feature_importances"])
        display_chart(fig_fi, key="fi_dt")

    with col_nb:
        st.markdown("### 🎲 Naive Bayes Classifier (Categorical)")
        render_kpi_row([
            {"title": "Accuracy", "value": f"{nb_res['accuracy']*100:.2f}%", "icon": "🎯"},
            {"title": "Precision", "value": f"{nb_res['precision']*100:.2f}%", "icon": "🔍"},
            {"title": "F1 Score", "value": f"{nb_res['f1_score']*100:.2f}%", "icon": "⚖️"},
        ])
        fig_nb_cm = plot_confusion_matrix(nb_res["confusion_matrix"], nb_res["classes"], title="Naive Bayes Confusion Matrix")
        display_chart(fig_nb_cm, key="cm_nb")

        st.markdown("#### 💡 Algorithm Behavior Analysis")
        st.write(
            "- **Decision Tree:** Builds explicit logical rule splits (e.g. `country_encoded <= 42` and `sport <= 18`). "
            "It captures non-linear interactions but can overfit at high depths.\n"
            "- **Naive Bayes:** Computes prior probabilities $P(C)$ and likelihoods $P(X_i|C)$. "
            "It is computationally fast and immune to feature scale, but assumes feature conditional independence."
        )
