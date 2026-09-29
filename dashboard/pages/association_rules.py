"""
Association Rule Mining Page for OLYMPIA.
Implements the Apriori algorithm on Olympic multi-sport market baskets:
  Transaction: Country + Olympic Games Edition
  Items: Distinct sports won.
"""

import streamlit as st
import pandas as pd
from dashboard.components.metrics import render_kpi_row, render_viva_note
from analytics.apriori import mine_association_rules


def render_page(df: pd.DataFrame):
    st.title("🔗 Association Rule Mining: Apriori Algorithm")
    st.caption("Discover co-occurrence patterns between Olympic sporting disciplines won within the same Games edition.")

    render_viva_note(
        "Association Rule Mining (Market Basket Analysis)",
        "Apriori identifies frequent itemsets using the Apriori property: 'All non-empty subsets of a frequent itemset must also be frequent.' It extracts rules: If Antecedent {A} then Consequent {B}.",
        "Key Metrics & No Causation: Support measures rule frequency P(A ∩ B); Confidence measures conditional probability P(B|A); Lift measures how much more often A and B occur together than expected by chance (P(B|A) / P(B)). Lift > 1 indicates positive affinity. An association rule indicates co-occurrence, NOT causation."
    )

    st.markdown("---")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        min_supp = st.slider(
            "Minimum Support Threshold",
            min_value=0.02,
            max_value=0.20,
            value=0.06,
            step=0.01,
            help="Minimum fraction of country-games appearances that must contain the sports."
        )
    with col_ctrl2:
        min_conf = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.20,
            max_value=0.90,
            value=0.40,
            step=0.05,
            help="Minimum conditional probability that B was won given that A was won."
        )

    with st.spinner("Mining frequent sport itemsets and association rules..."):
        apriori_res = mine_association_rules(
            df=df,
            min_support=min_supp,
            min_confidence=min_conf
        )

    render_kpi_row([
        {"title": "Country-Games Transactions", "value": f"{apriori_res['total_transactions']:,}", "icon": "🧺"},
        {"title": "Frequent Sport Itemsets", "value": f"{len(apriori_res['frequent_itemsets']):,}", "icon": "📦"},
        {"title": "Discovered Association Rules", "value": f"{len(apriori_res['rules']):,}", "icon": "🔗"},
    ])

    if apriori_res["status"] != "SUCCESS":
        st.warning(apriori_res["message"])
        return

    st.markdown("---")

    col_tab1, col_tab2 = st.tabs(["📜 Discovered Association Rules", "📦 Frequent Sport Itemsets"])

    with col_tab1:
        st.subheader("Discovered Multi-Sport Affinity Rules")
        rules_df = apriori_res["rules"]

        display_cols = ["rule_representation", "support", "confidence", "lift", "leverage"]
        st.dataframe(
            rules_df[display_cols],
            use_container_width=True,
            height=400
        )

        st.markdown("#### 💡 Rule Interpretation Example")
        top_rule = rules_df.iloc[0]
        st.info(
            f"**Top Rule:** `{top_rule['rule_representation']}`\n\n"
            f"- **Support = {top_rule['support'] * 100:.1f}%:** In {top_rule['support'] * 100:.1f}% of all historical country appearances, medals were won in both disciplines simultaneously.\n"
            f"- **Confidence = {top_rule['confidence'] * 100:.1f}%:** When a nation medaled in `{top_rule['antecedents_str']}`, there was a {top_rule['confidence'] * 100:.1f}% probability they also medaled in `{top_rule['consequents_str']}`.\n"
            f"- **Lift = {top_rule['lift']:.2f}:** Winning `{top_rule['antecedents_str']}` increases the likelihood of also winning `{top_rule['consequents_str']}` by **{top_rule['lift']:.2f}x** over independent chance."
        )

    with col_tab2:
        st.subheader("Frequent Sport Itemsets")
        itemsets_df = apriori_res["frequent_itemsets"]
        st.dataframe(
            itemsets_df[["length", "itemsets_display", "support"]],
            use_container_width=True
        )
