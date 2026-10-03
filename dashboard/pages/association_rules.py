"""
Association Rule Mining Page for OLYMPIA.
Implements the Apriori algorithm on Olympic multi-sport market baskets:
  Transaction: Country + Olympic Games Edition
  Items: Distinct sports won.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
from warehouse.warehouse import get_denormalized_medals
from dashboard.components.metrics import render_kpi_row, render_viva_note
from analytics.apriori import mine_association_rules


def render_page(df: pd.DataFrame = None):
    if df is None:
        df = get_denormalized_medals()

    # 1. Page Title & One-Sentence Summary
    st.title("🔗 Association Rule Mining: Apriori Algorithm")
    st.caption("Discover co-occurrence patterns and cross-sport affinities won by nations within the same Olympic Games edition.")

    # Honest Causation Warning
    st.info("💡 **Association vs. Causation:** These rules capture historical co-occurrence across sporting disciplines won by a nation in a single Olympic edition. Winning medals in Sport A does not *cause* medals in Sport B—it reflects multi-discipline national sporting investment.")

    # 2. Controls: Support, Confidence, Lift
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        min_supp = st.slider(
            "Minimum Support Threshold",
            min_value=0.02,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Minimum fraction of country-games appearances that must contain the sports."
        )
    with col_ctrl2:
        min_conf = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.20,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Minimum conditional probability that B was won given that A was won."
        )
    with col_ctrl3:
        min_lift = st.slider(
            "Minimum Lift Filter",
            min_value=1.0,
            max_value=4.0,
            value=1.1,
            step=0.1,
            help="Lift > 1 indicates positive affinity above random chance."
        )

    with st.spinner("Mining frequent sport itemsets and association rules..."):
        try:
            apriori_res = mine_association_rules(
                df=df,
                min_support=min_supp,
                min_confidence=min_conf
            )
        except Exception as e:
            st.error(f"Error executing Apriori algorithm: {e}")
            return

    # Filter rules by lift
    rules_df = apriori_res.get("rules", pd.DataFrame())
    if not rules_df.empty and "lift" in rules_df.columns:
        rules_df = rules_df[rules_df["lift"] >= min_lift]

    # 3. Key Metrics Row
    render_kpi_row([
        {"title": "Country-Games Baskets", "value": f"{apriori_res['total_transactions']:,}", "icon": "🧺"},
        {"title": "Frequent Sport Itemsets", "value": f"{len(apriori_res['frequent_itemsets']):,}", "icon": "📦"},
        {"title": "Discovered Rules (Lift ≥ Threshold)", "value": f"{len(rules_df):,}", "icon": "🔗"},
    ])

    st.markdown("---")

    if apriori_res["status"] != "SUCCESS" or rules_df.empty:
        st.warning("No association rules met the selected support, confidence, and lift thresholds. Try lowering the sliders.")
        return

    # 4. Human-Readable Discovered Rules Highlights
    st.subheader("💡 Key Discovered Sport Associations (Plain-English)")
    top_3_rules = rules_df.head(3)
    cols = st.columns(min(3, len(top_3_rules)))
    for idx, (_, row) in enumerate(top_3_rules.iterrows()):
        ant = row.get("antecedents_str", "Sport A")
        con = row.get("consequents_str", "Sport B")
        supp_pct = row.get("support", 0) * 100
        conf_pct = row.get("confidence", 0) * 100
        lift_val = row.get("lift", 1.0)

        with cols[idx]:
            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; height: 175px;">
                <div style="font-weight: 700; color: #38bdf8; font-size: 1rem; margin-bottom: 0.4rem;">
                    {ant} ➔ {con}
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
                    Nations winning medals in <strong>{ant}</strong> also won medals in <strong>{con}</strong> in <strong>{conf_pct:.1f}%</strong> of Olympic appearances.
                </div>
                <div style="font-size: 0.78rem; color: #fbbf24; margin-top: 0.5rem; font-weight: 600;">
                    Lift: {lift_val:.2f}x expected by random chance
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 5. Full Rules Table
    st.markdown("### 📋 Complete Association Rules Table")
    display_cols = [
        "rule_representation",
        "antecedents_str",
        "consequents_str",
        "support",
        "confidence",
        "lift"
    ]
    avail_cols = [c for c in display_cols if c in rules_df.columns]
    renamed = rules_df[avail_cols].rename(columns={
        "rule_representation": "Discovered Association Rule",
        "antecedents_str": "Antecedent (IF)",
        "consequents_str": "Consequent (THEN)",
        "support": "Support P(A ∩ B)",
        "confidence": "Confidence P(B|A)",
        "lift": "Lift Factor"
    })
    st.dataframe(renamed, use_container_width=True, hide_index=True)

    # Frequent Itemsets Expander
    with st.expander("📦 View Frequent Sport Itemsets"):
        if not apriori_res["frequent_itemsets"].empty:
            f_cols = ["itemsets_display", "length", "support"]
            f_renamed = apriori_res["frequent_itemsets"][[c for c in f_cols if c in apriori_res["frequent_itemsets"].columns]].rename(columns={
                "itemsets_display": "Sport Itemset",
                "length": "Size (k-itemset)",
                "support": "Support Frequency"
            })
            st.dataframe(f_renamed.head(50), use_container_width=True, hide_index=True)

    # 6. Optional Technical Details
    with st.expander("🛠️ Viva Technical Context: Apriori Property & Metrics"):
        render_viva_note(
            "Apriori Anti-Monotonicity Principle",
            "The Apriori algorithm exploits the anti-monotonic property: 'All non-empty subsets of a frequent itemset must also be frequent.' If a sport combination {A, B} is infrequent, any superset {A, B, C} is immediately pruned without database scanning.",
            "Support measures rule frequency P(A ∩ B); Confidence measures conditional probability P(B|A); Lift measures how much more often A and B occur together than expected under independence: P(B|A) / P(B). Lift > 1 indicates positive affinity."
        )


if __name__ == "__main__":
    render_page()
