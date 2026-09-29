"""
Apriori Association Rule Mining Module for OLYMPIA.
Constructs market-basket style transactions where:
  Transaction = Country + Olympic Games Edition (e.g., 'United States (2024 Paris)')
  Items = Distinct sports in which medals were won.
Mines frequent sport combinations and generates association rules with Support, Confidence, and Lift.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from typing import Dict, Any, List, Tuple
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from warehouse.warehouse import get_denormalized_medals


def build_transactions(df: pd.DataFrame = None) -> Tuple[List[List[str]], pd.DataFrame]:
    """
    Constructs Olympic transactions from country-games appearances.
    """
    if df is None:
        df = get_denormalized_medals()

    # Group by country and games to assemble basket of sports
    trans_grouped = df.groupby(["country", "games"])["sport"].apply(lambda s: sorted(list(set(s)))).reset_index()
    trans_grouped.columns = ["country", "games", "sports"]

    # Filter out empty baskets
    trans_grouped = trans_grouped[trans_grouped["sports"].apply(len) > 0]
    transactions = trans_grouped["sports"].tolist()

    return transactions, trans_grouped


def mine_association_rules(
    df: pd.DataFrame = None,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
    metric: str = "confidence"
) -> Dict[str, Any]:
    """
    Mines frequent itemsets and association rules using the Apriori algorithm.

    Parameters:
        df: Input denormalized DataFrame.
        min_support: Minimum frequency threshold for itemsets (e.g. 0.05 = 5%).
        min_confidence: Minimum conditional probability threshold for rules.
        metric: Rule filtering metric ('confidence', 'lift', 'support').

    Returns:
        Dict containing frequent_itemsets, rules_df, and summary statistics.
    """
    transactions, trans_df = build_transactions(df)

    if not transactions:
        return {
            "total_transactions": 0,
            "frequent_itemsets": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "status": "NO_TRANSACTIONS",
            "message": "No transactions available for the selected data."
        }

    # One-Hot Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)

    # 1. Mine Frequent Itemsets
    frequent_itemsets = apriori(
        basket_df,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        return {
            "total_transactions": len(transactions),
            "frequent_itemsets": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "status": "NO_ITEMSETS",
            "message": f"No frequent sport combinations met the minimum support of {min_support * 100:.1f}%. Try lowering min_support."
        }

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
    frequent_itemsets["itemsets_display"] = frequent_itemsets["itemsets"].apply(lambda s: ", ".join(sorted(list(s))))
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

    # 2. Mine Association Rules
    try:
        rules = association_rules(
            frequent_itemsets,
            metric=metric,
            min_threshold=min_confidence
        )
    except ValueError:
        rules = pd.DataFrame()

    if rules.empty:
        return {
            "total_transactions": len(transactions),
            "frequent_itemsets": frequent_itemsets,
            "rules": pd.DataFrame(),
            "status": "NO_RULES",
            "message": f"Frequent itemsets found, but no rules met the minimum confidence of {min_confidence * 100:.1f}%. Try lowering min_confidence."
        }

    # Format rules for presentation
    rules_display = rules.copy()
    rules_display["antecedents_str"] = rules_display["antecedents"].apply(lambda s: ", ".join(list(s)))
    rules_display["consequents_str"] = rules_display["consequents"].apply(lambda s: ", ".join(list(s)))
    rules_display["rule_representation"] = (
        rules_display["antecedents_str"] + "  ==>  " + rules_display["consequents_str"]
    )

    for col in ["support", "confidence", "lift", "leverage", "conviction"]:
        if col in rules_display.columns:
            rules_display[col] = rules_display[col].round(3)

    rules_display = rules_display.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)

    return {
        "total_transactions": len(transactions),
        "frequent_itemsets": frequent_itemsets,
        "rules": rules_display,
        "status": "SUCCESS",
        "message": f"Found {len(frequent_itemsets)} frequent itemsets and {len(rules_display)} association rules."
    }


if __name__ == "__main__":
    print("Testing Apriori Association Rule Mining...")
    res = mine_association_rules(min_support=0.08, min_confidence=0.4)
    print(f"Total Transactions: {res['total_transactions']}")
    print(f"Status: {res['status']} | {res['message']}")
    if not res["rules"].empty:
        print("\nTop 5 Association Rules:")
        print(res["rules"][["rule_representation", "support", "confidence", "lift"]].head(5).to_string(index=False))
