"""
Naive Bayes Classification Module for OLYMPIA.
Implements probabilistic classification of Olympic medal categories.
Compares performance directly with the Decision Tree model.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from analytics.decision_tree import prepare_classification_dataset


def train_naive_bayes(
    df: pd.DataFrame = None,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a Categorical Naive Bayes Classifier on Olympic Medal data.

    Returns:
        Dict containing metrics, confusion matrix, and prediction details.
    """
    X, y, encoders = prepare_classification_dataset(df)

    classes = ["Gold", "Silver", "Bronze"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # CategoricalNB is appropriate for integer-encoded categorical attributes
    # We use min_categories to prevent out-of-range unseen category index errors
    min_cats = [len(np.unique(X[col])) for col in X.columns]
    nb = CategoricalNB(min_categories=min_cats)
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    # Calculate Evaluation Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    return {
        "model_name": "Naive Bayes (Categorical)",
        "model": nb,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }


def compare_classifiers(dt_res: Dict[str, Any], nb_res: Dict[str, Any]) -> pd.DataFrame:
    """Creates a side-by-side comparison table of Decision Tree vs Naive Bayes metrics."""
    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1 Score (Weighted)", "Training Samples", "Testing Samples"],
        "Decision Tree": [
            f"{dt_res['accuracy'] * 100:.2f}%",
            f"{dt_res['precision'] * 100:.2f}%",
            f"{dt_res['recall'] * 100:.2f}%",
            f"{dt_res['f1_score'] * 100:.2f}%",
            f"{dt_res['train_samples']:,}",
            f"{dt_res['test_samples']:,}"
        ],
        "Naive Bayes": [
            f"{nb_res['accuracy'] * 100:.2f}%",
            f"{nb_res['precision'] * 100:.2f}%",
            f"{nb_res['recall'] * 100:.2f}%",
            f"{nb_res['f1_score'] * 100:.2f}%",
            f"{nb_res['train_samples']:,}",
            f"{nb_res['test_samples']:,}"
        ]
    })
    return comparison


if __name__ == "__main__":
    from analytics.decision_tree import train_decision_tree
    print("Testing Naive Bayes Module...")
    nb_res = train_naive_bayes()
    dt_res = train_decision_tree()
    print("Comparison:")
    comp = compare_classifiers(dt_res, nb_res)
    print(comp.to_string(index=False))
