"""
Decision Tree Classification Module for OLYMPIA.
Predicts Olympic Medal type (Gold, Silver, Bronze) based on contextual event features:
Country, Sport, Event Gender, Season, and Olympic Year.
Strictly avoids data leakage (excludes medal_points).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from warehouse.warehouse import get_denormalized_medals
from analytics.visualization import apply_dark_theme


def prepare_classification_dataset(df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """
    Prepares features (X) and target (y) for Olympic Medal Classification.
    Excludes medal_points to prevent target leakage.
    Encodes categorical features.
    """
    if df is None:
        df = get_denormalized_medals()

    # Features: Contextual attributes of the event
    feature_cols = ["season", "year", "country", "sport", "event_gender"]
    target_col = "medal"

    data = df[feature_cols + [target_col]].dropna().copy()

    encoders = {}
    X = pd.DataFrame(index=data.index)
    X["year"] = data["year"].astype(int)

    for col in ["season", "country", "sport", "event_gender"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    y = data[target_col].astype(str)
    return X, y, encoders


def train_decision_tree(
    df: pd.DataFrame = None,
    max_depth: int = 5,
    criterion: str = "gini",
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a Decision Tree Classifier on Olympic Medal data.

    Returns:
        Dict containing metrics, confusion matrix, feature importances, and model.
    """
    X, y, encoders = prepare_classification_dataset(df)

    classes = ["Gold", "Silver", "Bronze"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Calculate Evaluation Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Feature Importance
    feature_names = list(X.columns)
    importances = dict(zip(feature_names, [round(float(v), 4) for v in clf.feature_importances_]))

    return {
        "model_name": "Decision Tree",
        "model": clf,
        "max_depth": max_depth,
        "criterion": criterion,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "feature_importances": importances,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }


def plot_confusion_matrix(cm_data: list, classes: list, title: str = "Confusion Matrix") -> go.Figure:
    """Generates an interactive heatmap for the confusion matrix."""
    z = np.array(cm_data)
    text = [[str(val) for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=classes,
            y=classes,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "#ffffff"},
            colorscale="Blues",
            showscale=True
        )
    )

    return apply_dark_theme(
        fig,
        title=title,
        x_title="Predicted Label",
        y_title="True Label"
    )


def plot_feature_importance(importances: Dict[str, float]) -> go.Figure:
    """Generates a horizontal bar chart of feature importances."""
    items = sorted(importances.items(), key=lambda x: x[1])
    features = [x[0].replace("_", " ").title() for x in items]
    scores = [x[1] for x in items]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=features,
            orientation="h",
            marker={"color": "#38bdf8"}
        )
    )
    return apply_dark_theme(fig, title="Feature Importance Breakdown", x_title="Importance Score", y_title="Feature")


if __name__ == "__main__":
    print("Testing Decision Tree Module...")
    res = train_decision_tree()
    print(f"Accuracy: {res['accuracy']:.4f} | Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f} | F1: {res['f1_score']:.4f}")
    print("Feature Importances:", res["feature_importances"])
