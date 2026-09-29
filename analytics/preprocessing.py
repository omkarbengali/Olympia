"""
Data Preprocessing & Quality Analysis Module for OLYMPIA.
Implements:
1. Data Quality Assessment (Row counts, duplicates, missingness summary & percentages, dtypes).
2. Categorical Encoding (Label Encoding & One-Hot Encoding).
3. Feature Scaling (Min-Max Normalization & Z-score Standardization).
4. Dimensionality Reduction (Principal Component Analysis - PCA).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from warehouse.warehouse import get_denormalized_medals
from etl.extract import extract_data


def get_data_quality_report(
    raw_df: pd.DataFrame = None,
    cleaned_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Computes comprehensive data quality audit metrics comparing raw vs cleaned datasets.
    """
    if raw_df is None:
        raw_df, _ = extract_data()
    if cleaned_df is None:
        cleaned_df = get_denormalized_medals()

    # Raw metrics
    raw_rows = len(raw_df)
    raw_cols = len(raw_df.columns)
    raw_duplicates = int(raw_df.duplicated().sum())

    missing_counts = raw_df.isnull().sum()
    missing_pct = (missing_counts / raw_rows * 100).round(2)
    missing_summary = pd.DataFrame({
        "Column": raw_df.columns,
        "Data_Type": [str(t) for t in raw_df.dtypes],
        "Missing_Count": missing_counts.values,
        "Missing_Percentage": missing_pct.values,
        "Unique_Values": [raw_df[col].nunique() for col in raw_df.columns]
    })

    # Cleaned metrics
    cleaned_rows = len(cleaned_df)
    cleaned_cols = len(cleaned_df.columns)
    cleaned_duplicates = int(cleaned_df.duplicated().sum())
    cleaned_missing_total = int(cleaned_df.isnull().sum().sum())

    return {
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "raw_duplicates": raw_duplicates,
        "cleaned_rows": cleaned_rows,
        "cleaned_cols": cleaned_cols,
        "cleaned_duplicates": cleaned_duplicates,
        "cleaned_missing_total": cleaned_missing_total,
        "missing_summary_df": missing_summary,
    }


def demonstrate_label_encoding(
    df: pd.DataFrame,
    columns: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Demonstrates Label Encoding on specified categorical columns.
    Converts categorical text labels into ordinal integers (0, 1, 2, ...).
    """
    if columns is None:
        columns = ["event_gender", "medal", "season"]

    encoded_df = df.copy()
    mappings = {}

    for col in columns:
        if col in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[f"{col}_encoded"] = le.fit_transform(encoded_df[col].astype(str))
            mappings[col] = {label: int(code) for code, label in enumerate(le.classes_)}

    return encoded_df, mappings


def demonstrate_one_hot_encoding(
    df: pd.DataFrame,
    columns: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Demonstrates One-Hot Encoding on nominal categorical columns.
    Creates binary (0/1) indicator columns for each unique category.
    """
    if columns is None:
        columns = ["event_gender", "medal"]

    valid_cols = [c for c in columns if c in df.columns]
    ohe_df = pd.get_dummies(df, columns=valid_cols, prefix=valid_cols, dtype=int)
    new_cols = [c for c in ohe_df.columns if any(c.startswith(f"{v}_") for v in valid_cols)]
    return ohe_df, new_cols


def demonstrate_feature_scaling(
    df: pd.DataFrame,
    features: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Demonstrates Min-Max Normalization (scaled to [0, 1]) and
    Standardization / Z-Score Scaling (mean=0, std=1).
    """
    if features is None:
        features = ["medal_points"]

    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError(f"Features {features} not found in dataset.")

    # 1. Min-Max Normalization
    minmax = MinMaxScaler()
    norm_values = minmax.fit_transform(df[valid_features])
    norm_df = pd.DataFrame(
        norm_values,
        columns=[f"{f}_normalized" for f in valid_features],
        index=df.index
    )

    # 2. Standard Scaling (Z-Score)
    scaler = StandardScaler()
    std_values = scaler.fit_transform(df[valid_features])
    std_df = pd.DataFrame(
        std_values,
        columns=[f"{f}_standardized" for f in valid_features],
        index=df.index
    )

    return norm_df, std_df


def demonstrate_pca_reduction(
    df: pd.DataFrame,
    features: List[str],
    n_components: int = 2
) -> Tuple[pd.DataFrame, List[float], np.ndarray]:
    """
    Demonstrates PCA Dimensionality Reduction:
    Projects high-dimensional numerical features onto lower orthogonal principal components.

    Returns:
        reduced_df: DataFrame with PC1, PC2, etc.
        explained_variance: Proportion of variance explained by each component.
        components: Feature loading matrix.
    """
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) < n_components:
        n_components = len(valid_features)

    X = df[valid_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    reduced_values = pca.fit_transform(X_scaled)

    comp_cols = [f"PC{i+1}" for i in range(n_components)]
    reduced_df = pd.DataFrame(reduced_values, columns=comp_cols, index=df.index)
    explained_variance = pca.explained_variance_ratio_.tolist()

    return reduced_df, explained_variance, pca.components_


if __name__ == "__main__":
    df = get_denormalized_medals()
    print("Testing Preprocessing Module...")
    report = get_data_quality_report()
    print("Quality Report Summary:")
    print(f"  Raw Rows: {report['raw_rows']} | Cleaned Rows: {report['cleaned_rows']}")
    print(f"  Duplicates Removed: {report['raw_duplicates']}")

    enc_df, maps = demonstrate_label_encoding(df.head(10))
    print("\nLabel Encoding Mappings:", maps)

    ohe_df, new_cols = demonstrate_one_hot_encoding(df.head(5))
    print("\nOHE Columns:", new_cols)
