"""
Data cleaning utilities for the Credit Risk Modelling project.

This script contains reusable cleaning functions for handling missing values,
rare categories, categorical encoding, and feature preparation before modelling.
The functions are designed to be used by both notebooks and training scripts.
"""

import pandas as pd
import numpy as np

from src.config import HIGH_MISSING_THRESHOLD, RARE_CATEGORY_THRESHOLD


def get_columns_with_high_missing_values(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSING_THRESHOLD,
) -> list[str]:
    """
    Returns columns where the missing-value percentage is above the given threshold.
    """
    missing_ratio = df.isna().mean()
    return missing_ratio[missing_ratio > threshold].index.tolist()


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSING_THRESHOLD,
) -> pd.DataFrame:
    """
    Drops columns where the missing-value percentage is above the given threshold.
    """
    columns_to_drop = get_columns_with_high_missing_values(
        df=df,
        threshold=threshold,
    )

    return df.drop(columns=columns_to_drop)


def replace_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces positive and negative infinite values with missing values.
    """
    return df.replace([np.inf, -np.inf], np.nan)


def identify_column_types(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Separates columns into numeric and categorical feature lists.
    """
    if exclude_columns is None:
        exclude_columns = []

    feature_df = df.drop(columns=exclude_columns, errors="ignore")

    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_columns, categorical_columns


def fill_numeric_missing_values(
    df: pd.DataFrame,
    numeric_columns: list[str],
    fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Fills missing numeric values using median values.

    If fill_values are provided, they are reused. This is important when applying
    training-set cleaning rules to validation or test data.
    """
    df_clean = df.copy()

    if fill_values is None:
        fill_values = {}

        for column in numeric_columns:
            fill_values[column] = df_clean[column].median()

    for column in numeric_columns:
        if column in df_clean.columns:
            df_clean[column] = df_clean[column].fillna(fill_values.get(column, 0))

    return df_clean, fill_values


def fill_categorical_missing_values(
    df: pd.DataFrame,
    categorical_columns: list[str],
    fill_value: str = "Unknown",
) -> pd.DataFrame:
    """
    Fills missing categorical values with a fixed category.
    """
    df_clean = df.copy()

    for column in categorical_columns:
        if column in df_clean.columns:
            df_clean[column] = df_clean[column].fillna(fill_value).astype(str)

    return df_clean


def group_rare_categories(
    df: pd.DataFrame,
    categorical_columns: list[str],
    threshold: float = RARE_CATEGORY_THRESHOLD,
    category_maps: dict[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """
    Groups rare categories into 'Other'.

    If category_maps are provided, they are reused to ensure consistent category
    handling between train, validation, and test datasets.
    """
    df_clean = df.copy()

    if category_maps is None:
        category_maps = {}

        for column in categorical_columns:
            if column not in df_clean.columns:
                continue

            frequency = df_clean[column].value_counts(normalize=True, dropna=False)
            common_categories = set(frequency[frequency >= threshold].index.astype(str))
            category_maps[column] = common_categories

    for column in categorical_columns:
        if column not in df_clean.columns:
            continue

        allowed_categories = category_maps.get(column, set())
        df_clean[column] = df_clean[column].astype(str)
        df_clean[column] = df_clean[column].where(
            df_clean[column].isin(allowed_categories),
            "Other",
        )

    return df_clean, category_maps


def one_hot_encode_categoricals(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """
    One-hot encodes categorical columns using pandas get_dummies.
    """
    existing_categorical_columns = [
        column for column in categorical_columns if column in df.columns
    ]

    return pd.get_dummies(
        df,
        columns=existing_categorical_columns,
        dummy_na=False,
        drop_first=False,
    )


def align_train_test_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns train and test datasets so they contain the same feature columns.
    """
    train_aligned, test_aligned = train_df.align(
        test_df,
        join="left",
        axis=1,
        fill_value=0,
    )

    return train_aligned, test_aligned


def clean_application_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """
    Cleans the main application train and test datasets for modelling.

    Returns cleaned feature matrices, the target variable, and cleaning metadata
    that can be reused or inspected later.
    """
    train_clean = replace_infinite_values(train_df.copy())
    test_clean = replace_infinite_values(test_df.copy())

    y = train_clean[target_column].copy()

    train_features = train_clean.drop(columns=[target_column], errors="ignore")
    test_features = test_clean.copy()

    combined = pd.concat(
        [train_features, test_features],
        axis=0,
        ignore_index=True,
    )

    high_missing_columns = get_columns_with_high_missing_values(combined)
    combined = combined.drop(columns=high_missing_columns, errors="ignore")

    numeric_columns, categorical_columns = identify_column_types(
        combined,
        exclude_columns=[id_column],
    )

    combined, numeric_fill_values = fill_numeric_missing_values(
        df=combined,
        numeric_columns=numeric_columns,
    )

    combined = fill_categorical_missing_values(
        df=combined,
        categorical_columns=categorical_columns,
    )

    combined, category_maps = group_rare_categories(
        df=combined,
        categorical_columns=categorical_columns,
    )

    combined = one_hot_encode_categoricals(
        df=combined,
        categorical_columns=categorical_columns,
    )

    train_processed = combined.iloc[: len(train_features)].reset_index(drop=True)
    test_processed = combined.iloc[len(train_features):].reset_index(drop=True)

    metadata = {
        "high_missing_columns": high_missing_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_fill_values": numeric_fill_values,
        "category_maps": category_maps,
        "feature_columns": train_processed.columns.tolist(),
    }

    return train_processed, test_processed, y, metadata