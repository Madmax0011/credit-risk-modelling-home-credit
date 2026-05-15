"""
Feature engineering utilities for the Credit Risk Modelling project.

This script creates applicant-level modelling features from the secondary Home
Credit tables. Each secondary table may contain multiple rows per applicant, so
the functions aggregate those records to one row per SK_ID_CURR before merging
them with the main application dataset.
"""

import pandas as pd
import numpy as np

from src.config import ID_COLUMN


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divides two Series safely and replaces infinite values with missing values.
    """
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def flatten_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens multi-level column names created by groupby aggregation.
    """
    df = df.copy()

    flattened_columns = []

    for column in df.columns:
        if isinstance(column, tuple):
            clean_parts = [str(part) for part in column if str(part)]
            flattened_columns.append("_".join(clean_parts))
        else:
            flattened_columns.append(str(column))

    df.columns = flattened_columns

    return df


def aggregate_numeric_features(
    df: pd.DataFrame,
    group_column: str,
    prefix: str,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregates numeric columns by applicant ID using standard summary statistics.
    """
    if exclude_columns is None:
        exclude_columns = []

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns = [
        column
        for column in numeric_columns
        if column not in exclude_columns and column != group_column
    ]

    if not numeric_columns:
        return df[[group_column]].drop_duplicates().reset_index(drop=True)

    aggregated = df.groupby(group_column)[numeric_columns].agg(
        ["mean", "max", "min", "sum", "std"]
    )

    aggregated = flatten_column_names(aggregated.reset_index())

    rename_map = {
        column: f"{prefix}_{column}"
        for column in aggregated.columns
        if column != group_column
    }

    return aggregated.rename(columns=rename_map)


def aggregate_categorical_features(
    df: pd.DataFrame,
    group_column: str,
    prefix: str,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregates categorical columns by applicant ID using category proportions.
    """
    if exclude_columns is None:
        exclude_columns = []

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_columns = [
        column
        for column in categorical_columns
        if column not in exclude_columns and column != group_column
    ]

    if not categorical_columns:
        return df[[group_column]].drop_duplicates().reset_index(drop=True)

    encoded = pd.get_dummies(
        df[[group_column] + categorical_columns],
        columns=categorical_columns,
        dummy_na=True,
    )

    dummy_columns = [column for column in encoded.columns if column != group_column]

    aggregated = encoded.groupby(group_column)[dummy_columns].mean().reset_index()

    rename_map = {
        column: f"{prefix}_{column}_ratio"
        for column in aggregated.columns
        if column != group_column
    }

    return aggregated.rename(columns=rename_map)


def merge_feature_blocks(base_df: pd.DataFrame, feature_blocks: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple applicant-level feature blocks into the base application table.
    """
    merged = base_df.copy()

    for block in feature_blocks:
        if block is None or block.empty:
            continue

        if ID_COLUMN not in block.columns:
            raise ValueError(f"Feature block is missing required column: {ID_COLUMN}")

        merged = merged.merge(block, on=ID_COLUMN, how="left")

    return merged


def create_application_domain_features(application_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates direct applicant-level financial and demographic ratio features.
    """
    df = application_df.copy()

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["APP_CREDIT_TO_INCOME_RATIO"] = safe_divide(
            df["AMT_CREDIT"],
            df["AMT_INCOME_TOTAL"],
        )

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["APP_ANNUITY_TO_INCOME_RATIO"] = safe_divide(
            df["AMT_ANNUITY"],
            df["AMT_INCOME_TOTAL"],
        )

    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["APP_ANNUITY_TO_CREDIT_RATIO"] = safe_divide(
            df["AMT_ANNUITY"],
            df["AMT_CREDIT"],
        )

    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        df["APP_GOODS_PRICE_TO_CREDIT_RATIO"] = safe_divide(
            df["AMT_GOODS_PRICE"],
            df["AMT_CREDIT"],
        )

    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["APP_EMPLOYED_TO_AGE_RATIO"] = safe_divide(
            df["DAYS_EMPLOYED"],
            df["DAYS_BIRTH"],
        )

    if "DAYS_BIRTH" in df.columns:
        df["APP_AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25

    if "DAYS_EMPLOYED" in df.columns:
        df["APP_EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"]) / 365.25
        df["APP_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df.loc[df["DAYS_EMPLOYED"] == 365243, "APP_EMPLOYED_YEARS"] = np.nan

    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["APP_CHILDREN_TO_FAMILY_RATIO"] = safe_divide(
            df["CNT_CHILDREN"],
            df["CNT_FAM_MEMBERS"],
        )

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["APP_INCOME_PER_FAMILY_MEMBER"] = safe_divide(
            df["AMT_INCOME_TOTAL"],
            df["CNT_FAM_MEMBERS"],
        )

    return df


def create_bureau_features(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates applicant-level features from previous credit bureau records.
    """
    df = bureau_df.copy()

    if {"AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"}.issubset(df.columns):
        df["BUREAU_DEBT_TO_CREDIT_RATIO"] = safe_divide(
            df["AMT_CREDIT_SUM_DEBT"],
            df["AMT_CREDIT_SUM"],
        )

    if {"AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM"}.issubset(df.columns):
        df["BUREAU_OVERDUE_TO_CREDIT_RATIO"] = safe_divide(
            df["AMT_CREDIT_SUM_OVERDUE"],
            df["AMT_CREDIT_SUM"],
        )

    numeric_features = aggregate_numeric_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="BUREAU",
        exclude_columns=["SK_ID_BUREAU"],
    )

    categorical_features = aggregate_categorical_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="BUREAU",
        exclude_columns=["SK_ID_BUREAU"],
    )

    count_features = (
        df.groupby(ID_COLUMN)
        .size()
        .reset_index(name="BUREAU_RECORD_COUNT")
    )

    return merge_feature_blocks(
        base_df=count_features,
        feature_blocks=[numeric_features, categorical_features],
    )


def create_previous_application_features(previous_application_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates applicant-level features from previous loan application records.
    """
    df = previous_application_df.copy()

    if {"AMT_APPLICATION", "AMT_CREDIT"}.issubset(df.columns):
        df["PREV_APPLICATION_TO_CREDIT_RATIO"] = safe_divide(
            df["AMT_APPLICATION"],
            df["AMT_CREDIT"],
        )

    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["PREV_ANNUITY_TO_CREDIT_RATIO"] = safe_divide(
            df["AMT_ANNUITY"],
            df["AMT_CREDIT"],
        )

    numeric_features = aggregate_numeric_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="PREV",
        exclude_columns=["SK_ID_PREV"],
    )

    categorical_features = aggregate_categorical_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="PREV",
        exclude_columns=["SK_ID_PREV"],
    )

    count_features = (
        df.groupby(ID_COLUMN)
        .size()
        .reset_index(name="PREV_APPLICATION_COUNT")
    )

    return merge_feature_blocks(
        base_df=count_features,
        feature_blocks=[numeric_features, categorical_features],
    )


def create_pos_cash_features(pos_cash_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates applicant-level features from POS cash balance history.
    """
    df = pos_cash_df.copy()

    numeric_features = aggregate_numeric_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="POS",
        exclude_columns=["SK_ID_PREV"],
    )

    categorical_features = aggregate_categorical_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="POS",
        exclude_columns=["SK_ID_PREV"],
    )

    count_features = (
        df.groupby(ID_COLUMN)
        .size()
        .reset_index(name="POS_RECORD_COUNT")
    )

    return merge_feature_blocks(
        base_df=count_features,
        feature_blocks=[numeric_features, categorical_features],
    )


def create_installments_features(installments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates applicant-level features from instalment payment history.
    """
    df = installments_df.copy()

    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(df.columns):
        df["INSTALL_PAYMENT_TO_INSTALMENT_RATIO"] = safe_divide(
            df["AMT_PAYMENT"],
            df["AMT_INSTALMENT"],
        )
        df["INSTALL_PAYMENT_DIFFERENCE"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]

    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(df.columns):
        df["INSTALL_PAYMENT_DELAY_DAYS"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
        df["INSTALL_LATE_PAYMENT_FLAG"] = (df["INSTALL_PAYMENT_DELAY_DAYS"] > 0).astype(int)

    numeric_features = aggregate_numeric_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="INSTALL",
        exclude_columns=["SK_ID_PREV"],
    )

    count_features = (
        df.groupby(ID_COLUMN)
        .size()
        .reset_index(name="INSTALL_RECORD_COUNT")
    )

    return merge_feature_blocks(
        base_df=count_features,
        feature_blocks=[numeric_features],
    )


def create_credit_card_features(credit_card_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates applicant-level features from credit card balance history.
    """
    df = credit_card_df.copy()

    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(df.columns):
        df["CC_BALANCE_TO_LIMIT_RATIO"] = safe_divide(
            df["AMT_BALANCE"],
            df["AMT_CREDIT_LIMIT_ACTUAL"],
        )

    if {"AMT_DRAWINGS_CURRENT", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(df.columns):
        df["CC_DRAWINGS_TO_LIMIT_RATIO"] = safe_divide(
            df["AMT_DRAWINGS_CURRENT"],
            df["AMT_CREDIT_LIMIT_ACTUAL"],
        )

    if {"AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY"}.issubset(df.columns):
        df["CC_PAYMENT_TO_MIN_PAYMENT_RATIO"] = safe_divide(
            df["AMT_PAYMENT_CURRENT"],
            df["AMT_INST_MIN_REGULARITY"],
        )

    numeric_features = aggregate_numeric_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="CC",
        exclude_columns=["SK_ID_PREV"],
    )

    categorical_features = aggregate_categorical_features(
        df=df,
        group_column=ID_COLUMN,
        prefix="CC",
        exclude_columns=["SK_ID_PREV"],
    )

    count_features = (
        df.groupby(ID_COLUMN)
        .size()
        .reset_index(name="CC_RECORD_COUNT")
    )

    return merge_feature_blocks(
        base_df=count_features,
        feature_blocks=[numeric_features, categorical_features],
    )


def build_full_feature_dataset(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds full train and test feature datasets from all available Home Credit tables.
    """
    application_train = create_application_domain_features(data["application_train"])
    application_test = create_application_domain_features(data["application_test"])

    feature_blocks = []

    if "bureau" in data:
        feature_blocks.append(create_bureau_features(data["bureau"]))

    if "previous_application" in data:
        feature_blocks.append(create_previous_application_features(data["previous_application"]))

    if "pos_cash_balance" in data:
        feature_blocks.append(create_pos_cash_features(data["pos_cash_balance"]))

    if "installments_payments" in data:
        feature_blocks.append(create_installments_features(data["installments_payments"]))

    if "credit_card_balance" in data:
        feature_blocks.append(create_credit_card_features(data["credit_card_balance"]))

    train_features = merge_feature_blocks(
        base_df=application_train,
        feature_blocks=feature_blocks,
    )

    test_features = merge_feature_blocks(
        base_df=application_test,
        feature_blocks=feature_blocks,
    )

    return train_features, test_features