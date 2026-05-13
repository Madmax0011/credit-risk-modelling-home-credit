import pandas as pd
import numpy as np
from src.config import TARGET_COLUMN, ID_COLUMN


def get_basic_table_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Creates a high-level summary for each table.
    """
    rows = []

    for name, df in data.items():
        rows.append(
            {
                "table_name": name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "duplicate_rows": df.duplicated().sum(),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns missing value counts and percentages for a DataFrame.
    """
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_percent": missing_percent.values,
            "dtype": df.dtypes.astype(str).values,
        }
    )

    return summary.sort_values("missing_percent", ascending=False)


def get_target_distribution(application_train: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the target class distribution for application_train.
    TARGET = 1 means client had payment difficulties.
    TARGET = 0 means client did not have payment difficulties.
    """
    if TARGET_COLUMN not in application_train.columns:
        raise ValueError(f"{TARGET_COLUMN} column not found in application_train.")

    target_counts = application_train[TARGET_COLUMN].value_counts(dropna=False)
    target_percent = application_train[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100

    return pd.DataFrame(
        {
            "target": target_counts.index,
            "count": target_counts.values,
            "percentage": target_percent.round(2).to_numpy(),
        }
    )


def get_id_coverage(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    parent_name: str,
    child_name: str,
    id_col: str = ID_COLUMN,
) -> dict:
    """
    Checks how many customers in a child table are also present in the main application table.
    """
    if id_col not in parent_df.columns:
        raise ValueError(f"{id_col} not found in {parent_name}.")

    if id_col not in child_df.columns:
        raise ValueError(f"{id_col} not found in {child_name}.")

    parent_ids = set(parent_df[id_col].dropna().unique())
    child_ids = set(child_df[id_col].dropna().unique())

    overlap = parent_ids.intersection(child_ids)

    return {
        "parent_table": parent_name,
        "child_table": child_name,
        "parent_unique_ids": len(parent_ids),
        "child_unique_ids": len(child_ids),
        "overlap_ids": len(overlap),
        "child_ids_covered_in_parent_percent": round(len(overlap) / len(child_ids) * 100, 2)
        if child_ids
        else np.nan,
    }


def get_numeric_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for numeric columns.
    """
    numeric_df = df.select_dtypes(include=["number"])

    summary = numeric_df.describe().T.reset_index()
    summary = summary.rename(columns={"index": "column"})

    return summary