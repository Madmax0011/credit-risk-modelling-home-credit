"""
Data loading utilities for the Credit Risk Modelling project.

This script validates that all required Home Credit dataset files exist,
loads individual CSV files, and provides helper functions for loading the
full raw dataset into memory.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, REQUIRED_RAW_FILES


def validate_raw_data_files(raw_data_dir: Path = RAW_DATA_DIR) -> None:
    """
    Checks that all required raw dataset files exist before analysis begins.
    """
    missing_files = []

    for file_key, file_name in REQUIRED_RAW_FILES.items():
        file_path = raw_data_dir / file_name

        if not file_path.exists():
            missing_files.append(f"{file_key}: {file_path}")

    if missing_files:
        missing_text = "\n".join(missing_files)
        raise FileNotFoundError(
            "The following required dataset files are missing:\n"
            f"{missing_text}\n\n"
            "Please place all Home Credit Default Risk CSV files inside data/raw/."
        )


def load_csv_file(file_name: str, raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Loads a single CSV file from the raw data directory.
    """
    file_path = raw_data_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def load_raw_table(table_name: str, raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Loads one raw Home Credit table using its logical table name.
    """
    if table_name not in REQUIRED_RAW_FILES:
        valid_names = ", ".join(REQUIRED_RAW_FILES.keys())
        raise ValueError(
            f"Unknown table name: {table_name}. Valid table names are: {valid_names}"
        )

    file_name = REQUIRED_RAW_FILES[table_name]
    return load_csv_file(file_name=file_name, raw_data_dir=raw_data_dir)


def load_all_raw_data(raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """
    Loads all required Home Credit raw tables into a dictionary of DataFrames.
    """
    validate_raw_data_files(raw_data_dir=raw_data_dir)

    data = {}

    for table_name, file_name in REQUIRED_RAW_FILES.items():
        data[table_name] = load_csv_file(
            file_name=file_name,
            raw_data_dir=raw_data_dir,
        )

    return data


def get_dataset_shapes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Returns a summary of row and column counts for loaded datasets.
    """
    rows = []

    for table_name, df in data.items():
        rows.append(
            {
                "table_name": table_name,
                "rows": df.shape[0],
                "columns": df.shape[1],
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="rows",
        ascending=False,
    ).reset_index(drop=True)


def print_dataset_shapes(data: dict[str, pd.DataFrame]) -> None:
    """
    Prints a clean shape summary for quick notebook inspection.
    """
    shape_summary = get_dataset_shapes(data)

    for _, row in shape_summary.iterrows():
        print(
            f"{row['table_name']:<25} | "
            f"rows: {row['rows']:>10,} | "
            f"columns: {row['columns']:>4}"
        )


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: Path,
    index: bool = False,
) -> None:
    """
    Saves a processed DataFrame to CSV and creates the parent folder if required.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


def load_processed_dataset(file_path: Path) -> pd.DataFrame:
    """
    Loads a processed dataset from a CSV file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found: {file_path}")

    return pd.read_csv(file_path)