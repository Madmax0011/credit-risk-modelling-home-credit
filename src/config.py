"""
Central configuration file for the Credit Risk Modelling project.

This file stores project paths, dataset file names, key column names, and
default modelling settings. Keeping these values in one place makes the
project easier to maintain and prevents repeated hard-coded paths across
notebooks and scripts.
"""

from pathlib import Path


# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# Home Credit Default Risk dataset files
APPLICATION_TRAIN_FILE = "application_train.csv"
APPLICATION_TEST_FILE = "application_test.csv"
BUREAU_FILE = "bureau.csv"
BUREAU_BALANCE_FILE = "bureau_balance.csv"
PREVIOUS_APPLICATION_FILE = "previous_application.csv"
POS_CASH_BALANCE_FILE = "POS_CASH_balance.csv"
INSTALLMENTS_PAYMENTS_FILE = "installments_payments.csv"
CREDIT_CARD_BALANCE_FILE = "credit_card_balance.csv"


REQUIRED_RAW_FILES = {
    "application_train": APPLICATION_TRAIN_FILE,
    "application_test": APPLICATION_TEST_FILE,
    "bureau": BUREAU_FILE,
    "bureau_balance": BUREAU_BALANCE_FILE,
    "previous_application": PREVIOUS_APPLICATION_FILE,
    "pos_cash_balance": POS_CASH_BALANCE_FILE,
    "installments_payments": INSTALLMENTS_PAYMENTS_FILE,
    "credit_card_balance": CREDIT_CARD_BALANCE_FILE,
}


# Core dataset columns
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


# Processed dataset outputs
PROCESSED_TRAIN_FILE = "processed_train.csv"
PROCESSED_TEST_FILE = "processed_test.csv"


# Model outputs
BASELINE_MODEL_FILE = "baseline_logistic_regression.pkl"
TREE_MODEL_FILE = "credit_risk_lightgbm.pkl"
FINAL_MODEL_FILE = "credit_risk_model.pkl"


# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2


# Modelling defaults
HIGH_MISSING_THRESHOLD = 0.6
RARE_CATEGORY_THRESHOLD = 0.01


def create_project_directories() -> None:
    """
    Creates the main output directories if they do not already exist.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)