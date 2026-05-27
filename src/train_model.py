"""
Model training utilities for the Credit Risk Modelling project.

This script trains baseline and tree-based credit-risk models, saves trained
models, and returns model performance outputs that can be reviewed in notebooks
or reused by the API and dashboard layers.
"""

from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ID_COLUMN,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
    BASELINE_MODEL_FILE,
    TREE_MODEL_FILE,
)


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates model features from the binary target variable.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    y = df[target_column].copy()
    X = df.drop(columns=[target_column, id_column], errors="ignore").copy()

    return X, y


def create_train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Creates a stratified train-validation split for default-risk modelling.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_valid, y_train, y_valid


def get_numeric_columns(X: pd.DataFrame) -> list[str]:
    """
    Returns numeric feature columns suitable for model training.
    """
    return X.select_dtypes(include=["number", "bool"]).columns.tolist()

def sanitise_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans feature names so they are accepted by LightGBM.
    """
    X_clean = X.copy()
    X_clean.columns = (
        X_clean.columns.astype(str)
        .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return X_clean


def create_baseline_logistic_regression_model(
    numeric_columns: list[str],
) -> Pipeline:
    """
    Creates a regularised Logistic Regression baseline pipeline.
    """
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, numeric_columns),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def create_lightgbm_model() -> LGBMClassifier:
    """
    Creates a LightGBM classifier suitable for imbalanced credit-risk prediction.
    """
    return LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Trains the Logistic Regression baseline model.
    """
    numeric_columns = get_numeric_columns(X_train)

    if not numeric_columns:
        raise ValueError("No numeric columns found for baseline model training.")

    model = create_baseline_logistic_regression_model(numeric_columns)
    model.fit(X_train, y_train)

    return model


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LGBMClassifier:
    """
    Trains the LightGBM credit-risk model.
    """
    X_train_numeric = X_train.select_dtypes(include=["number", "bool"]).copy()
    X_train_numeric = sanitise_feature_names(X_train_numeric)
    X_train_numeric = X_train_numeric.replace([float("inf"), float("-inf")], float("nan"))

    model = create_lightgbm_model()
    model.fit(X_train_numeric, y_train)

    return model


def predict_default_probability(
    model,
    X: pd.DataFrame,
    numeric_only: bool = False,
) -> pd.Series:
    """
    Returns predicted probability of default for the positive class.
    """
    X_input = X.copy()

    if numeric_only:
        X_input = X_input.select_dtypes(include=["number", "bool"])
        X_input = sanitise_feature_names(X_input)
        X_input = X_input.replace([float("inf"), float("-inf")], float("nan"))
        
        if hasattr(model, "feature_name_"):
            X_input = X_input.reindex(columns=model.feature_name_, fill_value=0)

    probabilities = model.predict_proba(X_input)[:, 1]

    return pd.Series(probabilities, index=X.index, name="probability_of_default")


def calculate_validation_auc(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    numeric_only: bool = False,
) -> float:
    """
    Calculates validation ROC-AUC for a trained model.
    """
    valid_probabilities = predict_default_probability(
        model=model,
        X=X_valid,
        numeric_only=numeric_only,
    )

    return float(roc_auc_score(y_valid, valid_probabilities))


def save_model(model, file_name: str, models_dir: Path = MODELS_DIR) -> Path:
    """
    Saves a trained model as a pickle file.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path = models_dir / file_name
    joblib.dump(model, output_path)

    return output_path


def load_model(file_path: Path):
    """
    Loads a saved model from disk.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    return joblib.load(file_path)


def train_and_save_models(
    modelling_df: pd.DataFrame,
) -> dict:
    """
    Trains baseline and LightGBM models, evaluates them with ROC-AUC, and saves them.
    """
    X, y = split_features_and_target(modelling_df)

    X_train, X_valid, y_train, y_valid = create_train_validation_split(X, y)

    baseline_model = train_baseline_model(X_train, y_train)
    lightgbm_model = train_lightgbm_model(X_train, y_train)

    baseline_auc = calculate_validation_auc(
        model=baseline_model,
        X_valid=X_valid,
        y_valid=y_valid,
        numeric_only=False,
    )

    lightgbm_auc = calculate_validation_auc(
        model=lightgbm_model,
        X_valid=X_valid,
        y_valid=y_valid,
        numeric_only=True,
    )

    baseline_model_path = save_model(
        model=baseline_model,
        file_name=BASELINE_MODEL_FILE,
    )

    lightgbm_model_path = save_model(
        model=lightgbm_model,
        file_name=TREE_MODEL_FILE,
    )

    return {
        "baseline_model": baseline_model,
        "lightgbm_model": lightgbm_model,
        "baseline_auc": baseline_auc,
        "lightgbm_auc": lightgbm_auc,
        "baseline_model_path": baseline_model_path,
        "lightgbm_model_path": lightgbm_model_path,
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
    }