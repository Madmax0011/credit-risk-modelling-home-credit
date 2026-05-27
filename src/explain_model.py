"""
Model explainability utilities for the Credit Risk Modelling project.

This script uses feature importance and SHAP values to explain which variables
drive predicted default risk. The functions are designed for notebook analysis,
model reporting, and business-facing interpretation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import REPORTS_DIR


def get_model_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extracts feature importance from a trained tree-based model.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("The supplied model does not expose feature_importances_.")

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    return importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)


def plot_top_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 30,
    output_path: Path | None = None,
) -> None:
    """
    Plots the top model feature importances.
    """
    top_features = importance_df.head(top_n).sort_values(
        by="importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def create_shap_explainer(model):
    """
    Creates a SHAP TreeExplainer for a trained tree-based model.
    """
    return shap.TreeExplainer(model)


def calculate_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int = 5000,
    random_state: int = 42,
):
    """
    Calculates SHAP values on a sample of the feature matrix.

    Sampling is used to keep the explanation step fast and suitable for local
    machines while still giving a representative model explanation.
    """
    X_numeric = X.select_dtypes(include=["number", "bool"]).copy()
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    X_numeric = X_numeric.fillna(X_numeric.median())

    if len(X_numeric) > sample_size:
        X_sample = X_numeric.sample(
            n=sample_size,
            random_state=random_state,
        )
    else:
        X_sample = X_numeric.copy()

    explainer = create_shap_explainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return X_sample, shap_values


def create_shap_summary_plot(
    shap_values,
    X_sample: pd.DataFrame,
    max_display: int = 25,
    output_path: Path | None = None,
) -> None:
    """
    Creates a SHAP summary plot showing global feature impact.
    """
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def create_shap_bar_plot(
    shap_values,
    X_sample: pd.DataFrame,
    max_display: int = 25,
    output_path: Path | None = None,
) -> None:
    """
    Creates a SHAP bar plot showing mean absolute feature impact.
    """
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def get_mean_absolute_shap_importance(
    shap_values,
    X_sample: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converts SHAP values into a ranked feature-importance table.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "mean_absolute_shap_value": mean_abs_shap,
        }
    )

    return shap_importance_df.sort_values(
        by="mean_absolute_shap_value",
        ascending=False,
    ).reset_index(drop=True)


def explain_single_prediction(
    model,
    X: pd.DataFrame,
    applicant_index: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Explains the strongest positive and negative drivers for one applicant.
    """
    X_numeric = X.select_dtypes(include=["number", "bool"]).copy()
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    X_numeric = X_numeric.fillna(X_numeric.median())

    if applicant_index not in X_numeric.index:
        raise ValueError(f"Applicant index not found: {applicant_index}")

    applicant_row = X_numeric.loc[[applicant_index]]

    explainer = create_shap_explainer(model)
    shap_values = explainer.shap_values(applicant_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    explanation_df = pd.DataFrame(
        {
            "feature": applicant_row.columns,
            "feature_value": applicant_row.iloc[0].values,
            "shap_value": shap_values[0],
        }
    )

    explanation_df["absolute_shap_value"] = explanation_df["shap_value"].abs()

    return explanation_df.sort_values(
        by="absolute_shap_value",
        ascending=False,
    ).head(top_n).reset_index(drop=True)


def save_explainability_outputs(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    reports_dir: Path = REPORTS_DIR,
    sample_size: int = 5000,
) -> dict:
    """
    Saves feature-importance and SHAP explanation outputs to the reports folder.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    importance_df = get_model_feature_importance(
        model=model,
        feature_names=feature_names,
    )

    importance_path = reports_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    feature_importance_plot_path = reports_dir / "top_feature_importance.png"
    plot_top_feature_importance(
        importance_df=importance_df,
        top_n=30,
        output_path=feature_importance_plot_path,
    )

    X_sample, shap_values = calculate_shap_values(
        model=model,
        X=X,
        sample_size=sample_size,
    )

    shap_importance_df = get_mean_absolute_shap_importance(
        shap_values=shap_values,
        X_sample=X_sample,
    )

    shap_importance_path = reports_dir / "shap_importance.csv"
    shap_importance_df.to_csv(shap_importance_path, index=False)

    shap_bar_plot_path = reports_dir / "shap_bar_plot.png"
    create_shap_bar_plot(
        shap_values=shap_values,
        X_sample=X_sample,
        max_display=25,
        output_path=shap_bar_plot_path,
    )

    shap_summary_plot_path = reports_dir / "shap_summary_plot.png"
    create_shap_summary_plot(
        shap_values=shap_values,
        X_sample=X_sample,
        max_display=25,
        output_path=shap_summary_plot_path,
    )

    return {
        "feature_importance_path": importance_path,
        "feature_importance_plot_path": feature_importance_plot_path,
        "shap_importance_path": shap_importance_path,
        "shap_bar_plot_path": shap_bar_plot_path,
        "shap_summary_plot_path": shap_summary_plot_path,
    }