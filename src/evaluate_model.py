"""
Model evaluation utilities for the Credit Risk Modelling project.

This script provides reusable functions for evaluating credit-risk models using
classification, ranking, threshold-based and business-relevant metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_ks_statistic(
    y_true: pd.Series,
    y_probability: pd.Series,
) -> float:
    """
    Calculates the Kolmogorov-Smirnov statistic for binary classification.

    In credit-risk modelling, KS measures the maximum separation between the
    cumulative distributions of good and bad applicants.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    ks_statistic = max(tpr - fpr)

    return float(ks_statistic)


def calculate_gini_coefficient(
    y_true: pd.Series,
    y_probability: pd.Series,
) -> float:
    """
    Calculates the Gini coefficient from ROC-AUC.
    """
    auc = roc_auc_score(y_true, y_probability)
    gini = (2 * auc) - 1

    return float(gini)


def calculate_classification_metrics(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Calculates threshold-based classification metrics.
    """
    y_predicted = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_predicted),
        "precision": precision_score(y_true, y_predicted, zero_division=0),
        "recall": recall_score(y_true, y_predicted, zero_division=0),
        "f1_score": f1_score(y_true, y_predicted, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def calculate_ranking_metrics(
    y_true: pd.Series,
    y_probability: pd.Series,
) -> dict:
    """
    Calculates ranking metrics commonly used for credit-risk models.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_probability),
        "gini": calculate_gini_coefficient(y_true, y_probability),
        "ks_statistic": calculate_ks_statistic(y_true, y_probability),
    }


def evaluate_model_performance(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Calculates a complete model evaluation summary.
    """
    ranking_metrics = calculate_ranking_metrics(
        y_true=y_true,
        y_probability=y_probability,
    )

    classification_metrics = calculate_classification_metrics(
        y_true=y_true,
        y_probability=y_probability,
        threshold=threshold,
    )

    return {
        **ranking_metrics,
        **classification_metrics,
    }


def create_threshold_performance_table(
    y_true: pd.Series,
    y_probability: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluates model performance across different probability thresholds.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    rows = []

    for threshold in thresholds:
        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_probability=y_probability,
            threshold=threshold,
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


def create_decile_table(
    y_true: pd.Series,
    y_probability: pd.Series,
    number_of_bins: int = 10,
) -> pd.DataFrame:
    """
    Creates a decile table showing default rate by predicted-risk band.
    """
    evaluation_df = pd.DataFrame(
        {
            "actual_default": y_true.values,
            "probability_of_default": y_probability.values,
        }
    )

    evaluation_df["risk_decile"] = pd.qcut(
        evaluation_df["probability_of_default"],
        q=number_of_bins,
        labels=False,
        duplicates="drop",
    )

    decile_table = (
        evaluation_df.groupby("risk_decile")
        .agg(
            applicant_count=("actual_default", "count"),
            default_count=("actual_default", "sum"),
            average_probability_of_default=("probability_of_default", "mean"),
            actual_default_rate=("actual_default", "mean"),
            minimum_probability=("probability_of_default", "min"),
            maximum_probability=("probability_of_default", "max"),
        )
        .reset_index()
        .sort_values("risk_decile", ascending=False)
    )

    decile_table["applicant_share"] = (
        decile_table["applicant_count"] / decile_table["applicant_count"].sum()
    )

    decile_table["default_share"] = (
        decile_table["default_count"] / decile_table["default_count"].sum()
    )

    return decile_table


def create_gain_lift_table(
    y_true: pd.Series,
    y_probability: pd.Series,
    number_of_bins: int = 10,
) -> pd.DataFrame:
    """
    Creates a cumulative gain and lift table for model ranking performance.
    """
    decile_table = create_decile_table(
        y_true=y_true,
        y_probability=y_probability,
        number_of_bins=number_of_bins,
    )

    decile_table = decile_table.sort_values(
        "average_probability_of_default",
        ascending=False,
    ).reset_index(drop=True)

    decile_table["cumulative_applicant_share"] = decile_table["applicant_share"].cumsum()
    decile_table["cumulative_default_share"] = decile_table["default_share"].cumsum()

    overall_default_rate = y_true.mean()

    decile_table["lift"] = (
        decile_table["actual_default_rate"] / overall_default_rate
        if overall_default_rate > 0
        else np.nan
    )

    return decile_table


def compare_models(
    model_predictions: dict[str, pd.Series],
    y_true: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compares multiple models using the same validation target.
    """
    rows = []

    for model_name, probabilities in model_predictions.items():
        metrics = evaluate_model_performance(
            y_true=y_true,
            y_probability=probabilities,
            threshold=threshold,
        )

        metrics["model_name"] = model_name
        rows.append(metrics)

    comparison_df = pd.DataFrame(rows)

    ordered_columns = [
        "model_name",
        "roc_auc",
        "gini",
        "ks_statistic",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    return comparison_df[ordered_columns].sort_values(
        by="roc_auc",
        ascending=False,
    ).reset_index(drop=True)