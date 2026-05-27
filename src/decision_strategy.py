"""
Business decision strategy utilities for the Credit Risk Modelling project.

This script converts model probability outputs into practical credit-risk bands,
approval decisions, portfolio summaries, and threshold-based business impact
tables. The aim is to connect model performance with underwriting-style decision
making.
"""

import numpy as np
import pandas as pd


def assign_risk_band(
    probability_of_default: float,
    low_risk_threshold: float = 0.10,
    high_risk_threshold: float = 0.30,
) -> str:
    """
    Assigns a risk band based on predicted probability of default.
    """
    if pd.isna(probability_of_default):
        return "Unknown"

    if probability_of_default < low_risk_threshold:
        return "Low Risk"

    if probability_of_default < high_risk_threshold:
        return "Medium Risk"

    return "High Risk"


def assign_credit_decision(
    probability_of_default: float,
    low_risk_threshold: float = 0.10,
    high_risk_threshold: float = 0.30,
) -> str:
    """
    Converts predicted probability of default into a decision recommendation.
    """
    risk_band = assign_risk_band(
        probability_of_default=probability_of_default,
        low_risk_threshold=low_risk_threshold,
        high_risk_threshold=high_risk_threshold,
    )

    decision_map = {
        "Low Risk": "Approve",
        "Medium Risk": "Manual Review",
        "High Risk": "Decline",
        "Unknown": "Manual Review",
    }

    return decision_map[risk_band]


def create_decision_output(
    applicant_ids: pd.Series,
    probabilities: pd.Series,
    actual_target: pd.Series | None = None,
    low_risk_threshold: float = 0.10,
    high_risk_threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Creates an applicant-level decision table from predicted default probabilities.
    """
    decision_df = pd.DataFrame(
        {
            "applicant_id": applicant_ids.values,
            "probability_of_default": probabilities.values,
        }
    )

    decision_df["risk_band"] = decision_df["probability_of_default"].apply(
        lambda probability: assign_risk_band(
            probability_of_default=probability,
            low_risk_threshold=low_risk_threshold,
            high_risk_threshold=high_risk_threshold,
        )
    )

    decision_df["recommended_decision"] = decision_df["probability_of_default"].apply(
        lambda probability: assign_credit_decision(
            probability_of_default=probability,
            low_risk_threshold=low_risk_threshold,
            high_risk_threshold=high_risk_threshold,
        )
    )

    if actual_target is not None:
        decision_df["actual_default"] = actual_target.values

    return decision_df


def summarise_decision_distribution(decision_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarises the number and percentage of applicants in each decision category.
    """
    summary = (
        decision_df.groupby(["recommended_decision", "risk_band"])
        .agg(
            applicant_count=("applicant_id", "count"),
            average_probability_of_default=("probability_of_default", "mean"),
        )
        .reset_index()
    )

    total_applicants = summary["applicant_count"].sum()
    summary["applicant_share"] = summary["applicant_count"] / total_applicants

    return summary.sort_values(
        by="average_probability_of_default",
        ascending=True,
    ).reset_index(drop=True)


def summarise_decision_performance(decision_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarises actual default performance by decision category when target data exists.
    """
    if "actual_default" not in decision_df.columns:
        raise ValueError("actual_default column is required for decision performance analysis.")

    summary = (
        decision_df.groupby(["recommended_decision", "risk_band"])
        .agg(
            applicant_count=("applicant_id", "count"),
            default_count=("actual_default", "sum"),
            average_probability_of_default=("probability_of_default", "mean"),
            actual_default_rate=("actual_default", "mean"),
        )
        .reset_index()
    )

    total_applicants = summary["applicant_count"].sum()
    total_defaults = summary["default_count"].sum()

    summary["applicant_share"] = summary["applicant_count"] / total_applicants
    summary["default_share"] = (
        summary["default_count"] / total_defaults if total_defaults > 0 else np.nan
    )

    return summary.sort_values(
        by="average_probability_of_default",
        ascending=True,
    ).reset_index(drop=True)


def calculate_expected_loss(
    probability_of_default: pd.Series,
    exposure_at_default: pd.Series,
    loss_given_default: float = 0.45,
) -> pd.Series:
    """
    Calculates expected loss using PD, EAD and LGD.

    Expected Loss = Probability of Default * Exposure at Default * Loss Given Default
    """
    return probability_of_default * exposure_at_default * loss_given_default


def create_expected_loss_table(
    decision_df: pd.DataFrame,
    exposure_at_default: pd.Series,
    loss_given_default: float = 0.45,
) -> pd.DataFrame:
    """
    Adds expected loss estimates to an applicant-level decision table.
    """
    output_df = decision_df.copy()

    output_df["exposure_at_default"] = exposure_at_default.values
    output_df["loss_given_default"] = loss_given_default
    output_df["expected_loss"] = calculate_expected_loss(
        probability_of_default=output_df["probability_of_default"],
        exposure_at_default=output_df["exposure_at_default"],
        loss_given_default=loss_given_default,
    )

    return output_df


def summarise_expected_loss_by_decision(decision_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarises expected loss by decision category.
    """
    required_columns = {
        "recommended_decision",
        "risk_band",
        "applicant_id",
        "probability_of_default",
        "exposure_at_default",
        "expected_loss",
    }

    missing_columns = required_columns.difference(decision_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    summary = (
        decision_df.groupby(["recommended_decision", "risk_band"])
        .agg(
            applicant_count=("applicant_id", "count"),
            average_probability_of_default=("probability_of_default", "mean"),
            total_exposure=("exposure_at_default", "sum"),
            average_exposure=("exposure_at_default", "mean"),
            total_expected_loss=("expected_loss", "sum"),
            average_expected_loss=("expected_loss", "mean"),
        )
        .reset_index()
    )

    return summary.sort_values(
        by="average_probability_of_default",
        ascending=True,
    ).reset_index(drop=True)


def simulate_threshold_strategy(
    applicant_ids: pd.Series,
    probabilities: pd.Series,
    actual_target: pd.Series | None = None,
    exposure_at_default: pd.Series | None = None,
    low_risk_threshold: float = 0.10,
    high_risk_threshold: float = 0.30,
    loss_given_default: float = 0.45,
) -> dict:
    """
    Runs a complete decision-threshold simulation for a credit-risk model.
    """
    decision_df = create_decision_output(
        applicant_ids=applicant_ids,
        probabilities=probabilities,
        actual_target=actual_target,
        low_risk_threshold=low_risk_threshold,
        high_risk_threshold=high_risk_threshold,
    )

    decision_distribution = summarise_decision_distribution(decision_df)

    output = {
        "decision_table": decision_df,
        "decision_distribution": decision_distribution,
    }

    if actual_target is not None:
        output["decision_performance"] = summarise_decision_performance(decision_df)

    if exposure_at_default is not None:
        expected_loss_df = create_expected_loss_table(
            decision_df=decision_df,
            exposure_at_default=exposure_at_default,
            loss_given_default=loss_given_default,
        )

        output["expected_loss_table"] = expected_loss_df
        output["expected_loss_summary"] = summarise_expected_loss_by_decision(
            expected_loss_df
        )

    return output


def compare_threshold_strategies(
    applicant_ids: pd.Series,
    probabilities: pd.Series,
    actual_target: pd.Series,
    exposure_at_default: pd.Series | None = None,
    threshold_pairs: list[tuple[float, float]] | None = None,
    loss_given_default: float = 0.45,
) -> pd.DataFrame:
    """
    Compares different low-risk and high-risk thresholds from a business perspective.
    """
    if threshold_pairs is None:
        threshold_pairs = [
            (0.05, 0.20),
            (0.05, 0.25),
            (0.10, 0.25),
            (0.10, 0.30),
            (0.15, 0.35),
            (0.20, 0.40),
        ]

    rows = []

    for low_threshold, high_threshold in threshold_pairs:
        simulation = simulate_threshold_strategy(
            applicant_ids=applicant_ids,
            probabilities=probabilities,
            actual_target=actual_target,
            exposure_at_default=exposure_at_default,
            low_risk_threshold=low_threshold,
            high_risk_threshold=high_threshold,
            loss_given_default=loss_given_default,
        )

        decision_table = simulation["decision_table"]

        approved = decision_table["recommended_decision"] == "Approve"
        reviewed = decision_table["recommended_decision"] == "Manual Review"
        declined = decision_table["recommended_decision"] == "Decline"

        row = {
            "low_risk_threshold": low_threshold,
            "high_risk_threshold": high_threshold,
            "approval_rate": approved.mean(),
            "manual_review_rate": reviewed.mean(),
            "decline_rate": declined.mean(),
            "approved_default_rate": decision_table.loc[approved, "actual_default"].mean(),
            "manual_review_default_rate": decision_table.loc[reviewed, "actual_default"].mean(),
            "declined_default_rate": decision_table.loc[declined, "actual_default"].mean(),
        }

        if exposure_at_default is not None:
            expected_loss_table = simulation["expected_loss_table"]

            row["approved_expected_loss"] = expected_loss_table.loc[
                approved,
                "expected_loss",
            ].sum()

            row["total_expected_loss"] = expected_loss_table["expected_loss"].sum()

        rows.append(row)

    return pd.DataFrame(rows)