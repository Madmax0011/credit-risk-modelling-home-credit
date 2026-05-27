"""
Pydantic schemas for the Credit Risk Modelling API.

These schemas define the expected request and response structures for the
FastAPI application. They keep the API input and output clear, validated, and
easy to document automatically.
"""

from typing import Any

from pydantic import BaseModel, Field


class ApplicantFeatures(BaseModel):
    """
    Flexible applicant feature input for model prediction.

    The model may contain many engineered features, so the API accepts a
    dictionary of feature names and values. This keeps the endpoint reusable
    after feature engineering changes.
    """

    features: dict[str, Any] = Field(
        ...,
        description="Dictionary containing applicant feature names and values.",
        examples=[
            {
                "AMT_INCOME_TOTAL": 135000.0,
                "AMT_CREDIT": 568800.0,
                "AMT_ANNUITY": 20560.5,
                "DAYS_BIRTH": -12005,
                "DAYS_EMPLOYED": -4542,
                "CNT_CHILDREN": 0,
                "CNT_FAM_MEMBERS": 2,
                "EXT_SOURCE_2": 0.62,
                "EXT_SOURCE_3": 0.48,
            }
        ],
    )


class BatchApplicantFeatures(BaseModel):
    """
    Batch input schema for scoring multiple applicants.
    """

    applicants: list[dict[str, Any]] = Field(
        ...,
        description="List of applicant feature dictionaries.",
    )


class PredictionResponse(BaseModel):
    """
    Single-applicant prediction response.
    """

    probability_of_default: float = Field(
        ...,
        description="Predicted probability that the applicant defaults.",
    )
    risk_band: str = Field(
        ...,
        description="Assigned risk band based on the probability of default.",
    )
    recommended_decision: str = Field(
        ...,
        description="Credit decision recommendation generated from the risk band.",
    )


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response.
    """

    predictions: list[PredictionResponse]


class HealthCheckResponse(BaseModel):
    """
    API health-check response.
    """

    status: str
    model_loaded: bool
    model_path: str | None = None


class ErrorResponse(BaseModel):
    """
    Standard error response.
    """

    error: str
    details: str | None = None