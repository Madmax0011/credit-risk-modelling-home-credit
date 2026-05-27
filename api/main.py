"""
FastAPI application for serving the credit-risk model.

This API loads the trained model, accepts applicant-level feature inputs, returns
a predicted probability of default, and converts the prediction into a practical
risk band and credit decision.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    ApplicantFeatures,
    BatchApplicantFeatures,
    BatchPredictionResponse,
    HealthCheckResponse,
    PredictionResponse,
)
from src.config import FINAL_MODEL_FILE, MODELS_DIR
from src.decision_strategy import assign_credit_decision, assign_risk_band


MODEL_PATH = MODELS_DIR / FINAL_MODEL_FILE

app = FastAPI(
    title="Credit Risk Modelling API",
    description=(
        "API for predicting applicant probability of default and producing "
        "credit-risk decision recommendations."
    ),
    version="1.0.0",
)

model = None
model_feature_names: list[str] | None = None


def load_trained_model(model_path: Path = MODEL_PATH):
    """
    Loads the trained credit-risk model from disk.
    """
    if not model_path.exists():
        return None

    return joblib.load(model_path)


def get_model_feature_names(loaded_model) -> list[str] | None:
    """
    Attempts to extract expected feature names from a trained model.
    """
    if hasattr(loaded_model, "feature_name_"):
        return list(loaded_model.feature_name_)

    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)

    if hasattr(loaded_model, "named_steps"):
        final_step = list(loaded_model.named_steps.values())[-1]
        if hasattr(final_step, "feature_names_in_"):
            return list(final_step.feature_names_in_)

    return None


@app.on_event("startup")
def startup_event() -> None:
    """
    Loads the model once when the API starts.
    """
    global model, model_feature_names

    model = load_trained_model()
    model_feature_names = get_model_feature_names(model) if model is not None else None


@app.get("/health", response_model=HealthCheckResponse)
def health_check() -> HealthCheckResponse:
    """
    Returns API and model-loading status.
    """
    return HealthCheckResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
    )


def prepare_prediction_input(features: dict) -> pd.DataFrame:
    """
    Converts incoming applicant features into a one-row DataFrame.
    """
    input_df = pd.DataFrame([features])

    if model_feature_names is not None:
        input_df = input_df.reindex(columns=model_feature_names, fill_value=0)

    return input_df


def predict_single_applicant(features: dict) -> PredictionResponse:
    """
    Scores one applicant and returns a model prediction response.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Train and save the final model before "
                "starting the API."
            ),
        )

    try:
        input_df = prepare_prediction_input(features)
        probability_of_default = float(model.predict_proba(input_df)[0, 1])

        risk_band = assign_risk_band(probability_of_default)
        recommended_decision = assign_credit_decision(probability_of_default)

        return PredictionResponse(
            probability_of_default=probability_of_default,
            risk_band=risk_band,
            recommended_decision=recommended_decision,
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {error}",
        ) from error


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ApplicantFeatures) -> PredictionResponse:
    """
    Predicts probability of default for one applicant.
    """
    return predict_single_applicant(request.features)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchApplicantFeatures) -> BatchPredictionResponse:
    """
    Predicts probability of default for multiple applicants.
    """
    predictions = [
        predict_single_applicant(applicant_features)
        for applicant_features in request.applicants
    ]

    return BatchPredictionResponse(predictions=predictions)