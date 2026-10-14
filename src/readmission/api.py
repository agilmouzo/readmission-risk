"""FastAPI service: 30-day readmission risk for one diabetic patient.

Run it (after `python -m readmission.train`):

    uvicorn readmission.api:create_app --factory

Interactive documentation, with a ready-made example, is served at /docs.
Educational project: NOT a clinical tool.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from readmission import config
from readmission.artifact import load_artifact
from readmission.service import OutOfScopeError, predict_one

Drug = Literal["No", "Steady", "Up", "Down"]

EXAMPLE = {
    "age": 72,
    "gender": "Female",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 14,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "number_diagnoses": 8,
    "diag_1": "428",
    "diag_2": "250",
    "diag_3": "401",
    "payer_code": "MC",
    "medical_specialty": "InternalMedicine",
    "A1Cresult": ">8",
    "change": "Ch",
    "diabetesMed": "Yes",
    "insulin": "Steady",
    "metformin": "Steady",
}


class PatientInput(BaseModel):
    """One hospital encounter, with the fields the model uses (names as in the original dataset).

    Optional fields can be left out: diagnoses, payer and specialty default to "unknown",
    lab tests to "None" and every medication to "No".
    """

    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", json_schema_extra={"examples": [EXAMPLE]}
    )

    age: int = Field(ge=0, le=120, description="Age in years")
    gender: Literal["Female", "Male"]
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(
        ge=1, le=30, description="Codes 11, 13, 14, 19, 20, 21 (death/hospice) are rejected"
    )
    admission_source_id: int = Field(ge=1, le=26)
    time_in_hospital: int = Field(ge=1, le=14, description="Days")
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=20)
    num_medications: int = Field(ge=0, le=150)
    number_outpatient: int = Field(ge=0, le=200, description="Outpatient visits, previous year")
    number_emergency: int = Field(ge=0, le=200, description="Emergency visits, previous year")
    number_inpatient: int = Field(ge=0, le=200, description="Inpatient stays, previous year")
    number_diagnoses: int = Field(ge=1, le=30)

    diag_1: str | None = Field(None, description="Primary diagnosis, ICD-9 code (e.g. 250.83)")
    diag_2: str | None = Field(None, description="Secondary diagnosis, ICD-9 code")
    diag_3: str | None = Field(None, description="Additional diagnosis, ICD-9 code")
    payer_code: str | None = None
    medical_specialty: str | None = None

    max_glu_serum: Literal[">200", ">300", "Norm", "None"] = "None"
    A1Cresult: Literal[">7", ">8", "Norm", "None"] = "None"
    change: Literal["Ch", "No"] = "No"
    diabetesMed: Literal["Yes", "No"] = "No"

    insulin: Drug = "No"
    metformin: Drug = "No"
    repaglinide: Drug = "No"
    nateglinide: Drug = "No"
    glimepiride: Drug = "No"
    glipizide: Drug = "No"
    glyburide: Drug = "No"
    pioglitazone: Drug = "No"
    rosiglitazone: Drug = "No"
    acarbose: Drug = "No"
    glyburide_metformin: Drug = Field("No", alias="glyburide-metformin")


class Prediction(BaseModel):
    readmission_probability: float = Field(description="Predicted probability, 0 to 1")
    high_risk: bool = Field(description="True if in the 10% of patients with the highest risk")
    threshold: float = Field(description="Probability above which a patient counts as high risk")


class Health(BaseModel):
    status: str
    model_loaded: bool


def create_app(artifact: dict | None = None) -> FastAPI:
    """Build the app. Without an `artifact`, the saved model is loaded when the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.artifact is None:
            app.state.artifact = load_artifact(config.MODEL_PATH)
        yield

    app = FastAPI(
        title="Readmission risk API",
        version="0.1.0",
        description=(
            "Probability that a diabetic patient is readmitted within 30 days of discharge. "
            "Educational project, **not a clinical tool**."
        ),
        lifespan=lifespan,
    )
    app.state.artifact = artifact

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        return Health(status="ok", model_loaded=request.app.state.artifact is not None)

    @app.post(
        "/predict",
        response_model=Prediction,
        responses={422: {"description": "Invalid input, or a discharge the model does not cover"}},
    )
    def predict(patient: PatientInput, request: Request) -> Prediction:
        try:
            result = predict_one(request.app.state.artifact, patient.model_dump(by_alias=True))
        except OutOfScopeError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return Prediction(**result)

    return app
