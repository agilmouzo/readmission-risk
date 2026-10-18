"""FastAPI service: 30-day readmission risk for one diabetic patient.

Run it (after `python -m readmission.train`):

    uvicorn readmission.api:create_app --factory

Interactive documentation, with a ready-made example, is served at /docs.
Educational project: NOT a clinical tool.
"""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from readmission import codes, config, model_download, schema
from readmission.artifact import load_artifact
from readmission.service import OutOfScopeError, predict_one

Drug = Literal["No", "Steady", "Up", "Down"]
DRUG_HELP = "Drug status: No = not prescribed, Steady = same dose, Up/Down = dose changed"

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


def _codes_help(what: str, table: dict[int, str], common: list[int]) -> str:
    shown = "; ".join(f"{i} = {table[i]}" for i in common)
    return f"{what}. Common values: {shown}. All values: GET /codes."


class PatientInput(BaseModel):
    """One hospital encounter, with the fields the model uses (names as in the original dataset).

    Optional fields can be left out: diagnoses, payer and specialty default to "unknown",
    lab tests to "None" and every medication to "No".
    """

    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", json_schema_extra={"examples": [EXAMPLE]}
    )

    age: int = Field(ge=0, le=120, description="Age in years")
    gender: Literal["Female", "Male"] = Field(description="Patient gender as recorded")
    admission_type_id: int = Field(
        ge=1,
        le=8,
        description=_codes_help("How the patient was admitted", codes.ADMISSION_TYPE, [1, 2, 3]),
    )
    discharge_disposition_id: int = Field(
        ge=1,
        le=30,
        description=_codes_help(
            "Where the patient went after discharge", codes.DISCHARGE_DISPOSITION, [1, 3, 6]
        )
        + " Codes 11, 13, 14, 19, 20, 21 (death/hospice) are rejected.",
    )
    admission_source_id: int = Field(
        ge=1,
        le=26,
        description=_codes_help("Where the patient came from", codes.ADMISSION_SOURCE, [1, 4, 7]),
    )
    time_in_hospital: int = Field(ge=1, le=14, description="Length of stay, in days")
    num_lab_procedures: int = Field(ge=0, le=200, description="Lab tests done during the stay")
    num_procedures: int = Field(
        ge=0, le=20, description="Procedures (not lab tests) done during the stay"
    )
    num_medications: int = Field(ge=0, le=150, description="Distinct medications given")
    number_outpatient: int = Field(ge=0, le=200, description="Outpatient visits, previous year")
    number_emergency: int = Field(ge=0, le=200, description="Emergency visits, previous year")
    number_inpatient: int = Field(ge=0, le=200, description="Inpatient stays, previous year")
    number_diagnoses: int = Field(ge=1, le=30, description="Diagnoses entered for the stay")

    diag_1: str | None = Field(None, description="Primary diagnosis, ICD-9 code (e.g. 250.83)")
    diag_2: str | None = Field(None, description="Secondary diagnosis, ICD-9 code")
    diag_3: str | None = Field(None, description="Additional diagnosis, ICD-9 code")
    payer_code: str | None = Field(
        None, description="Insurer code as in the dataset (e.g. MC = Medicare, SP = self-pay)"
    )
    medical_specialty: str | None = Field(
        None, description="Admitting specialty as in the dataset (e.g. InternalMedicine)"
    )

    max_glu_serum: Literal[">200", ">300", "Norm", "None"] = Field(
        "None", description="Glucose serum test result; 'None' = not measured"
    )
    A1Cresult: Literal[">7", ">8", "Norm", "None"] = Field(
        "None", description="HbA1c test result (%); 'None' = not measured"
    )
    change: Literal["Ch", "No"] = Field(
        "No", description="'Ch' = diabetes medication was changed during the stay"
    )
    diabetesMed: Literal["Yes", "No"] = Field(
        "No", description="Any diabetes medication prescribed"
    )

    insulin: Drug = Field("No", description=DRUG_HELP)
    metformin: Drug = Field("No", description=DRUG_HELP)
    repaglinide: Drug = Field("No", description=DRUG_HELP)
    nateglinide: Drug = Field("No", description=DRUG_HELP)
    glimepiride: Drug = Field("No", description=DRUG_HELP)
    glipizide: Drug = Field("No", description=DRUG_HELP)
    glyburide: Drug = Field("No", description=DRUG_HELP)
    pioglitazone: Drug = Field("No", description=DRUG_HELP)
    rosiglitazone: Drug = Field("No", description=DRUG_HELP)
    acarbose: Drug = Field("No", description=DRUG_HELP)
    glyburide_metformin: Drug = Field("No", alias="glyburide-metformin", description=DRUG_HELP)


class Prediction(BaseModel):
    readmission_probability: float = Field(description="Predicted probability, 0 to 1")
    high_risk: bool = Field(
        description="True if above the high-risk threshold (set in validation to flag about 10%)"
    )
    threshold: float = Field(description="Probability above which a patient counts as high risk")
    baseline_risk: float = Field(description="Readmission rate of all patients (the reference)")
    relative_risk: float = Field(description="readmission_probability / baseline_risk")
    summary: str = Field(description="The result in one plain sentence")


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = Field(
        default=None,
        description=f"'{config.MODEL_RELEASE}' if it is the published model, 'custom' otherwise",
    )
    model_sha256: str | None = Field(
        default=None, description="SHA-256 of the model file that is loaded"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def create_app(artifact: dict | None = None) -> FastAPI:
    """Build the app. Without an `artifact`, the saved model is loaded when the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.artifact is None:
            app.state.artifact = load_artifact(config.MODEL_PATH)
            app.state.model_sha256 = _sha256(config.MODEL_PATH)
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
    app.state.model_sha256 = None  # only known when the model is loaded from a file

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/codes")
    def code_tables() -> dict:
        """Meaning of the numeric codes used by the three `*_id` fields."""
        return {
            "admission_type_id": codes.ADMISSION_TYPE,
            "discharge_disposition_id": codes.DISCHARGE_DISPOSITION,
            "admission_source_id": codes.ADMISSION_SOURCE,
            "rejected_discharge_codes": sorted(schema.EXCLUDED_DISCHARGE_CODES),
        }

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        sha = request.app.state.model_sha256
        published = sha is not None and sha == getattr(model_download, "MODEL_SHA256", None)
        return Health(
            status="ok",
            model_loaded=request.app.state.artifact is not None,
            model_version=None
            if sha is None
            else (config.MODEL_RELEASE if published else "custom"),
            model_sha256=sha,
        )

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
