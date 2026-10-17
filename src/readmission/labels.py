"""Human-readable names for the model variables and their coded values, for figures.

The tables in `reports/` keep the dataset's column names so that they can be traced back to the
data; only the figures use these labels.
"""

from __future__ import annotations

from readmission import codes, schema

VARIABLES = {
    "number_inpatient": "Previous inpatient stays",
    "number_emergency": "Previous emergency visits",
    "number_outpatient": "Previous outpatient visits",
    "prior_visits": "Total previous visits",
    "discharge_disposition_id": "Discharge destination",
    "admission_type_id": "Admission type",
    "admission_source_id": "Admission source",
    "number_diagnoses": "Number of diagnoses",
    "age_ord": "Age",
    "diag_1_group": "Primary diagnosis group",
    "diag_2_group": "Secondary diagnosis group",
    "diag_3_group": "Additional diagnosis group",
    "payer_code": "Insurance (payer code)",
    "medical_specialty": "Medical specialty",
    "diabetesMed": "Diabetes medication prescribed",
    "change": "Medication changed",
    "num_medications": "Number of medications",
    "num_lab_procedures": "Lab tests",
    "num_procedures": "Procedures",
    "time_in_hospital": "Days in hospital",
    "A1Cresult": "HbA1c result",
    "max_glu_serum": "Glucose serum result",
    "gender": "Gender",
    "insulin": "Insulin",
    "metformin": "Metformin",
    "glimepiride": "Glimepiride",
    "glyburide": "Glyburide",
    "glipizide": "Glipizide",
    "pioglitazone": "Pioglitazone",
    "rosiglitazone": "Rosiglitazone",
    "repaglinide": "Repaglinide",
    "nateglinide": "Nateglinide",
    "acarbose": "Acarbose",
    "glyburide-metformin": "Glyburide-metformin",
}

_CODE_TABLES = {
    "discharge_disposition_id": codes.DISCHARGE_DISPOSITION,
    "admission_type_id": codes.ADMISSION_TYPE,
    "admission_source_id": codes.ADMISSION_SOURCE,
}

MAX_VALUE_LENGTH = 30


def variable_label(variable: str) -> str:
    """Readable name of a variable; unknown names are returned with spaces for underscores."""
    return VARIABLES.get(variable, variable.replace("_", " "))


def _shorten(text: str) -> str:
    text = text.replace("Discharged/transferred", "Transferred")
    return text if len(text) <= MAX_VALUE_LENGTH else text[: MAX_VALUE_LENGTH - 1].rstrip() + "…"


def value_label(variable: str, value) -> str:
    """Readable value of a variable: decoded IDs, ages as decades, whole numbers without '.0'."""
    if value is None or value == "":
        return ""
    if variable == "age_ord":
        decade = int(value)
        return (
            f"under {(schema.MIN_AGE_ORD + 1) * 10}"
            if decade <= schema.MIN_AGE_ORD
            else f"{decade * 10}s"
        )
    if variable in _CODE_TABLES:
        try:
            return _shorten(_CODE_TABLES[variable][int(str(value))])
        except (KeyError, ValueError):
            return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _shorten(str(value))
