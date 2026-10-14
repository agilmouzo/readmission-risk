"""Score one patient: the logic behind the API, with no web code in it.

The patient goes through the SAME cleaning (`prepare`) and the SAME column selection
(`model_inputs`) as during training, so the API cannot drift from the model. Nothing is
re-implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from readmission import config, schema
from readmission.artifact import model_inputs
from readmission.features import prepare

OUT_OF_SCOPE_MESSAGE = (
    "Discharges to death or hospice (discharge_disposition_id in {codes}) are outside the scope "
    "of this model: those patients cannot be readmitted, so they were excluded from training."
)


class OutOfScopeError(ValueError):
    """The request is well formed, but the model was not built for this kind of patient."""


def age_bracket(age_years: int) -> str:
    """Age in years to the dataset's decade bracket: 72 -> '[70-80)'. 90 and over share one."""
    decade = min(max(int(age_years), 0) // 10, 9)
    return f"[{decade * 10}-{decade * 10 + 10})"


def build_frame(patient: Mapping[str, Any]) -> pd.DataFrame:
    """One-row DataFrame with the raw columns that `prepare` expects."""
    row = dict(patient)
    row["age"] = age_bracket(row["age"])
    return pd.DataFrame([row])


def predict_one(
    artifact: dict, patient: Mapping[str, Any], threshold: float = config.HIGH_RISK_THRESHOLD
) -> dict[str, Any]:
    """Probability of readmission within 30 days, and whether it is in the high-risk group."""
    if patient["discharge_disposition_id"] in schema.EXCLUDED_DISCHARGE_CODES:
        raise OutOfScopeError(OUT_OF_SCOPE_MESSAGE.format(codes=schema.EXCLUDED_DISCHARGE_CODES))
    frame = prepare(build_frame(patient))
    if frame.empty:  # defensive: `prepare` drops excluded rows
        raise OutOfScopeError("This patient is outside the scope of the model.")
    probability = float(artifact["pipeline"].predict_proba(model_inputs(frame, artifact))[0, 1])
    return {
        "readmission_probability": probability,
        "high_risk": probability >= threshold,
        "threshold": threshold,
    }
