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


def summarise(probability: float, high_risk: bool) -> str:
    """One plain-language sentence for the person reading the response."""
    ratio = probability / config.BASELINE_RISK
    average = f"{config.BASELINE_RISK:.0%} average"
    if ratio >= 1.15:
        comparison = f"{ratio:.1f}x the {average}"
    elif ratio <= 0.85:
        comparison = f"{1 / ratio:.1f}x below the {average}"
    else:
        comparison = f"close to the {average}"
    flag = "above" if high_risk else "below"
    return (
        f"Estimated {probability:.0%} chance of readmission within 30 days ({comparison}); "
        f"{flag} the high-risk threshold."
    )


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
    high_risk = probability >= threshold
    return {
        "readmission_probability": probability,
        "high_risk": high_risk,
        "threshold": threshold,
        "baseline_risk": config.BASELINE_RISK,
        "relative_risk": probability / config.BASELINE_RISK,
        "summary": summarise(probability, high_risk),
    }
