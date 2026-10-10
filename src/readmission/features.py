"""Row exclusions, derived variables and the scikit-learn preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from readmission import schema


def icd9_group(code: object) -> str:
    """Map an ICD-9 diagnosis code to a broad clinical group (~800 codes -> 9 groups)."""
    if pd.isna(code):
        return "Unknown"
    text = str(code)
    if text.startswith(("V", "E")):
        return "Other"
    try:
        n = int(float(text))
    except ValueError:
        return "Other"
    if n == 250:
        return "Diabetes"
    if 390 <= n <= 459 or n == 785:
        return "Circulatory"
    if 460 <= n <= 519 or n == 786:
        return "Respiratory"
    if 520 <= n <= 579 or n == 787:
        return "Digestive"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasms"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 800 <= n <= 999:
        return "Injury"
    return "Other"


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature columns, excluding IDs, targets and sensitive data."""
    numeric = [c for c in schema.NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in df.columns if c not in numeric and c not in schema.NON_FEATURE_COLS]
    return numeric, categorical


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules decided in the EDA. Does not modify the input.

    ID and sensitive columns (`patient_nbr`, `race`) are kept in the output: they are needed
    for the patient-level split and the subgroup analysis, but they are not features.
    """
    out = df[~df["discharge_disposition_id"].isin(schema.EXCLUDED_DISCHARGE_CODES)].copy()

    # ID columns are categories, not numbers; unmapped codes become "Unknown".
    for col, codes in schema.UNKNOWN_CODES.items():
        out[col] = np.where(out[col].isin(codes), "Unknown", out[col].astype(str))

    # Prior-visit total, computed from the raw counts before they are capped.
    out["prior_visits"] = out[["number_outpatient", "number_emergency", "number_inpatient"]].sum(
        axis=1
    )

    for col, upper in schema.CAPS.items():
        out[col] = out[col].clip(upper=upper)

    decade = out["age"].str.extract(r"\[(\d+)-", expand=False).astype(int) // 10
    out["age_ord"] = decade.clip(lower=schema.MIN_AGE_ORD)

    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = out[f"diag_{i}"].map(icd9_group)

    _, categorical = get_feature_columns(out)
    out[categorical] = out[categorical].fillna("Unknown")
    return out


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    min_frequency: int = 50,
    log_numeric: bool = False,
) -> ColumnTransformer:
    """Impute + scale numerics, one-hot encode categoricals.

    With `log_numeric=True`, `log1p` is applied before scaling. It tames the right-skewed counts
    (see the EDA) for linear models; tree models do not need it. All numeric features are >= 0.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if log_numeric:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(steps)
    categorical_enc = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=False,
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_enc, categorical)]
    )
    return preprocessor.set_output(transform="pandas")
