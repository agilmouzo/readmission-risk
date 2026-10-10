"""The saved model and the exact input columns it expects.

`get_feature_columns` works by exclusion (every column that is not on the banned list becomes a
feature), so any column added to a DataFrame later, such as predictions, would silently turn into
a feature. Whoever consumes the model (evaluation, API) must therefore select its inputs through
the column lists stored with the model, never by recomputing them.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from readmission import config

REQUIRED_KEYS = ("pipeline", "numeric", "categorical", "target")


def load_artifact(path: Path = config.MODEL_PATH) -> dict:
    """Load the dict written by `readmission.train` (pipeline + column lists + target name)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m readmission.train` first.")
    artifact = joblib.load(path)
    missing = [key for key in REQUIRED_KEYS if key not in artifact]
    if missing:
        raise ValueError(f"Model file is missing keys {missing}; retrain with `readmission.train`")
    return artifact


def model_inputs(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Select exactly the columns the model was trained on, in the same order."""
    columns = list(artifact["numeric"]) + list(artifact["categorical"])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"The DataFrame lacks columns the model needs: {missing}")
    return df[columns]
