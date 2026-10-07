from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from readmission import config


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV, turning the '?' placeholder into proper NaN."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(
        path, na_values=[config.MISSING_TOKEN], keep_default_na=False, low_memory=False)


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add `readmitted_30d` = 1 when the patient came back in under 30 days."""
    out = df.copy()
    out[config.TARGET] = (out[config.TARGET_RAW] == "<30").astype(np.int8)
    return out
