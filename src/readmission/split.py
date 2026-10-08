"""Train/test split by patient, so that no patient appears in both sets."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from readmission import config


def patient_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = config.RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold-out split grouped by `patient_nbr`."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[config.GROUP_COL]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
