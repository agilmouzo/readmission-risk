"""Fetch the trained model published as a GitHub Release asset.

Used by `scripts/download_model.py` and by the Streamlit demo, so that someone can try the
project without downloading the dataset and training.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from readmission import config

MODEL_URL = os.environ.get(
    "READMISSION_MODEL_URL",
    "https://github.com/agilmouzo/readmission-risk/releases/download/v0.1.0/model.joblib",
)


def download_model(target: Path = config.MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the model to `target` (skipped if it exists). Raises OSError on failure."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, partial.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except OSError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return target
