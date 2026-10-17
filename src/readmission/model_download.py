"""Fetch the trained model published as a GitHub Release asset.

Used by `scripts/download_model.py` and by the Streamlit demo, so that someone can try the
project without downloading the dataset and training.

The model file is a pickle, and loading a pickle runs code, so the download is checked against a
SHA-256 digest before it is kept. If you publish a new model, update `MODEL_SHA256` (print it
with `sha256sum models/model.joblib`) or pass `READMISSION_MODEL_SHA256` for your own build.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from readmission import config

MODEL_URL = os.environ.get(
    "READMISSION_MODEL_URL",
    "https://github.com/agilmouzo/readmission-risk/releases/download/v0.1.0/model.joblib",
)
MODEL_SHA256 = os.environ.get(
    "READMISSION_MODEL_SHA256",
    "b6eb65868979b1f2dc0c65edd29a7430cf5f6b274dfd84817f4a17f5130bc933",
)


class ModelIntegrityError(OSError):
    """The downloaded file is not the published model."""


def download_model(
    target: Path = config.MODEL_PATH, url: str = MODEL_URL, sha256: str = MODEL_SHA256
) -> Path:
    """Download the model to `target` (skipped if it exists) and verify it.

    Raises OSError if the download fails and ModelIntegrityError (an OSError) if the digest differs.
    """
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, partial.open("wb") as out:
            while chunk := resp.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256.lower():
            raise ModelIntegrityError(
                f"SHA-256 mismatch: got {digest.hexdigest()}, expected {sha256}. "
                "The file was discarded."
            )
    except OSError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return target
