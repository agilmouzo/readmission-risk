"""Download the trained model from the GitHub Release into models/model.joblib.

Lets you try the API or the demo without downloading the dataset and training.
The model was trained with the library versions pinned in constraints.txt, so install the
project in a clean environment (`pip install -e .`) before using it.

Usage:  python scripts/download_model.py
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

URL = os.environ.get(
    "READMISSION_MODEL_URL",
    "https://github.com/agilmouzo/readmission-risk/releases/download/v0.1.0/model.joblib",
)
TARGET = Path(
    os.environ.get(
        "READMISSION_MODEL_PATH",
        Path(__file__).resolve().parents[1] / "models" / "model.joblib",
    )
)


def main() -> None:
    if TARGET.exists():
        print(f"Already downloaded: {TARGET}")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    partial = TARGET.with_suffix(".part")
    try:
        with urllib.request.urlopen(URL, timeout=120) as resp, partial.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"Download failed ({error}).\nYou can train the model yourself instead: "
            "python scripts/download_data.py && python -m readmission.train"
        ) from error
    partial.replace(TARGET)
    print(f"Saved {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
