"""Download the trained model from the GitHub Release into models/model.joblib.

Lets you try the API or the demo without downloading the dataset and training.
The model was trained with the library versions pinned in constraints.txt, so install the
project in a clean environment (`pip install -e .`) before using it.

Usage:  python scripts/download_model.py
"""

from __future__ import annotations

from readmission import config
from readmission.model_download import MODEL_URL, download_model


def main() -> None:
    if config.MODEL_PATH.exists():
        print(f"Already downloaded: {config.MODEL_PATH}")
        return
    print(f"Downloading {MODEL_URL} ...")
    try:
        path = download_model()
    except OSError as error:
        raise SystemExit(
            f"Download failed ({error}).\nYou can train the model yourself instead: "
            "python scripts/download_data.py && python -m readmission.train"
        ) from error
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
