"""Download the UCI 'Diabetes 130-US hospitals (1999-2008)' dataset into data/raw/.

Source: https://archive.ics.uci.edu/dataset/296
Usage:  python scripts/download_data.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "diabetic_data.csv"
    if target.exists():
        print(f"Already downloaded: {target}")
        return
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(RAW_DIR)
    print(f"Extracted to {RAW_DIR}: {sorted(p.name for p in RAW_DIR.iterdir())}")


if __name__ == "__main__":
    main()
