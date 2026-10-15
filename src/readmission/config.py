import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "diabetic_data.csv"
# The path can be overridden (the Docker image keeps the model outside the installed package).
MODEL_PATH = Path(os.environ.get("READMISSION_MODEL_PATH", ROOT / "models" / "model.joblib"))

RANDOM_STATE = 42
TARGET_RAW = "readmitted"  # values: "<30", ">30", "NO"
TARGET = "readmitted_30d"  # binary: 1 if readmitted in < 30 days
GROUP_COL = "patient_nbr"  # split by patient to avoid leakage across encounters
MISSING_TOKEN = "?"

# Sensitive attribute: NOT used as a model feature, only for subgroup analysis.
SENSITIVE_COLS = ["race"]

# "High risk" = the 10% of patients with the highest predicted risk. The cut-off was chosen on
# out-of-fold predictions over the train split (see reports/operating_points.csv, row "top 10%")
# and is checked against that file in tests/test_api.py.
HIGH_RISK_THRESHOLD = 0.1971

# Readmission rate in the whole test set, used to put a prediction in context.
BASELINE_RISK = 0.112
