from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "diabetic_data.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"

RANDOM_STATE = 42
TARGET_RAW = "readmitted"      # values: "<30", ">30", "NO"
TARGET = "readmitted_30d"      # binary: 1 if readmitted in < 30 days
GROUP_COL = "patient_nbr"      # split by patient to avoid leakage across encounters
MISSING_TOKEN = "?"

# Sensitive attribute: NOT used as a model feature, only for subgroup analysis.
SENSITIVE_COLS = ["race"]
