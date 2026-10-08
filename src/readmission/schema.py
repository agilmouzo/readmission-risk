"""Dataset knowledge: which columns play which role, and the cleaning rules from the EDA."""

# Discharge codes for death / hospice (checked against IDS_mapping.csv).
# These patients cannot meaningfully be readmitted, so they are excluded.
EXCLUDED_DISCHARGE_CODES = [11, 13, 14, 19, 20, 21]

# "Not available / NULL / Not mapped" codes per ID column. Grouped as "Unknown".
UNKNOWN_CODES = {
    "admission_type_id": [5, 6, 8],
    "discharge_disposition_id": [18, 25, 26],
    "admission_source_id": [9, 15, 17, 20, 21],
}

# Sparse high counts are capped (see EDA: rates above these values are noise).
CAPS = {"number_inpatient": 8, "number_diagnoses": 10}

# Ages under 30 are merged into a single group (very few patients).
MIN_AGE_ORD = 2  # age_ord = decade index: [0-10) -> 0, [10-20) -> 1, [20-30) -> 2, ...

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "prior_visits",
    "age_ord",
]

# Medication columns where >= 99.9% of rows share one value (checked on the cleaned data).
LOW_INFO_COLS = [
    "examide",
    "citoglipton",
    "metformin-pioglitazone",
    "acetohexamide",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "troglitazone",
    "glipizide-metformin",
    "tolbutamide",
    "miglitol",
    "tolazamide",
    "chlorpropamide",
]

# Never used as model features.
NON_FEATURE_COLS = [
    "encounter_id",   # row identifier
    "patient_nbr",    # used only to split by patient
    "weight",         # ~97% missing
    "race",           # sensitive: subgroup analysis only
    "age",            # replaced by age_ord
    "diag_1",         # replaced by diag_N_group
    "diag_2",
    "diag_3",
    "readmitted",     # raw target: using it as a feature would be leakage
    "readmitted_30d", # binary target
    *LOW_INFO_COLS,
]
