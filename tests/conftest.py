import numpy as np
import pandas as pd
import pytest

from readmission.data import add_binary_target


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A tiny synthetic frame with the same columns/formats as the raw dataset."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": rng.integers(0, 120, n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", np.nan], n),
            "gender": rng.choice(["Male", "Female"], n),
            "age": rng.choice(["[0-10)", "[20-30)", "[50-60)", "[70-80)"], n),
            "weight": np.nan,
            "admission_type_id": rng.choice([1, 2, 3, 5, 6], n),
            "discharge_disposition_id": rng.choice([1, 3, 6, 11, 13, 18], n),
            "admission_source_id": rng.choice([1, 4, 7, 9, 17], n),
            "time_in_hospital": rng.integers(1, 15, n),
            "payer_code": rng.choice(["MC", "HM", np.nan], n),
            "medical_specialty": rng.choice(["Cardiology", "Surgery-General", np.nan], n),
            "num_lab_procedures": rng.integers(1, 100, n),
            "num_procedures": rng.integers(0, 6, n),
            "num_medications": rng.integers(1, 40, n),
            "number_outpatient": rng.integers(0, 5, n),
            "number_emergency": rng.integers(0, 5, n),
            "number_inpatient": rng.integers(0, 15, n),
            "diag_1": rng.choice(["250.83", "428", "486", "V57", "E909", "715"], n),
            "diag_2": rng.choice(["276", "585", np.nan], n),
            "diag_3": rng.choice(["401", "530", np.nan], n),
            "number_diagnoses": rng.integers(1, 17, n),
            "max_glu_serum": rng.choice(["None", ">200", "Norm"], n),
            "A1Cresult": rng.choice(["None", ">8", "Norm"], n),
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "examide": "No",
            "change": rng.choice(["Ch", "No"], n),
            "diabetesMed": rng.choice(["Yes", "No"], n),
            "readmitted": rng.choice(["<30", ">30", "NO"], n, p=[0.11, 0.35, 0.54]),
        }
    )
    return add_binary_target(df)
