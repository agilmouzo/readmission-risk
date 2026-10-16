"""Streamlit demo: fill in one hospital encounter and see the 30-day readmission risk.

Run it (needs models/model.joblib: `python scripts/download_model.py`):

    streamlit run app/streamlit_app.py

It calls the same `predict_one` as the API. Educational project: NOT a clinical tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Lets the app run from a plain checkout (Streamlit Community Cloud), without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from readmission import codes, config, schema
from readmission.api import EXAMPLE
from readmission.artifact import load_artifact
from readmission.model_download import download_model
from readmission.service import predict_one

DRUGS = ["No", "Steady", "Up", "Down"]
OTHER_DRUGS = [
    "repaglinide",
    "nateglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "glyburide-metformin",
]


@st.cache_resource
def get_model(path: str) -> dict:
    """Load the model, downloading it from the GitHub Release the first time."""
    target = Path(path)
    if not target.exists():
        download_model(target)
    return load_artifact(target)


def code_select(label: str, table: dict[int, str], default: int, exclude=()) -> int:
    options = [code for code in table if code not in exclude]
    return st.selectbox(
        label,
        options,
        index=options.index(default),
        format_func=lambda code: f"{code} - {table[code]}",
    )


def optional_text(label: str, default: str = "", help: str | None = None) -> str | None:
    return st.text_input(label, value=default, help=help).strip() or None


def main() -> None:
    st.set_page_config(page_title="Readmission risk", page_icon=":hospital:", layout="wide")
    st.title("30-day readmission risk, diabetic patients")
    st.caption(
        "Educational project, **not a clinical tool**. Model trained on US hospital data "
        "(1999-2008); see the repository for its limits."
    )
    try:
        artifact = get_model(str(config.MODEL_PATH))
    except OSError:
        st.error(
            "Could not get the model. Run `python scripts/download_model.py` "
            "(or train it) and reload this page."
        )
        st.stop()

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("The encounter")
        a, b = st.columns(2)
        age = a.number_input("Age (years)", 0, 120, EXAMPLE["age"])
        gender = b.selectbox("Gender", ["Female", "Male"])
        admission_type = code_select("How was the patient admitted?", codes.ADMISSION_TYPE, 1)
        admission_source = code_select(
            "Where did the patient come from?", codes.ADMISSION_SOURCE, 7
        )
        discharge = code_select(
            "Where did the patient go after discharge?",
            codes.DISCHARGE_DISPOSITION,
            1,
            exclude=schema.EXCLUDED_DISCHARGE_CODES,
        )
        st.caption("Discharges to death or hospice are not offered: the model does not cover them.")

        c1, c2, c3 = st.columns(3)
        time_in_hospital = c1.number_input("Days in hospital", 1, 14, EXAMPLE["time_in_hospital"])
        num_lab = c2.number_input("Lab tests", 0, 200, EXAMPLE["num_lab_procedures"])
        num_proc = c3.number_input("Procedures", 0, 20, EXAMPLE["num_procedures"])
        c1, c2, c3 = st.columns(3)
        num_med = c1.number_input("Medications", 0, 150, EXAMPLE["num_medications"])
        num_diag = c2.number_input("Diagnoses", 1, 30, EXAMPLE["number_diagnoses"])
        c3.empty()
        st.markdown("**Visits in the previous year**")
        c1, c2, c3 = st.columns(3)
        outpatient = c1.number_input("Outpatient", 0, 200, EXAMPLE["number_outpatient"])
        emergency = c2.number_input("Emergency", 0, 200, EXAMPLE["number_emergency"])
        inpatient = c3.number_input("Inpatient", 0, 200, EXAMPLE["number_inpatient"])

        st.markdown("**Diagnoses (ICD-9 codes, optional)**")
        c1, c2, c3 = st.columns(3)
        with c1:
            diag_1 = optional_text("Primary", EXAMPLE["diag_1"])
        with c2:
            diag_2 = optional_text("Secondary", EXAMPLE["diag_2"])
        with c3:
            diag_3 = optional_text("Additional", EXAMPLE["diag_3"])

        st.markdown("**Tests and diabetes treatment**")
        c1, c2 = st.columns(2)
        a1c = c1.selectbox("HbA1c result", ["None", "Norm", ">7", ">8"], index=3)
        glucose = c2.selectbox("Glucose serum result", ["None", "Norm", ">200", ">300"])
        c1, c2 = st.columns(2)
        diabetes_med = c1.selectbox("Diabetes medication prescribed?", ["Yes", "No"])
        change = c2.selectbox(
            "Medication changed during the stay?",
            ["Ch", "No"],
            format_func=lambda v: {"Ch": "Yes", "No": "No"}[v],
        )
        c1, c2 = st.columns(2)
        insulin = c1.selectbox("Insulin", DRUGS, index=1)
        metformin = c2.selectbox("Metformin", DRUGS, index=1)
        other = {}
        with st.expander("Other diabetes drugs"):
            for name in OTHER_DRUGS:
                other[name] = st.selectbox(name, DRUGS, key=name)
        with st.expander("Insurance and specialty (optional)"):
            payer = optional_text("Payer code (e.g. MC = Medicare)", EXAMPLE["payer_code"])
            specialty = optional_text("Medical specialty", EXAMPLE["medical_specialty"])

    patient = {
        "age": int(age),
        "gender": gender,
        "admission_type_id": admission_type,
        "discharge_disposition_id": discharge,
        "admission_source_id": admission_source,
        "time_in_hospital": int(time_in_hospital),
        "num_lab_procedures": int(num_lab),
        "num_procedures": int(num_proc),
        "num_medications": int(num_med),
        "number_outpatient": int(outpatient),
        "number_emergency": int(emergency),
        "number_inpatient": int(inpatient),
        "number_diagnoses": int(num_diag),
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "payer_code": payer,
        "medical_specialty": specialty,
        "max_glu_serum": glucose,
        "A1Cresult": a1c,
        "change": change,
        "diabetesMed": diabetes_med,
        "insulin": insulin,
        "metformin": metformin,
        **other,
    }
    result = predict_one(artifact, patient)

    with right:
        st.subheader("Result")
        st.metric(
            "Chance of readmission within 30 days",
            f"{result['readmission_probability']:.1%}",
            f"{result['relative_risk']:.1f}x the {result['baseline_risk']:.1%} average",
            delta_color="off",
        )
        st.progress(min(result["readmission_probability"], 1.0))
        if result["high_risk"]:
            st.warning("In the 10% of patients with the highest predicted risk.")
        else:
            st.success("Not in the 10% of patients with the highest predicted risk.")
        st.write(result["summary"])
        st.caption(
            "The model separates patients only modestly (AUC-ROC about 0.68): use it to "
            "explore, not to decide."
        )


main()
