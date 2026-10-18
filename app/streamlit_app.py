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

from readmission import config, i18n, schema
from readmission.api import EXAMPLE
from readmission.artifact import load_artifact
from readmission.features import icd9_group
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


def code_select(
    label: str, table: dict[int, str], default: int, key: str, exclude=(), help: str | None = None
) -> int:
    options = [code for code in table if code not in exclude]
    return st.selectbox(
        label,
        options,
        index=options.index(default),
        format_func=lambda code: f"{code} - {table[code]}",
        key=key,
        help=help,
    )


def diagnosis_input(label: str, default: str, key: str, help: str, text: dict, lang: str):
    """ICD-9 box that shows, below it, the broad group the model will use."""
    value = st.text_input(label, value=default, key=key, help=help).strip()
    if not value:
        return None
    if i18n.looks_like_icd9(value):
        group = i18n.ICD9_GROUPS[icd9_group(value)][lang == "es"]
        st.caption(text["diag_group"].format(group=group))
    else:
        st.caption(text["diag_bad"])
    return value


def main() -> None:
    lang = st.session_state.get("lang", "en")
    text = i18n.TEXT[lang]
    st.set_page_config(page_title=text["page_title"], page_icon=":hospital:", layout="wide")
    title, picker = st.columns([5, 1])
    with picker:
        lang = st.radio(
            i18n.TEXT[lang]["language"],
            list(i18n.LANGUAGES),
            format_func=i18n.LANGUAGES.get,
            horizontal=True,
            key="lang",
        )
    text = i18n.TEXT[lang]
    es = lang == "es"
    with title:
        st.title(text["title"])
        st.caption(text["caption"])
    try:
        artifact = get_model(str(config.MODEL_PATH))
    except OSError:
        st.error(text["model_error"])
        st.stop()

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader(text["encounter"])
        a, b = st.columns(2)
        age = a.number_input(text["age"], 0, 120, EXAMPLE["age"], key="age")
        gender = b.selectbox(
            text["gender"],
            ["Female", "Male"],
            format_func=lambda v: text[f"gender_{v}"],
            key="gender",
        )
        admission_type = code_select(
            text["admission_type"], i18n.code_labels(lang, "admission_type"), 1, "admission_type"
        )
        admission_source = code_select(
            text["admission_source"], i18n.code_labels(lang, "source"), 7, "admission_source"
        )
        discharge = code_select(
            text["discharge"],
            i18n.code_labels(lang, "discharge"),
            1,
            "discharge",
            exclude=schema.EXCLUDED_DISCHARGE_CODES,
        )
        st.caption(text["discharge_note"])

        c1, c2, c3 = st.columns(3)
        time_in_hospital = c1.number_input(
            text["days"], 1, 14, EXAMPLE["time_in_hospital"], key="days"
        )
        num_lab = c2.number_input(text["labs"], 0, 200, EXAMPLE["num_lab_procedures"], key="labs")
        num_proc = c3.number_input(
            text["procedures"], 0, 20, EXAMPLE["num_procedures"], key="procedures"
        )
        c1, c2, c3 = st.columns(3)
        num_med = c1.number_input(
            text["medications"], 0, 150, EXAMPLE["num_medications"], key="medications"
        )
        num_diag = c2.number_input(
            text["diagnoses"],
            1,
            30,
            EXAMPLE["number_diagnoses"],
            key="diagnoses",
            help=text["diagnoses_help"],
        )
        c3.empty()
        st.markdown(text["visits"])
        c1, c2, c3 = st.columns(3)
        outpatient = c1.number_input(
            text["outpatient"],
            0,
            200,
            EXAMPLE["number_outpatient"],
            key="outpatient",
            help=text["outpatient_help"],
        )
        emergency = c2.number_input(
            text["emergency"],
            0,
            200,
            EXAMPLE["number_emergency"],
            key="emergency",
            help=text["emergency_help"],
        )
        inpatient = c3.number_input(
            text["inpatient"],
            0,
            200,
            EXAMPLE["number_inpatient"],
            key="inpatient",
            help=text["inpatient_help"],
        )

        st.markdown(text["diag_title"], help=text["diag_help"])
        c1, c2, c3 = st.columns(3)
        with c1:
            diag_1 = diagnosis_input(
                text["diag_primary"],
                EXAMPLE["diag_1"],
                "diag_1",
                text["diag_primary_help"],
                text,
                lang,
            )
        with c2:
            diag_2 = diagnosis_input(
                text["diag_secondary"],
                EXAMPLE["diag_2"],
                "diag_2",
                text["diag_other_help"],
                text,
                lang,
            )
        with c3:
            diag_3 = diagnosis_input(
                text["diag_additional"],
                EXAMPLE["diag_3"],
                "diag_3",
                text["diag_other_help"],
                text,
                lang,
            )

        st.markdown(text["tests_title"])
        c1, c2 = st.columns(2)
        a1c = c1.selectbox(
            text["a1c"],
            ["None", "Norm", ">7", ">8"],
            index=3,
            format_func=lambda v: {
                "None": text["not_measured"],
                "Norm": text["normal"],
            }.get(v, v),
            key="a1c",
            help=text["a1c_help"],
        )
        glucose = c2.selectbox(
            text["glucose"],
            ["None", "Norm", ">200", ">300"],
            format_func=lambda v: {
                "None": text["not_measured"],
                "Norm": text["normal"],
            }.get(v, v),
            key="glucose",
            help=text["glucose_help"],
        )
        c1, c2 = st.columns(2)
        diabetes_med = c1.selectbox(
            text["diabetes_med"],
            ["Yes", "No"],
            format_func=lambda v: text[v.lower()],
            key="diabetes_med",
        )
        change = c2.selectbox(
            text["change"],
            ["Ch", "No"],
            format_func=lambda v: {"Ch": text["yes"], "No": text["no"]}[v],
            key="change",
            help=text["change_help"],
        )
        c1, c2 = st.columns(2)
        insulin = c1.selectbox(
            text["insulin"],
            DRUGS,
            index=1,
            format_func=lambda v: text[f"drug_{v}"],
            key="insulin",
            help=text["drug_help"],
        )
        metformin = c2.selectbox(
            text["metformin"],
            DRUGS,
            index=1,
            format_func=lambda v: text[f"drug_{v}"],
            key="metformin",
            help=text["drug_help"],
        )
        other = {}
        with st.expander(text["other_drugs"]):
            for name in OTHER_DRUGS:
                other[name] = st.selectbox(
                    i18n.DRUGS_ES[name] if es else name,
                    DRUGS,
                    format_func=lambda v: text[f"drug_{v}"],
                    key=name,
                )
        with st.expander(text["insurance_title"]):
            payer = st.selectbox(
                text["payer"],
                [None, *i18n.PAYERS],
                index=list(i18n.PAYERS).index(EXAMPLE["payer_code"]) + 1,
                format_func=lambda v: (
                    text["not_recorded"]
                    if v is None
                    else v
                    if i18n.PAYERS[v][es] == v
                    else f"{v} - {i18n.PAYERS[v][es]}"
                ),
                key="payer",
                help=text["payer_help"],
            )
            names = {value: (en_name, es_name)[es] for value, en_name, es_name in i18n.SPECIALTIES}
            specialty = st.selectbox(
                text["specialty"],
                [None, *names],
                index=list(names).index(EXAMPLE["medical_specialty"]) + 1,
                format_func=lambda v: text["not_recorded"] if v is None else names[v],
                key="specialty",
                help=text["specialty_help"],
            )

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
        st.subheader(text["result"])
        st.metric(
            text["chance"],
            i18n.decimal(lang, f"{result['readmission_probability']:.1%}"),
            i18n.decimal(
                lang,
                text["vs_average"].format(
                    ratio=result["relative_risk"], baseline=result["baseline_risk"]
                ),
            ),
            delta_color="off",
        )
        st.progress(min(result["readmission_probability"], 1.0))
        if result["high_risk"]:
            st.warning(text["high"])
        else:
            st.success(text["low"])
        st.write(i18n.summary(lang, result))
        st.caption(text["disclaimer"])


main()
