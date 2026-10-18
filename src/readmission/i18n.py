"""Texts of the Streamlit demo in English and Spanish, and the Spanish names of the dataset codes.

The model and the API are unchanged by the language: only what the demo shows is translated.
"""

from __future__ import annotations

import re

from readmission import codes

LANGUAGES = {"en": "English", "es": "Español"}

# Spanish names for the codes of the dataset. The "not recorded" codes (NULL, Not Available...)
# are not listed here: they are built from the English label in `code_labels`.
_ADMISSION_TYPE_ES = {
    1: "Urgencia",
    2: "Urgente",
    3: "Programado (electivo)",
    4: "Recién nacido",
    7: "Centro de trauma",
}

_DISCHARGE_ES = {
    1: "Alta a domicilio",
    2: "Traslado a otro hospital de agudos",
    3: "Alta/traslado a centro de enfermería especializada (SNF)",
    4: "Alta/traslado a centro de cuidados intermedios (ICF)",
    5: "Alta/traslado a otro tipo de institución de hospitalización",
    6: "Alta a domicilio con atención sanitaria domiciliaria",
    7: "Alta voluntaria (contra consejo médico)",
    8: "Alta a domicilio con tratamiento intravenoso domiciliario",
    9: "Ingresado como paciente hospitalizado en este mismo hospital",
    10: "Neonato trasladado a otro hospital para cuidados neonatales",
    11: "Fallecido",
    12: "Sigue ingresado o volverá para servicios ambulatorios",
    13: "Cuidados paliativos (hospice) / domicilio",
    14: "Cuidados paliativos (hospice) / centro sanitario",
    15: "Traslado dentro del centro a una cama «swing bed» aprobada por Medicare",
    16: "Alta/derivación a otra institución para servicios ambulatorios",
    17: "Alta/derivación a este centro para servicios ambulatorios",
    19: "Fallecido en domicilio (solo Medicaid, hospice)",
    20: "Fallecido en centro sanitario (solo Medicaid, hospice)",
    21: "Fallecido, lugar desconocido (solo Medicaid, hospice)",
    22: "Traslado a centro de rehabilitación (incluye unidades de rehabilitación de un hospital)",
    23: "Traslado a hospital de cuidados de larga duración",
    24: "Traslado a centro de enfermería certificado por Medicaid pero no por Medicare",
    27: "Traslado a un centro sanitario federal",
    28: "Traslado a un hospital psiquiátrico o a una unidad psiquiátrica de un hospital",
    29: "Traslado a un hospital de acceso crítico (CAH)",
    30: "Traslado a otro tipo de institución sanitaria no definida en otro lugar",
}

_ADMISSION_SOURCE_ES = {
    1: "Derivación de un médico",
    2: "Derivación de una clínica",
    3: "Derivación de una HMO (organización de mantenimiento de la salud)",
    4: "Traslado desde un hospital",
    5: "Traslado desde un centro de enfermería especializada (SNF)",
    6: "Traslado desde otro centro sanitario",
    7: "Urgencias",
    8: "Tribunal / fuerzas del orden",
    10: "Traslado desde un hospital de acceso crítico",
    11: "Parto normal",
    12: "Parto prematuro",
    13: "Bebé enfermo",
    14: "Nacimiento extrahospitalario",
    18: "Traslado desde otra agencia de atención domiciliaria",
    19: "Reingreso en la misma agencia de atención domiciliaria",
    22: "Traslado desde hospitalización del mismo centro, con facturación separada",
    23: "Nacido en este hospital",
    24: "Nacido fuera de este hospital",
    25: "Traslado desde un centro de cirugía ambulatoria",
    26: "Traslado desde un hospicio",
}

_SPANISH = {
    "admission_type": (codes.ADMISSION_TYPE, _ADMISSION_TYPE_ES),
    "discharge": (codes.DISCHARGE_DISPOSITION, _DISCHARGE_ES),
    "source": (codes.ADMISSION_SOURCE, _ADMISSION_SOURCE_ES),
}


def code_labels(lang: str, which: str) -> dict[int, str]:
    """Names of the codes of one table ("admission_type", "discharge" or "source")."""
    english, spanish = _SPANISH[which]
    if lang != "es":
        return english
    labels = {}
    for code, name in english.items():
        if code in spanish:
            labels[code] = spanish[code]
        else:  # "Unknown (recorded as 'NULL' in the dataset)", or the bare "NULL"
            found = re.search(r"'([^']+)'", name)
            original = found.group(1) if found else name
            labels[code] = f"Desconocido (figura como '{original}' en el dataset)"
    return labels


# (value the model receives, English name, Spanish name). Only the specialties frequent enough
# in the training data to be used by the model (the rest would be treated as "other").
SPECIALTIES = [
    ("Cardiology", "Cardiology", "Cardiología"),
    ("Emergency/Trauma", "Emergency / trauma", "Urgencias / traumatología"),
    ("Endocrinology", "Endocrinology", "Endocrinología"),
    ("Family/GeneralPractice", "Family / general practice", "Medicina familiar / general"),
    ("Gastroenterology", "Gastroenterology", "Gastroenterología"),
    ("Hematology", "Hematology", "Hematología"),
    ("Hematology/Oncology", "Hematology / oncology", "Hematología / oncología"),
    ("Hospitalist", "Hospitalist", "Hospitalista (médico de planta)"),
    ("InternalMedicine", "Internal medicine", "Medicina interna"),
    ("Nephrology", "Nephrology", "Nefrología"),
    ("Neurology", "Neurology", "Neurología"),
    ("ObstetricsandGynecology", "Obstetrics and gynecology", "Obstetricia y ginecología"),
    ("Oncology", "Oncology", "Oncología"),
    ("Orthopedics", "Orthopedics", "Ortopedia"),
    ("Orthopedics-Reconstructive", "Orthopedics, reconstructive", "Ortopedia reconstructiva"),
    ("Otolaryngology", "Otolaryngology (ENT)", "Otorrinolaringología"),
    ("Pediatrics", "Pediatrics", "Pediatría"),
    ("Pediatrics-CriticalCare", "Pediatrics, critical care", "Pediatría: cuidados críticos"),
    ("Pediatrics-Endocrinology", "Pediatrics, endocrinology", "Pediatría: endocrinología"),
    (
        "PhysicalMedicineandRehabilitation",
        "Physical medicine and rehabilitation",
        "Medicina física y rehabilitación",
    ),
    ("Podiatry", "Podiatry", "Podología"),
    ("Psychiatry", "Psychiatry", "Psiquiatría"),
    ("Psychology", "Psychology", "Psicología"),
    ("Pulmonology", "Pulmonology", "Neumología"),
    ("Radiologist", "Radiology", "Radiología"),
    ("Surgery-Cardiovascular", "Surgery, cardiovascular", "Cirugía cardiovascular"),
    (
        "Surgery-Cardiovascular/Thoracic",
        "Surgery, cardiovascular / thoracic",
        "Cirugía cardiotorácica",
    ),
    ("Surgery-General", "Surgery, general", "Cirugía general"),
    ("Surgery-Neuro", "Surgery, neuro", "Neurocirugía"),
    ("Surgery-Thoracic", "Surgery, thoracic", "Cirugía torácica"),
    ("Surgery-Vascular", "Surgery, vascular", "Cirugía vascular"),
    ("Urology", "Urology", "Urología"),
]

# Payer codes the model uses. The dataset does not define them; the descriptions are the usual
# reading for US insurance and only given where it is widely agreed.
PAYERS = {
    "BC": ("Blue Cross", "Blue Cross"),
    "CH": ("CH", "CH"),
    "CM": ("CM", "CM"),
    "CP": ("CP", "CP"),
    "DM": ("DM", "DM"),
    "HM": ("HMO (health maintenance organisation)", "HMO (organización de mantenimiento)"),
    "MC": ("Medicare (public insurance, mostly 65+)", "Medicare (seguro público, sobre todo 65+)"),
    "MD": ("Medicaid (public insurance, low income)", "Medicaid (seguro público, bajos ingresos)"),
    "MP": ("MP", "MP"),
    "OG": ("OG", "OG"),
    "OT": ("OT", "OT"),
    "PO": ("PO", "PO"),
    "SP": ("Self-pay (no insurance)", "Pago privado (sin seguro)"),
    "UN": ("UN", "UN"),
    "WC": ("WC", "WC"),
}

# Spanish spelling of the generic names of the diabetes drugs.
DRUGS_ES = {
    "repaglinide": "repaglinida",
    "nateglinide": "nateglinida",
    "glimepiride": "glimepirida",
    "glipizide": "glipizida",
    "glyburide": "gliburida",
    "pioglitazone": "pioglitazona",
    "rosiglitazone": "rosiglitazona",
    "acarbose": "acarbosa",
    "glyburide-metformin": "gliburida-metformina",
}

ICD9_GROUPS = {
    "Diabetes": ("Diabetes", "Diabetes"),
    "Circulatory": ("Circulatory", "Circulatorio"),
    "Respiratory": ("Respiratory", "Respiratorio"),
    "Digestive": ("Digestive", "Digestivo"),
    "Genitourinary": ("Genitourinary", "Genitourinario"),
    "Neoplasms": ("Neoplasms (tumours)", "Neoplasias (tumores)"),
    "Musculoskeletal": ("Musculoskeletal", "Musculoesquelético"),
    "Injury": ("Injury", "Traumatismos"),
    "Other": ("Other", "Otros"),
    "Unknown": ("Not given", "Sin indicar"),
}

# Plausible ICD-9 shape: 3 digits (+ up to 2 decimals), or V/E code.
_ICD9 = re.compile(r"^(\d{3}(\.\d{1,2})?|[VE]\d{2,3}(\.\d{1,2})?)$")


def looks_like_icd9(code: str) -> bool:
    return bool(_ICD9.match(code.strip().upper()))


def decimal(lang: str, text: str) -> str:
    """Spanish writes decimals with a comma."""
    return text.replace(".", ",") if lang == "es" else text


def summary(lang: str, result: dict) -> str:
    """The sentence of the API response, in the language of the demo."""
    if lang != "es":
        return result["summary"]
    probability = result["readmission_probability"]
    baseline = result["baseline_risk"]
    ratio = result["relative_risk"]
    average = f"la media del {baseline:.0%}"
    if ratio >= 1.15:
        comparison = f"{decimal(lang, f'{ratio:.1f}')} veces {average}"
    elif ratio <= 0.85:
        comparison = f"{decimal(lang, f'{1 / ratio:.1f}')} veces por debajo de {average}"
    else:
        comparison = f"cercano a {average}"
    side = "por encima" if result["high_risk"] else "por debajo"
    return (
        f"Probabilidad estimada de reingreso en 30 días: {probability:.0%} ({comparison}); "
        f"{side} del umbral de alto riesgo."
    )


TEXT: dict[str, dict[str, str]] = {
    "en": {
        "page_title": "Readmission risk",
        "title": "30-day readmission risk, diabetic patients",
        "caption": (
            "Educational project, **not a clinical tool**. Model trained on US hospital data "
            "(1999-2008); see the repository for its limits."
        ),
        "language": "Language",
        "model_error": (
            "Could not get the model. Run `python scripts/download_model.py` "
            "(or train it) and reload this page."
        ),
        "encounter": "The encounter",
        "age": "Age (years)",
        "gender": "Gender",
        "gender_Female": "Female",
        "gender_Male": "Male",
        "admission_type": "How was the patient admitted?",
        "admission_source": "Where did the patient come from?",
        "discharge": "Where did the patient go after discharge?",
        "discharge_note": "Discharges to death or hospice are not offered: the model does not "
        "cover them.",
        "days": "Days in hospital",
        "labs": "Lab tests",
        "procedures": "Procedures",
        "medications": "Medications",
        "diagnoses": "Diagnoses",
        "diagnoses_help": (
            "Total number of diagnoses recorded in the stay. The three codes below are only "
            "the first three."
        ),
        "visits": "**Visits in the previous year**",
        "outpatient": "Outpatient",
        "emergency": "Emergency",
        "inpatient": "Inpatient",
        "outpatient_help": "Consultations without staying overnight.",
        "emergency_help": "Visits to the emergency department.",
        "inpatient_help": "Previous stays in hospital. It is the variable that weighs most in the "
        "model.",
        "diag_title": "**Diagnoses (ICD-9 codes, optional)**",
        "diag_help": (
            "ICD-9 is the code system for diagnoses used in this dataset. Type the code with or "
            "without decimals, for example 428 (heart failure), 250.83 (diabetes), "
            "401 (hypertension) or 486 (pneumonia). Codes starting with V or E (for example V45) "
            "are accepted. The model only uses the broad group, shown under each box. "
            "Leave it empty if unknown."
        ),
        "diag_primary": "Primary",
        "diag_secondary": "Secondary",
        "diag_additional": "Additional",
        "diag_primary_help": "Main reason for the stay.",
        "diag_other_help": "Other condition of the patient during the stay.",
        "diag_group": "Group: {group}",
        "diag_bad": "This does not look like an ICD-9 code (for example 428 or V45): it will "
        "count as 'Other'.",
        "tests_title": "**Tests and diabetes treatment**",
        "a1c": "HbA1c result",
        "a1c_help": (
            "Blood test of average sugar over the last 2-3 months. Not measured, normal, "
            "above 7% or above 8%."
        ),
        "glucose": "Glucose serum result",
        "glucose_help": "Blood sugar test (mg/dL). Not measured, normal, above 200 or above 300.",
        "not_measured": "Not measured",
        "normal": "Normal",
        "diabetes_med": "Diabetes medication prescribed?",
        "change": "Medication changed during the stay?",
        "change_help": "Any change of drug or dose during the stay.",
        "yes": "Yes",
        "no": "No",
        "insulin": "Insulin",
        "metformin": "Metformin",
        "drug_help": (
            "No: not prescribed. Steady: same dose. Up: dose increased. Down: dose decreased."
        ),
        "drug_No": "No",
        "drug_Steady": "Steady",
        "drug_Up": "Up",
        "drug_Down": "Down",
        "other_drugs": "Other diabetes drugs",
        "insurance_title": "Insurance and specialty (optional)",
        "payer": "Insurance (payer code)",
        "payer_help": (
            "Who pays for the stay. The dataset does not define the codes; the usual reading is "
            "shown where it is clear (MC Medicare, MD Medicaid, HM HMO, SP self-pay, BC Blue "
            "Cross). Only the codes in the list are used by the model."
        ),
        "specialty": "Medical specialty",
        "specialty_help": (
            "Specialty of the doctor who admitted the patient. Only the specialties frequent "
            "enough in the training data are listed."
        ),
        "not_recorded": "Not recorded",
        "result": "Result",
        "chance": "Chance of readmission within 30 days",
        "vs_average": "{ratio:.1f}x the {baseline:.1%} average",
        "high": "Above the high-risk threshold (about the top 10% in validation).",
        "low": "Below the high-risk threshold (about the top 10% in validation).",
        "disclaimer": (
            "Educational estimate from historical US data (1999-2008). The model separates "
            "patients only modestly (AUC-ROC about 0.68) and ranks groups better than "
            "individuals: use it to explore, not to decide."
        ),
    },
    "es": {
        "page_title": "Riesgo de reingreso",
        "title": "Riesgo de reingreso a 30 días, pacientes diabéticos",
        "caption": (
            "Proyecto educativo, **no es una herramienta clínica**. Modelo entrenado con datos "
            "de hospitales de EE. UU. (1999-2008); sus límites están en el repositorio."
        ),
        "language": "Idioma",
        "model_error": (
            "No se pudo obtener el modelo. Ejecuta `python scripts/download_model.py` "
            "(o entrénalo) y recarga la página."
        ),
        "encounter": "El ingreso",
        "age": "Edad (años)",
        "gender": "Sexo",
        "gender_Female": "Mujer",
        "gender_Male": "Hombre",
        "admission_type": "¿Cómo ingresó el paciente?",
        "admission_source": "¿De dónde venía el paciente?",
        "discharge": "¿Adónde fue el paciente al recibir el alta?",
        "discharge_note": "No se ofrecen las altas por fallecimiento ni a cuidados paliativos: el "
        "modelo no las cubre.",
        "days": "Días en el hospital",
        "labs": "Pruebas de laboratorio",
        "procedures": "Procedimientos",
        "medications": "Medicamentos",
        "diagnoses": "Diagnósticos",
        "diagnoses_help": (
            "Número total de diagnósticos registrados en el ingreso. Los tres códigos de abajo "
            "son solo los tres primeros."
        ),
        "visits": "**Visitas en el año anterior**",
        "outpatient": "Consultas ambulatorias",
        "emergency": "Urgencias",
        "inpatient": "Ingresos",
        "outpatient_help": "Consultas sin pasar la noche en el hospital.",
        "emergency_help": "Visitas al servicio de urgencias.",
        "inpatient_help": "Ingresos hospitalarios previos. Es la variable que más pesa en el "
        "modelo.",
        "diag_title": "**Diagnósticos (códigos CIE-9, opcional)**",
        "diag_help": (
            "CIE-9 (ICD-9 en inglés) es el sistema de códigos de diagnóstico que usa este "
            "dataset. Escribe el código con o sin decimales, por ejemplo 428 (insuficiencia "
            "cardiaca), 250.83 (diabetes), 401 (hipertensión) o 486 (neumonía). Se aceptan "
            "códigos que empiecen por V o E (por ejemplo V45). El modelo solo usa el grupo "
            "general, que aparece bajo cada casilla. Déjalo vacío si no lo sabes."
        ),
        "diag_primary": "Principal",
        "diag_secondary": "Secundario",
        "diag_additional": "Adicional",
        "diag_primary_help": "Motivo principal del ingreso.",
        "diag_other_help": "Otra enfermedad del paciente durante el ingreso.",
        "diag_group": "Grupo: {group}",
        "diag_bad": "Esto no parece un código CIE-9 (por ejemplo 428 o V45): contará como 'Otros'.",
        "tests_title": "**Pruebas y tratamiento de la diabetes**",
        "a1c": "Resultado de HbA1c",
        "a1c_help": (
            "Análisis de sangre del azúcar medio de los últimos 2-3 meses. No medida, normal, "
            "mayor de 7 % o mayor de 8 %."
        ),
        "glucose": "Resultado de glucosa en suero",
        "glucose_help": (
            "Análisis del azúcar en sangre (mg/dL). No medida, normal, mayor de 200 o mayor de 300."
        ),
        "not_measured": "No medida",
        "normal": "Normal",
        "diabetes_med": "¿Se recetó medicación para la diabetes?",
        "change": "¿Se cambió la medicación durante el ingreso?",
        "change_help": "Cualquier cambio de fármaco o de dosis durante el ingreso.",
        "yes": "Sí",
        "no": "No",
        "insulin": "Insulina",
        "metformin": "Metformina",
        "drug_help": (
            "No: no se receta. Steady: misma dosis. Up: dosis aumentada. Down: dosis reducida."
        ),
        "drug_No": "No (no se receta)",
        "drug_Steady": "Steady (misma dosis)",
        "drug_Up": "Up (dosis aumentada)",
        "drug_Down": "Down (dosis reducida)",
        "other_drugs": "Otros fármacos para la diabetes",
        "insurance_title": "Seguro y especialidad (opcional)",
        "payer": "Seguro (código del pagador)",
        "payer_help": (
            "Quién paga el ingreso. El dataset no define los códigos; se muestra la lectura "
            "habitual donde está clara (MC Medicare, MD Medicaid, HM HMO, SP pago privado, BC "
            "Blue Cross). El modelo solo usa los códigos de la lista."
        ),
        "specialty": "Especialidad médica",
        "specialty_help": (
            "Especialidad del médico que ingresó al paciente. Solo se listan las especialidades "
            "con suficientes casos en los datos de entrenamiento."
        ),
        "not_recorded": "Sin indicar",
        "result": "Resultado",
        "chance": "Probabilidad de reingreso en 30 días",
        "vs_average": "{ratio:.1f} veces la media del {baseline:.1%}",
        "high": "Por encima del umbral de alto riesgo (aprox. el 10 % superior en la validación).",
        "low": "Por debajo del umbral de alto riesgo (aprox. el 10 % superior en la validación).",
        "disclaimer": (
            "Estimación educativa con datos históricos de EE. UU. (1999-2008). El modelo separa "
            "a los pacientes solo de forma modesta (AUC-ROC cercano a 0,68) y ordena mejor "
            "grupos que individuos: úsalo para explorar, no para decidir."
        ),
    },
}
