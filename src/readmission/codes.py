"""Meaning of the numeric codes in the dataset (copied from its IDS_mapping.csv).

Only used to document the API and the demo: the model itself receives the numbers.

The dataset has several labels for "not recorded" ("NULL", "Not Available", "Not Mapped",
"Unknown/Invalid"). The model groups them all as "Unknown" (see schema.UNKNOWN_CODES), so they are
shown here with one clear wording that keeps the original label.
"""

_NOT_RECORDED = ("NULL", "Not Available", "Not Mapped", "Unknown/Invalid")

ADMISSION_TYPE: dict[int, str] = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped",
}

DISCHARGE_DISPOSITION: dict[int, str] = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    16: "Discharged/transferred/referred another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home. Medicaid only, hospice.",
    20: "Expired in a medical facility. Medicaid only, hospice.",
    21: "Expired, place unknown. Medicaid only, hospice.",
    22: "Discharged/transferred to another rehab fac including rehab units of a hospital",
    23: "Discharged/transferred to a long term care hospital",
    24: (
        "Discharged/transferred to a nursing facility certified under Medicaid "
        "but not certified under Medicare"
    ),
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility",
    28: (
        "Discharged/transferred/referred to a psychiatric hospital of psychiatric "
        "distinct part unit of a hospital"
    ),
    29: "Discharged/transferred to a Critical Access Hospital (CAH)",
    30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
}

ADMISSION_SOURCE: dict[int, str] = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a hospital",
    5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critical access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Available",
    17: "NULL",
    18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency",
    20: "Not Mapped",
    21: "Unknown/Invalid",
    22: "Transfer from hospital inpt/same fac reslt in a sep claim",
    23: "Born inside this hospital",
    24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center",
    26: "Transfer from Hospice",
}


def _readable(table: dict[int, str]) -> dict[int, str]:
    return {
        code: f"Unknown (recorded as '{name}' in the dataset)" if name in _NOT_RECORDED else name
        for code, name in table.items()
    }


ADMISSION_TYPE = _readable(ADMISSION_TYPE)
DISCHARGE_DISPOSITION = _readable(DISCHARGE_DISPOSITION)
ADMISSION_SOURCE = _readable(ADMISSION_SOURCE)
