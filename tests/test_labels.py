import pytest

from readmission.labels import VARIABLES, value_label, variable_label


def test_every_model_variable_has_a_readable_label(real_like_artifact):
    model_variables = real_like_artifact["numeric"] + real_like_artifact["categorical"]
    assert [v for v in model_variables if v not in VARIABLES] == []


def test_unknown_variable_falls_back_to_a_spaced_name():
    assert variable_label("some_new_column") == "some new column"


@pytest.mark.parametrize(
    ("variable", "value", "expected"),
    [
        ("discharge_disposition_id", "1", "Discharged to home"),
        ("discharge_disposition_id", "Unknown", "Unknown"),
        ("admission_source_id", "7", "Emergency Room"),
        ("admission_type_id", 1, "Emergency"),
        ("age_ord", 7, "70s"),
        ("age_ord", 2, "under 30"),
        ("number_inpatient", 3.0, "3"),
        ("payer_code", "MC", "MC"),
        ("number_inpatient", "", ""),
    ],
)
def test_value_label(variable, value, expected):
    assert value_label(variable, value) == expected


def test_long_values_are_shortened():
    text = value_label("discharge_disposition_id", "5")
    assert len(text) <= 30
    assert text.endswith("…")
