import joblib
import pandas as pd
import pytest

from readmission import schema
from readmission.artifact import REQUIRED_KEYS, load_artifact, model_inputs
from readmission.features import get_feature_columns, prepare

# The feature contract on the synthetic frame (the real dataset has more medication columns, which
# is why the real count is larger). If this changes, a feature was added or removed ON PURPOSE:
# update the list in the same commit, and think about whether the saved model is still valid.
EXPECTED_CATEGORICAL = [
    "A1Cresult",
    "admission_source_id",
    "admission_type_id",
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "discharge_disposition_id",
    "gender",
    "insulin",
    "max_glu_serum",
    "medical_specialty",
    "payer_code",
]


def test_feature_contract_is_frozen(raw_df):
    numeric, categorical = get_feature_columns(prepare(raw_df))
    assert numeric == schema.NUMERIC_FEATURES
    assert sorted(categorical) == EXPECTED_CATEGORICAL


def _artifact(df):
    numeric, categorical = get_feature_columns(df)
    return {"pipeline": None, "numeric": numeric, "categorical": categorical, "target": "t"}


def test_model_inputs_select_exactly_the_saved_columns_in_order(signal_df):
    artifact = _artifact(signal_df)
    polluted = signal_df.assign(proba=0.5, anything_else="x")
    inputs = model_inputs(polluted, artifact)
    assert list(inputs.columns) == artifact["numeric"] + artifact["categorical"]
    assert "proba" not in inputs.columns


def test_model_inputs_reports_missing_columns(signal_df):
    artifact = _artifact(signal_df)
    with pytest.raises(KeyError, match="insulin"):
        model_inputs(signal_df.drop(columns=["insulin"]), artifact)


def test_load_artifact_explains_how_to_create_a_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="readmission.train"):
        load_artifact(tmp_path / "nope.joblib")


def test_load_artifact_rejects_an_incomplete_file(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"pipeline": object()}, path)
    with pytest.raises(ValueError, match="missing keys"):
        load_artifact(path)


def test_load_artifact_roundtrip(tmp_path):
    path = tmp_path / "ok.joblib"
    payload = {key: pd.Series([1]) for key in REQUIRED_KEYS}
    joblib.dump(payload, path)
    assert set(load_artifact(path)) == set(REQUIRED_KEYS)
