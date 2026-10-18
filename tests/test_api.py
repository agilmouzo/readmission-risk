import csv
import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from readmission import config, schema
from readmission.api import EXAMPLE, PatientInput, create_app
from readmission.features import prepare
from readmission.service import OutOfScopeError, age_bracket, build_frame, predict_one

# The model inputs of the REAL dataset (10 numeric + 24 categorical). If the model changes on
# purpose, update this list in the same commit as the API fields.
REAL_CATEGORICAL = {
    "A1Cresult",
    "acarbose",
    "admission_source_id",
    "admission_type_id",
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "discharge_disposition_id",
    "gender",
    "glimepiride",
    "glipizide",
    "glyburide",
    "glyburide-metformin",
    "insulin",
    "max_glu_serum",
    "medical_specialty",
    "metformin",
    "nateglinide",
    "payer_code",
    "pioglitazone",
    "repaglinide",
    "rosiglitazone",
}


def _payload(**changes):
    return {**EXAMPLE, **changes}


@pytest.fixture
def client(real_like_artifact):
    with TestClient(create_app(real_like_artifact)) as client:
        yield client


def test_api_covers_every_input_of_the_real_model():
    patient = PatientInput(**EXAMPLE).model_dump(by_alias=True)
    prepared = prepare(build_frame(patient))
    available = set(prepared.columns)
    assert set(schema.NUMERIC_FEATURES) <= available
    assert REAL_CATEGORICAL <= available


def test_the_synthetic_model_has_the_same_inputs_as_the_real_one(real_like_artifact):
    assert real_like_artifact["numeric"] == schema.NUMERIC_FEATURES
    assert set(real_like_artifact["categorical"]) == REAL_CATEGORICAL


def test_high_risk_threshold_matches_the_evaluation_report():
    with open(config.ROOT / "reports" / "operating_points.csv") as handle:
        rows = {row["operating_point"]: row for row in csv.DictReader(handle)}
    assert config.HIGH_RISK_THRESHOLD == pytest.approx(
        float(rows["top 10%"]["threshold"]), abs=1e-4
    )


def test_root_redirects_to_the_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_every_input_field_is_documented():
    from readmission.api import PatientInput

    missing = [n for n, f in PatientInput.model_fields.items() if not f.description]
    assert not missing


def test_prediction_comes_with_context(client):
    body = client.post("/predict", json=EXAMPLE).json()
    assert body["baseline_risk"] == config.BASELINE_RISK
    assert body["relative_risk"] == pytest.approx(
        body["readmission_probability"] / config.BASELINE_RISK
    )
    assert "30 days" in body["summary"]


@pytest.mark.parametrize(
    ("probability", "high_risk", "expected"),
    [
        (0.30, True, "2.7x the 11% average"),
        (0.112, False, "close to the 11% average"),
        (0.05, False, "below the 11% average"),
    ],
)
def test_summary_wording(probability, high_risk, expected):
    from readmission.service import summarise

    text = summarise(probability, high_risk)
    assert expected in text
    assert ("; above the high-risk threshold" in text) == high_risk
    assert ("; below the high-risk threshold" in text) == (not high_risk)


def test_codes_endpoint_describes_every_accepted_code(client):
    body = client.get("/codes").json()
    assert set(map(int, body["admission_type_id"])) == set(range(1, 9))
    assert set(map(int, body["discharge_disposition_id"])) == set(range(1, 31))
    assert body["rejected_discharge_codes"] == sorted(schema.EXCLUDED_DISCHARGE_CODES)
    assert body["discharge_disposition_id"]["1"] == "Discharged to home"


def test_age_bracket():
    assert age_bracket(0) == "[0-10)"
    assert age_bracket(72) == "[70-80)"
    assert age_bracket(90) == age_bracket(104) == "[90-100)"


def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": None,  # the test model is injected, not loaded from a file
        "model_sha256": None,
    }


def test_health_reports_which_model_file_is_loaded(real_like_artifact, tmp_path, monkeypatch):
    import hashlib

    import joblib
    from fastapi.testclient import TestClient

    from readmission import model_download
    from readmission.api import create_app

    path = tmp_path / "model.joblib"
    joblib.dump(real_like_artifact, path)
    monkeypatch.setattr(config, "MODEL_PATH", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with TestClient(create_app()) as local:
        body = local.get("/health").json()
        assert body["model_sha256"] == digest
        assert body["model_version"] == "custom"
        monkeypatch.setattr(model_download, "MODEL_SHA256", digest, raising=False)
        assert local.get("/health").json()["model_version"] == config.MODEL_RELEASE


def test_release_tag_matches_the_download_url():
    from readmission import model_download

    assert f"/download/{config.MODEL_RELEASE}/" in model_download.MODEL_URL


def test_predict_returns_a_probability(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert body["threshold"] == config.HIGH_RISK_THRESHOLD
    assert body["high_risk"] == (body["readmission_probability"] >= body["threshold"])


def test_predict_matches_the_model_applied_to_the_training_frame(real_like_artifact, full_raw_df):
    """The API must give exactly what the model gives on the same patient in batch."""
    batch = prepare(full_raw_df)
    pipeline = real_like_artifact["pipeline"]
    columns = real_like_artifact["numeric"] + real_like_artifact["categorical"]
    expected = pipeline.predict_proba(batch[columns])[:, 1]
    raw = full_raw_df.loc[batch.index]
    checked = 0
    for position, (_, row) in enumerate(raw.iterrows()):
        if checked == 15:
            break
        patient = {
            key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in row.items()
            if key in PatientInput.model_fields or key in ("age", "glyburide-metformin")
        }
        patient["age"] = int(row["age"].strip("[)").split("-")[0]) + 5
        result = predict_one(real_like_artifact, PatientInput(**patient).model_dump(by_alias=True))
        assert result["readmission_probability"] == pytest.approx(expected[position], abs=1e-9)
        checked += 1
    assert checked == 15


def test_threshold_decides_the_label(real_like_artifact):
    patient = PatientInput(**EXAMPLE).model_dump(by_alias=True)
    probability = predict_one(real_like_artifact, patient)["readmission_probability"]
    assert predict_one(real_like_artifact, patient, threshold=probability)["high_risk"]
    assert not predict_one(real_like_artifact, patient, threshold=probability + 1e-6)["high_risk"]


@pytest.mark.parametrize("code", schema.EXCLUDED_DISCHARGE_CODES)
def test_excluded_discharges_get_a_clear_422(client, code):
    response = client.post("/predict", json=_payload(discharge_disposition_id=code))
    assert response.status_code == 422
    assert "outside the scope" in response.json()["detail"]


def test_service_raises_out_of_scope_for_excluded_discharge(real_like_artifact):
    patient = PatientInput(**EXAMPLE).model_dump(by_alias=True)
    patient["discharge_disposition_id"] = 11
    with pytest.raises(OutOfScopeError):
        predict_one(real_like_artifact, patient)


def test_optional_fields_can_be_left_out(client):
    minimal = {
        key: value
        for key, value in EXAMPLE.items()
        if key not in {"diag_1", "diag_2", "diag_3", "payer_code", "medical_specialty", "metformin"}
    }
    assert client.post("/predict", json=minimal).status_code == 200


def test_unseen_category_does_not_break_the_model(client):
    response = client.post("/predict", json=_payload(medical_specialty="SomethingNew"))
    assert response.status_code == 200


def test_hyphenated_field_is_accepted_by_its_dataset_name(client):
    assert (
        client.post("/predict", json=_payload(**{"glyburide-metformin": "Up"})).status_code == 200
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"age": -1},
        {"age": 150},
        {"gender": "Other"},
        {"time_in_hospital": 0},
        {"insulin": "Maybe"},
        {"A1Cresult": ">9"},
        {"unknown_field": 1},
    ],
)
def test_invalid_input_is_rejected(client, changes):
    assert client.post("/predict", json=_payload(**changes)).status_code == 422


def test_missing_required_field_is_rejected(client):
    body = dict(EXAMPLE)
    del body["age"]
    assert client.post("/predict", json=body).status_code == 422


def test_patient_input_validates_without_the_web_layer():
    with pytest.raises(ValidationError):
        PatientInput(**_payload(number_inpatient=-3))


def test_missing_model_file_fails_with_a_helpful_message(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError, match="readmission.train"):
        with TestClient(create_app()):
            pass


def test_not_recorded_codes_are_labelled_unknown_exactly_where_the_model_groups_them():
    from readmission import codes, schema

    for column, table in {
        "admission_type_id": codes.ADMISSION_TYPE,
        "discharge_disposition_id": codes.DISCHARGE_DISPOSITION,
        "admission_source_id": codes.ADMISSION_SOURCE,
    }.items():
        labelled = {code for code, name in table.items() if name.startswith("Unknown (")}
        assert labelled == set(schema.UNKNOWN_CODES[column])
