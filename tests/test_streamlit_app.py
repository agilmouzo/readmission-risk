from pathlib import Path

import joblib
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from readmission import config  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_demo_shows_a_prediction(real_like_artifact, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(real_like_artifact, path)
    monkeypatch.setattr(config, "MODEL_PATH", path)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert "chance of readmission" in app.metric[0].label.lower()
    assert app.metric[0].value.endswith("%")


def _app(real_like_artifact, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(real_like_artifact, path)
    monkeypatch.setattr(config, "MODEL_PATH", path)
    return AppTest.from_file(APP, default_timeout=60).run()


def test_demo_works_in_spanish_and_keeps_the_values(real_like_artifact, tmp_path, monkeypatch):
    app = _app(real_like_artifact, tmp_path, monkeypatch)
    app.number_input(key="inpatient").set_value(4).run()
    changed = app.metric[0].value
    app.radio(key="lang").set_value("es").run()
    assert not app.exception
    assert "reingreso" in app.metric[0].label.lower()
    assert app.number_input(key="inpatient").value == 4  # the language does not reset the form
    assert app.metric[0].value == changed.replace(".", ",")  # same number, Spanish decimals
    assert any("umbral de alto riesgo" in m.value for m in app.markdown)


def test_dropdown_values_survive_a_language_change(real_like_artifact, tmp_path, monkeypatch):
    app = _app(real_like_artifact, tmp_path, monkeypatch)
    app.selectbox(key="gender_en").set_value("Male").run()
    app.radio(key="lang").set_value("es").run()
    assert not app.exception
    assert app.selectbox(key="gender_es").value == "Male"
    assert "Hombre" in app.selectbox(key="gender_es").options
    assert app.selectbox(key="admission_type_es").options[0] == "1 - Urgencia"
    app.radio(key="lang").set_value("en").run()
    assert app.selectbox(key="gender_en").value == "Male"


def test_demo_shows_the_diagnosis_group_and_flags_bad_codes(
    real_like_artifact, tmp_path, monkeypatch
):
    app = _app(real_like_artifact, tmp_path, monkeypatch)
    captions = [c.value for c in app.caption]
    assert "Group: Circulatory" in captions  # 428
    assert "Group: Diabetes" in captions  # 250
    app.text_input(key="diag_1").set_value("hello").run()
    assert any("does not look like an ICD-9" in c.value for c in app.caption)


def test_demo_explains_a_missing_model(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("readmission.model_download.download_model", fail)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "nope.joblib")
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert "download_model.py" in app.error[0].value


def test_cloud_requirements_pin_what_constraints_pin():
    root = Path(__file__).resolve().parents[1]
    pinned = {line for line in (root / "constraints.txt").read_text().split("\n") if line}
    requirements = set((root / "requirements.txt").read_text().split("\n"))
    assert pinned <= requirements
    assert any(line.startswith("streamlit==") for line in requirements)
