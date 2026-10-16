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
