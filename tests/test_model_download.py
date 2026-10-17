import hashlib

import pytest

from readmission.model_download import ModelIntegrityError, download_model


def test_download_keeps_a_file_with_the_expected_digest(tmp_path):
    source = tmp_path / "published.joblib"
    source.write_bytes(b"model bytes")
    target = tmp_path / "models" / "model.joblib"
    digest = hashlib.sha256(b"model bytes").hexdigest()
    download_model(target, url=source.as_uri(), sha256=digest)
    assert target.read_bytes() == b"model bytes"


def test_download_discards_a_file_with_the_wrong_digest(tmp_path):
    source = tmp_path / "published.joblib"
    source.write_bytes(b"tampered")
    target = tmp_path / "model.joblib"
    with pytest.raises(ModelIntegrityError):
        download_model(target, url=source.as_uri(), sha256="0" * 64)
    assert not target.exists()
    assert not target.with_suffix(".part").exists()


def test_an_existing_model_is_not_downloaded_again(tmp_path):
    target = tmp_path / "model.joblib"
    target.write_bytes(b"mine")
    download_model(target, url="file:///does/not/exist", sha256="0" * 64)
    assert target.read_bytes() == b"mine"
