import joblib
import numpy as np
import pandas as pd
import pytest

from readmission import config
from readmission.features import build_preprocessor, get_feature_columns, prepare
from readmission.train import compare_models, fit_final, save_results

FAST_LGBM = {"n_estimators": 20, "min_child_samples": 5}


@pytest.fixture
def signal_df(raw_df):
    """Prepared frame whose target really depends on prior inpatient visits (plus noise)."""
    df = prepare(raw_df)
    rng = np.random.default_rng(1)
    latent = df["number_inpatient"] + rng.normal(0, 1.5, len(df))
    df[config.TARGET] = (latent > latent.quantile(0.75)).astype(int)
    return df


def test_log_numeric_option_changes_scaling_only_for_numeric(signal_df):
    numeric, categorical = get_feature_columns(signal_df)
    X = signal_df[numeric + categorical]
    plain = build_preprocessor(numeric, categorical).fit_transform(X)
    logged = build_preprocessor(numeric, categorical, log_numeric=True).fit_transform(X)
    assert plain.shape == logged.shape
    assert list(plain.columns) == list(logged.columns)
    assert np.isfinite(logged.to_numpy()).all()
    assert not np.allclose(plain["num__number_outpatient"], logged["num__number_outpatient"])


def test_compare_models_ranks_models_above_the_dummy(signal_df):
    numeric, categorical = get_feature_columns(signal_df)
    results = compare_models(signal_df, numeric, categorical, n_splits=3, lgbm_params=FAST_LGBM)
    assert list(results.index) == ["dummy (prevalence)", "logistic regression", "lightgbm"]
    assert results.loc["dummy (prevalence)", "auc_roc_mean"] == pytest.approx(0.5)
    assert results.loc["logistic regression", "auc_roc_mean"] > 0.65
    assert results.loc["lightgbm", "auc_roc_mean"] > 0.6


def test_fit_final_artifact_roundtrips_through_joblib(signal_df, tmp_path):
    numeric, categorical = get_feature_columns(signal_df)
    artifact = fit_final(signal_df, numeric, categorical, lgbm_params=FAST_LGBM)
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    loaded = joblib.load(path)
    X = signal_df[loaded["numeric"] + loaded["categorical"]].head(5)
    proba = loaded["pipeline"].predict_proba(X)
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_save_results_writes_the_csv(tmp_path):
    path = tmp_path / "reports" / "cv.csv"
    assert save_results(pd.DataFrame({"auc_pr_mean": [0.2]}), path) is True
    assert path.exists()


def test_save_results_survives_an_unwritable_location(tmp_path, capsys):
    blocker = tmp_path / "reports"
    blocker.write_text("this is a file, so it cannot be used as a directory")
    assert save_results(pd.DataFrame({"a": [1]}), blocker / "cv.csv") is False
    assert "WARNING" in capsys.readouterr().out
