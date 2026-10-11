import numpy as np
import pandas as pd
import pytest

from readmission import config
from readmission import explain as ex
from readmission.features import get_feature_columns
from readmission.modeling import make_lgbm

pytest.importorskip("shap")


@pytest.fixture
def fitted(signal_df):
    numeric, categorical = get_feature_columns(signal_df)
    model = make_lgbm(numeric, categorical, n_estimators=60, min_child_samples=10).fit(
        signal_df[numeric + categorical], signal_df[config.TARGET]
    )
    artifact = {
        "pipeline": model,
        "numeric": numeric,
        "categorical": categorical,
        "target": config.TARGET,
    }
    return artifact, signal_df


def test_original_variable_maps_every_encoded_column(fitted):
    artifact, df = fitted
    matrix = ex.encoded_matrix(
        artifact["pipeline"], df[artifact["numeric"] + artifact["categorical"]]
    )
    owners = {
        ex.original_variable(c, artifact["numeric"], artifact["categorical"])
        for c in matrix.columns
    }
    assert owners == set(artifact["numeric"]) | set(artifact["categorical"])


def test_original_variable_rejects_unknown_names():
    with pytest.raises(ValueError):
        ex.original_variable("other__x", ["a"], ["b"])
    with pytest.raises(ValueError):
        ex.original_variable("cat__zzz_1", ["a"], ["b"])


def test_contributions_add_up_to_the_model_output(fitted):
    artifact, df = fitted
    explanation = ex.explain_sample(artifact, df, n_samples=150)
    X = explanation.inputs
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    np.testing.assert_allclose(explanation.probability, proba, atol=1e-6)
    assert list(explanation.contributions.columns) == (
        artifact["numeric"] + artifact["categorical"]
    )


def test_signal_variable_is_the_most_important(fitted):
    artifact, df = fitted
    explanation = ex.explain_sample(artifact, df, n_samples=250)
    table = ex.global_importance(explanation)
    assert table.index[0] in {"number_inpatient", "prior_visits"}
    assert table["share"].sum() == pytest.approx(1.0)


def test_explain_patient_sums_to_the_total(fitted):
    artifact, df = fitted
    explanation = ex.explain_sample(artifact, df, n_samples=100)
    table = ex.explain_patient(explanation, 0, top=3)
    total = explanation.contributions.iloc[0].sum()
    assert table["contribution"].sum() == pytest.approx(total)
    assert len(table) == 4  # 3 variables + "other"


def test_explain_sample_is_reproducible(fitted):
    artifact, df = fitted
    a = ex.explain_sample(artifact, df, n_samples=80)
    b = ex.explain_sample(artifact, df, n_samples=80)
    pd.testing.assert_frame_equal(a.contributions, b.contributions)


def test_tables_and_figures_are_written(fitted, tmp_path):
    artifact, df = fitted
    explanation = ex.explain_sample(artifact, df, n_samples=200)
    assert ex.write_tables(explanation, tmp_path)[0].exists()
    names = {p.name for p in ex.write_figures(explanation, tmp_path / "figs")}
    assert names == {"shap_importance.png", "shap_patients.png", "shap_numeric.png"}


def test_original_variable_prefers_the_longest_matching_name():
    categorical = ["glyburide", "glyburide-metformin", "metformin"]
    pick = lambda name: ex.original_variable(name, [], categorical)  # noqa: E731
    assert pick("cat__glyburide_No") == "glyburide"
    assert pick("cat__glyburide_metformin_No") == "glyburide-metformin"
    assert pick("cat__glyburide_metformin_infrequent_sklearn") == "glyburide-metformin"
    assert pick("cat__metformin_Steady") == "metformin"
