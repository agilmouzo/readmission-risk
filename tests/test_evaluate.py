import pytest

from readmission.evaluate import evaluate_model
from readmission.evaluation import F1_THRESHOLD_NAME
from readmission.features import get_feature_columns
from readmission.reporting import write_figures, write_tables
from readmission.split import patient_split
from readmission.train import fit_final

FAST = {"n_estimators": 20, "min_child_samples": 5}


@pytest.fixture
def result(signal_df):
    numeric, categorical = get_feature_columns(signal_df)
    train, test = patient_split(signal_df, test_size=0.3)
    artifact = fit_final(train, numeric, categorical, lgbm_params=FAST)
    return evaluate_model(
        train,
        test,
        artifact,
        n_splits=3,
        n_boot=30,
        lgbm_params=FAST,
        subgroup_min_n=5,
        subgroup_min_events=1,
    )


def test_evaluate_model_produces_every_piece(result):
    assert set(result.thresholds) == {F1_THRESHOLD_NAME, "top 5%", "top 10%", "top 20%", "top 30%"}
    top = [result.thresholds[f"top {k}%"] for k in (5, 10, 20, 30)]
    assert top == sorted(top, reverse=True)  # flagging fewer people needs a higher threshold
    assert result.test_metrics.loc["auc_roc", "estimate"] > 0.5
    assert (result.test_metrics["ci_low"] <= result.test_metrics["ci_high"]).all()
    assert len(result.lift) == 10
    assert set(result.calibration_summary) == {"slope", "intercept", "ece"}
    assert set(result.subgroups["variable"]) == {"race", "gender", "age_band"}
    assert len(result.y_test) == len(result.p_test)
    assert ((result.p_test >= 0) & (result.p_test <= 1)).all()


def test_reports_are_written(result, tmp_path):
    tables = write_tables(result, tmp_path)
    figures = write_figures(result, tmp_path / "figures")
    assert len(tables) == 6 and len(figures) == 4
    for path in [*tables, *figures]:
        assert path.exists() and path.stat().st_size > 0


def test_the_subgroup_figure_is_skipped_when_no_group_is_reliable(result, tmp_path):
    result.subgroups["reliable"] = False
    figures = write_figures(result, tmp_path)
    assert [p.name for p in figures] == ["pr_curve.png", "calibration.png", "risk_deciles.png"]
