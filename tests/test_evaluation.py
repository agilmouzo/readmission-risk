import numpy as np
import pandas as pd
import pytest

from readmission import evaluation as ev


@pytest.fixture
def scored():
    """Skewed risks and outcomes drawn from them: calibrated by construction."""
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.beta(1, 8, n)
    y = (rng.random(n) < p).astype(int)
    groups = rng.integers(0, 1500, n)
    return y, p, groups


def test_operating_point_on_a_hand_checked_example():
    y = [1, 1, 0, 0, 0, 1]
    p = [0.9, 0.8, 0.7, 0.2, 0.1, 0.3]
    point = ev.operating_point(y, p, 0.5)
    assert point["flagged_share"] == pytest.approx(0.5)
    assert point["precision"] == pytest.approx(2 / 3)
    assert point["recall"] == pytest.approx(2 / 3)
    assert point["specificity"] == pytest.approx(2 / 3)
    assert point["f1"] == pytest.approx(2 / 3)
    assert point["lift"] == pytest.approx((2 / 3) / 0.5)


def test_operating_point_flagging_nobody_has_undefined_precision():
    point = ev.operating_point([1, 0, 0], [0.1, 0.2, 0.3], 0.9)
    assert np.isnan(point["precision"])
    assert point["recall"] == 0
    assert point["f1"] == 0


def test_threshold_max_f1_is_at_least_as_good_as_any_other_threshold(scored):
    y, p, _ = scored
    best = ev.operating_point(y, p, ev.threshold_max_f1(y, p))["f1"]
    for threshold in np.linspace(0.02, 0.9, 40):
        assert best >= ev.operating_point(y, p, threshold)["f1"] - 1e-9


def test_threshold_for_top_fraction_flags_about_that_share(scored):
    _, p, _ = scored
    for fraction in (0.05, 0.2, 0.5):
        flagged = (p >= ev.threshold_for_top_fraction(p, fraction)).mean()
        assert flagged == pytest.approx(fraction, abs=0.01)
    with pytest.raises(ValueError):
        ev.threshold_for_top_fraction(p, 1.5)


def test_lift_table_structure_and_ordering(scored):
    y, p, _ = scored
    table = ev.lift_table(y, p)
    assert table["n"].sum() == len(y)
    assert table["readmissions"].sum() == y.sum()
    assert table["cum_capture"].iloc[-1] == pytest.approx(1.0)
    assert table["cum_capture"].is_monotonic_increasing
    assert table["rate"].iloc[0] > table["rate"].iloc[-1]
    assert table["lift"].iloc[0] > 1 > table["lift"].iloc[-1]


def test_precision_at_fraction_with_a_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    assert ev.precision_at_fraction(y, y.astype(float), 0.10) == pytest.approx(1.0)
    assert ev.precision_at_fraction(y, y.astype(float), 0.20) == pytest.approx(0.5)
    assert ev.lift_at_fraction(y, y.astype(float), 0.10) == pytest.approx(10.0)


@pytest.mark.parametrize("metric", [ev.auc_pr, ev.auc_roc, ev.brier, ev._precision_top10])
def test_weighted_metrics_equal_metrics_on_replicated_rows(scored, metric):
    """The bootstrap relies on integer weights meaning 'this row was drawn k times'."""
    y, p, _ = scored
    weights = np.random.default_rng(3).integers(0, 4, len(y))
    weighted = metric(y, p, weights)
    replicated = metric(np.repeat(y, weights), np.repeat(p, weights))
    assert weighted == pytest.approx(replicated, rel=1e-6)


def test_cluster_bootstrap_is_reproducible_and_ordered(scored):
    y, p, groups = scored
    metrics = ev.default_metrics(threshold=0.12)
    a = ev.cluster_bootstrap(y, p, groups, metrics, n_boot=60, seed=1)
    b = ev.cluster_bootstrap(y, p, groups, metrics, n_boot=60, seed=1)
    c = ev.cluster_bootstrap(y, p, groups, metrics, n_boot=60, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert (a["ci_low"] <= a["ci_high"]).all()
    assert set(a.index) >= {"auc_pr", "auc_roc", "brier", "lift_top10", "recall_at_f1_threshold"}
    assert a.loc["auc_roc", "ci_low"] < a.loc["auc_roc", "estimate"] < a.loc["auc_roc", "ci_high"]


def test_resampling_patients_gives_wider_intervals_than_resampling_rows():
    """Four identical encounters per patient: the evidence is 500 patients, not 2000 rows."""
    rng = np.random.default_rng(5)
    p = rng.beta(1, 8, 500)
    y = (rng.random(500) < p).astype(int)
    y4, p4 = np.repeat(y, 4), np.repeat(p, 4)
    by_patient = np.repeat(np.arange(500), 4)
    by_row = np.arange(2000)
    metrics = {"auc_roc": ev.auc_roc}
    wide = ev.cluster_bootstrap(y4, p4, by_patient, metrics, n_boot=300)
    narrow = ev.cluster_bootstrap(y4, p4, by_row, metrics, n_boot=300)
    width = lambda t: t.loc["auc_roc", "ci_high"] - t.loc["auc_roc", "ci_low"]  # noqa: E731
    assert width(wide) > 1.4 * width(narrow)


def test_cluster_bootstrap_rejects_missing_groups(scored):
    y, p, groups = scored
    bad = pd.Series(groups).astype(float)
    bad.iloc[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        ev.cluster_bootstrap(y, p, bad, {"auc_roc": ev.auc_roc}, n_boot=5)


def test_calibration_of_calibrated_predictions(scored):
    y, p, _ = scored
    table = ev.calibration_table(y, p)
    assert len(table) == 10
    assert ev.expected_calibration_error(table) < 0.03
    slope, intercept = ev.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.25)
    assert intercept == pytest.approx(0.0, abs=0.35)


def test_calibration_slope_detects_overconfident_predictions(scored):
    y, p, _ = scored
    logit = np.log(p / (1 - p))
    overconfident = 1 / (1 + np.exp(-2.5 * logit))
    slope, _ = ev.calibration_slope_intercept(y, overconfident)
    assert slope < 0.6


def test_age_band():
    bands = ev.age_band(pd.Series([2, 3, 4, 5, 6, 7, 8, 9]))
    assert bands.tolist() == ["<40", "<40", "40-59", "40-59", "60-79", "60-79", "80+", "80+"]


def test_subgroup_report_adds_patient_bootstrap_ci_for_reliable_groups():
    rng = np.random.default_rng(1)
    n = 600
    p = rng.beta(1, 8, n)
    df = pd.DataFrame(
        {
            "g": ["A"] * 400 + ["C"] * 200,
            "p": p,
            "y": (rng.random(n) < p).astype(int),
            "patient": rng.integers(0, 300, n),
        }
    )
    report = ev.subgroup_report(
        df, "g", "y", "p", 0.15, min_n=100, min_events=10, patient_col="patient", n_boot=100
    ).set_index("group")
    row = report.loc["A"]
    assert row["auc_roc_low"] <= row["auc_roc"] <= row["auc_roc_high"]


def test_subgroup_report_labels_missing_and_flags_small_groups():
    rng = np.random.default_rng(0)
    n_a, n_b, n_c = 400, 300, 12
    groups = ["A"] * n_a + [None] * n_b + ["C"] * n_c
    p = rng.beta(1, 8, len(groups))
    df = pd.DataFrame({"g": groups, "p": p, "y": (rng.random(len(groups)) < p).astype(int)})
    report = ev.subgroup_report(df, "g", "y", "p", threshold=0.15, min_n=100, min_events=10)
    by_group = report.set_index("group")
    assert set(by_group.index) == {"A", "C", "Unknown"}
    assert by_group.loc["A", "reliable"] and by_group.loc["Unknown", "reliable"]
    assert not by_group.loc["C", "reliable"]
    assert np.isnan(by_group.loc["C", "auc_roc"])
    assert 0 < by_group.loc["A", "auc_roc"] < 1
    assert by_group["n"].to_dict() == {"A": n_a, "C": n_c, "Unknown": n_b}


def test_subgroup_gaps_ignores_unreliable_groups():
    report = pd.DataFrame(
        {
            "variable": ["v", "v", "v"],
            "group": ["a", "b", "c"],
            "reliable": [True, True, False],
            "auc_roc": [0.70, 0.64, 0.10],
            "recall": [0.50, 0.40, 0.99],
        }
    )
    gaps = ev.subgroup_gaps(report).iloc[0]
    assert gaps["groups"] == 2
    assert gaps["auc_roc_gap"] == pytest.approx(0.06)
    assert gaps["recall_gap"] == pytest.approx(0.10)
