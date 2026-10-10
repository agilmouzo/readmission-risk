"""Evaluate the final model on the hold-out TEST set.

Usage:
    python -m readmission.evaluate

This is the first and only place where the test set is used. Run it once the model is final:
every decision (model, hyper-parameters, threshold) has already been made on the train split.

The decision threshold is chosen on out-of-fold predictions over TRAIN, then applied to the test
set. Confidence intervals resample patients, not rows.
"""

from __future__ import annotations

import argparse

import pandas as pd

from readmission import config
from readmission.artifact import load_artifact, model_inputs
from readmission.data import add_binary_target, load_raw
from readmission.evaluation import (
    F1_THRESHOLD_NAME,
    TOP_FRACTIONS,
    EvaluationResult,
    age_band,
    calibration_slope_intercept,
    calibration_table,
    cluster_bootstrap,
    default_metrics,
    expected_calibration_error,
    lift_table,
    operating_points,
    subgroup_gaps,
    subgroup_report,
    threshold_for_top_fraction,
    threshold_max_f1,
)
from readmission.features import prepare
from readmission.modeling import make_lgbm, out_of_fold_predictions
from readmission.reporting import REPORTS_DIR, write_figures, write_tables
from readmission.split import patient_split

SUBGROUP_VARIABLES = ("race", "gender", "age_band")


def evaluate_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    artifact: dict,
    n_splits: int = 5,
    n_boot: int = 1000,
    lgbm_params: dict | None = None,
    subgroup_min_n: int = 200,
    subgroup_min_events: int = 20,
) -> EvaluationResult:
    """Pick thresholds on train (out-of-fold), then measure everything on test."""
    target = artifact["target"]
    y_train = train[target]

    # 1. Thresholds from out-of-fold predictions on train: the test set plays no part here.
    fresh_model = make_lgbm(artifact["numeric"], artifact["categorical"], **(lgbm_params or {}))
    oof = out_of_fold_predictions(
        fresh_model, model_inputs(train, artifact), y_train, train[config.GROUP_COL], n_splits
    )
    thresholds = {F1_THRESHOLD_NAME: threshold_max_f1(y_train, oof)}
    for fraction in TOP_FRACTIONS:
        thresholds[f"top {fraction:.0%}"] = threshold_for_top_fraction(oof, fraction)

    # 2. The final model (fitted on the whole train split) predicts the test set, once.
    y_test = test[target].to_numpy()
    p_test = artifact["pipeline"].predict_proba(model_inputs(test, artifact))[:, 1]

    # 3. Headline metrics with patient-level bootstrap intervals.
    test_metrics = cluster_bootstrap(
        y_test,
        p_test,
        test[config.GROUP_COL],
        default_metrics(thresholds[F1_THRESHOLD_NAME]),
        n_boot=n_boot,
    )

    # 4. Operating points, lift and calibration.
    calibration = calibration_table(y_test, p_test)
    slope, intercept = calibration_slope_intercept(y_test, p_test)
    calibration_summary = {
        "slope": slope,
        "intercept": intercept,
        "ece": expected_calibration_error(calibration),
    }

    # 5. Subgroups, including the sensitive attribute that was kept out of the model.
    scored = test.assign(proba=p_test, age_band=age_band(test["age_ord"]))
    subgroups = pd.concat(
        [
            subgroup_report(
                scored,
                variable,
                target,
                "proba",
                thresholds[F1_THRESHOLD_NAME],
                subgroup_min_n,
                subgroup_min_events,
                patient_col=config.GROUP_COL,
                n_boot=min(n_boot, 500),
            )
            for variable in SUBGROUP_VARIABLES
        ],
        ignore_index=True,
    )

    return EvaluationResult(
        thresholds=thresholds,
        test_metrics=test_metrics,
        operating_points=operating_points(y_test, p_test, thresholds),
        lift=lift_table(y_test, p_test),
        calibration=calibration,
        calibration_summary=calibration_summary,
        subgroups=subgroups,
        y_test=y_test,
        p_test=p_test,
    )


def print_summary(result: EvaluationResult) -> None:
    pd.options.display.float_format = "{:.3f}".format
    print("\nThresholds (chosen on out-of-fold predictions over TRAIN):")
    print({name: round(value, 4) for name, value in result.thresholds.items()})
    print("\nTest metrics (95% CI, bootstrap resampling patients):")
    print(result.test_metrics.to_string())
    print("\nOperating points on the test set:")
    print(result.operating_points.to_string())
    print("\nTest set by risk decile (1 = highest predicted risk):")
    print(result.lift.to_string(index=False))
    print("\nCalibration (ideal: slope 1, intercept 0, ECE 0):")
    print({name: round(value, 4) for name, value in result.calibration_summary.items()})
    print("\nSubgroups (reliable=False means too few cases to judge):")
    print(result.subgroups.to_string(index=False))
    gaps = subgroup_gaps(result.subgroups)
    if not gaps.empty:
        print("\nLargest gap between subgroups with at least 1,000 patients:")
        print(gaps.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", type=int, default=5, help="folds for the out-of-fold step")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples")
    args = parser.parse_args(argv)

    artifact = load_artifact()
    df = prepare(add_binary_target(load_raw()))
    train, test = patient_split(df)
    print(
        f"train: {len(train)} rows | test: {len(test)} rows "
        f"| model inputs: {len(artifact['numeric'])} numeric + "
        f"{len(artifact['categorical'])} categorical"
    )
    print("NOTE: this opens the TEST set. Run it once the model is final.\n")

    result = evaluate_model(train, test, artifact, n_splits=args.folds, n_boot=args.bootstrap)
    print_summary(result)

    write_tables(result, REPORTS_DIR)
    write_figures(result, REPORTS_DIR / "figures")
    print(f"\nSaved tables to {REPORTS_DIR} and figures to {REPORTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
