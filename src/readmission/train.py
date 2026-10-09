"""Compare models with patient-level cross-validation and save the final model.

Usage:
    python -m readmission.train

Only the TRAIN split is used here. The hold-out test set is reserved for the evaluation phase,
so that the final numbers have not influenced any modelling decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from readmission import config
from readmission.data import add_binary_target, load_raw
from readmission.features import get_feature_columns, prepare
from readmission.modeling import (
    cross_validate_by_patient,
    make_baseline,
    make_dummy,
    make_lgbm,
    summarise_scores,
)
from readmission.split import patient_split

REPORTS_DIR = config.ROOT / "reports"
CV_RESULTS_CSV = REPORTS_DIR / "cv_results.csv"


def compare_models(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    n_splits: int = 5,
    lgbm_params: dict | None = None,
) -> pd.DataFrame:
    """Cross-validate the dummy floor, the logistic baseline and LightGBM on the same folds."""
    X = train[numeric + categorical]
    y, groups = train[config.TARGET], train[config.GROUP_COL]
    models = {
        "dummy (prevalence)": make_dummy(),
        "logistic regression": make_baseline(numeric, categorical),
        "lightgbm": make_lgbm(numeric, categorical, **(lgbm_params or {})),
    }
    rows = []
    for name, model in models.items():
        print(f"Cross-validating {name} ...", flush=True)
        scores = cross_validate_by_patient(model, X, y, groups, n_splits=n_splits)
        rows.append(summarise_scores(name, scores))
    return pd.DataFrame(rows).set_index("model")


def fit_final(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    lgbm_params: dict | None = None,
) -> dict:
    """Fit LightGBM on the whole train split and bundle it with the columns it expects."""
    pipeline = make_lgbm(numeric, categorical, **(lgbm_params or {}))
    pipeline.fit(train[numeric + categorical], train[config.TARGET])
    return {
        "pipeline": pipeline,
        "numeric": numeric,
        "categorical": categorical,
        "target": config.TARGET,
    }


def save_results(results: pd.DataFrame, path: Path = CV_RESULTS_CSV) -> bool:
    """Write the comparison table. A failure here must never throw the results away."""
    try:
        path.parent.mkdir(exist_ok=True)
        results.to_csv(path)
    except OSError as exc:
        print(f"WARNING: could not save {path} ({exc}). The results are shown above.")
        return False
    print(f"Saved {path}")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    args = parser.parse_args(argv)

    df = prepare(add_binary_target(load_raw()))
    numeric, categorical = get_feature_columns(df)
    train, test = patient_split(df)
    print(f"train: {len(train)} rows | test (untouched): {len(test)} rows")

    results = compare_models(train, numeric, categorical, n_splits=args.folds)
    print(f"\nCross-validation on train ({args.folds} folds, split by patient):")
    print(results.round(4).to_string())
    save_results(results)

    print("\nFitting the final LightGBM on the whole train split ...")
    artifact = fit_final(train, numeric, categorical)
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, config.MODEL_PATH)
    print(f"Saved {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
