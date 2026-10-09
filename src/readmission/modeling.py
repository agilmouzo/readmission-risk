"""Model definitions and patient-level cross-validation."""

from __future__ import annotations

import re

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from readmission import config
from readmission.features import build_preprocessor

# Moderately regularised: the signal is weak and noisy, so shallow trees and a small learning
# rate. Class weights are NOT used: probabilities stay honest and the decision threshold is
# chosen later, in the evaluation phase.
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "random_state": config.RANDOM_STATE,
    "verbose": -1,
}

SCORING = {
    "auc_pr": "average_precision",
    "auc_roc": "roc_auc",
    "neg_brier": "neg_brier_score",
}


def _sanitize_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM rejects some characters in column names (e.g. ':' or ','); replace them."""
    X = X.rename(columns=lambda c: re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)))
    if X.columns.duplicated().any():
        raise ValueError("Feature names collide after sanitising")
    return X


def make_dummy() -> DummyClassifier:
    """Predicts the training prevalence for everyone: the floor any model must beat."""
    return DummyClassifier(strategy="prior")


def make_baseline(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Logistic regression; skewed counts are log-transformed (see the EDA)."""
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric, categorical, log_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


def make_lgbm(numeric: list[str], categorical: list[str], **overrides) -> Pipeline:
    """Gradient-boosted trees (LightGBM) on the shared preprocessing pipeline."""
    params = {**LGBM_PARAMS, **overrides}
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric, categorical)),
            ("names", FunctionTransformer(_sanitize_feature_names)),
            ("clf", LGBMClassifier(**params)),
        ]
    )


def make_patient_cv(n_splits: int = 5, random_state: int = config.RANDOM_STATE):
    """Stratified K-fold that never puts the same patient in two folds."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_by_patient(
    model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5
) -> pd.DataFrame:
    """Per-fold scores (rows = folds). The whole pipeline is refitted inside each fold."""
    result = cross_validate(
        model, X, y, groups=groups, cv=make_patient_cv(n_splits), scoring=SCORING
    )
    return pd.DataFrame(
        {
            "auc_pr": result["test_auc_pr"],
            "auc_roc": result["test_auc_roc"],
            "brier": -result["test_neg_brier"],
        }
    )


def summarise_scores(name: str, scores: pd.DataFrame) -> dict[str, float | str]:
    """One row of the comparison table: mean and std of every metric."""
    row: dict[str, float | str] = {"model": name}
    for metric in scores.columns:
        row[f"{metric}_mean"] = scores[metric].mean()
        row[f"{metric}_std"] = scores[metric].std()
    return row
