import numpy as np
import pandas as pd
import pytest

from readmission import config
from readmission.features import get_feature_columns, prepare
from readmission.modeling import (
    _sanitize_feature_names,
    cross_validate_by_patient,
    make_baseline,
    make_dummy,
    make_lgbm,
    make_patient_cv,
    out_of_fold_predictions,
)


@pytest.fixture
def xy(raw_df):
    df = prepare(raw_df)
    numeric, categorical = get_feature_columns(df)
    return df[numeric + categorical], df[config.TARGET], df[config.GROUP_COL], numeric, categorical


def test_models_fit_and_return_valid_probabilities(xy):
    X, y, _, numeric, categorical = xy
    models = [
        make_dummy(),
        make_baseline(numeric, categorical),
        make_lgbm(numeric, categorical, n_estimators=10, min_child_samples=5),
    ]
    for model in models:
        proba = model.fit(X, y).predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert ((proba >= 0) & (proba <= 1)).all()


def test_sanitize_feature_names_replaces_special_characters():
    X = pd.DataFrame({"cat__a>b": [1], "cat__c d,e": [2]})
    assert list(_sanitize_feature_names(X).columns) == ["cat__a_b", "cat__c_d_e"]


def test_sanitize_feature_names_rejects_collisions():
    X = pd.DataFrame({"a>b": [1], "a<b": [2]})
    with pytest.raises(ValueError, match="collide"):
        _sanitize_feature_names(X)


def test_patient_cv_folds_never_share_patients(xy):
    X, y, groups, _, _ = xy
    for train_idx, valid_idx in make_patient_cv(3).split(X, y, groups):
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[valid_idx]))


def test_cross_validate_by_patient_returns_one_row_per_fold(xy):
    X, y, groups, numeric, categorical = xy
    scores = cross_validate_by_patient(
        make_baseline(numeric, categorical), X, y, groups, n_splits=3
    )
    assert len(scores) == 3
    assert set(scores.columns) == {"auc_pr", "auc_roc", "brier"}
    assert scores["auc_roc"].between(0, 1).all()


def test_out_of_fold_predictions_cover_every_row_once(xy):
    X, y, groups, numeric, categorical = xy
    model = make_baseline(numeric, categorical)
    oof = out_of_fold_predictions(model, X, y, groups, n_splits=3)
    assert len(oof) == len(X)
    assert oof.index.equals(X.index)
    assert oof.between(0, 1).all()
    # not the same thing as predicting with a model fitted on every row
    in_sample = model.fit(X, y).predict_proba(X)[:, 1]
    assert not np.allclose(oof.to_numpy(), in_sample)
