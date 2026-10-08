import pandas as pd
import pytest

from readmission import config, schema
from readmission.features import build_preprocessor, get_feature_columns, icd9_group, prepare


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("250.83", "Diabetes"),
        ("428", "Circulatory"),
        ("785", "Circulatory"),
        ("486", "Respiratory"),
        ("530", "Digestive"),
        ("585", "Genitourinary"),
        ("199", "Neoplasms"),
        ("715", "Musculoskeletal"),
        ("820", "Injury"),
        ("V57", "Other"),
        ("E909", "Other"),
        ("276", "Other"),
        (float("nan"), "Unknown"),
    ],
)
def test_icd9_group(code, expected):
    assert icd9_group(code) == expected


def test_prepare_excludes_death_and_hospice(raw_df):
    out = prepare(raw_df)
    assert not out["discharge_disposition_id"].isin(schema.EXCLUDED_DISCHARGE_CODES).any()
    # the input is not mutated
    assert raw_df["discharge_disposition_id"].isin(schema.EXCLUDED_DISCHARGE_CODES).any()


def test_prepare_groups_unknown_codes_as_strings(raw_df):
    out = prepare(raw_df)
    for col in schema.UNKNOWN_CODES:
        assert out[col].map(type).eq(str).all()
    assert (out["admission_type_id"] == "Unknown").any()
    assert not out["admission_type_id"].isin(["5", "6", "8"]).any()


def test_prepare_caps_and_prior_visits(raw_df):
    out = prepare(raw_df)
    assert out["number_inpatient"].max() <= schema.CAPS["number_inpatient"]
    assert out["number_diagnoses"].max() <= schema.CAPS["number_diagnoses"]
    expected = (
        raw_df.loc[out.index, ["number_outpatient", "number_emergency", "number_inpatient"]]
        .sum(axis=1)
    )
    pd.testing.assert_series_equal(out["prior_visits"], expected, check_names=False)


def test_prepare_age_ord_merges_young_groups(raw_df):
    out = prepare(raw_df)
    assert out["age_ord"].min() == schema.MIN_AGE_ORD
    assert out.loc[raw_df["age"] == "[70-80)", "age_ord"].eq(7).all()


def test_prepare_keeps_patient_and_race_but_they_are_not_features(raw_df):
    out = prepare(raw_df)
    numeric, categorical = get_feature_columns(out)
    assert config.GROUP_COL in out.columns and "race" in out.columns
    features = set(numeric) | set(categorical)
    for banned in ["patient_nbr", "race", "encounter_id", "weight", "examide",
                   config.TARGET, config.TARGET_RAW]:
        assert banned not in features


def test_no_nulls_left_in_features(raw_df):
    out = prepare(raw_df)
    numeric, categorical = get_feature_columns(out)
    assert not out[numeric + categorical].isna().any().any()


def test_preprocessor_output_is_numeric_and_handles_unseen_category(raw_df):
    out = prepare(raw_df)
    numeric, categorical = get_feature_columns(out)
    pre = build_preprocessor(numeric, categorical, min_frequency=5)
    pre.fit(out[numeric + categorical])
    new = out[numeric + categorical].head(3).copy()
    new["medical_specialty"] = "A-specialty-never-seen"
    result = pre.transform(new)
    assert result.shape[0] == 3
    assert result.notna().all().all()
