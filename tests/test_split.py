from readmission import config
from readmission.split import patient_split


def test_patient_split_has_no_patient_overlap(raw_df):
    train, test = patient_split(raw_df, test_size=0.25)
    assert set(train[config.GROUP_COL]).isdisjoint(set(test[config.GROUP_COL]))
    assert len(train) + len(test) == len(raw_df)


def test_patient_split_is_reproducible(raw_df):
    a_train, _ = patient_split(raw_df)
    b_train, _ = patient_split(raw_df)
    assert a_train.index.equals(b_train.index)
