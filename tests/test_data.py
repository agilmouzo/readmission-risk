import pandas as pd

from readmission import config
from readmission.data import add_binary_target, load_raw


def test_add_binary_target():
    df = pd.DataFrame({config.TARGET_RAW: ["<30", ">30", "NO", "<30"]})
    out = add_binary_target(df)
    assert out[config.TARGET].tolist() == [1, 0, 0, 1]
    assert config.TARGET not in df.columns  # input is not mutated


def test_load_raw_replaces_question_marks(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("race,readmitted\n?,NO\nCaucasian,<30\n")
    df = load_raw(csv)
    assert df["race"].isna().tolist() == [True, False]

def test_load_raw_keeps_none_as_a_value(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("A1Cresult,readmitted\nNone,NO\n>8,<30\n")
    df = load_raw(csv)
    assert df["A1Cresult"].tolist() == ["None", ">8"]
    