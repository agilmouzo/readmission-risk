import pandas as pd
import pytest

from readmission import hospital_value as hv


@pytest.fixture
def deciles() -> pd.DataFrame:
    # Two groups of 500 patients: the first with a 30% rate, the second with a 2% rate.
    return pd.DataFrame({"decile": [1, 2], "n": [500, 500], "readmissions": [150, 10]})


def test_net_saving_by_cutoff_matches_hand_calculation(deciles):
    table = hv.net_saving_by_cutoff(deciles, cost=300, effect=0.2, readmission_cost=10_000)
    # Treat nobody: 0. Treat the first half: 150 readmissions per 1,000 x 0.2 x 10,000 =
    # 300,000, minus 500 x 300 = 150,000. Treat everyone: 160 x 0.2 x 10,000 - 1,000 x 300.
    assert table["net_saving"].tolist() == pytest.approx([0, 150_000, 20_000])
    assert table["treated"].tolist() == pytest.approx([0, 500, 1000])


def test_cutoff_with_most_saving_is_the_one_whose_last_group_still_pays_off(deciles):
    table = hv.net_saving_by_cutoff(deciles, cost=300, effect=0.2, readmission_cost=10_000)
    assert table["net_saving"].idxmax() == 1
    # The second group (2% rate) is below the break-even probability, 300 / (0.2 x 10,000).
    assert hv.break_even_probability(300, 0.2, 10_000) == pytest.approx(0.15)
    assert 0.02 < 0.15 < 0.30


def test_with_no_effect_the_best_choice_is_to_treat_nobody(deciles):
    table = hv.net_saving_by_cutoff(deciles, cost=300, effect=0.0, readmission_cost=10_000)
    assert table["net_saving"].idxmax() == 0


def test_published_deciles_give_a_consistent_grid():
    deciles = pd.read_csv(hv.DECILES_CSV)
    grid = hv.scenario_grid(deciles)
    assert len(grid) == len(hv.INTERVENTION_COSTS) * len(hv.EFFECTS)
    # Treating the best cut-off can never be worse than treating nobody or everybody.
    assert (grid["best_net_saving"] >= 0).all()
    assert (grid["gain_over_treating_everyone"] >= -1e-9).all()
