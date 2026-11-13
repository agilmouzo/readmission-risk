"""Illustrative economics of using the model in a hospital.

Answers one question: given what an intervention costs, what it prevents and what a
readmission costs, how many of the highest-risk patients are worth treating, and how much
would that save per 1,000 discharges?

The calculation uses only reports/lift_deciles.csv (test set, patients sorted from highest to
lowest predicted risk). The costs and the effect sizes are ASSUMPTIONS, not results of this
project: see the section "How it could be used in a hospital" of the README.

Usage:  python -m readmission.hospital_value
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from readmission import config

DECILES_CSV = config.ROOT / "reports" / "lift_deciles.csv"
OUTPUT_CSV = config.ROOT / "reports" / "hospital_value.csv"

# Mean cost of a 30-day readmission for endocrine, nutritional and metabolic conditions
# (US$, 2020), AHRQ HCUP Statistical Brief #304. The brief gives US$ 16,300 for all conditions.
READMISSION_COST = 15_200

# Cost of the intervention per treated patient (US$) and share of readmissions it prevents.
INTERVENTION_COSTS = (100, 300, 1_000, 3_000)
EFFECTS = (0.10, 0.20, 0.30)


def net_saving_by_cutoff(
    deciles: pd.DataFrame, cost: float, effect: float, readmission_cost: float = READMISSION_COST
) -> pd.DataFrame:
    """Net saving per 1,000 discharges when treating the top 0, 1, ..., 10 deciles of risk.

    Net saving = readmissions prevented x cost of a readmission - patients treated x cost of
    the intervention. The effect is assumed equal in every risk group.
    """
    ordered = deciles.sort_values("decile")
    total = ordered["n"].sum()
    treated = ordered["n"].cumsum() / total * 1000
    caught = ordered["readmissions"].cumsum() / total * 1000
    table = pd.DataFrame(
        {
            "deciles_treated": range(1, len(ordered) + 1),
            "treated": treated.to_numpy(),
            "readmissions_in_treated": caught.to_numpy(),
        }
    )
    none = pd.DataFrame(
        {"deciles_treated": [0], "treated": [0.0], "readmissions_in_treated": [0.0]}
    )
    table = pd.concat([none, table], ignore_index=True)
    table["net_saving"] = (
        table["readmissions_in_treated"] * effect * readmission_cost - table["treated"] * cost
    )
    return table


def break_even_probability(cost: float, effect: float, readmission_cost: float) -> float:
    """Treat a patient only if their readmission probability exceeds cost / (effect x R)."""
    return cost / (effect * readmission_cost)


def scenario_grid(deciles: pd.DataFrame) -> pd.DataFrame:
    """Best cut-off for every combination of cost and effect, against the two extremes."""
    rows = []
    for cost in INTERVENTION_COSTS:
        for effect in EFFECTS:
            table = net_saving_by_cutoff(deciles, cost, effect)
            best = table.loc[table["net_saving"].idxmax()]
            rows.append(
                {
                    "intervention_cost": cost,
                    "effect": effect,
                    "break_even_probability": break_even_probability(
                        cost, effect, READMISSION_COST
                    ),
                    "best_deciles_treated": int(best["deciles_treated"]),
                    "best_net_saving": best["net_saving"],
                    "treat_everyone_net_saving": table.iloc[-1]["net_saving"],
                }
            )
    grid = pd.DataFrame(rows)
    grid["gain_over_treating_everyone"] = (
        grid["best_net_saving"] - grid["treat_everyone_net_saving"]
    )
    return grid


def main(deciles_path: Path = DECILES_CSV, output_path: Path = OUTPUT_CSV) -> None:
    deciles = pd.read_csv(deciles_path)
    grid = scenario_grid(deciles)
    grid.to_csv(output_path, index=False)
    print(f"Readmission cost: US$ {READMISSION_COST:,}. Net saving per 1,000 discharges (US$)")
    print(grid.round(3).to_string(index=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
