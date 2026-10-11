"""Explain the final model with SHAP.

Usage:
    python -m readmission.explain

SHAP splits each prediction into one contribution per variable: starting from the average risk of
the model, every variable pushes the patient's risk up (positive) or down (negative). Contributions
are in log-odds, the scale on which the model adds things up; only their sign and relative size
matter for reading them.

The model one-hot encodes categorical variables, so SHAP first produces one contribution per
encoded column (`insulin_Steady`, `insulin_No`, ...). They are summed back into one contribution
per ORIGINAL variable, which is what a person wants to read. The sum is exact because SHAP values
are additive.

The explanations use a sample of the TRAIN split: the test set was already spent in the evaluation
phase and nothing here may feed back into modelling decisions.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from readmission import config
from readmission.artifact import load_artifact, model_inputs
from readmission.data import add_binary_target, load_raw
from readmission.features import prepare
from readmission.reporting import (
    _RC,
    INK,
    INK_2,
    REFERENCE,
    REPORTS_DIR,
    SERIES,
)
from readmission.split import patient_split

DEFAULT_SAMPLE = 5000
TOP_N = 15


@dataclass
class Explanation:
    """SHAP results for a sample of patients."""

    inputs: pd.DataFrame  # original (readable) values of the model inputs
    contributions: pd.DataFrame  # SHAP per ORIGINAL variable, log-odds
    base_value: float  # average model output (log-odds)
    logit: np.ndarray  # model output (log-odds) for each row
    numeric: list[str]
    categorical: list[str]

    @property
    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit))


def encoded_matrix(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every step of the pipeline except the classifier (encoding, scaling, renaming)."""
    return pipeline[:-1].transform(X)


def _sanitised(name: str) -> str:
    """The same renaming the pipeline applies to column names (`-` becomes `_`, etc.)."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name)


def original_variable(encoded_name: str, numeric: list[str], categorical: list[str]) -> str:
    """Map an encoded column name (`cat__insulin_Steady`, `num__age_ord`) to its variable.

    Encoded names come out of the pipeline already sanitised (`glyburide-metformin` became
    `glyburide_metformin`), so variable names are compared in sanitised form. The longest name
    wins: `glyburide_metformin_No` belongs to `glyburide-metformin`, not to `glyburide`.
    """
    if encoded_name.startswith("num__"):
        candidates, name = numeric, encoded_name[len("num__") :]
    elif encoded_name.startswith("cat__"):
        candidates, name = categorical, encoded_name[len("cat__") :]
    else:
        raise ValueError(f"Unexpected encoded column name: {encoded_name!r}")
    for variable in sorted(candidates, key=lambda v: len(_sanitised(v)), reverse=True):
        key = _sanitised(variable)
        if name == key or (encoded_name.startswith("cat__") and name.startswith(f"{key}_")):
            return variable
    raise ValueError(f"Cannot match {encoded_name!r} to any model variable")


def group_contributions(
    shap_encoded: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Sum the SHAP values of the encoded columns that come from the same variable."""
    owner = [original_variable(c, numeric, categorical) for c in shap_encoded.columns]
    return shap_encoded.T.groupby(owner).sum().T[[*numeric, *categorical]]


def _positive_class_values(values) -> np.ndarray:
    """SHAP returns a list [negative, positive] in some versions and one array in others."""
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, 1]
    return values


def explain_sample(
    artifact: dict,
    df: pd.DataFrame,
    n_samples: int = DEFAULT_SAMPLE,
    seed: int = config.RANDOM_STATE,
) -> Explanation:
    """SHAP values for a random sample of the rows of `df`."""
    import shap

    sample = df.sample(n=min(n_samples, len(df)), random_state=seed)
    X = model_inputs(sample, artifact)
    pipeline = artifact["pipeline"]
    matrix = encoded_matrix(pipeline, X)

    # Calling the explainer (instead of `.shap_values`) is the current API and avoids a warning
    # about the output format of LightGBM binary models.
    result = shap.TreeExplainer(pipeline[-1])(matrix)
    values = _positive_class_values(result.values)
    base = float(np.ravel(np.asarray(result.base_values)[..., -1])[0])
    shap_encoded = pd.DataFrame(values, columns=matrix.columns, index=X.index)

    contributions = group_contributions(shap_encoded, artifact["numeric"], artifact["categorical"])
    return Explanation(
        inputs=X,
        contributions=contributions,
        base_value=base,
        logit=base + contributions.sum(axis=1).to_numpy(),
        numeric=list(artifact["numeric"]),
        categorical=list(artifact["categorical"]),
    )


def global_importance(explanation: Explanation) -> pd.DataFrame:
    """Mean absolute contribution of each variable: how much it moves predictions on average."""
    importance = explanation.contributions.abs().mean().sort_values(ascending=False)
    table = importance.rename("mean_abs_shap").to_frame()
    table["share"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    return table.rename_axis("variable")


def numeric_effects(explanation: Explanation, variable: str) -> pd.DataFrame:
    """Average contribution of a numeric variable at each of its values."""
    frame = pd.DataFrame(
        {
            "value": explanation.inputs[variable].to_numpy(),
            "shap": explanation.contributions[variable].to_numpy(),
        }
    )
    return frame.groupby("value")["shap"].agg(["mean", "size"]).reset_index()


def explain_patient(explanation: Explanation, row: int, top: int = 8) -> pd.DataFrame:
    """Largest contributions for one patient (`row` is a position in the sample)."""
    contributions = explanation.contributions.iloc[row]
    order = contributions.abs().sort_values(ascending=False).index
    shown = list(order[:top])
    rest = float(contributions.drop(shown).sum())
    table = pd.DataFrame(
        {
            "variable": shown,
            "value": [explanation.inputs.iloc[row][v] for v in shown],
            "contribution": [contributions[v] for v in shown],
        }
    )
    if len(order) > top:
        other = pd.DataFrame(
            {
                "variable": [f"{len(order) - top} other variables"],
                "value": [""],
                "contribution": [rest],
            }
        )
        table = pd.concat([table, other], ignore_index=True)
    return table


def pick_example_patients(explanation: Explanation) -> dict[str, int]:
    """Positions of the highest-risk and the lowest-risk patient in the sample."""
    probability = explanation.probability
    return {"highest risk": int(np.argmax(probability)), "lowest risk": int(np.argmin(probability))}


# --------------------------------------------------------------------------- output


def write_tables(explanation: Explanation, out_dir=REPORTS_DIR) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shap_importance.csv"
    global_importance(explanation).to_csv(path, float_format="%.6f")
    return [path]


def write_figures(explanation: Explanation, out_dir) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(_RC):
        for name, draw in {
            "shap_importance.png": _plot_importance,
            "shap_patients.png": _plot_patients,
            "shap_numeric.png": _plot_numeric_effects,
        }.items():
            fig = draw(explanation, plt)
            if fig is None:
                continue
            fig.tight_layout()
            path = out_dir / name
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def _plot_importance(explanation: Explanation, plt):
    table = global_importance(explanation).head(TOP_N)
    positions = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(table) + 1.4))
    ax.barh(positions, table["mean_abs_shap"], height=0.62, color=SERIES)
    offset = table["mean_abs_shap"].max() * 0.01
    for pos, value, share in zip(positions, table["mean_abs_shap"], table["share"], strict=True):
        ax.text(value + offset, pos, f"{share:.0%}", va="center", color=INK_2, fontsize=8.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(table.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average impact on the prediction (mean |SHAP|, log-odds)")
    ax.set_title(f"Which variables move the risk most (top {len(table)})")
    ax.set_xlim(0, table["mean_abs_shap"].max() * 1.15)
    return fig


def _plot_patients(explanation: Explanation, plt):
    examples = pick_example_patients(explanation)
    fig, axes = plt.subplots(1, len(examples), figsize=(11.5, 4.6), sharex=True)
    limit = 0.0
    tables = {}
    for label, row in examples.items():
        tables[label] = explain_patient(explanation, row)
        limit = max(limit, tables[label]["contribution"].abs().max())
    for ax, (label, row) in zip(axes, examples.items(), strict=True):
        table = tables[label]
        positions = np.arange(len(table))[::-1]
        colours = [SERIES if c > 0 else REFERENCE for c in table["contribution"]]
        ax.barh(positions, table["contribution"], height=0.6, color=colours)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        names = [
            f"{v} = {val}" if val != "" else str(v)
            for v, val in zip(table["variable"], table["value"], strict=True)
        ]
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=8.5)
        for pos, c in zip(positions, table["contribution"], strict=True):
            ax.text(
                c + (0.02 * limit if c >= 0 else -0.02 * limit),
                pos,
                f"{c:+.2f}",
                va="center",
                ha="left" if c >= 0 else "right",
                color=INK_2,
                fontsize=8.5,
            )
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-limit * 1.3, limit * 1.3)
        ax.set_title(
            f"{label.capitalize()}: {explanation.probability[row]:.0%} predicted risk",
            fontsize=10.5,
        )
        ax.set_xlabel("Contribution (log-odds; blue raises risk, grey lowers it)")
    average = 1.0 / (1.0 + np.exp(-explanation.base_value))
    fig.suptitle(
        f"Two example patients (the average patient is at {average:.0%})",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    return fig


def _plot_numeric_effects(explanation: Explanation, plt):
    importance = global_importance(explanation)
    variables = [v for v in importance.index if v in explanation.numeric][:4]
    if not variables:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
    for ax, variable in zip(axes.ravel(), variables, strict=False):
        effect = numeric_effects(explanation, variable)
        # Values seen in very few patients give a noisy average: only draw where there is data.
        effect = effect[effect["size"] >= 30]
        if effect.empty:
            ax.text(0.5, 0.5, "too few patients", ha="center", va="center", color=INK_2)
            ax.set_title(variable, fontsize=10.5)
            continue
        ax.axhline(0, color=REFERENCE, linewidth=1.0)
        ax.plot(
            effect["value"], effect["mean"], color=SERIES, linewidth=1.8, marker="o", markersize=4
        )
        ax.set_title(variable, fontsize=10.5)
        ax.set_xlabel("Value")
        ax.set_ylabel("Average contribution (log-odds)")
    for ax in axes.ravel()[len(variables) :]:
        ax.set_visible(False)
    fig.suptitle(
        "How the main numeric variables change the risk",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=INK,
    )
    return fig


def print_summary(explanation: Explanation) -> None:
    pd.options.display.float_format = "{:.3f}".format
    average = 1.0 / (1.0 + np.exp(-explanation.base_value))
    print(f"\nExplained {len(explanation.inputs)} patients; average model risk {average:.1%}")
    print("\nMost influential variables (share = percentage of the total average impact):")
    print(global_importance(explanation).head(TOP_N).to_string())
    for label, row in pick_example_patients(explanation).items():
        print(f"\nExample, {label} patient (predicted risk {explanation.probability[row]:.1%}):")
        print(explain_patient(explanation, row).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE, help="patients to explain")
    args = parser.parse_args(argv)

    artifact = load_artifact()
    df = prepare(add_binary_target(load_raw()))
    train, _ = patient_split(df)  # the test split is deliberately left alone
    explanation = explain_sample(artifact, train, n_samples=args.samples)
    print_summary(explanation)

    write_tables(explanation, REPORTS_DIR)
    write_figures(explanation, REPORTS_DIR / "figures")
    print(f"\nSaved tables to {REPORTS_DIR} and figures to {REPORTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
