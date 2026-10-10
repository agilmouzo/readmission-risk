"""Write the evaluation results: CSV tables (the numbers) and PNG figures (the pictures).

The figures follow a simple house style: one series colour, hairline solid grid, thin marks,
text in ink colours and a few direct labels instead of a number on every point. Every figure has
a table twin in `reports/*.csv`, so nothing is readable from colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from readmission import config
from readmission.evaluation import F1_THRESHOLD_NAME, EvaluationResult

REPORTS_DIR = config.ROOT / "reports"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
SERIES = "#2a78d6"
REFERENCE = "#a5a49f"
GRID = "#e6e5e0"
AXIS = "#c9c8c3"

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
}


def write_tables(result: EvaluationResult, out_dir: Path = REPORTS_DIR) -> list[Path]:
    """Save every table as CSV (plus the calibration summary as JSON)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "test_metrics.csv": result.test_metrics.rename_axis("metric"),
        "operating_points.csv": result.operating_points.rename_axis("operating_point"),
        "lift_deciles.csv": result.lift.set_index("decile"),
        "calibration.csv": result.calibration.rename_axis("bin"),
        "subgroups.csv": result.subgroups.set_index("variable"),
    }
    paths = []
    for name, table in tables.items():
        path = out_dir / name
        table.to_csv(path, float_format="%.6f")
        paths.append(path)
    summary_path = out_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(result.calibration_summary, indent=2) + "\n")
    return [*paths, summary_path]


def write_figures(result: EvaluationResult, out_dir: Path) -> list[Path]:
    """Save the PNG figures. matplotlib is imported here, so only this step needs it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    drawers = {
        "pr_curve.png": _plot_pr_curve,
        "calibration.png": _plot_calibration,
        "risk_deciles.png": _plot_risk_deciles,
        "subgroup_auc.png": _plot_subgroups,
    }
    paths = []
    with plt.rc_context(_RC):
        for name, draw in drawers.items():
            fig = draw(result, plt)
            if fig is None:  # nothing reliable to show
                continue
            path = out_dir / name
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def _percent_axis(ax, axis: str = "y") -> None:
    from matplotlib.ticker import PercentFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _plot_pr_curve(result: EvaluationResult, plt):
    from sklearn.metrics import precision_recall_curve

    y, p = result.y_test, result.p_test
    precision, recall, _ = precision_recall_curve(y, p)
    prevalence = float(np.mean(y))
    auc = result.test_metrics.loc["auc_pr", "estimate"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(recall, precision, color=SERIES, linewidth=2)
    ax.axhline(prevalence, color=REFERENCE, linewidth=1.2)
    ax.text(
        0.99,
        prevalence + 0.008,
        f"No model: {prevalence:.1%}",
        ha="right",
        va="bottom",
        color=INK_2,
        fontsize=9,
    )

    point = result.operating_points.loc[F1_THRESHOLD_NAME]
    ax.plot(
        point["recall"],
        point["precision"],
        "o",
        color=SERIES,
        markersize=9,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=3,
    )
    ax.annotate(
        f"{F1_THRESHOLD_NAME} threshold:\nrecall {point['recall']:.0%}, "
        f"precision {point['precision']:.0%}",
        xy=(point["recall"], point["precision"]),
        xytext=(12, 14),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )

    informative = precision[recall >= 0.02]
    top = min(1.0, max(0.4, float(informative.max()) * 1.1)) if informative.size else 1.0
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top)
    _percent_axis(ax, "x")
    _percent_axis(ax, "y")
    ax.set_xlabel("Recall (share of readmissions flagged)")
    ax.set_ylabel("Precision (share of flagged who are readmitted)")
    ax.set_title(f"Precision–recall on the test set (AUC-PR {auc:.3f})")
    return fig


def _plot_calibration(result: EvaluationResult, plt):
    cal = result.calibration
    summary = result.calibration_summary
    limit = float(max(cal["mean_pred"].max(), cal["observed"].max())) * 1.15

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, limit], [0, limit], color=REFERENCE, linewidth=1.2)
    ax.text(
        limit * 0.98,
        limit * 0.93,
        "Perfect calibration",
        ha="right",
        va="top",
        color=INK_2,
        fontsize=9,
        rotation=0,
    )
    ax.plot(cal["mean_pred"], cal["observed"], color=SERIES, linewidth=1.5)
    ax.plot(
        cal["mean_pred"],
        cal["observed"],
        "o",
        color=SERIES,
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=3,
    )
    ax.text(
        0.98,
        0.04,
        f"slope {summary['slope']:.2f} · ECE {summary['ece']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=INK_2,
        fontsize=9,
    )

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    _percent_axis(ax, "x")
    _percent_axis(ax, "y")
    ax.set_xlabel("Mean predicted risk (by decile)")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Calibration on the test set")
    return fig


def _plot_risk_deciles(result: EvaluationResult, plt):
    lift = result.lift
    prevalence = float(np.mean(result.y_test))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(lift["decile"], lift["rate"], width=0.62, color=SERIES)
    ax.axhline(prevalence, color=REFERENCE, linewidth=1.2)
    ax.text(
        len(lift) + 0.45,
        prevalence + 0.004,
        f"All patients: {prevalence:.1%}",
        ha="right",
        va="bottom",
        color=INK_2,
        fontsize=9,
    )
    first = lift.iloc[0]
    ax.text(
        first["decile"],
        first["rate"] + 0.004,
        f"{first['rate']:.1%} ({first['lift']:.1f}× average)",
        ha="left",
        va="bottom",
        color=INK,
        fontsize=9,
    )

    ax.set_xticks(lift["decile"])
    ax.set_xlim(0.4, len(lift) + 0.6)
    ax.grid(axis="x", visible=False)
    _percent_axis(ax, "y")
    ax.set_xlabel("Predicted-risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Readmission rate by predicted-risk decile (test set)")
    return fig


def _plot_subgroups(result: EvaluationResult, plt):
    report = result.subgroups[result.subgroups["reliable"]]
    if report.empty:
        return None
    labels, is_header, values, lows, highs = [], [], [], [], []
    titles = {"race": "Race", "gender": "Gender", "age_band": "Age band"}
    for variable, sub in report.groupby("variable", sort=False):
        labels.append(titles.get(variable, variable))
        is_header.append(True)
        values.append(np.nan)
        lows.append(np.nan)
        highs.append(np.nan)
        for _, row in sub.iterrows():
            labels.append(f"{row['group']}  (n={int(row['n']):,})")
            is_header.append(False)
            values.append(row["auc_roc"])
            lows.append(row.get("auc_roc_low", np.nan))
            highs.append(row.get("auc_roc_high", np.nan))

    positions = np.arange(len(labels))[::-1]
    overall = float(result.test_metrics.loc["auc_roc", "estimate"])
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(labels) + 1.6))
    ax.axvline(overall, color=REFERENCE, linewidth=1.2)
    lows_a, highs_a = np.array(lows), np.array(highs)
    has_ci = ~np.isnan(lows_a) & ~np.isnan(highs_a)
    ax.hlines(
        positions[has_ci], lows_a[has_ci], highs_a[has_ci], color=SERIES, linewidth=1.5, zorder=2
    )
    ax.scatter(values, positions, s=80, color=SERIES, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.text(
        overall,
        positions.max() + 0.9,
        f"All patients: {overall:.3f}",
        ha="center",
        va="bottom",
        color=INK_2,
        fontsize=9,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    for tick, header in zip(ax.get_yticklabels(), is_header, strict=True):
        if header:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    finite = [v for v in [*values, *lows_a[has_ci], *highs_a[has_ci]] if not np.isnan(v)]
    ax.set_xlim(min(finite + [overall]) - 0.03, max(finite + [overall]) + 0.03)
    ax.set_ylim(-0.7, positions.max() + 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC-ROC on the test set (bars: 95% CI, resampling patients)")
    ax.set_title("AUC-ROC by subgroup", loc="left")
    return fig
