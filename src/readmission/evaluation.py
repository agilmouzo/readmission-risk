"""Evaluation metrics: thresholds, lift, patient-level bootstrap, calibration and subgroups.

Everything here is a pure function over arrays / DataFrames (no files, no plotting), so it is easy
to test. `readmission.evaluate` wires it to the data and `readmission.reporting` writes the output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from readmission import config

# A metric takes (y, p, weights) and returns a float. Weights are None for the point estimate and
# the bootstrap multiplicities of each row inside a resample.
MetricFn = Callable[[np.ndarray, np.ndarray, "np.ndarray | None"], float]

TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
F1_THRESHOLD_NAME = "F1-optimal"
AGE_BAND_LABELS = ["<40", "40-59", "60-79", "80+"]


# --------------------------------------------------------------------------- thresholds


def threshold_max_f1(y, p) -> float:
    """Probability threshold that maximises F1 (choose it on out-of-fold predictions)."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # the last point has no threshold
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_top_fraction(p, fraction: float) -> float:
    """Threshold that flags (about) the `fraction` of patients with the highest predicted risk."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    return float(np.quantile(np.asarray(p, dtype=float), 1 - fraction))


def operating_point(y, p, threshold: float) -> dict[str, float]:
    """What happens if every patient with predicted risk >= threshold is flagged."""
    y = np.asarray(y).astype(int)
    flagged = np.asarray(p, dtype=float) >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    prevalence = y.mean()
    return {
        "threshold": float(threshold),
        "flagged_share": flagged.mean(),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": f1,
        "lift": precision / prevalence if prevalence else float("nan"),
    }


def operating_points(y, p, thresholds: Mapping[str, float]) -> pd.DataFrame:
    """One row per named threshold."""
    rows = {name: operating_point(y, p, thr) for name, thr in thresholds.items()}
    return pd.DataFrame(rows).T


def lift_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Readmission rate by risk bin (bin 1 = highest predicted risk) and cumulative capture."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    prevalence, total_positives = y.mean(), y.sum()
    rows, captured = [], 0.0
    for i, idx in enumerate(np.array_split(np.arange(len(y)), n_bins), start=1):
        positives = y_sorted[idx].sum()
        captured += positives
        rate = positives / len(idx)
        rows.append(
            {
                "decile": i,
                "n": len(idx),
                "readmissions": int(positives),
                "rate": rate,
                "mean_pred": p_sorted[idx].mean(),
                "lift": rate / prevalence,
                "cum_capture": captured / total_positives,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- bootstrap


def _weights(y: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    return np.ones(len(y)) if w is None else np.asarray(w, dtype=float)


def auc_pr(y, p, w=None) -> float:
    return float(average_precision_score(np.asarray(y).astype(int), p, sample_weight=w))


def auc_roc(y, p, w=None) -> float:
    return float(roc_auc_score(np.asarray(y).astype(int), p, sample_weight=w))


def safe_auc_roc(y, p, w=None) -> float:
    """AUC-ROC, or NaN when a resample has no cases of one of the two classes."""
    y = np.asarray(y)
    weights = _weights(y, w)
    positives = weights[y == 1].sum()
    if positives == 0 or positives == weights.sum():
        return float("nan")
    return auc_roc(y, p, w)


def brier(y, p, w=None) -> float:
    return float(brier_score_loss(np.asarray(y).astype(int), p, sample_weight=w))


def precision_at_fraction(y, p, fraction: float, w=None) -> float:
    """Share of true readmissions among the `fraction` of patients with the highest risk.

    With integer weights ("this row was drawn k times") the result equals the one computed on the
    replicated rows: the row where the cut falls is only partly counted.
    """
    y = np.asarray(y, dtype=float)
    weights = _weights(y, w)
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")
    weights_sorted, y_sorted = weights[order], y[order]
    cumulative = np.cumsum(weights_sorted)
    target = np.ceil(fraction * cumulative[-1])  # size of the top group, in (replicated) rows
    k = int(np.searchsorted(cumulative, target, side="left"))
    selected = weights_sorted[: k + 1].copy()
    selected[-1] -= cumulative[k] - target
    return float((selected * y_sorted[: k + 1]).sum() / selected.sum())


def lift_at_fraction(y, p, fraction: float, w=None) -> float:
    """`precision_at_fraction` divided by the overall readmission rate."""
    y = np.asarray(y, dtype=float)
    return precision_at_fraction(y, p, fraction, w) / float(np.average(y, weights=_weights(y, w)))


def precision_at_threshold(threshold: float) -> MetricFn:
    def metric(y, p, w=None) -> float:
        weights = _weights(np.asarray(y), w)
        flagged = np.asarray(p, dtype=float) >= threshold
        denominator = (weights * flagged).sum()
        if denominator == 0:
            return float("nan")
        return float((weights * np.asarray(y) * flagged).sum() / denominator)

    return metric


def recall_at_threshold(threshold: float) -> MetricFn:
    def metric(y, p, w=None) -> float:
        weights = _weights(np.asarray(y), w)
        flagged = np.asarray(p, dtype=float) >= threshold
        positives = (weights * np.asarray(y)).sum()
        return float((weights * np.asarray(y) * flagged).sum() / positives)

    return metric


def _precision_top10(y, p, w=None) -> float:
    return precision_at_fraction(y, p, 0.10, w)


def _lift_top10(y, p, w=None) -> float:
    return lift_at_fraction(y, p, 0.10, w)


def default_metrics(threshold: float | None = None) -> dict[str, MetricFn]:
    """The headline metrics; with a threshold, also its precision and recall."""
    metrics: dict[str, MetricFn] = {
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "brier": brier,
        "precision_top10": _precision_top10,
        "lift_top10": _lift_top10,
    }
    if threshold is not None:
        metrics["precision_at_f1_threshold"] = precision_at_threshold(threshold)
        metrics["recall_at_f1_threshold"] = recall_at_threshold(threshold)
    return metrics


def cluster_bootstrap(
    y,
    p,
    groups,
    metrics: Mapping[str, MetricFn],
    n_boot: int = 1000,
    seed: int = config.RANDOM_STATE,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Point estimate and percentile CI for each metric, resampling PATIENTS with replacement.

    Resampling rows would treat several encounters of one patient as independent evidence and make
    the intervals too narrow. Each resample is implemented as integer weights per patient (how many
    times the patient was drawn), which is equivalent and much faster than copying rows.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    codes, uniques = pd.factorize(pd.Series(np.asarray(groups)))
    if (codes < 0).any():
        raise ValueError("groups must not contain missing values")
    n_groups = len(uniques)
    rng = np.random.default_rng(seed)

    draws = {name: np.empty(n_boot) for name in metrics}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        weights = counts[codes].astype(float)
        for name, fn in metrics.items():
            draws[name][b] = fn(y, p, weights)

    rows = {}
    for name, fn in metrics.items():
        low, high = np.nanquantile(draws[name], [alpha / 2, 1 - alpha / 2])
        rows[name] = {"estimate": fn(y, p, None), "ci_low": low, "ci_high": high}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- calibration


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted risk vs observed rate in equal-sized bins of predicted risk."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"], n_bins, labels=False, duplicates="drop")
    table = frame.groupby("bin").agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")
    )
    return table.reset_index(drop=True)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Average gap between predicted and observed risk across bins, weighted by bin size."""
    gap = (table["observed"] - table["mean_pred"]).abs()
    return float(np.average(gap, weights=table["n"]))


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic regression of the outcome on logit(p). Perfect calibration: slope 1, intercept 0.

    A slope below 1 means the predictions are too extreme (overconfident); above 1, too timid.
    """
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    # A huge C makes the penalty irrelevant (unpenalised fit) in every scikit-learn version.
    fit = LogisticRegression(C=1e9, max_iter=1000).fit(logit, np.asarray(y).astype(int))
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


# --------------------------------------------------------------------------- subgroups


def age_band(age_ord: pd.Series) -> pd.Series:
    """Coarse age bands from the decade index (age_ord 3 = 30-39 years, 9 = 90-99)."""
    return pd.cut(age_ord, bins=[-np.inf, 3, 5, 7, np.inf], labels=AGE_BAND_LABELS).astype(str)


def subgroup_report(
    df: pd.DataFrame,
    group_col: str,
    y_col: str,
    p_col: str,
    threshold: float,
    min_n: int = 200,
    min_events: int = 20,
    patient_col: str | None = None,
    n_boot: int = 0,
) -> pd.DataFrame:
    """Performance per subgroup. Groups too small to judge are listed with `reliable=False`.

    With `patient_col` and `n_boot > 0`, the AUC-ROC of reliable groups also gets a patient-level
    bootstrap interval (`auc_roc_low`, `auc_roc_high`): a gap between groups only means something
    if the intervals do not overlap.
    """
    labels = df[group_col].astype("string").fillna("Unknown")
    rows = []
    for name in sorted(labels.unique()):
        sub = df[labels == name]
        y = sub[y_col].to_numpy().astype(int)
        p = sub[p_col].to_numpy(dtype=float)
        events = int(y.sum())
        reliable = len(sub) >= min_n and events >= min_events and len(sub) - events >= min_events
        row = {
            "variable": group_col,
            "group": name,
            "n": len(sub),
            "readmissions": events,
            "observed_rate": y.mean(),
            "mean_pred": p.mean(),
            "reliable": reliable,
        }
        if reliable:
            point = operating_point(y, p, threshold)
            row |= {
                "auc_roc": auc_roc(y, p),
                "auc_pr": auc_pr(y, p),
                "flagged_share": point["flagged_share"],
                "precision": point["precision"],
                "recall": point["recall"],
            }
            if patient_col is not None and n_boot > 0:
                ci = cluster_bootstrap(
                    y, p, sub[patient_col], {"auc_roc": safe_auc_roc}, n_boot=n_boot
                )
                row["auc_roc_low"] = ci.loc["auc_roc", "ci_low"]
                row["auc_roc_high"] = ci.loc["auc_roc", "ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_gaps(
    report: pd.DataFrame,
    columns: Sequence[str] = ("auc_roc", "recall"),
    min_n: int = 1000,
) -> pd.DataFrame:
    """Largest difference between groups of each variable, for each metric.

    Only reliable groups with at least `min_n` patients are compared: in small groups the metrics
    move a lot by chance, and the "gap" would mostly measure noise.
    """
    rows = []
    comparable = report[report["reliable"] & (report["n"] >= min_n)]
    for variable, sub in comparable.groupby("variable"):
        if len(sub) < 2:
            continue
        row = {"variable": variable, "groups": len(sub)}
        for column in columns:
            row[f"{column}_gap"] = sub[column].max() - sub[column].min()
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- result container


@dataclass
class EvaluationResult:
    """Everything the evaluation produces; consumed by `readmission.reporting`."""

    thresholds: dict[str, float]
    test_metrics: pd.DataFrame
    operating_points: pd.DataFrame
    lift: pd.DataFrame
    calibration: pd.DataFrame
    calibration_summary: dict[str, float]
    subgroups: pd.DataFrame
    y_test: np.ndarray
    p_test: np.ndarray
