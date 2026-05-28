""" ANDD stratified  regression-to-the-mean 

 test predictions,
 ANDD stratified antigen-level test split,
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ,,
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/seqproft_xdg_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/seqproft_matplotlib_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
POOLED_PATH = (
    ROOT
    / "outputs"
    / "andd_antibody_v2_stratified"
    / "all_cdr_pooled"
    / "andd_antibody_v2_stratified_all_cdr_pooled_test_predictions.csv"
)
CROSS_ATTENTION_PATH = (
    ROOT
    / "outputs"
    / "andd_antibody_v2_stratified"
    / "cross_attention_all_cdrs"
    / "test_predictions.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "final_reports" / "figures"
FIGURE_PATH = OUTPUT_DIR / "regression_to_mean_scatter.png"
SUMMARY_PATH = OUTPUT_DIR / "regression_to_mean_scatter_summary.md"

TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    """ prediction CSV """
    frame = pd.read_csv(path)
    required = {"sample_id", TRUE_COLUMN, PRED_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}  columns: {missing}")

    result = frame[["sample_id", TRUE_COLUMN, PRED_COLUMN]].copy()
    result[TRUE_COLUMN] = pd.to_numeric(result[TRUE_COLUMN], errors="coerce")
    result[PRED_COLUMN] = pd.to_numeric(result[PRED_COLUMN], errors="coerce")
    result = result.dropna(subset=[TRUE_COLUMN, PRED_COLUMN])
    result["residual"] = result[PRED_COLUMN] - result[TRUE_COLUMN]
    result["model"] = model_name
    return result


def validate_same_test_set(pooled: pd.DataFrame, cross_attention: pd.DataFrame) -> None:
    """ test samples  target"""
    pooled_truth = pooled.set_index("sample_id")[TRUE_COLUMN].sort_index()
    cross_truth = cross_attention.set_index("sample_id")[TRUE_COLUMN].sort_index()
    if not pooled_truth.index.equals(cross_truth.index):
        raise ValueError("pooled  cross-attention prediction CSV  sample_id ")
    if not np.allclose(pooled_truth.to_numpy(), cross_truth.to_numpy()):
        raise ValueError("pooled  cross-attention prediction CSV  true target ")


def diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    """ regression-to-the-mean """
    true = frame[TRUE_COLUMN]
    pred = frame[PRED_COLUMN]
    residual = frame["residual"]
    true_std = float(true.std(ddof=1))
    pred_std = float(pred.std(ddof=1))
    return {
        "rows": float(len(frame)),
        "pred_std_true_std": pred_std / true_std if true_std else float("nan"),
        "error_vs_true_pearson": float(residual.corr(true, method="pearson")),
        "residual_slope": float(np.polyfit(true, residual, 1)[0]),
        "prediction_slope": float(np.polyfit(true, pred, 1)[0]),
    }


def draw_scatter(pooled: pd.DataFrame, cross_attention: pd.DataFrame) -> None:
    """ true-vs-predicted  residual-vs-true  panel"""
    styles = {
        "All-CDR pooled": {"color": "#247BA0", "marker": "o"},
        "All-CDR cross-attention": {"color": "#C65A4A", "marker": "^"},
    }
    combined = pd.concat([pooled, cross_attention], ignore_index=True)
    values = np.concatenate(
        [combined[TRUE_COLUMN].to_numpy(), combined[PRED_COLUMN].to_numpy()]
    )
    padding = max((values.max() - values.min()) * 0.06, 0.15)
    lower, upper = values.min() - padding, values.max() + padding

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#A8ADB4",
            "axes.titleweight": "bold",
            "xtick.color": "#39424E",
            "ytick.color": "#39424E",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for model_name, frame in [
        ("All-CDR pooled", pooled),
        ("All-CDR cross-attention", cross_attention),
    ]:
        style = styles[model_name]
        axes[0].scatter(
            frame[TRUE_COLUMN],
            frame[PRED_COLUMN],
            label=model_name,
            color=style["color"],
            marker=style["marker"],
            s=42,
            alpha=0.62,
            edgecolors="white",
            linewidths=0.35,
        )
        axes[1].scatter(
            frame[TRUE_COLUMN],
            frame["residual"],
            label=model_name,
            color=style["color"],
            marker=style["marker"],
            s=42,
            alpha=0.62,
            edgecolors="white",
            linewidths=0.35,
        )

        # , residual  true target 
        true_grid = np.linspace(frame[TRUE_COLUMN].min(), frame[TRUE_COLUMN].max(), 120)
        pred_fit = np.polyfit(frame[TRUE_COLUMN], frame[PRED_COLUMN], 1)
        residual_fit = np.polyfit(frame[TRUE_COLUMN], frame["residual"], 1)
        axes[0].plot(
            true_grid,
            np.polyval(pred_fit, true_grid),
            color=style["color"],
            linewidth=1.5,
            alpha=0.95,
        )
        axes[1].plot(
            true_grid,
            np.polyval(residual_fit, true_grid),
            color=style["color"],
            linewidth=1.5,
            alpha=0.95,
        )

    axes[0].plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="#4C5560",
        linewidth=1.2,
        label="Ideal y = x",
    )
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("True vs predicted affinity")
    axes[0].set_xlabel("True neg_log10_affinity")
    axes[0].set_ylabel("Predicted neg_log10_affinity")

    axes[1].axhline(
        0, linestyle="--", color="#4C5560", linewidth=1.2, label="Ideal residual = 0"
    )
    axes[1].set_title("Residual vs true affinity")
    axes[1].set_xlabel("True neg_log10_affinity")
    axes[1].set_ylabel("Prediction - true")
    axes[1].text(
        0.04,
        0.06,
        "Downward residual trend = regression to the mean",
        transform=axes[1].transAxes,
        fontsize=9.5,
        color="#374151",
        bbox={
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "boxstyle": "round,pad=0.35",
            "alpha": 0.92,
        },
    )

    for axis in axes:
        axis.grid(color="#DDE2E8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=9, loc="best")

    fig.suptitle(
        "ANDD stratified test set: regression-to-the-mean diagnostic",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        "Same stratified antigen-level test split (117 rows); dashed reference lines show ideal behavior. "
        "Fitted lines visualize compression/bias, not a new trained model.",
        fontsize=9,
        color="#4A5560",
    )
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(
    pooled: pd.DataFrame,
    cross_attention: pd.DataFrame,
    pooled_stats: dict[str, float],
    cross_stats: dict[str, float],
) -> None:
    """ beginner-friendly """
    report = f"""# Regression-To-The-Mean Scatter Summary

## 

 ANDD antibody v2 stratified antigen-level test set  predictions:

- All-CDR pooled:`{POOLED_PATH.relative_to(ROOT)}`
- All-CDR cross-attention:`{CROSS_ATTENTION_PATH.relative_to(ROOT)}`

 predictions  `sample_id`  true target , `{len(pooled)}`  test samples,

## 

### True vs Predicted Scatter

-  affinity target, prediction
-  `y=x` 
-  target , prediction , **prediction spread **: affinity

### Residual vs True Scatter

- residual  `prediction - true`
-  `y=0` 
- , residual  true target , target , target , **regression-to-the-mean**

##  Split 

| Model | Test rows | pred_std / true_std | error vs true Pearson | residual trend slope | prediction trend slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| All-CDR pooled | {int(pooled_stats['rows'])} | {pooled_stats['pred_std_true_std']:.4f} | {pooled_stats['error_vs_true_pearson']:.4f} | {pooled_stats['residual_slope']:.4f} | {pooled_stats['prediction_slope']:.4f} |
| All-CDR cross-attention | {int(cross_stats['rows'])} | {cross_stats['pred_std_true_std']:.4f} | {cross_stats['error_vs_true_pearson']:.4f} | {cross_stats['residual_slope']:.4f} | {cross_stats['prediction_slope']:.4f} |

:

- `pred_std / true_std`  `1`,
- `error vs true Pearson`  residual trend slope  `0`,regression-to-the-mean 

## 

-  regression-to-the-mean: `y=x` ,residual 
- All-CDR pooled  `pred_std / true_std = {pooled_stats['pred_std_true_std']:.4f}`,
- All-CDR cross-attention  `pred_std / true_std = {cross_stats['pred_std_true_std']:.4f}`, pooled  `1`
- Cross-attention  `error vs true Pearson = {cross_stats['error_vs_true_pearson']:.4f}`, pooled  `{pooled_stats['error_vs_true_pearson']:.4f}`  `0`
- , stratified test split ,**cross-attention  regression-to-the-mean **, overall MAE 

:pooled all-CDR  MAE , cross-attention  prediction spreadranking  high-affinity tail 
"""
    SUMMARY_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled = load_predictions(POOLED_PATH, "All-CDR pooled")
    cross_attention = load_predictions(CROSS_ATTENTION_PATH, "All-CDR cross-attention")
    validate_same_test_set(pooled, cross_attention)

    pooled_stats = diagnostics(pooled)
    cross_stats = diagnostics(cross_attention)
    draw_scatter(pooled, cross_attention)
    write_summary(pooled, cross_attention, pooled_stats, cross_stats)

    print(f"Saved figure: {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Saved summary: {SUMMARY_PATH.relative_to(ROOT)}")
    print("Pooled pred_std/true_std:", f"{pooled_stats['pred_std_true_std']:.4f}")
    print(
        "Cross-attention pred_std/true_std:",
        f"{cross_stats['pred_std_true_std']:.4f}",
    )
    print("Pooled error_vs_true Pearson:", f"{pooled_stats['error_vs_true_pearson']:.4f}")
    print(
        "Cross-attention error_vs_true Pearson:",
        f"{cross_stats['error_vs_true_pearson']:.4f}",
    )


if __name__ == "__main__":
    main()
