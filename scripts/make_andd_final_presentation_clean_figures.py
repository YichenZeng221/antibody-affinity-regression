"""Generate clean final presentation figures for ANDD antibody v2.

 CSV / predictions; dataset
 `_clean`  Figure 1 scatter 
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "final_reports" / "figures"

MULTISEED = ROOT / "outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.csv"
CONTACT_METRICS = ROOT / (
    "outputs/andd_antibody_v2_stratified/contact_feature_audit/"
    "cdr3_contact_augmented_metrics.csv"
)
PRED_DIR = ROOT / "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions"


BLUE = "#4C78A8"
GREEN = "#54A24B"
RED = "#E45756"
ORANGE = "#F58518"
GRID = "#C8C8C8"


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def add_value_labels(ax, xs, means, errors=None, fmt="{:.3f}", pad_fraction=0.035) -> None:
    errors = np.zeros(len(means)) if errors is None else np.asarray(errors)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * pad_fraction
    for x, y, err in zip(xs, means, errors):
        ax.text(x, y + err + pad, fmt.format(y), ha="center", va="bottom", fontsize=10)


def set_padded_ylim(ax, values, errors=None, min_zero=True, top_extra=0.22, include_one=False) -> None:
    values = np.asarray(values, dtype=float)
    errors = np.zeros_like(values) if errors is None else np.asarray(errors, dtype=float)
    low = float(np.nanmin(values - errors))
    high = float(np.nanmax(values + errors))
    if include_one:
        high = max(high, 1.0)
    bottom = 0.0 if min_zero else low - (high - low) * 0.12
    top = high + max(high - bottom, 0.1) * top_extra
    ax.set_ylim(bottom, top)


def metric_from_predictions(df: pd.DataFrame) -> dict[str, float]:
    true = df["true_neg_log10_affinity"]
    pred = df["predicted_neg_log10_affinity"]
    err = pred - true
    return {
        "rows": len(df),
        "MAE": float(err.abs().mean()),
        "Spearman": float(true.corr(pred, method="spearman")),
        "pred_std_true_std": float(pred.std(ddof=1) / true.std(ddof=1)),
    }


def figure1_train_val_test_scatter() -> Path:
    files = {
        "Train split": PRED_DIR / "all_cdr_cross_attention_train_predictions.csv",
        "Validation split": PRED_DIR / "all_cdr_cross_attention_val_predictions.csv",
        "Test split": PRED_DIR / "all_cdr_cross_attention_test_predictions.csv",
    }
    frames = {label: pd.read_csv(path) for label, path in files.items()}
    all_true = pd.concat([df["true_neg_log10_affinity"] for df in frames.values()])
    all_pred = pd.concat([df["predicted_neg_log10_affinity"] for df in frames.values()])
    lo = min(float(all_true.min()), float(all_pred.min())) - 0.35
    hi = max(float(all_true.max()), float(all_pred.max())) + 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), sharex=True, sharey=True)
    for ax, (label, df) in zip(axes, frames.items()):
        true = df["true_neg_log10_affinity"]
        pred = df["predicted_neg_log10_affinity"]
        stats = metric_from_predictions(df)
        ax.scatter(true, pred, s=22, color=ORANGE, alpha=0.66, edgecolor="white", linewidth=0.3)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.2, label="ideal y=x")

        # ; 1 
        slope, intercept = np.polyfit(true, pred, 1)
        xs = np.linspace(lo, hi, 120)
        ax.plot(xs, slope * xs + intercept, color=RED, linewidth=2.0, label="fitted trend")

        ax.set_title(label, fontsize=14)
        ax.set_xlabel("True -log10(Kd)", fontsize=12)
        ax.grid(alpha=0.22, color=GRID)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(
            0.04,
            0.96,
            (
                f"n={stats['rows']}\n"
                f"MAE={stats['MAE']:.3f}\n"
                f"Spearman={stats['Spearman']:.3f}\n"
                f"pred/true std={stats['pred_std_true_std']:.3f}\n"
                f"trend slope={slope:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
        )
    axes[0].set_ylabel("Predicted -log10(Kd)", fontsize=12)
    axes[-1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle(
        "Figure 1. Train/Validation/Test True vs Predicted: "
        "Prediction Compression Persists on Training Data",
        fontsize=17,
        y=1.02,
    )
    path = FIG_DIR / "figure1_train_val_test_true_vs_pred_scatter.png"
    savefig(path)
    return path


def figure4_multiseed_clean() -> Path:
    df = pd.read_csv(MULTISEED)
    primary = df[(df["summary_type"].isin(["mean", "std"])) & (df["policy"] == "best_val_tail_mae")]
    groups = ["unweighted", "tailaware_w2"]
    group_labels = ["Unweighted", "Tail-aware w2"]
    metrics = [
        ("MAE", "MAE (lower)", False),
        ("Spearman", "Spearman (higher)", False),
        ("pred_std_true_std", "pred_std / true_std (closer to 1)", True),
        ("tail_MAE", "P10/P90 Tail MAE (lower)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    axes = axes.ravel()
    colors = [BLUE, GREEN]
    x = np.arange(2)
    for ax, (metric, title, include_one) in zip(axes, metrics):
        means, stds = [], []
        for group in groups:
            means.append(float(primary[(primary["summary_type"] == "mean") & (primary["group"] == group)][metric].iloc[0]))
            stds.append(float(primary[(primary["summary_type"] == "std") & (primary["group"] == group)][metric].iloc[0]))
        set_padded_ylim(ax, means, stds, min_zero=True, top_extra=0.28, include_one=include_one)
        ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="white", linewidth=0.8)
        add_value_labels(ax, x, means, stds)
        if include_one:
            ax.axhline(1.0, linestyle="--", color="black", linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=12, ha="right")
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.25, color=GRID)
    fig.suptitle(
        "Figure 4. Multi-Seed Validation: Tail-Aware W2 Improves Spread/Tails but Not Overall Ranking",
        fontsize=16,
        y=1.02,
    )
    path = FIG_DIR / "figure4_multiseed_unweighted_vs_tailaware_w2_clean.png"
    savefig(path)
    return path


def figure5_contact_clean() -> Path:
    df = pd.read_csv(CONTACT_METRICS)
    df = df[df["sequence_baseline"] == "tailaware_w2_best_val_tail_mae"].copy()
    df["method_label"] = df["method"].map(
        {
            "sequence_only_prediction": "Sequence only",
            "sequence_plus_contact_ridge_residual": "+ CDR3 contact",
        }
    )
    df["subset_label"] = df["subset"].map(
        {
            "hcdr3_lcdr3_contact_safe": "HCDR3+LCDR3 safe\n(test n=58)",
            "all_cdr_contact_safe": "All-CDR safe\n(test n=49)",
        }
    )

    subsets = ["HCDR3+LCDR3 safe\n(test n=58)", "All-CDR safe\n(test n=49)"]
    methods = ["Sequence only", "+ CDR3 contact"]
    metrics = [
        ("MAE", "MAE (lower)", False),
        ("Spearman", "Spearman (higher)", False),
        ("tail_MAE", "P10/P90 Tail MAE (lower)", False),
        ("pred_std_true_std", "pred_std / true_std (closer to 1)", True),
    ]
    colors = [BLUE, RED]
    width = 0.34
    x = np.arange(len(subsets))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    axes = axes.ravel()
    handles = None
    for ax, (metric, title, include_one) in zip(axes, metrics):
        all_vals = []
        for method in methods:
            vals = []
            for subset in subsets:
                val = float(df[(df["subset_label"] == subset) & (df["method_label"] == method)][metric].iloc[0])
                vals.append(val)
            all_vals.extend(vals)
        set_padded_ylim(ax, all_vals, min_zero=True, top_extra=0.24, include_one=include_one)

        bars_for_legend = []
        for i, method in enumerate(methods):
            vals = []
            for subset in subsets:
                val = float(df[(df["subset_label"] == subset) & (df["method_label"] == method)][metric].iloc[0])
                vals.append(val)
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors[i],
                edgecolor="white",
                linewidth=0.8,
                label=method,
            )
            bars_for_legend.append(bars[0])
            add_value_labels(ax, x + offset, vals, fmt="{:.3f}", pad_fraction=0.03)

        if include_one:
            ax.axhline(1.0, linestyle="--", color="black", linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(subsets)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.25, color=GRID)
        handles = bars_for_legend

    fig.suptitle(
        "Figure 5. CDR3 Contact Augmentation: Small Subset Gains, Compression Remains",
        fontsize=16,
        y=1.03,
    )
    fig.legend(handles, methods, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.text(
        0.5,
        -0.055,
        "Contact-covered subset only; not directly comparable to full 1,168-row benchmark.",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    path = FIG_DIR / "figure5_cdr3_contact_augmentation_clean.png"
    savefig(path)
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        figure4_multiseed_clean(),
        figure5_contact_clean(),
        figure1_train_val_test_scatter(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
