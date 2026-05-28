"""Create the final 4-figure presentation set for ANDD antibody v2.

:
 CSV / prediction outputs, pandas + matplotlib 
 PNG dataset
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
PRED_DIR = ROOT / "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions"
MULTISEED = ROOT / "outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.csv"
CONTACT_METRICS = ROOT / (
    "outputs/andd_antibody_v2_stratified/contact_feature_audit/"
    "cdr3_contact_augmented_metrics.csv"
)

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
BLACK = "#111111"
GRID = "#D0D0D0"


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    }
)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_cross_attention_predictions() -> dict[str, pd.DataFrame]:
    files = {
        "Train": PRED_DIR / "all_cdr_cross_attention_train_predictions.csv",
        "Validation": PRED_DIR / "all_cdr_cross_attention_val_predictions.csv",
        "Test": PRED_DIR / "all_cdr_cross_attention_test_predictions.csv",
    }
    frames = {}
    for split, path in files.items():
        df = pd.read_csv(path)
        required = {"true_neg_log10_affinity", "predicted_neg_log10_affinity"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        df = df.copy()
        df["residual"] = df["predicted_neg_log10_affinity"] - df["true_neg_log10_affinity"]
        frames[split] = df
    return frames


def prediction_stats(df: pd.DataFrame) -> dict[str, float]:
    true = df["true_neg_log10_affinity"]
    pred = df["predicted_neg_log10_affinity"]
    error = pred - true
    return {
        "n": len(df),
        "mae": float(error.abs().mean()),
        "spearman": float(true.corr(pred, method="spearman")),
        "std_ratio": float(pred.std(ddof=1) / true.std(ddof=1)),
    }


def padded_limits(values: list[pd.Series | np.ndarray], pad: float = 0.35) -> tuple[float, float]:
    concat = np.concatenate([np.asarray(v, dtype=float) for v in values])
    return float(np.nanmin(concat) - pad), float(np.nanmax(concat) + pad)


def set_metric_ylim(ax, values, errors=None, include_one=False, top_extra=0.22) -> None:
    values = np.asarray(values, dtype=float)
    errors = np.zeros_like(values) if errors is None else np.asarray(errors, dtype=float)
    high = float(np.nanmax(values + errors))
    if include_one:
        high = max(high, 1.0)
    top = high + max(high, 0.2) * top_extra
    ax.set_ylim(0, top)


def add_bar_labels(ax, xs, means, errors=None) -> None:
    errors = np.zeros(len(means)) if errors is None else np.asarray(errors, dtype=float)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.025
    for x, mean, err in zip(xs, means, errors):
        ax.text(x, mean + err + pad, f"{mean:.3f}", ha="center", va="bottom", fontsize=10)


def figure1_prediction_compression() -> Path:
    frames = load_cross_attention_predictions()
    lo, hi = padded_limits(
        [
            *(df["true_neg_log10_affinity"] for df in frames.values()),
            *(df["predicted_neg_log10_affinity"] for df in frames.values()),
        ],
        pad=0.35,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), sharex=True, sharey=True)
    for ax, (split, df) in zip(axes, frames.items()):
        true = df["true_neg_log10_affinity"]
        pred = df["predicted_neg_log10_affinity"]
        stats = prediction_stats(df)
        slope, intercept = np.polyfit(true, pred, 1)
        xs = np.linspace(lo, hi, 120)

        ax.scatter(true, pred, s=22, color=ORANGE, alpha=0.65, edgecolor="white", linewidth=0.3)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=BLACK, linewidth=1.2, label="ideal y=x")
        ax.plot(xs, slope * xs + intercept, color=RED, linewidth=2.0, label="fitted trend")
        ax.set_title(split)
        ax.set_xlabel("True -log10(Kd)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22, color=GRID)
        ax.text(
            0.04,
            0.96,
            (
                f"n={stats['n']}\n"
                f"MAE={stats['mae']:.3f}\n"
                f"Spearman={stats['spearman']:.3f}\n"
                f"pred/true std={stats['std_ratio']:.3f}\n"
                f"trend slope={slope:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
        )
    axes[0].set_ylabel("Predicted -log10(Kd)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Figure 1. Prediction Compression Across Splits", y=1.02)
    path = FIG_DIR / "final_fig1_prediction_compression_across_splits.png"
    savefig(path)
    return path


def figure2_residual_trend() -> Path:
    frames = load_cross_attention_predictions()
    xlo, xhi = padded_limits([df["true_neg_log10_affinity"] for df in frames.values()], pad=0.25)
    y_abs = max(float(df["residual"].abs().max()) for df in frames.values()) + 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), sharex=True, sharey=True)
    for ax, (split, df) in zip(axes, frames.items()):
        true = df["true_neg_log10_affinity"]
        residual = df["residual"]
        slope, intercept = np.polyfit(true, residual, 1)
        xs = np.linspace(xlo, xhi, 120)

        ax.scatter(true, residual, s=22, color=ORANGE, alpha=0.65, edgecolor="white", linewidth=0.3)
        ax.axhline(0, linestyle="--", color=BLACK, linewidth=1.1)
        ax.plot(xs, slope * xs + intercept, color=RED, linewidth=2.0)
        ax.set_title(split)
        ax.set_xlabel("True -log10(Kd)")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-y_abs, y_abs)
        ax.grid(alpha=0.22, color=GRID)
        ax.text(
            0.04,
            0.94,
            f"residual trend slope={slope:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
        )
    axes[0].set_ylabel("Residual = predicted - true")
    fig.suptitle("Figure 2. Residual Trend Shows Regression-to-the-Mean", y=1.02)
    path = FIG_DIR / "final_fig2_residual_trend_regression_to_mean.png"
    savefig(path)
    return path


def figure3_multiseed_tailaware() -> Path:
    df = pd.read_csv(MULTISEED)
    primary = df[(df["summary_type"].isin(["mean", "std"])) & (df["policy"] == "best_val_tail_mae")]
    groups = ["unweighted", "tailaware_w2"]
    labels = ["Unweighted\ncross-attention", "Tail-aware w2"]
    colors = [BLUE, GREEN]
    metrics = [
        ("MAE", "MAE (lower)", False),
        ("Spearman", "Spearman (higher)", False),
        ("pred_std_true_std", "pred_std / true_std (closer to 1)", True),
        ("tail_MAE", "P10/P90 Tail MAE (lower)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.2))
    axes = axes.ravel()
    x = np.arange(2)
    for ax, (metric, title, include_one) in zip(axes, metrics):
        means, stds = [], []
        for group in groups:
            means.append(float(primary[(primary["summary_type"] == "mean") & (primary["group"] == group)][metric].iloc[0]))
            stds.append(float(primary[(primary["summary_type"] == "std") & (primary["group"] == group)][metric].iloc[0]))
        set_metric_ylim(ax, means, stds, include_one=include_one)
        ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="white", linewidth=0.8)
        add_bar_labels(ax, x, means, stds)
        if include_one:
            ax.axhline(1.0, linestyle="--", color=BLACK, linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=8, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, color=GRID)
    fig.subplots_adjust(hspace=0.42, wspace=0.20, top=0.88, bottom=0.12)
    fig.suptitle("Figure 3. Multi-Seed Validation of Tail-Aware W2", y=1.02)
    path = FIG_DIR / "final_fig3_multiseed_tailaware_w2.png"
    savefig(path)
    return path


def figure4_contact_augmentation() -> Path:
    df = pd.read_csv(CONTACT_METRICS)
    df = df[df["sequence_baseline"] == "tailaware_w2_best_val_tail_mae"].copy()
    df["method_label"] = df["method"].map(
        {
            "sequence_only_prediction": "Sequence only",
            "sequence_plus_contact_ridge_residual": "Sequence + CDR3 contact",
        }
    )
    df["subset_label"] = df["subset"].map(
        {
            "hcdr3_lcdr3_contact_safe": "HCDR3+LCDR3 safe\n(test n=58)",
            "all_cdr_contact_safe": "All-CDR safe\n(test n=49)",
        }
    )

    subsets = ["HCDR3+LCDR3 safe\n(test n=58)", "All-CDR safe\n(test n=49)"]
    methods = ["Sequence only", "Sequence + CDR3 contact"]
    metrics = [
        ("MAE", "MAE (lower)", False),
        ("Spearman", "Spearman (higher)", False),
        ("tail_MAE", "P10/P90 Tail MAE (lower)", False),
        ("pred_std_true_std", "pred_std / true_std (closer to 1)", True),
    ]
    colors = [BLUE, RED]
    width = 0.34
    x = np.arange(len(subsets))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.5))
    axes = axes.ravel()
    legend_handles = []
    for ax, (metric, title, include_one) in zip(axes, metrics):
        all_values = []
        for method in methods:
            for subset in subsets:
                all_values.append(
                    float(df[(df["subset_label"] == subset) & (df["method_label"] == method)][metric].iloc[0])
                )
        set_metric_ylim(ax, all_values, include_one=include_one)
        for i, method in enumerate(methods):
            values = []
            for subset in subsets:
                values.append(
                    float(df[(df["subset_label"] == subset) & (df["method_label"] == method)][metric].iloc[0])
                )
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[i],
                edgecolor="white",
                linewidth=0.8,
                label=method,
            )
            if len(legend_handles) < 2:
                legend_handles.append(bars[0])
            add_bar_labels(ax, x + offset, values)
        if include_one:
            ax.axhline(1.0, linestyle="--", color=BLACK, linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(subsets)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, color=GRID)

    fig.subplots_adjust(hspace=0.42, wspace=0.20, top=0.88, bottom=0.18)
    fig.suptitle("Figure 4. CDR3 Contact Augmentation", y=0.98)
    fig.legend(legend_handles, methods, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.06))
    fig.text(
        0.5,
        0.01,
        "Contact-covered subset only; not directly comparable to full 1,168-row benchmark.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )
    path = FIG_DIR / "final_fig4_cdr3_contact_augmentation.png"
    savefig(path)
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        figure1_prediction_compression(),
        figure2_residual_trend(),
        figure3_multiseed_tailaware(),
        figure4_contact_augmentation(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
