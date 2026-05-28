"""Make final presentation figures for ANDD antibody v2 affinity regression.

:
 prediction / metrics CSV, pandas + matplotlib 
, dataset,, final_reports/figures
 presentation figures
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
SUMMARY_DIR = ROOT / "outputs" / "final_reports"

FIT_METRICS = ROOT / "outputs/andd_antibody_v2_stratified/fit_diagnosis/fit_metrics_by_split.csv"
POOLED_TEST = ROOT / (
    "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions/"
    "all_cdr_pooled_test_predictions.csv"
)
CROSS_TEST = ROOT / (
    "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions/"
    "all_cdr_cross_attention_test_predictions.csv"
)
TAIL_W2_TEST = ROOT / (
    "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware_w2/"
    "tailaware_w2_test_predictions_best_val_tail_mae.csv"
)
MULTISEED = ROOT / "outputs/andd_antibody_v2_stratified/multiseed/multiseed_summary.csv"
CONTACT_METRICS = ROOT / (
    "outputs/andd_antibody_v2_stratified/contact_feature_audit/"
    "cdr3_contact_augmented_metrics.csv"
)


COLORS = {
    "Pooled all-CDR": "#4C78A8",
    "Cross-attention": "#F58518",
    "Tail-aware w2": "#54A24B",
    "Sequence only": "#4C78A8",
    "+ CDR3 contact": "#E45756",
    "Unweighted": "#4C78A8",
    "Tail-aware w2 multi-seed": "#54A24B",
}


def savefig(path: Path) -> None:
    """Save a figure cleanly for slides / Markdown preview."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def add_diag_line(ax, true_values: pd.Series) -> None:
    lo = float(true_values.min()) - 0.2
    hi = float(true_values.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2, label="ideal y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def load_prediction(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"true_neg_log10_affinity", "predicted_neg_log10_affinity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df.copy()
    out["model_label"] = label
    out["residual"] = out["predicted_neg_log10_affinity"] - out["true_neg_log10_affinity"]
    return out


def figure1_prediction_spread() -> Path:
    """Train/val/test prediction spread diagnosis."""

    metrics = pd.read_csv(FIT_METRICS)
    metrics = metrics[metrics["model"].isin(["all_cdr_pooled", "all_cdr_cross_attention"])].copy()
    metrics["model_label"] = metrics["model"].map(
        {
            "all_cdr_pooled": "Pooled all-CDR",
            "all_cdr_cross_attention": "Cross-attention",
        }
    )
    split_order = ["train", "val", "test"]
    model_order = ["Pooled all-CDR", "Cross-attention"]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(split_order))
    width = 0.34
    for i, model in enumerate(model_order):
        sub = metrics.set_index(["split", "model_label"]).loc[(slice(None), model), :].reset_index()
        values = [float(sub[sub["split"] == split]["pred_std_true_std"].iloc[0]) for split in split_order]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[model],
            label=model,
            edgecolor="white",
            linewidth=0.8,
        )
        for xx, yy in zip(x + offset, values):
            ax.text(xx, yy + 0.018, f"{yy:.2f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="ideal spread = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "Validation", "Test"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("prediction std / true std")
    ax.set_title("Figure 1. Train/Val/Test Prediction Spread Diagnosis")
    ax.text(
        0.01,
        0.94,
        "Values far below 1 mean predictions are compressed toward the mean.",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.9),
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    path = FIG_DIR / "figure1_train_val_test_prediction_spread.png"
    savefig(path)
    return path


def figure2_true_vs_predicted(predictions: list[pd.DataFrame]) -> Path:
    """True-vs-predicted scatter with equal axes."""

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    labels = ["Pooled all-CDR", "Cross-attention", "Tail-aware w2"]
    for ax, label in zip(axes, labels):
        df = next(d for d in predictions if d["model_label"].iloc[0] == label)
        ax.scatter(
            df["true_neg_log10_affinity"],
            df["predicted_neg_log10_affinity"],
            s=28,
            alpha=0.72,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.35,
        )
        add_diag_line(ax, df["true_neg_log10_affinity"])
        ax.set_title(label)
        ax.set_xlabel("True -log10(Kd)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Predicted -log10(Kd)")
    fig.suptitle("Figure 2. True vs Predicted: Prediction Compression Is Visible", y=1.02)
    path = FIG_DIR / "figure2_true_vs_predicted_scatter.png"
    savefig(path)
    return path


def figure3_residual_vs_true(predictions: list[pd.DataFrame]) -> Path:
    """Residual-vs-true scatter to show regression-to-the-mean."""

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    labels = ["Pooled all-CDR", "Cross-attention", "Tail-aware w2"]
    all_df = pd.concat(predictions, ignore_index=True)
    x_min = all_df["true_neg_log10_affinity"].min() - 0.2
    x_max = all_df["true_neg_log10_affinity"].max() + 0.2
    y_lim = max(abs(all_df["residual"].min()), abs(all_df["residual"].max())) + 0.3

    for ax, label in zip(axes, labels):
        df = next(d for d in predictions if d["model_label"].iloc[0] == label)
        ax.scatter(
            df["true_neg_log10_affinity"],
            df["residual"],
            s=28,
            alpha=0.72,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.35,
        )
        # : low target high target 
        coeff = np.polyfit(df["true_neg_log10_affinity"], df["residual"], 1)
        xs = np.linspace(x_min, x_max, 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], color="black", linewidth=1.4)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("True -log10(Kd)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(alpha=0.22)
        ax.text(
            0.03,
            0.93,
            f"slope={coeff[0]:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.88),
        )
    axes[0].set_ylabel("Residual = predicted - true")
    fig.suptitle("Figure 3. Residual vs True: Downward Trend = Regression-to-the-Mean", y=1.02)
    path = FIG_DIR / "figure3_residual_vs_true_scatter.png"
    savefig(path)
    return path


def figure4_multiseed() -> tuple[Path, pd.DataFrame]:
    """Multi-seed unweighted vs tail-aware w2 summary."""

    df = pd.read_csv(MULTISEED)
    primary = df[(df["summary_type"].isin(["mean", "std"])) & (df["policy"] == "best_val_tail_mae")]
    groups = ["unweighted", "tailaware_w2"]
    metrics = [
        ("MAE", "MAE", "lower"),
        ("Spearman", "Spearman", "higher"),
        ("pred_std_true_std", "pred std / true std", "closer to 1"),
        ("tail_MAE", "tail MAE", "lower"),
    ]
    rows = []
    for group in groups:
        mean_row = primary[(primary["summary_type"] == "mean") & (primary["group"] == group)].iloc[0]
        std_row = primary[(primary["summary_type"] == "std") & (primary["group"] == group)].iloc[0]
        for metric, label, direction in metrics:
            rows.append(
                {
                    "group": group,
                    "group_label": "Unweighted" if group == "unweighted" else "Tail-aware w2",
                    "metric": metric,
                    "metric_label": label,
                    "direction": direction,
                    "mean": float(mean_row[metric]),
                    "std": float(std_row[metric]),
                }
            )
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    axes = axes.ravel()
    for ax, (metric, label, direction) in zip(axes, metrics):
        sub = summary[summary["metric"] == metric]
        x = np.arange(len(sub))
        colors = [COLORS["Unweighted"], COLORS["Tail-aware w2 multi-seed"]]
        ax.bar(x, sub["mean"], yerr=sub["std"], color=colors, capsize=5, edgecolor="white")
        for xx, yy, ee in zip(x, sub["mean"], sub["std"]):
            ax.text(xx, yy + ee + 0.02 * max(sub["mean"]), f"{yy:.3f}", ha="center", fontsize=9)
        if metric == "pred_std_true_std":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(["Unweighted", "Tail-aware w2"], rotation=20, ha="right")
        ax.set_title(f"{label} ({direction})")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 4. Multi-Seed Validation: Tail-Aware W2 Helps Spread/Tails, Not Overall MAE", y=1.02)
    path = FIG_DIR / "figure4_multiseed_unweighted_vs_tailaware_w2.png"
    savefig(path)
    return path, summary


def figure5_contact_augmentation() -> tuple[Path, pd.DataFrame]:
    """Contact-covered subset result for sequence-only vs CDR3 contact correction."""

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
    metrics = [
        ("MAE", "MAE", "lower"),
        ("Spearman", "Spearman", "higher"),
        ("tail_MAE", "tail MAE", "lower"),
        ("pred_std_true_std", "pred std / true std", "closer to 1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axes = axes.ravel()
    subsets = ["HCDR3+LCDR3 safe\n(test n=58)", "All-CDR safe\n(test n=49)"]
    methods = ["Sequence only", "+ CDR3 contact"]
    width = 0.34
    x = np.arange(len(subsets))
    for ax, (metric, label, direction) in zip(axes, metrics):
        for i, method in enumerate(methods):
            vals = []
            for subset in subsets:
                row = df[(df["subset_label"] == subset) & (df["method_label"] == method)].iloc[0]
                vals.append(float(row[metric]))
            offset = (i - 0.5) * width
            color = COLORS[method]
            ax.bar(x + offset, vals, width=width, color=color, edgecolor="white", label=method)
            for xx, yy in zip(x + offset, vals):
                ax.text(xx, yy + 0.02 * max(vals), f"{yy:.3f}", ha="center", fontsize=8)
        if metric == "pred_std_true_std":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(subsets)
        ax.set_title(f"{label} ({direction})")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 5. CDR3 Contact Augmentation: Small Subset Gains, Compression Remains", y=1.02)
    path = FIG_DIR / "figure5_cdr3_contact_augmentation_subset.png"
    savefig(path)
    return path, df


def write_summary(paths: dict[str, Path], multiseed_summary: pd.DataFrame, contact_summary: pd.DataFrame) -> None:
    """Write a short Markdown guide for slide narration."""

    csv_path = FIG_DIR / "andd_final_presentation_figure_metrics.csv"
    combined = []
    ms = multiseed_summary.copy()
    ms["figure"] = "figure4_multiseed"
    combined.append(ms)
    ct = contact_summary[
        [
            "subset",
            "subset_description",
            "sequence_baseline",
            "method",
            "MAE",
            "RMSE",
            "Spearman",
            "pred_std_true_std",
            "error_vs_true_Pearson",
            "tail_MAE",
            "train_rows",
            "test_rows",
        ]
    ].copy()
    ct["figure"] = "figure5_contact_augmentation"
    combined.append(ct)
    pd.concat(combined, ignore_index=True, sort=False).to_csv(csv_path, index=False)

    summary_path = SUMMARY_DIR / "andd_final_presentation_figures_summary.md"
    lines = [
        "# ANDD Final Presentation Figures Summary",
        "",
        " 5  CSV / prediction outputs;, dataset",
        "",
        "## Figures",
    ]
    for name, path in paths.items():
        lines.append(f"- {name}: `{path.relative_to(ROOT)}`")
    lines += [
        f"- Metrics CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## How To Present",
        "",
        "1. **Figure 1**: train/val/test  prediction spread  1, compression  train , overfit",
        "2. **Figure 2**: true vs predicted ,",
        "3. **Figure 3**: residual vs true  downward trend, low target high target , regression-to-the-mean",
        "4. **Figure 4**: multi-seed  tail-aware w2  spread  tail MAE, MAE/Spearman, single-seed overclaim",
        "5. **Figure 5**: CDR3 contact features  contact-covered subset , pred_std/true_std  1,",
        "",
        "## One-Sentence Takeaway",
        "",
        "> The model's main bottleneck is not a simple code or split issue; cross-attention, tail-aware loss, and CDR3 contact features each help part of the symptom, but richer structure/contact-aware representation is still needed.",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pooled = load_prediction(POOLED_TEST, "Pooled all-CDR")
    cross = load_prediction(CROSS_TEST, "Cross-attention")
    tail_w2 = load_prediction(TAIL_W2_TEST, "Tail-aware w2")

    paths: dict[str, Path] = {}
    paths["Figure 1 - Train/Val/Test prediction spread"] = figure1_prediction_spread()
    paths["Figure 2 - True vs predicted"] = figure2_true_vs_predicted([pooled, cross, tail_w2])
    paths["Figure 3 - Residual vs true"] = figure3_residual_vs_true([pooled, cross, tail_w2])
    fig4_path, multiseed_summary = figure4_multiseed()
    paths["Figure 4 - Multi-seed unweighted vs tail-aware w2"] = fig4_path
    fig5_path, contact_summary = figure5_contact_augmentation()
    paths["Figure 5 - CDR3 contact augmentation subset result"] = fig5_path

    write_summary(paths, multiseed_summary, contact_summary)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(f"Summary: {SUMMARY_DIR / 'andd_final_presentation_figures_summary.md'}")


if __name__ == "__main__":
    main()
