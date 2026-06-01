"""Generate a presentation figure for ESM2 backbone scaling experiments.

This script reads existing experiment outputs only. It does not train models or
modify datasets. The 650M bs4 run was stopped early, so only validation
snapshots recorded in its summary are included.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_reports" / "figures"
FIGURE_PATH = OUTPUT_DIR / "backbone_scaling_8M_35M_150M_650M.png"
CSV_PATH = OUTPUT_DIR / "backbone_scaling_8M_35M_150M_650M.csv"

HISTORY_FILES = {
    "ESM2 35M, seed 42, bs1": ROOT
    / "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_esm35M_unweighted/training_history.csv",
    "ESM2 150M, seed 42, bs1": ROOT
    / "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_esm150M_unweighted/training_history.csv",
    "ESM2 150M, seed 123, bs1": ROOT
    / "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_esm150M_unweighted_seed123/training_history.csv",
}

METRICS_35M = ROOT / (
    "outputs/andd_antibody_v2_stratified/"
    "cross_attention_all_cdrs_esm35M_unweighted/"
    "train_val_test_metrics_best_val_tail_mae.csv"
)
METRICS_150M_S42 = ROOT / (
    "outputs/andd_antibody_v2_stratified/"
    "cross_attention_all_cdrs_esm150M_unweighted/"
    "train_val_test_metrics_best_val_tail_mae.csv"
)
REPORT_150M_S123 = ROOT / (
    "outputs/andd_antibody_v2_stratified/"
    "cross_attention_all_cdrs_esm150M_unweighted_seed123/"
    "esm150M_cross_attention_report.md"
)
HISTORY_650M_BS4 = ROOT / (
    "outputs/andd_antibody_v2_stratified/"
    "cross_attention_all_cdrs_esm650M_unweighted_s2026_lr1e-5_e150_bs4/"
    "training_history_completed_epochs.csv"
)


def recorded_650m_validation_snapshots() -> pd.DataFrame:
    """Return validation snapshots documented in the existing pilot summaries."""

    rows = [
        # bs14 25-epoch pilot summary snapshots.
        ("ESM2 650M, seed 2026, bs14 pilot", 1, 2.3289, -0.0054),
        ("ESM2 650M, seed 2026, bs14 pilot", 5, 1.0490, 0.2360),
        ("ESM2 650M, seed 2026, bs14 pilot", 10, 1.0218, 0.3468),
        ("ESM2 650M, seed 2026, bs14 pilot", 15, 1.0098, 0.3978),
        ("ESM2 650M, seed 2026, bs14 pilot", 20, 0.9899, 0.4501),
        ("ESM2 650M, seed 2026, bs14 pilot", 25, 0.9910, 0.4385),
    ]
    frame = pd.DataFrame(rows, columns=["model", "epoch", "val_mae", "val_spearman"])
    frame["record_type"] = "validation_snapshot"
    return frame


def read_history() -> pd.DataFrame:
    """Read validation curves from saved training histories."""

    frames = []
    for model, path in HISTORY_FILES.items():
        frame = pd.read_csv(path)
        frame["model"] = model
        frame["record_type"] = "validation_history"
        frames.append(frame[["model", "record_type", "epoch", "val_mae", "val_spearman"]])
    bs4 = pd.read_csv(HISTORY_650M_BS4)
    bs4["model"] = "ESM2 650M, seed 2026, bs4 stopped"
    bs4["record_type"] = "validation_history"
    frames.append(bs4[["model", "record_type", "epoch", "val_mae", "val_spearman"]])
    return pd.concat([*frames, recorded_650m_validation_snapshots()], ignore_index=True)


def test_row_from_csv(path: Path) -> dict[str, float]:
    """Read the test row from a train/validation/test metrics CSV."""

    frame = pd.read_csv(path)
    row = frame.loc[frame["split"] == "test"].iloc[0]
    return {
        "MAE": float(row["mae"]),
        "Spearman": float(row["spearman"]),
    }


def test_row_from_report(path: Path) -> dict[str, float]:
    """Read the cross-attention test row from the seed-123 markdown report."""

    text = path.read_text(encoding="utf-8")
    pattern = (
        r"\| `cross_attention_all_cdrs` \|\s*\d+\s*\|\s*"
        r"([0-9.]+)\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)"
    )
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse test metrics from {path}")
    return {"MAE": float(match.group(1)), "Spearman": float(match.group(2))}


def read_test_metrics() -> pd.DataFrame:
    """Collect comparable completed test evaluations."""

    rows = [
        {"model": "ESM2 8M baseline", "MAE": 0.9523, "Spearman": 0.3861},
        {"model": "ESM2 35M, seed 42", **test_row_from_csv(METRICS_35M)},
        {"model": "ESM2 150M, seed 42", **test_row_from_csv(METRICS_150M_S42)},
        {"model": "ESM2 150M, seed 123", **test_row_from_report(REPORT_150M_S123)},
    ]
    frame = pd.DataFrame(rows)
    frame["record_type"] = "test_evaluation"
    return frame


def annotate_bars(axis: plt.Axes, bars, values: pd.Series) -> None:
    """Add compact values above bars."""

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_figure(validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """Render the 2x2 comparison figure."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {
        "ESM2 35M, seed 42, bs1": "#4C78A8",
        "ESM2 150M, seed 42, bs1": "#59A14F",
        "ESM2 150M, seed 123, bs1": "#8CD17D",
        "ESM2 650M, seed 2026, bs14 pilot": "#F28E2B",
        "ESM2 650M, seed 2026, bs4 stopped": "#E15759",
    }

    for axis, metric, title, ylabel in [
        (axes[0, 0], "val_mae", "Validation MAE During Training", "Validation MAE (lower is better)"),
        (
            axes[0, 1],
            "val_spearman",
            "Validation Spearman During Training",
            "Validation Spearman (higher is better)",
        ),
    ]:
        for model, frame in validation.groupby("model", sort=False):
            is_snapshot = frame["record_type"].iloc[0] == "validation_snapshot"
            axis.plot(
                frame["epoch"],
                frame[metric],
                color=colors[model],
                label=model,
                linewidth=2 if not is_snapshot else 1.8,
                linestyle="--" if is_snapshot else "-",
                marker="o" if is_snapshot else None,
                markersize=5,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    bar_colors = ["#9C755F", "#4C78A8", "#59A14F", "#8CD17D"]
    for axis, metric, title, ylabel in [
        (axes[1, 0], "MAE", "Completed Test Evaluation: MAE", "Test MAE (lower is better)"),
        (
            axes[1, 1],
            "Spearman",
            "Completed Test Evaluation: Spearman",
            "Test Spearman (higher is better)",
        ),
    ]:
        bars = axis.bar(test["model"], test[metric], color=bar_colors, alpha=0.9)
        annotate_bars(axis, bars, test[metric])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.16)

    axes[0, 0].legend(loc="upper right", fontsize=8)
    figure.suptitle("ESM2 Backbone Scaling: Capacity Helps, but 650M Pilot Plateaus", fontsize=15)
    figure.text(
        0.5,
        0.015,
        "Top: validation trajectories. 650M lines contain recorded pilot snapshots only. "
        "Bottom: completed test evaluations only; 650M is excluded because no comparable test evaluation was run.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    figure.savefig(FIGURE_PATH, dpi=300)
    plt.close(figure)


def main() -> None:
    """Generate figure and its source-data CSV."""

    validation = read_history()
    test = read_test_metrics()
    validation_export = validation.copy()
    validation_export["split"] = "validation"
    test_export = test.copy()
    test_export["split"] = "test"
    pd.concat([validation_export, test_export], ignore_index=True, sort=False).to_csv(CSV_PATH, index=False)
    make_figure(validation, test)
    print(f"Saved figure to {FIGURE_PATH}")
    print(f"Saved plotted data to {CSV_PATH}")


if __name__ == "__main__":
    main()
