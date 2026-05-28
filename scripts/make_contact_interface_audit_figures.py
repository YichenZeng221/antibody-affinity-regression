"""Generate presentation figures for contact/interface audit steps.

中文说明：
只读取已有 contact audit CSV，生成两张汇报用 PNG：
1. Contact/interface availability funnel
2. CDR mapping validation summary

不训练模型、不修改 dataset、不覆盖已有实验结果。
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
CONTACT_AVAILABILITY = ROOT / (
    "outputs/andd_antibody_v2_stratified/contact_feature_audit/"
    "contact_feature_availability.csv"
)
CDR_MAPPING = ROOT / (
    "outputs/andd_antibody_v2_stratified/contact_feature_audit/"
    "cdr_mapping_availability.csv"
)

BLUE = "#4C78A8"
GREEN = "#54A24B"
ORANGE = "#F58518"
RED = "#E45756"
GRAY = "#777777"
GRID = "#D0D0D0"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
    }
)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def as_bool(series: pd.Series) -> pd.Series:
    """兼容 bool / string bool 两种 CSV 读入形式。"""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def add_bar_labels(ax, bars, suffix="") -> None:
    xmax = ax.get_xlim()[1]
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}{suffix}",
            va="center",
            ha="left",
            fontsize=10,
        )


def figure_contact_availability_funnel() -> Path:
    df = pd.read_csv(CONTACT_AVAILABILITY)
    total = len(df)
    counts = [
        ("Total ANDD antibody v2\nstratified rows", total),
        ("Structure file found", int(as_bool(df["any_structure_exists"]).sum())),
        ("Complete H/L/antigen\nchain metadata option", int((df["complete_chain_mapping_option_count"] > 0).sum())),
        ("Unambiguous viable\nchain mapping", int(as_bool(df["unambiguous_chain_mapping"]).sum())),
        ("Basic interface\nfeature-ready", int(as_bool(df["basic_interface_features_ready_for_extraction"]).sum())),
    ]

    labels = [x[0] for x in counts]
    values = [x[1] for x in counts]
    colors = [BLUE, BLUE, BLUE, ORANGE, GREEN]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Rows")
    ax.set_title("Contact / Interface Availability Audit")
    ax.grid(axis="x", alpha=0.25, color=GRID)
    add_bar_labels(ax, bars)

    for i, (_, value) in enumerate(counts):
        pct = value / total * 100 if total else 0
        ax.text(
            value / 2,
            i,
            f"{pct:.1f}%",
            ha="center",
            va="center",
            color="white" if value > total * 0.35 else "#222222",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.01,
        -0.18,
        "Conservative rule: ambiguous chain mappings were not used for contact extraction.",
        transform=ax.transAxes,
        fontsize=10,
        color="#444444",
    )

    path = FIG_DIR / "final_fig5_contact_interface_availability_funnel.png"
    savefig(path)
    return path


def figure_cdr_mapping_validation() -> Path:
    df = pd.read_csv(CDR_MAPPING)
    cdrs = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]
    success_rates = []
    success_counts = []
    total = len(df)
    for cdr in cdrs:
        status_col = f"{cdr}_mapping_status"
        ok = df[status_col].astype(str).str.lower().eq("success")
        success_counts.append(int(ok.sum()))
        success_rates.append(float(ok.mean() * 100))

    safe_counts = [
        ("HCDR3 only", int(as_bool(df["hcdr3_contact_feature_eligible"]).sum())),
        ("LCDR3 only", int(as_bool(df["lcdr3_contact_feature_eligible"]).sum())),
        ("HCDR3+LCDR3", int(as_bool(df["hcdr3_lcdr3_contact_feature_eligible"]).sum())),
        ("All six CDRs", int(as_bool(df["cdr_contact_feature_eligible"]).sum())),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    x = np.arange(len(cdrs))
    bars = ax.bar(x, success_rates, color=[BLUE, BLUE, ORANGE, BLUE, BLUE, GREEN], edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cdrs, rotation=20, ha="right")
    ax.set_ylim(88, 101.5)
    ax.set_ylabel("Mapping success rate (%)")
    ax.set_title("CDR-to-Structure Mapping Success")
    ax.grid(axis="y", alpha=0.25, color=GRID)
    for bar, rate, count in zip(bars, success_rates, success_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.25,
            f"{rate:.1f}%\n({count}/{total})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    labels = [x[0] for x in safe_counts]
    values = [x[1] for x in safe_counts]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=[ORANGE, GREEN, GREEN, BLUE], edgecolor="white", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, total * 1.18)
    ax.set_xlabel("Rows")
    ax.set_title("Contact-Safe Subsets")
    ax.grid(axis="x", alpha=0.25, color=GRID)
    add_bar_labels(ax, bars)
    for i, value in enumerate(values):
        ax.text(
            value / 2,
            i,
            f"{value / total * 100:.1f}%",
            ha="center",
            va="center",
            color="white" if value > total * 0.35 else "#222222",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("CDR Mapping Validation for Contact Features", y=1.03)
    fig.text(
        0.5,
        -0.04,
        "Input is the 472-row unambiguous chain-mapping pilot subset; ambiguous mappings are excluded.",
        ha="center",
        fontsize=10,
        color="#444444",
    )

    path = FIG_DIR / "final_fig6_cdr_mapping_validation.png"
    savefig(path)
    return path


def main() -> None:
    paths = [figure_contact_availability_funnel(), figure_cdr_mapping_validation()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
