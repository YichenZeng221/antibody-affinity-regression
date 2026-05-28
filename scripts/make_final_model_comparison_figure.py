"""Create final two-benchmark model comparison figure.

中文人话说明：
这张图用于 today summary。

重点：
- unified_no_high_risk 和 ANDD antibody v2 是不同 test set。
- 所以图里分成两个 panel / section。
- 可以在同一个 benchmark 内比较模型，但不要跨 benchmark 直接比较 MAE。

图形编码：
- bar: MAE，越低越好。
- dot/line: Spearman，越高越好。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/final_reports/figures"
OUTPUT_PATH = OUTPUT_DIR / "final_model_comparison_two_benchmarks.png"
CSV_PATH = OUTPUT_DIR / "final_model_comparison_two_benchmarks.csv"


def build_results_table() -> pd.DataFrame:
    """Hard-code final reported metrics from the week-1 summaries."""

    rows = [
        {
            "benchmark": "unified_no_high_risk",
            "model": "Whole-sequence",
            "mae": 1.1083,
            "spearman": 0.4557,
        },
        {
            "benchmark": "unified_no_high_risk",
            "model": "All-CDR pooled",
            "mae": 0.9975,
            "spearman": 0.4497,
        },
        {
            "benchmark": "unified_no_high_risk",
            "model": "HCDR3+LCDR3 pooled",
            "mae": 1.0204,
            "spearman": 0.4438,
        },
        {
            "benchmark": "unified_no_high_risk",
            "model": "Dot-product interaction",
            "mae": 1.0504,
            "spearman": 0.4126,
        },
        {
            "benchmark": "unified_no_high_risk",
            "model": "All-CDR cross-attention",
            "mae": 1.0515,
            "spearman": 0.5018,
        },
        {
            "benchmark": "ANDD antibody v2",
            "model": "All-CDR pooled",
            "mae": 0.9066,
            "spearman": 0.3817,
        },
        {
            "benchmark": "ANDD antibody v2",
            "model": "Weighted all-CDR pooled",
            "mae": 0.9042,
            "spearman": 0.3697,
        },
    ]
    return pd.DataFrame(rows)


def make_plot(results: pd.DataFrame) -> None:
    """Draw MAE bars and Spearman dots with benchmark separation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(CSV_PATH, index=False)

    x_positions = list(range(len(results)))
    colors = [
        "#4C78A8" if benchmark == "unified_no_high_risk" else "#F58518"
        for benchmark in results["benchmark"]
    ]

    fig, ax_mae = plt.subplots(figsize=(12, 6.5))

    bars = ax_mae.bar(
        x_positions,
        results["mae"],
        color=colors,
        alpha=0.82,
        label="MAE (lower is better)",
    )
    ax_mae.set_ylabel("MAE (log10 affinity units, lower is better)", color="#1f2933")
    ax_mae.set_ylim(0, max(results["mae"]) * 1.22)
    ax_mae.tick_params(axis="y", labelcolor="#1f2933")

    for bar, value in zip(bars, results["mae"]):
        ax_mae.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax_spearman = ax_mae.twinx()
    ax_spearman.plot(
        x_positions,
        results["spearman"],
        color="#111827",
        marker="o",
        linewidth=2,
        label="Spearman (higher is better)",
    )
    ax_spearman.set_ylabel("Spearman correlation (higher is better)", color="#111827")
    ax_spearman.set_ylim(0.30, 0.55)
    ax_spearman.tick_params(axis="y", labelcolor="#111827")

    for x, value in zip(x_positions, results["spearman"]):
        ax_spearman.text(x, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    labels = [label.replace(" ", "\n") for label in results["model"]]
    ax_mae.set_xticks(x_positions)
    ax_mae.set_xticklabels(labels, fontsize=9)

    # 灰色竖线把两个 benchmark 隔开，提醒读者不要跨不同 test set 直接比较。
    separator_x = 4.5
    ax_mae.axvline(separator_x, color="#9CA3AF", linestyle="--", linewidth=1.4)
    ax_mae.text(2, ax_mae.get_ylim()[1] * 0.96, "unified_no_high_risk\n605-row benchmark", ha="center", va="top", fontsize=11)
    ax_mae.text(5.5, ax_mae.get_ylim()[1] * 0.96, "ANDD antibody v2\nlarger benchmark", ha="center", va="top", fontsize=11)

    ax_mae.set_title("Final Model Comparison Across Two Benchmarks", fontsize=15, pad=16)
    ax_mae.text(
        0.5,
        -0.24,
        "Note: benchmarks use different test sets. Compare models within each section, not directly across sections.",
        transform=ax_mae.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#4B5563",
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8", alpha=0.82, label="unified_no_high_risk MAE"),
        plt.Rectangle((0, 0), 1, 1, color="#F58518", alpha=0.82, label="ANDD antibody v2 MAE"),
        plt.Line2D([0], [0], color="#111827", marker="o", linewidth=2, label="Spearman"),
    ]
    ax_mae.legend(handles=legend_handles, loc="upper right", frameon=False)
    ax_mae.grid(axis="y", alpha=0.18)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Entry point."""

    results = build_results_table()
    make_plot(results)
    print(f"Saved figure to {OUTPUT_PATH}")
    print(f"Saved source table to {CSV_PATH}")


if __name__ == "__main__":
    main()
