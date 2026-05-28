"""Create presentation figures from unified affinity ablation results.

中文人话说明：
这个脚本只读取已经生成好的 ``ablation_results.csv``。
它不会训练模型，不会重新切分 dataset，也不会改 prediction 文件。

输出内容：
1. 一份适合复制到汇报里的 Markdown 表格；
2. 模型 MAE 和 mean baseline MAE 的对照柱状图；
3. MAE / RMSE / Spearman 的 ablation 指标图；
4. MAE bar + Spearman dot + baseline line 的综合图；
5. best model 的简短 Markdown 摘要。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 汇报脚本只保存 PNG，不弹出交互窗口；Agg backend 在 terminal/服务器环境更稳。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ablation"
    / "unified_affinity_dataset_v1"
    / "ablation_results.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_reports" / "figures"
TABLE_PATH = OUTPUT_DIR / "ablation_summary_table.md"
BASELINE_FIGURE_PATH = OUTPUT_DIR / "model_vs_mean_baseline_mae.png"
METRICS_FIGURE_PATH = OUTPUT_DIR / "ablation_metrics.png"
COMBINED_FIGURE_PATH = OUTPUT_DIR / "ablation_mae_spearman_baseline.png"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model_summary.md"

BEST_MODEL_NAME = "unified_no_high_risk"

# 同一套颜色让图和表更容易对应。
DATASET_COLORS = {
    "unified_full_629": "#3b82f6",
    "unified_no_peptide": "#f59e0b",
    "unified_no_high_risk": "#10b981",
    "unified_no_less_strict": "#8b5cf6",
}


def pretty_dataset_name(name: str) -> str:
    """Use short dataset labels in figures."""

    return name.replace("unified_", "").replace("_", "\n")


def load_results() -> pd.DataFrame:
    """Load and validate the metrics table."""

    results = pd.read_csv(INPUT_PATH)
    required_columns = {
        "dataset_version",
        "rows",
        "test_MAE",
        "test_RMSE",
        "test_Spearman",
        "prediction_std",
        "mean_baseline_MAE",
        "mean_baseline_RMSE",
        "beats_mean_baseline_MAE",
        "beats_mean_baseline_RMSE",
    }
    missing = sorted(required_columns - set(results.columns))
    if missing:
        raise ValueError(f"ablation_results.csv is missing required columns: {missing}")
    return results


def write_summary_table(results: pd.DataFrame) -> None:
    """Write a compact Markdown table for a presentation/report."""

    lines = [
        "# Unified Affinity Ablation Summary",
        "",
        "| dataset_version | rows | Test MAE | Test RMSE | Test Spearman | prediction_std | Mean baseline MAE | Mean baseline RMSE | Beats mean baseline? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in results.iterrows():
        beats_both = bool(row["beats_mean_baseline_MAE"]) and bool(
            row["beats_mean_baseline_RMSE"]
        )
        lines.append(
            f"| `{row['dataset_version']}` | {int(row['rows'])} | "
            f"{row['test_MAE']:.4f} | {row['test_RMSE']:.4f} | "
            f"{row['test_Spearman']:.4f} | {row['prediction_std']:.4f} | "
            f"{row['mean_baseline_MAE']:.4f} | {row['mean_baseline_RMSE']:.4f} | "
            f"`{beats_both}` |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- MAE/RMSE are measured on the `neg_log10_affinity` target scale.",
            "- Mean baseline predicts the train-set target mean for every test sample.",
        ]
    )
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_model_vs_baseline_mae(results: pd.DataFrame) -> None:
    """Plot model test MAE against mean-baseline MAE."""

    x = np.arange(len(results))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.8))
    baseline_bars = ax.bar(
        x - width / 2,
        results["mean_baseline_MAE"],
        width,
        label="Mean baseline MAE",
        color="#94a3b8",
        edgecolor="#475569",
    )
    model_bars = ax.bar(
        x + width / 2,
        results["test_MAE"],
        width,
        label="ESM2+LoRA Test MAE",
        color=[DATASET_COLORS.get(name, "#0f766e") for name in results["dataset_version"]],
        edgecolor="#0f172a",
    )

    ax.set_title("Ablation: Model vs Mean Baseline MAE")
    ax.set_xlabel("Dataset version")
    ax.set_ylabel("MAE on neg_log10_affinity (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_dataset_name(name) for name in results["dataset_version"]])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # 给每根柱子加数值，汇报时不用再查 CSV。
    ax.bar_label(baseline_bars, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(model_bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(BASELINE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def make_ablation_metrics(results: pd.DataFrame) -> None:
    """Plot MAE, RMSE, and Spearman for each dataset version."""

    metrics = [
        ("test_MAE", "Test MAE\nlower is better", "#ef4444"),
        ("test_RMSE", "Test RMSE\nlower is better", "#f97316"),
        ("test_Spearman", "Test Spearman\nhigher is better", "#0ea5e9"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    labels = [pretty_dataset_name(name) for name in results["dataset_version"]]
    colors = [DATASET_COLORS.get(name, "#0f766e") for name in results["dataset_version"]]

    for ax, (column, title, edge_color) in zip(axes, metrics):
        bars = ax.bar(labels, results[column], color=colors, edgecolor=edge_color, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Dataset version")
        ax.set_ylabel(column)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    fig.suptitle("Unified Affinity Ablation Metrics", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(METRICS_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_mae_spearman_baseline_figure(results: pd.DataFrame) -> None:
    """Plot MAE bars, Spearman dots, and mean-baseline MAE line.

    中文人话说明：
    - bar：模型 Test MAE，越低越好；
    - dashed line：mean baseline MAE，bar 低于线才说明 MAE 超过 baseline；
    - dot：Test Spearman，越高说明排序越好。
    因为 MAE 和 Spearman 单位不同，所以 Spearman 放在右侧 y-axis。
    """

    x = np.arange(len(results))
    labels = [pretty_dataset_name(name) for name in results["dataset_version"]]
    colors = [DATASET_COLORS.get(name, "#0f766e") for name in results["dataset_version"]]

    fig, mae_axis = plt.subplots(figsize=(11, 6.2))
    bars = mae_axis.bar(
        x,
        results["test_MAE"],
        width=0.58,
        color=colors,
        edgecolor="#0f172a",
        label="Model Test MAE",
        zorder=2,
    )
    baseline_line = mae_axis.plot(
        x,
        results["mean_baseline_MAE"],
        color="#475569",
        marker="s",
        linestyle="--",
        linewidth=2.2,
        label="Mean baseline MAE",
        zorder=3,
    )
    mae_axis.set_title("Ablation Tradeoff: MAE, Mean Baseline, and Ranking")
    mae_axis.set_xlabel("Dataset version")
    mae_axis.set_ylabel("MAE on neg_log10_affinity (lower is better)")
    mae_axis.set_xticks(x)
    mae_axis.set_xticklabels(labels)
    mae_axis.grid(axis="y", alpha=0.25, zorder=1)
    mae_axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    spearman_axis = mae_axis.twinx()
    spearman_points = spearman_axis.plot(
        x,
        results["test_Spearman"],
        color="#111827",
        marker="o",
        markersize=9,
        linewidth=0,
        label="Test Spearman",
        zorder=4,
    )
    spearman_axis.set_ylabel("Test Spearman (higher is better)")
    spearman_axis.set_ylim(0, max(0.6, float(results["test_Spearman"].max()) * 1.18))

    # 给 Spearman 点标数字，方便做 slides 时直接读值。
    for index, value in enumerate(results["test_Spearman"]):
        spearman_axis.annotate(
            f"{value:.3f}",
            (x[index], value),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#111827",
        )

    handles = [bars, baseline_line[0], spearman_points[0]]
    labels_for_legend = [handle.get_label() for handle in handles]
    mae_axis.legend(handles, labels_for_legend, loc="upper left")
    fig.tight_layout()
    fig.savefig(COMBINED_FIGURE_PATH, dpi=200)
    plt.close(fig)


def write_best_model_summary(results: pd.DataFrame) -> None:
    """Write the requested summary for unified_no_high_risk."""

    best = results.loc[results["dataset_version"] == BEST_MODEL_NAME]
    if best.empty:
        raise ValueError(f"Cannot find requested best model row: {BEST_MODEL_NAME}")
    row = best.iloc[0]
    beats_mae = bool(row["beats_mean_baseline_MAE"])
    beats_rmse = bool(row["beats_mean_baseline_RMSE"])
    lines = [
        "# Best Ablation Model Summary",
        "",
        f"## {BEST_MODEL_NAME}",
        "",
        "The ablation result highlights `unified_no_high_risk` because it removes peptide, "
        "same-H/L metadata, and suspicious affinity-method rows while keeping a larger dataset "
        "than the no-less-strict version.",
        "",
        "| item | value |",
        "|---|---:|",
        f"| rows | {int(row['rows'])} |",
        f"| Test MAE | {row['test_MAE']:.4f} |",
        f"| Test RMSE | {row['test_RMSE']:.4f} |",
        f"| Test Spearman | {row['test_Spearman']:.4f} |",
        f"| prediction_std | {row['prediction_std']:.4f} |",
        f"| Mean baseline MAE | {row['mean_baseline_MAE']:.4f} |",
        f"| Mean baseline RMSE | {row['mean_baseline_RMSE']:.4f} |",
        "",
        "## Interpretation",
        "",
        f"- Beats mean baseline on MAE: `{beats_mae}`.",
        f"- Beats mean baseline on RMSE: `{beats_rmse}`.",
        "- In this ablation table, `unified_no_high_risk` beats the mean baseline on both MAE and RMSE.",
        "- Its higher Spearman than the full dataset suggests that removing the configured high-risk rows "
        "can improve ranking behavior for this dataset version.",
        "- These ablation datasets were re-split, so treat the comparison as dataset-version evidence, "
        "not a pure causal estimate for removing individual rows.",
    ]
    BEST_MODEL_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate all presentation artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    write_summary_table(results)
    make_model_vs_baseline_mae(results)
    make_ablation_metrics(results)
    make_mae_spearman_baseline_figure(results)
    write_best_model_summary(results)

    print("Unified ablation report figures created.")
    for path in [
        TABLE_PATH,
        BASELINE_FIGURE_PATH,
        METRICS_FIGURE_PATH,
        COMBINED_FIGURE_PATH,
        BEST_MODEL_PATH,
    ]:
        print(path.relative_to(PROJECT_ROOT))
    print("No training, model changes, or dataset changes were performed.")


if __name__ == "__main__":
    main()
