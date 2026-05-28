"""诊断 ANDD antibody v2 的 target distribution，不训练模型。

中文人话说明：
affinity regression 的 label 是 `neg_log10_affinity_candidate`。
如果训练数据几乎都集中在中间范围，模型容易学成“猜中间值”；
如果 train/val/test 分布差异很大，测试表现也可能被 split 影响。

本脚本只读取已经存在的 CSV，并输出统计报告和图片，不修改 dataset。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# 无图形界面的终端中只保存 PNG，不弹窗口。
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DATA_DIR = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2_cdr_annotated"
OUTPUT_DIR = ROOT / "outputs/andd_antibody_v2/target_distribution"
TARGET_COLUMN = "neg_log10_affinity_candidate"
SPLITS = ["train", "val", "test"]
QUANTILES = [0.0, 0.10, 0.25, 1 / 3, 0.50, 2 / 3, 0.75, 0.90, 1.0]
BIN_LABELS = ["low_target", "mid_target", "high_target"]
COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}


def load_splits() -> dict[str, pd.DataFrame]:
    """读取三个 split，并确认 regression target 可以用于统计。"""

    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")
        frame = pd.read_csv(path)
        if TARGET_COLUMN not in frame.columns:
            raise KeyError(f"{path.name} is missing target column: {TARGET_COLUMN}")
        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if frame[TARGET_COLUMN].isna().any():
            missing_count = int(frame[TARGET_COLUMN].isna().sum())
            raise ValueError(f"{path.name} contains {missing_count} non-numeric/missing target values.")
        frames[split] = frame
    return frames


def summarize_targets(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """计算每个 split 的 count / mean / std / 范围和 quantiles。"""

    rows: list[dict] = []
    json_summary: dict[str, dict] = {}
    for split, frame in frames.items():
        target = frame[TARGET_COLUMN]
        values = {
            "count": int(target.count()),
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        }
        quantile_values = target.quantile(QUANTILES)
        for quantile, value in quantile_values.items():
            label = f"q{quantile * 100:.1f}".replace(".", "_")
            values[label] = float(value)
        rows.append({"split": split, **values})
        json_summary[split] = values
    return pd.DataFrame(rows), json_summary


def assign_train_tertile_bin(target: pd.Series, low_edge: float, high_edge: float) -> pd.Series:
    """使用只从 train 学到的阈值给所有 split 分 bin，避免偷看 test。"""

    return pd.cut(
        target,
        bins=[-np.inf, low_edge, high_edge, np.inf],
        labels=BIN_LABELS,
        include_lowest=True,
        right=True,
    )


def summarize_train_bins(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict], float, float]:
    """按训练集三分位阈值统计三个 split 的 low/mid/high 比例。"""

    train_target = frames["train"][TARGET_COLUMN]
    low_edge = float(train_target.quantile(1 / 3))
    high_edge = float(train_target.quantile(2 / 3))
    if low_edge >= high_edge:
        raise ValueError("Train tertile thresholds are identical; cannot create three target bins.")

    rows: list[dict] = []
    json_summary: dict[str, dict] = {}
    for split, frame in frames.items():
        bins = assign_train_tertile_bin(frame[TARGET_COLUMN], low_edge, high_edge)
        counts = bins.value_counts(sort=False).reindex(BIN_LABELS, fill_value=0)
        json_summary[split] = {}
        for bin_name in BIN_LABELS:
            count = int(counts[bin_name])
            proportion = float(count / len(frame))
            rows.append(
                {
                    "split": split,
                    "target_bin": bin_name,
                    "count": count,
                    "proportion": proportion,
                }
            )
            json_summary[split][bin_name] = {"count": count, "proportion": proportion}
    return pd.DataFrame(rows), json_summary, low_edge, high_edge


def summarize_extreme_tails(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, float, float]:
    """额外统计 train 外侧 10% 阈值，用来区分 tertile 与真正少量 extreme tails。"""

    lower_edge = float(frames["train"][TARGET_COLUMN].quantile(0.10))
    upper_edge = float(frames["train"][TARGET_COLUMN].quantile(0.90))
    rows: list[dict] = []
    for split, frame in frames.items():
        target = frame[TARGET_COLUMN]
        low_count = int((target <= lower_edge).sum())
        high_count = int((target >= upper_edge).sum())
        rows.extend(
            [
                {
                    "split": split,
                    "tail": "bottom_10pct_by_train",
                    "threshold": lower_edge,
                    "count": low_count,
                    "proportion": low_count / len(frame),
                },
                {
                    "split": split,
                    "tail": "top_10pct_by_train",
                    "threshold": upper_edge,
                    "count": high_count,
                    "proportion": high_count / len(frame),
                },
            ]
        )
    return pd.DataFrame(rows), lower_edge, upper_edge


def plot_target_histogram(frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    """用共同的 bin edges 比较 train/val/test target 分布。"""

    all_targets = pd.concat([frames[split][TARGET_COLUMN] for split in SPLITS], ignore_index=True)
    global_min = float(all_targets.min())
    global_max = float(all_targets.max())
    shared_bins = np.linspace(global_min, global_max, 21)

    plt.figure(figsize=(8, 5))
    for split in SPLITS:
        plt.hist(
            frames[split][TARGET_COLUMN],
            bins=shared_bins,
            density=True,
            alpha=0.42,
            color=COLORS[split],
            label=f"{split} (n={len(frames[split])})",
        )
    # 比较三个 split 时必须共用同一套 bin edges，否则柱子的范围不同会误导比较。
    plt.xlabel("neg_log10_affinity_candidate")
    plt.ylabel("Density")
    plt.title("ANDD antibody v2 target distribution (shared histogram bins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """生成不依赖额外 Markdown package 的简单表格。"""

    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.{float_digits}f}")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_report(
    target_summary: pd.DataFrame,
    bin_summary: pd.DataFrame,
    tail_summary: pd.DataFrame,
    low_edge: float,
    high_edge: float,
    lower_tail_edge: float,
    upper_tail_edge: float,
) -> None:
    """把统计结果和结论写成便于复核的 Markdown 报告。"""

    bin_display = bin_summary.copy()
    bin_display["proportion"] = bin_display["proportion"] * 100
    bin_display = bin_display.rename(columns={"proportion": "proportion_pct"})
    tail_display = tail_summary.copy()
    tail_display["proportion"] = tail_display["proportion"] * 100
    tail_display = tail_display.rename(columns={"proportion": "proportion_pct"})

    train_bins = bin_summary[bin_summary["split"] == "train"].set_index("target_bin")
    low_pct = float(train_bins.loc["low_target", "proportion"] * 100)
    high_pct = float(train_bins.loc["high_target", "proportion"] * 100)

    lines = [
        "# ANDD Antibody v2 Target Distribution Audit",
        "",
        "## Scope",
        "",
        "- 本分析只读取 CDR-annotated train/val/test CSV；没有训练模型，也没有修改数据。",
        f"- Target column: `{TARGET_COLUMN}`。",
        "- 目的：检查 label 分布是否能解释已观察到的 regression-to-the-mean。",
        "",
        "## 1. Target Summary",
        "",
        markdown_table(target_summary),
        "",
        "## 2. Quantile / Range Interpretation",
        "",
        f"- Train target 范围为 `{target_summary.loc[target_summary['split'] == 'train', 'min'].iloc[0]:.4f}` 到 "
        f"`{target_summary.loc[target_summary['split'] == 'train', 'max'].iloc[0]:.4f}`；"
        f"test 范围为 `{target_summary.loc[target_summary['split'] == 'test', 'min'].iloc[0]:.4f}` 到 "
        f"`{target_summary.loc[target_summary['split'] == 'test', 'max'].iloc[0]:.4f}`。",
        "- Train 中存在比 val/test 更高的少量 high-target tail，因此总体范围较宽。",
        "",
        "## 3. Low / Mid / High Bins Defined From Train Tertiles",
        "",
        "这里用 train 的三分位阈值定义所有 split 的区间；test 没有参与阈值选择：",
        "",
        f"- `low_target`: target <= `{low_edge:.4f}`",
        f"- `mid_target`: `{low_edge:.4f}` < target <= `{high_edge:.4f}`",
        f"- `high_target`: target > `{high_edge:.4f}`",
        "",
        markdown_table(bin_display),
        "",
        "重要解释：因为阈值就是从 train tertiles 定义的，train 的三档数量本来就会被切成接近三等份。"
        "因此这个表适合检查 val/test 是否发生 target distribution shift，不能用来证明训练集中真正的极端样本很多。",
        "",
        "## 4. Extreme Tail Context",
        "",
        f"作为补充，下面用 train 的 P10 (`{lower_tail_edge:.4f}`) 与 P90 (`{upper_tail_edge:.4f}`) 阈值统计外侧尾部样本：",
        "",
        markdown_table(tail_display),
        "",
        "## 5. Does Target Imbalance Explain Regression-To-The-Mean?",
        "",
        f"- 在 tertile 层面，train 的 low/high 各约 `{low_pct:.1f}%` / `{high_pct:.1f}%`，并没有明显少于 mid。",
        "- val/test 的 low/mid/high 比例也可与 train 直接比较；若差异不大，不能把 prediction range 压缩主要归因于粗粒度 label imbalance。",
        "- 真正很极端的 target 仍只占尾部少数样本；这可能让极端 affinity 的学习更困难，但不是“low/high 三档没有训练样本”的问题。",
        "",
        "## 6. Modeling Implication",
        "",
        "- 若 low/high 三档在 train 中明显稀缺，优先尝试可验证的 sampling 或平滑 weighting，比直接换 Huber loss 更针对覆盖不足问题。",
        "- 当前按 train tertiles 看不到明显 low/high 缺口。`HuberLoss` 会降低大残差样本的影响，若极端样本是真实信号，反而可能进一步削弱学习极端值。",
        "- 因此更合理的下一步是：继续评估 post-hoc calibration、验证集驱动的 weighting/checkpoint policy，"
        "以及更能表示 binding interaction 的模型表示；Huber 只适合作为“极端 label 疑似噪声很大”时的对照实验。",
        "",
        "## Files",
        "",
        "- `target_summary.csv`: count / mean / std / range / quantiles",
        "- `target_bin_counts.csv`: train-tertile low/mid/high counts and proportions",
        "- `extreme_tail_counts.csv`: train P10/P90 tail context",
        "- `target_distribution_histogram.png`: shared-bin target histogram",
    ]
    (OUTPUT_DIR / "target_distribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_splits()
    target_summary, json_summary = summarize_targets(frames)
    bin_summary, bin_json, low_edge, high_edge = summarize_train_bins(frames)
    tail_summary, lower_tail_edge, upper_tail_edge = summarize_extreme_tails(frames)

    target_summary.to_csv(OUTPUT_DIR / "target_summary.csv", index=False)
    bin_summary.to_csv(OUTPUT_DIR / "target_bin_counts.csv", index=False)
    tail_summary.to_csv(OUTPUT_DIR / "extreme_tail_counts.csv", index=False)
    plot_target_histogram(frames, OUTPUT_DIR / "target_distribution_histogram.png")
    write_report(
        target_summary,
        bin_summary,
        tail_summary,
        low_edge,
        high_edge,
        lower_tail_edge,
        upper_tail_edge,
    )

    summary_json = {
        "target_column": TARGET_COLUMN,
        "train_tertile_edges": {"low_upper": low_edge, "mid_upper": high_edge},
        "train_tail_edges": {"p10": lower_tail_edge, "p90": upper_tail_edge},
        "target_summary": json_summary,
        "target_bins": bin_json,
    }
    (OUTPUT_DIR / "target_distribution_summary.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    print("ANDD antibody v2 target distribution audit completed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train tertile edges: low <= {low_edge:.4f}, high > {high_edge:.4f}")
    print("提醒：train tertiles 会按定义使 train low/mid/high 接近均衡；请结合 tail counts 看极端样本。")


if __name__ == "__main__":
    main()
