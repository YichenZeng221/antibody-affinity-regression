"""生成汇报用 model progression 指标图。

这张图不是把不同 dataset 的模型强行排成一个排行榜，而是展示：
1. 在同一 curated split 中，输入表示和 interaction 设计如何演进；
2. ANDD 数据扩展后的独立 benchmark 结果；
3. 在同一 ANDD stratified split 中，pooled 与 cross-attention 的公平比较。

所有数值都从已经存在的 Markdown 实验报告中读取。报告中没有的数值会保留为 NaN，
避免为了画图而人工猜测结果。
"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# 将字体和 matplotlib 缓存放到临时目录，避免在项目里重新制造缓存文件，
# 也避免受限环境下 Fontconfig 尝试写用户目录而失败。
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/seqproft_xdg_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/seqproft_matplotlib_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_reports" / "figures"
FIGURE_PATH = OUTPUT_DIR / "model_progression_metrics.png"
CSV_PATH = OUTPUT_DIR / "model_progression_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "model_progression_metrics_summary.md"

UNIFIED_REPORT = (
    ROOT
    / "outputs"
    / "cross_attention"
    / "unified_no_high_risk"
    / "all_cdrs_antigen"
    / "cross_attention_report.md"
)
ANDD_ORIGINAL_REPORT = (
    ROOT
    / "outputs"
    / "andd_antibody_v2"
    / "all_cdr_pooled"
    / "andd_antibody_v2_all_cdr_pooled_report.md"
)
ANDD_STRATIFIED_REPORT = (
    ROOT
    / "outputs"
    / "andd_antibody_v2_stratified"
    / "cross_attention_all_cdrs"
    / "cross_attention_report.md"
)


def read_markdown_table(path: Path, heading: str) -> pd.DataFrame:
    """读取指定 heading 后面的第一张 Markdown 表格。"""
    text = path.read_text(encoding="utf-8")
    if heading not in text:
        return pd.DataFrame()

    following = text.split(heading, maxsplit=1)[1].splitlines()
    table_lines: list[str] = []
    started = False
    for line in following:
        if line.strip().startswith("|"):
            table_lines.append(line)
            started = True
        elif started:
            break

    if len(table_lines) < 3:
        return pd.DataFrame()

    table = pd.read_csv(StringIO("\n".join(table_lines)), sep="|", engine="python")
    table = table.dropna(axis=1, how="all")
    table.columns = [str(column).strip() for column in table.columns]
    table = table.map(lambda value: value.strip() if isinstance(value, str) else value)
    separator_mask = table.apply(
        lambda row: all(
            str(value).replace("-", "").replace(":", "").strip() == ""
            for value in row
        ),
        axis=1,
    )
    return table.loc[~separator_mask].reset_index(drop=True)


def clean_name(value: object) -> str:
    """去掉 Markdown code quotes，便于按 model id 找到表格行。"""
    return str(value).replace("`", "").strip()


def number(value: object, take_after_slash: bool = False) -> float:
    """安全读取数字；缺失或格式异常时返回 NaN。"""
    if value is None or pd.isna(value):
        return float("nan")
    text = str(value).strip().replace("`", "")
    if take_after_slash and "/" in text:
        text = text.split("/")[-1].strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def select_row(table: pd.DataFrame, key_column: str, key: str) -> pd.Series | None:
    """从表格中找到指定模型行；找不到就返回 None，让输出保留 NaN。"""
    if table.empty or key_column not in table.columns:
        return None
    mask = table[key_column].map(clean_name) == key
    if not mask.any():
        return None
    return table.loc[mask].iloc[0]


def metric(row: pd.Series | None, column: str, *, slash_value: bool = False) -> float:
    if row is None or column not in row.index:
        return float("nan")
    return number(row[column], take_after_slash=slash_value)


def build_metrics_dataframe() -> pd.DataFrame:
    """从三份现有报告整理八个模型的一致 dataframe。"""
    unified_metrics = read_markdown_table(UNIFIED_REPORT, "## Test Metrics")
    unified_bins = read_markdown_table(UNIFIED_REPORT, "## Target-Bin MAE")
    andd_metrics = read_markdown_table(ANDD_ORIGINAL_REPORT, "## Test Metrics")
    andd_bins = read_markdown_table(ANDD_ORIGINAL_REPORT, "## Target-Bin MAE")
    strat_metrics = read_markdown_table(ANDD_STRATIFIED_REPORT, "## Test Metrics")
    strat_bins = read_markdown_table(
        ANDD_STRATIFIED_REPORT, "## Train-Defined Target-Bin MAE"
    )
    strat_tails = read_markdown_table(ANDD_STRATIFIED_REPORT, "## Train-Defined Tail MAE")

    rows: list[dict[str, object]] = []

    unified_models = [
        ("whole_chain_pooled", "whole_sequence", "Whole chain\ncurated"),
        ("all_cdr_pooled", "all_cdrs_pooled", "All CDR pooled\ncurated"),
        ("hcdr3_lcdr3_pooled", "hcdr3_lcdr3_pooled", "HCDR3+LCDR3\ncurated"),
        (
            "dot_product_interaction",
            "simple_interaction_hcdr3_lcdr3",
            "Dot product\ncurated",
        ),
        ("cross_attention", "all_cdrs_cross_attention", "Cross-attn\ncurated"),
    ]
    for name, report_key, label in unified_models:
        metrics_row = select_row(unified_metrics, "model", report_key)
        bins_row = select_row(unified_bins, "model", report_key)
        rows.append(
            {
                "model": name,
                "plot_label": label,
                "benchmark_split": "unified_no_high_risk / antigen-group split",
                "fair_comparison_group": "curated_same_split",
                "MAE": metric(metrics_row, "MAE"),
                "Spearman": metric(metrics_row, "Spearman"),
                "pred_std_true_std": metric(metrics_row, "pred std / true std"),
                "high_tail_MAE_or_above_train_p90_MAE": metric(
                    bins_row, "high target MAE"
                ),
                "tail_metric_definition": "high target MAE (test-target quantile bin)",
                "source_report": str(UNIFIED_REPORT.relative_to(ROOT)),
            }
        )

    andd_row = select_row(andd_metrics, "model", "cdr_aware")
    andd_bin_row = select_row(andd_bins, "model", "cdr_aware")
    rows.append(
        {
            "model": "andd_v2_all_cdr_pooled",
            "plot_label": "All CDR pooled\nANDD original",
            "benchmark_split": "ANDD antibody v2 / original antigen split",
            "fair_comparison_group": "andd_original_context_only",
            "MAE": metric(andd_row, "MAE"),
            "Spearman": metric(andd_row, "Spearman"),
            "pred_std_true_std": metric(andd_row, "pred std / true std"),
            "high_tail_MAE_or_above_train_p90_MAE": metric(
                andd_bin_row, "high target MAE"
            ),
            "tail_metric_definition": "high target MAE (test-target quantile bin)",
            "source_report": str(ANDD_ORIGINAL_REPORT.relative_to(ROOT)),
        }
    )

    stratified_models = [
        (
            "andd_v2_stratified_all_cdr_pooled",
            "all_cdrs_pooled",
            "All CDR pooled\nANDD strat.",
        ),
        (
            "andd_v2_stratified_cross_attention",
            "cross_attention_all_cdrs",
            "Cross-attn\nANDD strat.",
        ),
    ]
    for name, report_key, label in stratified_models:
        metrics_row = select_row(strat_metrics, "model", report_key)
        bins_row = select_row(strat_bins, "model", report_key)
        tail_row = select_row(strat_tails, "model", report_key)
        rows.append(
            {
                "model": name,
                "plot_label": label,
                "benchmark_split": "ANDD antibody v2 / stratified antigen split",
                "fair_comparison_group": "andd_stratified_same_split",
                "MAE": metric(metrics_row, "MAE"),
                "Spearman": metric(metrics_row, "Spearman"),
                "pred_std_true_std": metric(metrics_row, "pred std / true std"),
                "high_tail_MAE_or_above_train_p90_MAE": metric(
                    tail_row, "above train P90 rows / MAE", slash_value=True
                ),
                "tail_metric_definition": "above train P90 MAE",
                "high_target_MAE_reference": metric(
                    bins_row, "high rows / MAE", slash_value=True
                ),
                "source_report": str(ANDD_STRATIFIED_REPORT.relative_to(ROOT)),
            }
        )

    frame = pd.DataFrame(rows)
    if "high_target_MAE_reference" not in frame.columns:
        frame["high_target_MAE_reference"] = np.nan
    return frame


def draw_figure(frame: pd.DataFrame) -> None:
    """绘制 2x2 模型演进图，颜色同时表示 benchmark/split 组别。"""
    colors = {
        "curated_same_split": "#247BA0",
        "andd_original_context_only": "#D9902F",
        "andd_stratified_same_split": "#C65A4A",
    }
    bar_colors = [colors[group] for group in frame["fair_comparison_group"]]
    x = np.arange(len(frame))

    metrics = [
        ("MAE", "MAE (lower is better)", None),
        ("Spearman", "Spearman (higher is better)", None),
        ("pred_std_true_std", "Prediction spread ratio (closer to 1 is better)", 1.0),
        (
            "high_tail_MAE_or_above_train_p90_MAE",
            "High-tail MAE (lower is better; definitions differ*)",
            None,
        ),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#A8ADB4",
            "axes.labelcolor": "#2D3440",
            "xtick.color": "#39424E",
            "ytick.color": "#39424E",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5), constrained_layout=True)

    for axis, (column, title, ideal_line) in zip(axes.flat, metrics):
        values = frame[column].to_numpy(dtype=float)
        bars = axis.bar(x, values, color=bar_colors, width=0.7, alpha=0.92)
        axis.set_title(title, fontsize=12, pad=10)
        axis.grid(axis="y", color="#DDE2E8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(x, frame["plot_label"], rotation=35, ha="right", fontsize=9)

        # ANDD original 与新的 stratified split 数据定义不同，用虚线分隔。
        axis.axvline(5.5, color="#59636F", linestyle=(0, (3, 3)), linewidth=1)
        # curated benchmark 与 ANDD 数据扩展同样不属于同一 test set。
        axis.axvline(4.5, color="#B5BBC3", linestyle=(0, (2, 3)), linewidth=0.9)
        if ideal_line is not None:
            axis.axhline(
                ideal_line,
                color="#2D6A4F",
                linestyle="--",
                linewidth=1,
                label="ideal spread ratio = 1",
            )

        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            upper_limit = max(finite_values.max() * 1.23, 0.05)
            if ideal_line is not None:
                upper_limit = max(upper_limit, ideal_line * 1.1)
            axis.set_ylim(0, upper_limit)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "NaN",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )
            else:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#202630",
                )

    legend_items = [
        Patch(color=colors["curated_same_split"], label="curated: same split"),
        Patch(color=colors["andd_original_context_only"], label="ANDD original: context"),
        Patch(color=colors["andd_stratified_same_split"], label="ANDD stratified: same split"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.suptitle(
        "Model progression: CDR focus, interaction modeling, and ANDD expansion",
        fontsize=16,
        fontweight="bold",
        y=1.075,
    )
    fig.text(
        0.01,
        -0.02,
        "* Curated and ANDD-original bars use high-target MAE; ANDD-stratified bars use above-train-P90 MAE. "
        "Compare absolute heights only within matching benchmark/split definitions. Single-seed baselines.",
        fontsize=9,
        color="#4A5560",
    )
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """轻量输出 Markdown table，避免为了报告额外依赖 `tabulate`。"""
    display = frame.copy()
    for column in display.select_dtypes(include=["float", "float64"]).columns:
        display[column] = display[column].map(
            lambda value: "NaN" if pd.isna(value) else f"{value:.4f}"
        )
    columns = list(display.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_summary(frame: pd.DataFrame) -> None:
    """生成供汇报时阅读的中文说明，并列出指标缺失情况。"""
    missing = []
    metric_columns = [
        "MAE",
        "Spearman",
        "pred_std_true_std",
        "high_tail_MAE_or_above_train_p90_MAE",
    ]
    for _, row in frame.iterrows():
        for column in metric_columns:
            if pd.isna(row[column]):
                missing.append(f"`{row['model']}` 的 `{column}`")
    missing_text = "没有；所需四项指标均从已有报告中找到。" if not missing else "、".join(missing)

    table_frame = frame[
        [
            "model",
            "benchmark_split",
            "MAE",
            "Spearman",
            "pred_std_true_std",
            "high_tail_MAE_or_above_train_p90_MAE",
            "tail_metric_definition",
        ]
    ].copy()
    table_frame = table_frame.rename(
        columns={
            "benchmark_split": "benchmark / split",
            "pred_std_true_std": "pred_std / true_std",
            "high_tail_MAE_or_above_train_p90_MAE": "tail MAE",
            "tail_metric_definition": "tail definition",
        }
    )
    table_md = dataframe_to_markdown(table_frame)

    summary = f"""# Model Progression Metrics Summary

## 图表目的

这张图展示模型路线如何逐步处理前一个阶段暴露的问题：从 whole-chain pooled input，到 CDR-focused input，再到显式 interaction 与 cross-attention，最后进入更大的 ANDD antibody-only benchmark 以及 tail-aware stratified split。

需要注意：这是一张 **progression context** 图，不是把所有柱子当作同一 test set 的排行榜。

## 指标表

{table_md}

## 哪些比较是公平的

- `whole_chain_pooled`、`all_cdr_pooled`、`hcdr3_lcdr3_pooled`、`dot_product_interaction`、`cross_attention` 都来自 `unified_no_high_risk` 的同一 antigen-group split，可以在这一组内直接比较。
- `andd_v2_all_cdr_pooled` 来自扩展后的 ANDD 原始 split，是新的 benchmark context，不能与前面 605-row benchmark 直接宣称胜负。
- `andd_v2_stratified_all_cdr_pooled` 与 `andd_v2_stratified_cross_attention` 来自同一个 stratified antigen-level split，可以在这两者之间公平比较。
- 第四张 subplot 中，curated / ANDD original 使用 `high-target MAE`；ANDD stratified 使用更明确的 `above train P90 MAE`。因此 tail MAE 主要应在各自相同定义的比较组内解读。

## 模型演进解释

1. `all_cdr_pooled` 相比 `whole_chain_pooled` 降低 MAE/RMSE，说明把输入聚焦到标准 IMGT CDR 后，减少了 whole-chain framework region 带来的非关键噪声。
2. `dot_product_interaction` 尝试解决 pooled representation 没有显式 CDR-antigen interaction 的问题，但简单 dot-product summary 没有带来提升，说明过早压缩 interaction matrix 过于粗糙。
3. `cross_attention` 尝试解决 dot-product summary 太粗的问题：在原始 curated benchmark 上，它提高 Spearman 与 prediction spread，并改善 high-target MAE，但 overall MAE 不如 pooled all-CDR。
4. ANDD 扩展带来一个更大的 antibody-only benchmark；它的绝对指标只能在自身 split 语境下解释，不能当作对原始 benchmark 的直接升级成绩。
5. 在同一 ANDD stratified split 上，pooled all-CDR 的 MAE 更好（`0.9373` vs `0.9523`），但 cross-attention 改善 Spearman（`0.3861` vs `0.3699`）、`pred_std / true_std`（`0.3925` vs `0.3170`）以及 upper-tail MAE（`1.8483` vs `2.0887`）。

## 诚实结论

- CDR-focused input 明确改善了整体误差。
- Learnable cross-attention 对 ranking、prediction spread 和高 affinity tail 有信号，但当前尚未赢 overall MAE。
- Regression-to-the-mean 仍然存在，且在 tail-covering stratified split 上仍能观察到，因此它不只是 split artifact。
- 当前所有结果都是 single-seed baseline；下一步需要在固定 split 下做更稳定的 seed/checkpoint 比较，并继续研究 calibration、tail-aware training 或 structure/contact-aware features。

## 缺失指标检查

{missing_text}

## 数据来源

- `{UNIFIED_REPORT.relative_to(ROOT)}`
- `{ANDD_ORIGINAL_REPORT.relative_to(ROOT)}`
- `{ANDD_STRATIFIED_REPORT.relative_to(ROOT)}`
"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_metrics_dataframe()
    frame.to_csv(CSV_PATH, index=False)
    draw_figure(frame)
    write_summary(frame)

    print(f"Saved metrics CSV: {CSV_PATH.relative_to(ROOT)}")
    print(f"Saved figure: {FIGURE_PATH.relative_to(ROOT)}")
    print(f"Saved summary: {SUMMARY_PATH.relative_to(ROOT)}")
    print("NaN metric count:", int(frame[["MAE", "Spearman", "pred_std_true_std", "high_tail_MAE_or_above_train_p90_MAE"]].isna().sum().sum()))


if __name__ == "__main__":
    main()
