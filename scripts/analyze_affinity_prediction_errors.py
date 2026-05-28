"""Analyze test prediction errors for TDC plus SAbDab supplement v1.

中文人话说明：
这个脚本只分析已经保存好的 prediction CSV，不训练模型。

我们把两份文件对齐：
1. predictions CSV：有 true / predicted / error。
2. test CSV：有 source 和 antigen_sequence 等 metadata。

这样可以回答：
- 模型到底是数值误差大，还是排序完全坏了？
- 预测范围有没有比真实 target 范围窄很多？
- 低/中/高 target 哪个区间 MAE 最大？
- source 或 antigen length 是否和误差有关？
"""

from __future__ import annotations

import os
from pathlib import Path
import json
import math


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "affinity_tdc_plus_sabdab_supplement_v1_test_predictions.csv"
TEST_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "tdc_plus_sabdab_supplement_v1"
    / "antigen_group_split"
    / "test.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis" / "tdc_plus_sabdab_supplement_v1"
JSON_REPORT_PATH = OUTPUT_DIR / "error_analysis_report.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "error_analysis_report.md"

# Matplotlib 在脚本模式下也会写字体缓存。
# 缓存放在项目 outputs 里，避免 IDE/沙盒环境写用户 home 失败。
MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / "matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"
ERROR_COLUMN = "error"
ABS_ERROR_COLUMN = "absolute_error"
TARGET_BIN_ORDER = ["low", "mid", "high"]
LENGTH_BIN_ORDER = ["short", "medium", "long"]


def load_and_merge_inputs() -> pd.DataFrame:
    """Read predictions and test metadata, then align them by sample_id."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Cannot find predictions file: {PREDICTIONS_PATH}")
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Cannot find test CSV: {TEST_CSV_PATH}")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    test = pd.read_csv(TEST_CSV_PATH)

    prediction_required = {"sample_id", TRUE_COLUMN, PRED_COLUMN}
    test_required = {"sample_id", "source", "antigen_sequence"}
    missing_prediction = prediction_required - set(predictions.columns)
    missing_test = test_required - set(test.columns)
    if missing_prediction:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing_prediction)}")
    if missing_test:
        raise ValueError(f"Test CSV missing columns: {sorted(missing_test)}")

    metadata_columns = ["sample_id", "source", "antigen_sequence"]
    if "supplement_candidate_id" in test.columns:
        metadata_columns.append("supplement_candidate_id")

    metadata = test[metadata_columns].copy()
    metadata = metadata.rename(columns={"source": "test_source"})
    merged = predictions.merge(metadata, on="sample_id", how="left", validate="one_to_one")

    if merged["test_source"].isna().any():
        missing_ids = merged.loc[merged["test_source"].isna(), "sample_id"].astype(str).tolist()
        raise ValueError(f"Prediction sample IDs missing from test CSV: {missing_ids[:10]}")

    # 重新算 error，避免旧 CSV 中 error 列被手动改坏后分析还悄悄继续。
    merged[TRUE_COLUMN] = pd.to_numeric(merged[TRUE_COLUMN], errors="raise")
    merged[PRED_COLUMN] = pd.to_numeric(merged[PRED_COLUMN], errors="raise")
    merged[ERROR_COLUMN] = merged[PRED_COLUMN] - merged[TRUE_COLUMN]
    merged[ABS_ERROR_COLUMN] = merged[ERROR_COLUMN].abs()
    merged["antigen_len"] = merged["antigen_sequence_y"].fillna("").astype(str).str.len()
    merged["source_for_analysis"] = merged["test_source"].astype(str)
    return merged


def numeric_summary(series: pd.Series) -> dict:
    """Return count/min/max/mean/std for one numeric series."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    """Return correlation or None if pandas cannot compute it."""

    value = pd.to_numeric(left, errors="coerce").corr(pd.to_numeric(right, errors="coerce"), method=method)
    return None if pd.isna(value) else float(value)


def regression_metrics(dataframe: pd.DataFrame) -> dict:
    """Compute MAE, RMSE, and Spearman from saved predictions."""

    errors = dataframe[ERROR_COLUMN].astype(float)
    mae = float(dataframe[ABS_ERROR_COLUMN].mean())
    rmse = float(math.sqrt((errors * errors).mean()))
    spearman = safe_correlation(dataframe[TRUE_COLUMN], dataframe[PRED_COLUMN], "spearman")
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "approx_mae_fold_error": float(10 ** mae),
        "approx_rmse_fold_error": float(10 ** rmse),
    }


def quantile_bin(series: pd.Series, labels: list[str]) -> pd.Series:
    """Split a series into quantile bins while keeping duplicate edges safe."""

    ranked = pd.to_numeric(series, errors="coerce").rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels)


def mae_group_summary(dataframe: pd.DataFrame, group_column: str) -> dict:
    """Summarize MAE/RMSE and target/prediction means by a group column."""

    summary = {}
    for group_name, group in dataframe.groupby(group_column, observed=True, dropna=False):
        errors = group[ERROR_COLUMN].astype(float)
        summary[str(group_name)] = {
            "count": int(len(group)),
            "mae": float(group[ABS_ERROR_COLUMN].mean()),
            "rmse": float(math.sqrt((errors * errors).mean())),
            "true_mean": float(group[TRUE_COLUMN].mean()),
            "pred_mean": float(group[PRED_COLUMN].mean()),
        }
        if group_column == "antigen_length_bin":
            summary[str(group_name)]["antigen_len_min"] = int(group["antigen_len"].min())
            summary[str(group_name)]["antigen_len_max"] = int(group["antigen_len"].max())
            summary[str(group_name)]["antigen_len_mean"] = float(group["antigen_len"].mean())
    return summary


def add_analysis_bins(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add low/mid/high target bins and short/medium/long antigen length bins."""

    analyzed = dataframe.copy()
    analyzed["target_bin"] = quantile_bin(analyzed[TRUE_COLUMN], TARGET_BIN_ORDER)
    analyzed["antigen_length_bin"] = quantile_bin(analyzed["antigen_len"], LENGTH_BIN_ORDER)
    return analyzed


def build_report(dataframe: pd.DataFrame) -> dict:
    """Build JSON report with all requested error analysis statistics."""

    true_summary = numeric_summary(dataframe[TRUE_COLUMN])
    pred_summary = numeric_summary(dataframe[PRED_COLUMN])
    true_std = true_summary["std"]
    pred_std = pred_summary["std"]
    pred_std_ratio = None if true_std in {None, 0} else float(pred_std / true_std)

    report = {
        "inputs": {
            "predictions_csv": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "test_csv": str(TEST_CSV_PATH.relative_to(PROJECT_ROOT)),
        },
        "sample_count": int(len(dataframe)),
        "metrics": regression_metrics(dataframe),
        "true_target_summary": true_summary,
        "prediction_summary": pred_summary,
        "pred_std_over_true_std": pred_std_ratio,
        "error_vs_true_target_correlation": {
            "pearson": safe_correlation(dataframe[ERROR_COLUMN], dataframe[TRUE_COLUMN], "pearson"),
            "spearman": safe_correlation(dataframe[ERROR_COLUMN], dataframe[TRUE_COLUMN], "spearman"),
        },
        "absolute_error_vs_true_target_correlation": {
            "pearson": safe_correlation(dataframe[ABS_ERROR_COLUMN], dataframe[TRUE_COLUMN], "pearson"),
            "spearman": safe_correlation(dataframe[ABS_ERROR_COLUMN], dataframe[TRUE_COLUMN], "spearman"),
        },
        "mae_by_target_bin": mae_group_summary(dataframe, "target_bin"),
        "mae_by_source": mae_group_summary(dataframe, "source_for_analysis"),
        "mae_by_antigen_length_bin": mae_group_summary(dataframe, "antigen_length_bin"),
        "notes": {
            "spearman_note": "Spearman measures ranking tendency; MAE/RMSE measure numeric error size.",
            "residual_note": "error = predicted - true. Strong negative error-vs-true correlation suggests regression-to-mean.",
            "bin_note": "Target and antigen-length bins are quantile bins so this small test set is split into comparable row counts.",
        },
    }
    return report


def parity_limits(dataframe: pd.DataFrame) -> tuple[float, float]:
    """Use equal true/pred axes for fair parity plotting."""

    min_value = min(dataframe[TRUE_COLUMN].min(), dataframe[PRED_COLUMN].min())
    max_value = max(dataframe[TRUE_COLUMN].max(), dataframe[PRED_COLUMN].max())
    padding = (max_value - min_value) * 0.05
    if padding == 0:
        padding = 0.5
    return float(min_value - padding), float(max_value + padding)


def save_true_vs_predicted(dataframe: pd.DataFrame) -> Path:
    """Save parity plot with true target on x-axis and prediction on y-axis."""

    output_path = OUTPUT_DIR / "true_vs_predicted.png"
    min_value, max_value = parity_limits(dataframe)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        dataframe[TRUE_COLUMN],
        dataframe[PRED_COLUMN],
        alpha=0.78,
        color="#4C78A8",
        label="test samples",
    )
    plt.plot([min_value, max_value], [min_value, max_value], "--", color="#E45756", label="y = x")
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("True vs Predicted Affinity (Test Set)")
    plt.xlabel("True neg_log10_affinity")
    plt.ylabel("Predicted neg_log10_affinity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_residual_vs_true(dataframe: pd.DataFrame) -> Path:
    """Save residual plot for regression-to-mean diagnosis."""

    output_path = OUTPUT_DIR / "residual_vs_true.png"

    plt.figure(figsize=(8, 6))
    for source_name, group in dataframe.groupby("source_for_analysis"):
        plt.scatter(
            group[TRUE_COLUMN],
            group[ERROR_COLUMN],
            alpha=0.78,
            label=str(source_name),
        )
    plt.axhline(0, linestyle="--", color="#E45756", label="zero error")
    plt.title("Residual vs True Target")
    plt.xlabel("True neg_log10_affinity")
    plt.ylabel("Prediction error = predicted - true")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_abs_error_by_target_bin(dataframe: pd.DataFrame) -> Path:
    """Save MAE bar chart for low/mid/high target bins."""

    output_path = OUTPUT_DIR / "abs_error_by_target_bin.png"
    mae_values = (
        dataframe.groupby("target_bin", observed=True)[ABS_ERROR_COLUMN]
        .mean()
        .reindex(TARGET_BIN_ORDER)
    )

    plt.figure(figsize=(8, 6))
    plt.bar(mae_values.index.astype(str), mae_values.values, color=["#4C78A8", "#54A24B", "#F58518"], label="MAE")
    plt.title("Absolute Error by Target Bin")
    plt.xlabel("Target quantile bin")
    plt.ylabel("Mean absolute error (log10 units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_prediction_distribution(dataframe: pd.DataFrame) -> Path:
    """Save true and predicted target histograms with shared bins."""

    output_path = OUTPUT_DIR / "prediction_distribution.png"
    all_values = pd.concat([dataframe[TRUE_COLUMN], dataframe[PRED_COLUMN]], ignore_index=True)
    bins = np.linspace(float(all_values.min()), float(all_values.max()), 20)

    plt.figure(figsize=(9, 6))
    plt.hist(dataframe[TRUE_COLUMN], bins=bins, alpha=0.58, color="#4C78A8", label="true target")
    plt.hist(dataframe[PRED_COLUMN], bins=bins, alpha=0.58, color="#F58518", label="prediction")
    plt.title("True and Predicted Target Distribution")
    plt.xlabel("neg_log10_affinity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_figures(dataframe: pd.DataFrame) -> list[str]:
    """Create all requested PNG figures and return relative paths."""

    figure_paths = [
        save_true_vs_predicted(dataframe),
        save_residual_vs_true(dataframe),
        save_abs_error_by_target_bin(dataframe),
        save_prediction_distribution(dataframe),
    ]
    return [str(path.relative_to(PROJECT_ROOT)) for path in figure_paths]


def markdown_group_table(title: str, group_summary: dict) -> list[str]:
    """Render one grouped MAE summary table."""

    lines = [
        f"## {title}",
        "",
        "| group | count | MAE | RMSE | true mean | pred mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group_name, summary in group_summary.items():
        lines.append(
            f"| `{group_name}` | {summary['count']} | {summary['mae']:.4f} | {summary['rmse']:.4f} | "
            f"{summary['true_mean']:.4f} | {summary['pred_mean']:.4f} |"
        )
    lines.append("")
    return lines


def write_markdown_report(report: dict) -> None:
    """Write a readable error-analysis report beside JSON."""

    metrics = report["metrics"]
    lines = [
        "# Affinity Prediction Error Analysis",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{report['inputs']['predictions_csv']}`",
        f"- Test metadata: `{report['inputs']['test_csv']}`",
        "",
        "## Core Metrics",
        "",
        f"- Samples: {report['sample_count']}",
        f"- MAE: {metrics['mae']:.4f} log10 units",
        f"- RMSE: {metrics['rmse']:.4f} log10 units",
        f"- Spearman: {metrics['spearman']:.4f}",
        f"- Approx fold error from MAE: {metrics['approx_mae_fold_error']:.1f}x",
        f"- Approx fold error from RMSE: {metrics['approx_rmse_fold_error']:.1f}x",
        "",
        "## Prediction Range",
        "",
        f"- True target summary: `{report['true_target_summary']}`",
        f"- Prediction summary: `{report['prediction_summary']}`",
        f"- `pred_std / true_std`: `{report['pred_std_over_true_std']:.4f}`",
        "",
        "## Error Correlations",
        "",
        f"- Error vs true target: `{report['error_vs_true_target_correlation']}`",
        f"- Absolute error vs true target: `{report['absolute_error_vs_true_target_correlation']}`",
        "",
    ]
    lines.extend(markdown_group_table("MAE By Low/Mid/High Target Bin", report["mae_by_target_bin"]))
    lines.extend(markdown_group_table("MAE By Source", report["mae_by_source"]))
    lines.extend(markdown_group_table("MAE By Antigen Length Bin", report["mae_by_antigen_length_bin"]))
    lines.extend(
        [
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in report["figures"])
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {text}" for text in report["notes"].values())
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: dict) -> None:
    """Print short terminal summary after report and figures are saved."""

    metrics = report["metrics"]
    print("Affinity prediction error analysis complete.")
    print(f"Samples: {report['sample_count']}")
    print(f"MAE / RMSE / Spearman: {metrics['mae']:.4f} / {metrics['rmse']:.4f} / {metrics['spearman']:.4f}")
    print(f"pred_std / true_std: {report['pred_std_over_true_std']:.4f}")
    print(f"Error vs true target correlation: {report['error_vs_true_target_correlation']}")
    print(f"MAE by target bin: {report['mae_by_target_bin']}")
    print(f"MAE by source: {report['mae_by_source']}")
    print(f"MAE by antigen length bin: {report['mae_by_antigen_length_bin']}")
    print("Generated figures:")
    for path in report["figures"]:
        print(f"  {path}")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Run prediction error analysis and save report artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_and_merge_inputs()
    analyzed = add_analysis_bins(merged)
    report = build_report(analyzed)
    report["figures"] = save_figures(analyzed)

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_markdown_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()
