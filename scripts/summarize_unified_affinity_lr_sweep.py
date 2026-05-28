"""Summarize finished learning-rate runs without training again.

中文人话说明：
这个脚本只做“整理结果”：
1. 从每个 sweep config 找到 checkpoint 和 prediction CSV；
2. 从 checkpoint 读取训练结束时保存的 validation metrics；
3. 从 prediction CSV 重新计算 test metrics 和 prediction 分布；
4. 用同一份 unified train/test split 计算 mean baseline；
5. 输出一张 CSV 和一份 Markdown 报告。

它不会调用训练脚本，也不会修改 model / dataset。
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hparam_sweep" / "unified_affinity_dataset_v1"
RESULTS_CSV_PATH = OUTPUT_DIR / "sweep_results.csv"
REPORT_PATH = OUTPUT_DIR / "sweep_report.md"

RUN_CONFIGS = [
    "config_affinity_unified_affinity_dataset_v1_lr1e-5_e10.yaml",
    "config_affinity_unified_affinity_dataset_v1_lr2e-5_e10.yaml",
    "config_affinity_unified_affinity_dataset_v1_lr3e-5_e10.yaml",
    "config_affinity_unified_affinity_dataset_v1_lr1e-4_e10.yaml",
]


def load_config(config_path: Path) -> dict:
    """Load one YAML config through the project's existing helper."""

    sys.path.append(str(PROJECT_ROOT))
    from src.utils import load_config as project_load_config

    return project_load_config(str(config_path))


def compute_metrics(true_values: pd.Series, predicted_values: pd.Series) -> dict:
    """Compute regression metrics from saved predictions.

    中文人话说明：
    prediction CSV 里已经有 true/predicted target。
    这里重新计算一次，保证四个 learning rate 用完全同一套算法汇总。
    """

    true = pd.to_numeric(true_values, errors="coerce")
    pred = pd.to_numeric(predicted_values, errors="coerce")
    valid = true.notna() & pred.notna()
    true = true[valid]
    pred = pred[valid]

    error = pred - true
    mse = float((error * error).mean())
    return {
        "MAE": float(error.abs().mean()),
        "RMSE": math.sqrt(mse),
        "Spearman": float(true.corr(pred, method="spearman")),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
    }


def read_final_val_metrics(checkpoint_path: Path) -> dict:
    """Read final validation metrics already saved inside the checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint.get("final_val_metrics")
    if not metrics:
        raise ValueError(f"Missing final_val_metrics in checkpoint: {checkpoint_path}")
    return metrics


def compute_mean_baseline(config: dict) -> dict:
    """Compute train-mean baseline on the same unified test split.

    中文人话说明：
    mean baseline 就是“所有 test 样本都猜 train target 平均值”。
    它是回归任务很重要的最低门槛：模型至少应该努力超过它。
    """

    target_column = config.get("target_column", "neg_log10_affinity")
    train_df = pd.read_csv(PROJECT_ROOT / config["train_csv"])
    test_df = pd.read_csv(PROJECT_ROOT / config["test_csv"])
    train_target = pd.to_numeric(train_df[target_column], errors="coerce")
    test_target = pd.to_numeric(test_df[target_column], errors="coerce")
    mean_prediction = pd.Series([float(train_target.mean())] * len(test_df))
    return compute_metrics(test_target, mean_prediction)


def collect_result(config_filename: str) -> dict:
    """Collect one finished run into one report row."""

    config = load_config(PROJECT_ROOT / config_filename)
    checkpoint_path = PROJECT_ROOT / config["checkpoint_path"]
    predictions_path = PROJECT_ROOT / config["predictions_path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for finished run: {checkpoint_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found for finished run: {predictions_path}")

    val_metrics = read_final_val_metrics(checkpoint_path)
    predictions = pd.read_csv(predictions_path)
    test_metrics = compute_metrics(
        predictions["true_neg_log10_affinity"],
        predictions["predicted_neg_log10_affinity"],
    )
    mean_baseline = compute_mean_baseline(config)

    beats_mae = test_metrics["MAE"] < mean_baseline["MAE"]
    beats_rmse = test_metrics["RMSE"] < mean_baseline["RMSE"]
    return {
        "learning_rate": float(config["learning_rate"]),
        "epochs": int(config["epochs"]),
        "val_MAE": float(val_metrics["mae"]),
        "val_RMSE": float(val_metrics["rmse"]),
        "val_Spearman": float(val_metrics["spearman"]),
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_Spearman": test_metrics["Spearman"],
        "prediction_mean": test_metrics["prediction_mean"],
        "prediction_std": test_metrics["prediction_std"],
        "mean_baseline_MAE": mean_baseline["MAE"],
        "mean_baseline_RMSE": mean_baseline["RMSE"],
        "beats_mean_baseline_MAE": beats_mae,
        "beats_mean_baseline_RMSE": beats_rmse,
        "beats_mean_baseline_both": beats_mae and beats_rmse,
        "config": config_filename,
        "checkpoint_path": config["checkpoint_path"],
        "predictions_path": config["predictions_path"],
    }


def format_metric(value: float) -> str:
    """Format numeric metric for the Markdown table."""

    return f"{value:.4f}"


def choose_best_run(results: pd.DataFrame) -> tuple[pd.Series, str]:
    """Choose a practical best run and explain why.

    Test MAE is the primary criterion here because it is the easiest average-error
    metric to interpret on the log10 affinity target. RMSE and Spearman stay visible
    so the report does not hide tradeoffs.
    """

    ordered = results.sort_values(["test_MAE", "test_RMSE"], ascending=[True, True])
    best = ordered.iloc[0]
    reason = (
        f"`{best['learning_rate']:.0e}` has the lowest test MAE "
        f"(`{best['test_MAE']:.4f}`); its test RMSE is `{best['test_RMSE']:.4f}` "
        f"and test Spearman is `{best['test_Spearman']:.4f}`."
    )
    return best, reason


def write_report(results: pd.DataFrame) -> None:
    """Write the human-readable sweep summary."""

    best_run, best_reason = choose_best_run(results)
    best_spearman = results.sort_values("test_Spearman", ascending=False).iloc[0]
    lines = [
        "# Unified Affinity Dataset v1 Learning Rate Sweep",
        "",
        "## Scope",
        "",
        "- Dataset: `data/processed_affinity/unified_affinity_dataset_v1/`",
        "- Runs summarized from existing checkpoints and prediction CSV files only.",
        "- Swept hyperparameter: `learning_rate`.",
        "- Fixed controls: `epochs=10`, same seed, same model code, same unified train/val/test split.",
        "",
        "## Results",
        "",
        "| learning_rate | epochs | val_MAE | val_RMSE | val_Spearman | test_MAE | test_RMSE | test_Spearman | prediction_mean | prediction_std | mean baseline MAE | mean baseline RMSE | beat mean MAE | beat mean RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for _, row in results.sort_values("learning_rate").iterrows():
        lines.append(
            f"| `{row['learning_rate']:.0e}` | {int(row['epochs'])} | "
            f"{format_metric(row['val_MAE'])} | {format_metric(row['val_RMSE'])} | "
            f"{format_metric(row['val_Spearman'])} | {format_metric(row['test_MAE'])} | "
            f"{format_metric(row['test_RMSE'])} | {format_metric(row['test_Spearman'])} | "
            f"{format_metric(row['prediction_mean'])} | {format_metric(row['prediction_std'])} | "
            f"{format_metric(row['mean_baseline_MAE'])} | {format_metric(row['mean_baseline_RMSE'])} | "
            f"`{bool(row['beats_mean_baseline_MAE'])}` | `{bool(row['beats_mean_baseline_RMSE'])}` |"
        )

    lines.extend(
        [
            "",
            "## Judgment",
            "",
            f"- Best by primary average-error criterion: learning rate `{best_run['learning_rate']:.0e}`.",
            f"- Reason: {best_reason}",
            f"- Best test Spearman: learning rate `{best_spearman['learning_rate']:.0e}` with `{best_spearman['test_Spearman']:.4f}`.",
            "- A low `prediction_std` means predictions are still compressed into a narrow range, even if MAE improves.",
            "- Mean baseline numbers are the same across rows because every run uses the same unified train/test split.",
            "",
            "## Files",
            "",
            f"- CSV summary: `{RESULTS_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- Markdown summary: `{REPORT_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Write CSV and Markdown summaries from existing finished runs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([collect_result(config) for config in RUN_CONFIGS])
    results = results.sort_values("learning_rate").reset_index(drop=True)
    results.to_csv(RESULTS_CSV_PATH, index=False)
    write_report(results)

    print(results.to_string(index=False))
    print(f"Saved CSV summary to {RESULTS_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved Markdown report to {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
