"""Post-hoc linear calibration for the ANDD antibody v2 all-CDR pooled model.

中文人话说明：
模型已经训练结束了，这个脚本不会重新训练模型、不会修改 checkpoint、不会改 dataset。

它做的是“预测值校准”：
1. 如果没有 validation predictions，就从已有 best checkpoint 在 val set 上跑 inference。
2. 只用 validation set 学一个简单直线：
       calibrated_prediction = a * raw_prediction + b
3. 把同一条直线应用到 test prediction。
4. 对比 raw vs calibrated 是否让预测范围更接近真实范围，以及 low/high target MAE 是否下降。

重要原则：
- 校准参数只从 validation set 学，不可以用 test labels 拟合。
- 若 a > 0，线性校准不会改变样本排名，所以 Spearman 理论上基本不变。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# matplotlib 缓存写进项目目录，避免系统用户 cache 不可写导致 warning。
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.affinity_cdr_dataset import CDRAwareAffinityDataset
from src.affinity_cdr_evaluate import evaluate_cdr_affinity_model
from src.affinity_cdr_model import SeqProFTCDRAwareAffinityRegressor
from src.utils import get_device, load_config


DEFAULT_CONFIG = "config_affinity_andd_antibody_v2_all_cdr_pooled_lr3e-5_e10.yaml"
TRUE_COLUMN = "true_neg_log10_affinity"
RAW_COLUMN = "raw_predicted_neg_log10_affinity"
CALIBRATED_COLUMN = "calibrated_predicted_neg_log10_affinity"
TARGET_BINS = ["low_target", "mid_target", "high_target"]


def parse_args() -> argparse.Namespace:
    """Read config and output directory arguments."""

    parser = argparse.ArgumentParser(description="Calibrate ANDD antibody v2 CDR-aware predictions.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", default="outputs/andd_antibody_v2/calibration")
    return parser.parse_args()


def target_bins(true_values: pd.Series) -> pd.Series:
    """Build equally sized low/mid/high bins from true target ranks."""

    ranks = pd.to_numeric(true_values, errors="raise").rank(method="first")
    return pd.qcut(ranks, q=3, labels=TARGET_BINS)


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    """Compute correlation, returning None if it is undefined."""

    result = pd.to_numeric(left, errors="raise").corr(pd.to_numeric(right, errors="raise"), method=method)
    return None if pd.isna(result) else float(result)


def compute_metrics(frame: pd.DataFrame, prediction_column: str) -> dict:
    """Compute regression and regression-to-the-mean diagnostics."""

    true_values = pd.to_numeric(frame[TRUE_COLUMN], errors="raise")
    predictions = pd.to_numeric(frame[prediction_column], errors="raise")
    errors = predictions - true_values
    absolute_errors = errors.abs()
    mse = float((errors**2).mean())
    true_std = float(true_values.std())
    prediction_std = float(predictions.std())

    working = frame.copy()
    working["target_bin"] = target_bins(true_values)
    working["analysis_absolute_error"] = absolute_errors
    working["analysis_error"] = errors
    bin_mae = {
        str(name): float(group["analysis_absolute_error"].mean())
        for name, group in working.groupby("target_bin", observed=True)
    }
    bin_mean_error = {
        str(name): float(group["analysis_error"].mean())
        for name, group in working.groupby("target_bin", observed=True)
    }

    return {
        "mae": float(absolute_errors.mean()),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "spearman": safe_corr(true_values, predictions, "spearman"),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": prediction_std,
        "true_mean": float(true_values.mean()),
        "true_std": true_std,
        "pred_std_over_true_std": prediction_std / true_std if true_std else None,
        "error_vs_true_pearson": safe_corr(errors, true_values, "pearson"),
        "low_target_mae": bin_mae.get("low_target"),
        "mid_target_mae": bin_mae.get("mid_target"),
        "high_target_mae": bin_mae.get("high_target"),
        "low_target_mean_error": bin_mean_error.get("low_target"),
        "mid_target_mean_error": bin_mean_error.get("mid_target"),
        "high_target_mean_error": bin_mean_error.get("high_target"),
    }


def prediction_rows(dataset: CDRAwareAffinityDataset, true_values: list[float], predictions: list[float]) -> pd.DataFrame:
    """Pair inference outputs with essential validation metadata."""

    rows = []
    for row, true_value, predicted_value in zip(dataset.data.to_dict("records"), true_values, predictions):
        rows.append(
            {
                "sample_id": row.get("sample_id", row.get("candidate_id", "")),
                TRUE_COLUMN: float(true_value),
                RAW_COLUMN: float(predicted_value),
                "source": row.get("source", ""),
                "pdb_id": row.get("pdb_id", ""),
                "ag_name": row.get("ag_name", ""),
            }
        )
    return pd.DataFrame(rows)


def generate_val_predictions(config: dict, output_path: Path) -> pd.DataFrame:
    """Run validation inference from existing checkpoint, without any training."""

    device = get_device()
    print(f"Generating validation predictions with device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = CDRAwareAffinityDataset(
        csv_path=config["val_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        input_cdr_fields=config.get("input_cdr_fields"),
    )
    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    model = SeqProFTCDRAwareAffinityRegressor(config).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, true_values, predicted_values = evaluate_cdr_affinity_model(model, dataloader, device)
    frame = prediction_rows(dataset, true_values, predicted_values)
    frame.to_csv(output_path, index=False)
    print(f"Saved validation predictions to {output_path}")
    return frame


def load_test_predictions(config: dict) -> pd.DataFrame:
    """Load original test predictions without overwriting them."""

    raw = pd.read_csv(config["predictions_path"])
    required = {TRUE_COLUMN, "predicted_neg_log10_affinity"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Original test predictions are missing columns: {sorted(missing)}")
    raw = raw.copy()
    raw[RAW_COLUMN] = pd.to_numeric(raw["predicted_neg_log10_affinity"], errors="raise")
    return raw


def fit_linear_calibration(validation: pd.DataFrame) -> tuple[float, float]:
    """Fit y_true = a * raw_prediction + b using validation only."""

    raw_predictions = pd.to_numeric(validation[RAW_COLUMN], errors="raise").to_numpy()
    true_values = pd.to_numeric(validation[TRUE_COLUMN], errors="raise").to_numpy()
    design = np.column_stack([raw_predictions, np.ones_like(raw_predictions)])
    coefficients, _, _, _ = np.linalg.lstsq(design, true_values, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def add_calibration_columns(frame: pd.DataFrame, slope: float, intercept: float) -> pd.DataFrame:
    """Apply fitted calibration and add interpretable raw/calibrated errors."""

    output = frame.copy()
    output[CALIBRATED_COLUMN] = slope * output[RAW_COLUMN] + intercept
    output["raw_error"] = output[RAW_COLUMN] - output[TRUE_COLUMN]
    output["raw_absolute_error"] = output["raw_error"].abs()
    output["calibrated_error"] = output[CALIBRATED_COLUMN] - output[TRUE_COLUMN]
    output["calibrated_absolute_error"] = output["calibrated_error"].abs()
    output["calibrated_fold_error"] = 10 ** output["calibrated_absolute_error"]
    output["target_bin"] = target_bins(output[TRUE_COLUMN])
    return output


def metrics_rows(raw_metrics: dict, calibrated_metrics: dict) -> pd.DataFrame:
    """Convert metric dictionaries to one tidy CSV table."""

    rows = []
    for name, metrics in [("raw", raw_metrics), ("calibrated", calibrated_metrics)]:
        rows.append({"prediction_type": name, **metrics})
    return pd.DataFrame(rows)


def fmt(value: float | None) -> str:
    """Small report-format helper."""

    return "NA" if value is None else f"{value:.4f}"


def write_report(
    output_path: Path,
    slope: float,
    intercept: float,
    val_raw: dict,
    val_calibrated: dict,
    test_raw: dict,
    test_calibrated: dict,
    calibrated_predictions_path: Path,
    figure_path: Path,
) -> None:
    """Write readable calibration conclusions."""

    lines = [
        "# ANDD Antibody v2 All-CDR Pooled: Post-hoc Linear Calibration",
        "",
        "## Scope",
        "",
        "- This is post-hoc calibration only; the ESM2+LoRA model was not retrained.",
        "- Calibration was fitted on validation predictions only and applied once to the test predictions.",
        "- Original test predictions and original model results were not overwritten.",
        "",
        "## Calibration Formula",
        "",
        f"`calibrated_pred = {slope:.6f} * raw_pred + {intercept:.6f}`",
        "",
        "A positive slope preserves ranking order, so Spearman is expected to remain essentially unchanged; the main question is whether calibration improves scale/spread and absolute errors.",
        "",
        "## Validation Metrics Used To Fit Calibration",
        "",
        "| Prediction | MAE | RMSE | Spearman | pred_std / true_std | error vs true Pearson | low MAE | mid MAE | high MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| raw | {val_raw['mae']:.4f} | {val_raw['rmse']:.4f} | {fmt(val_raw['spearman'])} | "
            f"{fmt(val_raw['pred_std_over_true_std'])} | {fmt(val_raw['error_vs_true_pearson'])} | "
            f"{val_raw['low_target_mae']:.4f} | {val_raw['mid_target_mae']:.4f} | {val_raw['high_target_mae']:.4f} |"
        ),
        (
            f"| calibrated | {val_calibrated['mae']:.4f} | {val_calibrated['rmse']:.4f} | {fmt(val_calibrated['spearman'])} | "
            f"{fmt(val_calibrated['pred_std_over_true_std'])} | {fmt(val_calibrated['error_vs_true_pearson'])} | "
            f"{val_calibrated['low_target_mae']:.4f} | {val_calibrated['mid_target_mae']:.4f} | {val_calibrated['high_target_mae']:.4f} |"
        ),
        "",
        "## Test Metrics: Raw vs Calibrated",
        "",
        "| Prediction | MAE | RMSE | Spearman | prediction std | pred_std / true_std | error vs true Pearson | low MAE | mid MAE | high MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| raw | {test_raw['mae']:.4f} | {test_raw['rmse']:.4f} | {fmt(test_raw['spearman'])} | "
            f"{test_raw['prediction_std']:.4f} | {fmt(test_raw['pred_std_over_true_std'])} | "
            f"{fmt(test_raw['error_vs_true_pearson'])} | {test_raw['low_target_mae']:.4f} | "
            f"{test_raw['mid_target_mae']:.4f} | {test_raw['high_target_mae']:.4f} |"
        ),
        (
            f"| calibrated | {test_calibrated['mae']:.4f} | {test_calibrated['rmse']:.4f} | {fmt(test_calibrated['spearman'])} | "
            f"{test_calibrated['prediction_std']:.4f} | {fmt(test_calibrated['pred_std_over_true_std'])} | "
            f"{fmt(test_calibrated['error_vs_true_pearson'])} | {test_calibrated['low_target_mae']:.4f} | "
            f"{test_calibrated['mid_target_mae']:.4f} | {test_calibrated['high_target_mae']:.4f} |"
        ),
        "",
        "## Reading The Result",
        "",
        f"- Change in test `pred_std / true_std`: `{test_raw['pred_std_over_true_std']:.4f}` -> `{test_calibrated['pred_std_over_true_std']:.4f}`.",
        f"- Change in test low-target MAE: `{test_raw['low_target_mae']:.4f}` -> `{test_calibrated['low_target_mae']:.4f}`.",
        f"- Change in test high-target MAE: `{test_raw['high_target_mae']:.4f}` -> `{test_calibrated['high_target_mae']:.4f}`.",
        f"- Change in test overall MAE: `{test_raw['mae']:.4f}` -> `{test_calibrated['mae']:.4f}`.",
        "",
        "Post-hoc calibration can correct prediction scale, but it does not add new biological information or improve ranking when the fitted slope is positive.",
        "",
        "## Output Files",
        "",
        f"- Calibrated test predictions: `{calibrated_predictions_path}`",
        f"- True vs predicted figure: `{figure_path}`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def draw_true_vs_predicted(test: pd.DataFrame, output_path: Path) -> None:
    """Draw raw and calibrated test parity plots with equal axes."""

    true_values = test[TRUE_COLUMN]
    all_predictions = pd.concat([test[RAW_COLUMN], test[CALIBRATED_COLUMN]])
    lower = float(min(true_values.min(), all_predictions.min()))
    upper = float(max(true_values.max(), all_predictions.max()))
    margin = 0.15
    lower -= margin
    upper += margin

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, column, title, color in [
        (axes[0], RAW_COLUMN, "Raw test predictions", "#4C78A8"),
        (axes[1], CALIBRATED_COLUMN, "Calibrated test predictions", "#F58518"),
    ]:
        ax.scatter(test[TRUE_COLUMN], test[column], alpha=0.72, color=color, s=28)
        ax.plot([lower, upper], [lower, upper], "--", color="#333333", linewidth=1)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("True neg_log10_affinity")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted neg_log10_affinity")
    fig.suptitle("ANDD Antibody v2: Post-hoc Linear Calibration", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Fit calibration on validation predictions and apply it to test predictions."""

    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_predictions_path = output_dir / "validation_raw_predictions.csv"
    if val_predictions_path.exists():
        print(f"Using cached validation predictions: {val_predictions_path}")
        validation = pd.read_csv(val_predictions_path)
    else:
        validation = generate_val_predictions(config, val_predictions_path)

    if "predicted_neg_log10_affinity" in validation.columns and RAW_COLUMN not in validation.columns:
        validation[RAW_COLUMN] = validation["predicted_neg_log10_affinity"]

    slope, intercept = fit_linear_calibration(validation)
    print(f"Calibration formula: calibrated_pred = {slope:.6f} * raw_pred + {intercept:.6f}")

    calibrated_validation = add_calibration_columns(validation, slope, intercept)
    test = load_test_predictions(config)
    calibrated_test = add_calibration_columns(test, slope, intercept)

    val_raw_metrics = compute_metrics(calibrated_validation, RAW_COLUMN)
    val_calibrated_metrics = compute_metrics(calibrated_validation, CALIBRATED_COLUMN)
    test_raw_metrics = compute_metrics(calibrated_test, RAW_COLUMN)
    test_calibrated_metrics = compute_metrics(calibrated_test, CALIBRATED_COLUMN)

    calibrated_predictions_path = output_dir / "calibrated_test_predictions.csv"
    calibrated_test.to_csv(calibrated_predictions_path, index=False)
    calibrated_validation.to_csv(output_dir / "calibrated_validation_predictions.csv", index=False)

    metrics = pd.concat(
        [
            metrics_rows(val_raw_metrics, val_calibrated_metrics).assign(split="val"),
            metrics_rows(test_raw_metrics, test_calibrated_metrics).assign(split="test"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
    (output_dir / "calibration_parameters.json").write_text(
        json.dumps({"slope_a": slope, "intercept_b": intercept}, indent=2),
        encoding="utf-8",
    )

    figure_path = output_dir / "true_vs_predicted_raw_vs_calibrated.png"
    draw_true_vs_predicted(calibrated_test, figure_path)
    write_report(
        output_dir / "calibration_report.md",
        slope,
        intercept,
        val_raw_metrics,
        val_calibrated_metrics,
        test_raw_metrics,
        test_calibrated_metrics,
        calibrated_predictions_path,
        figure_path,
    )

    print(
        f"Test raw MAE/RMSE: {test_raw_metrics['mae']:.4f} / {test_raw_metrics['rmse']:.4f}; "
        f"calibrated: {test_calibrated_metrics['mae']:.4f} / {test_calibrated_metrics['rmse']:.4f}"
    )
    print(
        f"Test pred_std/true_std: {test_raw_metrics['pred_std_over_true_std']:.4f} "
        f"-> {test_calibrated_metrics['pred_std_over_true_std']:.4f}"
    )
    print(f"Saved report to {output_dir / 'calibration_report.md'}")


if __name__ == "__main__":
    main()
