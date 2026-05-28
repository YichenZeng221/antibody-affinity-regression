"""Summarize CDR ablation test predictions into one CSV and markdown table.

:

 CDR mode  evaluation  predictions CSV
 predictions , CDR  signal
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.affinity_evaluate import compute_regression_metrics
from src.utils import load_config


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cdr_ablation" / "unified_no_high_risk"
SUMMARY_CSV = OUTPUT_DIR / "cdr_ablation_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "cdr_ablation_summary.md"
TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"
TARGET_BIN_LABELS = ["low_target", "mid_target", "high_target"]

CONFIGS = [
    "config_affinity_unified_no_high_risk_cdr_aware_lr3e-5_e10.yaml",
    "config_affinity_unified_no_high_risk_hcdr3_antigen_lr3e-5_e10.yaml",
    "config_affinity_unified_no_high_risk_hcdr3_lcdr3_antigen_lr3e-5_e10.yaml",
    "config_affinity_unified_no_high_risk_heavy_cdrs_antigen_lr3e-5_e10.yaml",
    "config_affinity_unified_no_high_risk_light_cdrs_antigen_lr3e-5_e10.yaml",
]


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    """Return numeric correlation or None if a metric is undefined."""

    value = pd.to_numeric(left, errors="coerce").corr(
        pd.to_numeric(right, errors="coerce"),
        method=method,
    )
    return None if pd.isna(value) else float(value)


def prediction_metrics(predictions: pd.DataFrame) -> dict:
    """Compute requested ablation metrics from one saved prediction file."""

    true_values = pd.to_numeric(predictions[TRUE_COLUMN], errors="raise")
    predicted_values = pd.to_numeric(predictions[PRED_COLUMN], errors="raise")
    error = predicted_values - true_values
    absolute_error = error.abs()
    basic = compute_regression_metrics(true_values.tolist(), predicted_values.tolist())

    ranked = true_values.rank(method="first")
    target_bins = pd.qcut(ranked, q=3, labels=TARGET_BIN_LABELS)
    bin_frame = pd.DataFrame({"target_bin": target_bins, "absolute_error": absolute_error})
    target_bin_mae = {
        str(bin_name): float(group["absolute_error"].mean())
        for bin_name, group in bin_frame.groupby("target_bin", observed=True)
    }

    true_std = float(true_values.std())
    pred_std = float(predicted_values.std())
    return {
        "rows": int(len(predictions)),
        "test_MAE": basic["mae"],
        "test_RMSE": basic["rmse"],
        "test_Spearman": basic["spearman"],
        "prediction_std": pred_std,
        "true_std": true_std,
        "pred_std_over_true_std": pred_std / true_std if true_std else None,
        "error_vs_true_Pearson": safe_corr(error, true_values, "pearson"),
        "low_target_MAE": target_bin_mae.get("low_target"),
        "mid_target_MAE": target_bin_mae.get("mid_target"),
        "high_target_MAE": target_bin_mae.get("high_target"),
    }


def collect_rows() -> list[dict]:
    """Collect rows for modes whose predictions already exist."""

    rows = []
    for config_path in CONFIGS:
        config = load_config(config_path)
        predictions_path = PROJECT_ROOT / config["predictions_path"]
        row = {
            "mode": config["mode_name"],
            "input_cdr_fields": ",".join(config["input_cdr_fields"]),
            "config": config_path,
            "predictions_path": config["predictions_path"],
            "status": "ready" if predictions_path.exists() else "missing_predictions",
        }
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            row.update(prediction_metrics(predictions))
        rows.append(row)
    return rows


def format_value(value) -> str:
    """Make markdown numbers compact."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_markdown(summary: pd.DataFrame) -> None:
    """Write human-readable ablation summary."""

    columns = [
        "mode",
        "input_cdr_fields",
        "status",
        "test_MAE",
        "test_RMSE",
        "test_Spearman",
        "prediction_std",
        "pred_std_over_true_std",
        "error_vs_true_Pearson",
        "low_target_MAE",
        "mid_target_MAE",
        "high_target_MAE",
    ]
    lines = [
        "# Unified No High Risk CDR Ablation Summary",
        "",
        "This report is generated from saved CDR ablation test prediction CSVs. Missing rows mean that mode has not been evaluated yet.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in summary[columns].to_dict("records"):
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- Lower MAE/RMSE means smaller absolute target error.",
            "- Higher Spearman means stronger ranking ability.",
            "- `pred_std_over_true_std` nearer 1 and `error_vs_true_Pearson` nearer 0 indicate less regression-to-mean compression.",
            "- Compare high-target MAE carefully because affinity extremes were difficult for earlier baselines.",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Write ablation summary files from available predictions."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(collect_rows())
    summary.to_csv(SUMMARY_CSV, index=False)
    write_markdown(summary)
    print(f"Saved CDR ablation summary CSV: {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved CDR ablation summary markdown: {SUMMARY_MD.relative_to(PROJECT_ROOT)}")
    print(summary[["mode", "status", "test_MAE", "test_Spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
