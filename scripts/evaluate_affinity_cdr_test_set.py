"""Evaluate CDR-aware affinity regression and compare with whole-sequence baseline.

中文人话说明：
这个脚本在你训练完 CDR-aware checkpoint 后运行。
它会：
1. 在 filtered CDR-aware test set 上预测。
2. 保存每个样本 prediction。
3. 计算 MAE/RMSE/Spearman 和 regression-to-mean 诊断。
4. 如果 whole-sequence best predictions 存在，在同一批 sample_id 上做对照。
5. 写 `cdr_aware_report.md`，方便你直接看实验结论。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.affinity_cdr_dataset import CDRAwareAffinityDataset
from src.affinity_cdr_evaluate import evaluate_cdr_affinity_model
from src.affinity_cdr_model import SeqProFTCDRAwareAffinityRegressor
from src.affinity_evaluate import compute_regression_metrics
from src.utils import get_device, load_config


TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"
TARGET_BIN_LABELS = ["low_target", "mid_target", "high_target"]


def parse_args() -> argparse.Namespace:
    """Read CDR-aware config path."""

    parser = argparse.ArgumentParser(description="Evaluate CDR-aware affinity test set.")
    parser.add_argument(
        "--config",
        default="config_affinity_unified_no_high_risk_cdr_aware_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def quantile_target_bins(true_values: pd.Series) -> pd.Series:
    """Create low/mid/high bins with near-equal sample counts."""

    ranked = pd.to_numeric(true_values, errors="coerce").rank(method="first")
    return pd.qcut(ranked, q=3, labels=TARGET_BIN_LABELS)


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    """Return correlation or None when it is undefined."""

    value = pd.to_numeric(left, errors="coerce").corr(
        pd.to_numeric(right, errors="coerce"),
        method=method,
    )
    return None if pd.isna(value) else float(value)


def metric_bundle(predictions: pd.DataFrame) -> dict:
    """Compute metrics and regression-to-mean diagnostics from one predictions table."""

    true_values = pd.to_numeric(predictions[TRUE_COLUMN], errors="raise")
    predicted_values = pd.to_numeric(predictions[PRED_COLUMN], errors="raise")
    error = predicted_values - true_values
    absolute_error = error.abs()

    metrics = compute_regression_metrics(true_values.tolist(), predicted_values.tolist())
    true_std = float(true_values.std())
    prediction_std = float(predicted_values.std())
    predictions = predictions.copy()
    predictions["target_bin"] = quantile_target_bins(true_values)
    predictions["absolute_error_for_metrics"] = absolute_error
    bin_mae = {
        str(bin_name): float(group["absolute_error_for_metrics"].mean())
        for bin_name, group in predictions.groupby("target_bin", observed=True)
    }
    bin_mean_error = {
        str(bin_name): float(
            (
                pd.to_numeric(group[PRED_COLUMN], errors="raise")
                - pd.to_numeric(group[TRUE_COLUMN], errors="raise")
            ).mean()
        )
        for bin_name, group in predictions.groupby("target_bin", observed=True)
    }

    return {
        **metrics,
        "prediction_mean": float(predicted_values.mean()),
        "prediction_std": prediction_std,
        "true_mean": float(true_values.mean()),
        "true_std": true_std,
        "prediction_std_over_true_std": prediction_std / true_std if true_std else None,
        "error_vs_true_target_pearson": safe_corr(error, true_values, "pearson"),
        "target_bin_mae": bin_mae,
        "target_bin_mean_error": bin_mean_error,
        "rows": int(len(predictions)),
    }


def build_prediction_rows(dataset: CDRAwareAffinityDataset, true_values: list, predicted_values: list) -> pd.DataFrame:
    """Pair filtered test metadata with predictions in DataLoader order."""

    rows = []
    for row, true_value, predicted_value in zip(
        dataset.data.to_dict("records"),
        true_values,
        predicted_values,
    ):
        error = float(predicted_value - true_value)
        output_row = {
            "sample_id": row.get("sample_id", ""),
            TRUE_COLUMN: float(true_value),
            PRED_COLUMN: float(predicted_value),
            "error": error,
            "absolute_error": abs(error),
            "fold_error": 10 ** abs(error),
            "source": row.get("source", ""),
            "pdb_or_antibody_id": row.get("pdb_or_antibody_id", ""),
            "antigen_id": row.get("antigen_id", ""),
            "antigen_sequence": row.get("antigen_sequence", ""),
        }
        for column_name in ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]:
            output_row[column_name] = row.get(column_name, "")
        rows.append(output_row)
    return pd.DataFrame(rows)


def load_whole_sequence_predictions(config: dict, cdr_predictions: pd.DataFrame) -> pd.DataFrame | None:
    """Load best whole-sequence predictions on matching sample_ids if available."""

    whole_path = Path(config.get("whole_sequence_predictions_path", ""))
    if not whole_path.exists():
        return None

    whole_predictions = pd.read_csv(whole_path)
    needed = {"sample_id", TRUE_COLUMN, PRED_COLUMN}
    if needed - set(whole_predictions.columns):
        raise ValueError(f"{whole_path} is missing whole-sequence prediction columns: {needed}")

    sample_ids = set(cdr_predictions["sample_id"].astype(str))
    matched = whole_predictions[
        whole_predictions["sample_id"].astype(str).isin(sample_ids)
    ].copy()
    if len(matched) != len(cdr_predictions):
        raise ValueError(
            "Whole-sequence comparison predictions do not match the filtered CDR test rows. "
            f"Expected {len(cdr_predictions)}, found {len(matched)}."
        )
    return matched


def fmt(value: float | None) -> str:
    """Format metric values for markdown."""

    return "NA" if value is None else f"{value:.4f}"


def write_report(
    config: dict,
    cdr_metrics: dict,
    cdr_predictions: pd.DataFrame,
    test_dataset: CDRAwareAffinityDataset,
    whole_metrics: dict | None,
) -> Path:
    """Write experiment comparison report after CDR-aware evaluation."""

    report_path = Path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Unified No High Risk CDR-Aware Baseline Report",
        "",
        "## Experiment",
        "",
        "- CDR backend: standard AbNumber + IMGT annotated CSVs.",
        f"- CDR-aware input fields: `{test_dataset.input_cdr_fields} + antigen_sequence`.",
        "- Model family: shared ESM-2 8M + LoRA + mean pooling + scalar regression head.",
        f"- CDR-aware test rows kept after extraction-status filtering: `{len(test_dataset)}` / `{test_dataset.raw_row_count}`.",
        "- Loader accepts CDR statuses `success` and `ok` as successful annotations.",
        "",
        "## Test Metrics",
        "",
        "| model | rows | MAE | RMSE | Spearman | prediction std | true std | pred std / true std | error vs true Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `cdr_aware` | {cdr_metrics['rows']} | {cdr_metrics['mae']:.4f} | "
            f"{cdr_metrics['rmse']:.4f} | {cdr_metrics['spearman']:.4f} | "
            f"{cdr_metrics['prediction_std']:.4f} | {cdr_metrics['true_std']:.4f} | "
            f"{fmt(cdr_metrics['prediction_std_over_true_std'])} | "
            f"{fmt(cdr_metrics['error_vs_true_target_pearson'])} |"
        ),
    ]
    if whole_metrics:
        lines.append(
            f"| `whole_sequence_best` | {whole_metrics['rows']} | {whole_metrics['mae']:.4f} | "
            f"{whole_metrics['rmse']:.4f} | {whole_metrics['spearman']:.4f} | "
            f"{whole_metrics['prediction_std']:.4f} | {whole_metrics['true_std']:.4f} | "
            f"{fmt(whole_metrics['prediction_std_over_true_std'])} | "
            f"{fmt(whole_metrics['error_vs_true_target_pearson'])} |"
        )

    lines.extend(
        [
            "",
            "## Target-Bin MAE",
            "",
            "Low/mid/high bins are test-target quantile bins, so the bin comparison is on similarly sized groups.",
            "",
            "| model | low target MAE | mid target MAE | high target MAE |",
            "|---|---:|---:|---:|",
            (
                f"| `cdr_aware` | {cdr_metrics['target_bin_mae'].get('low_target', math.nan):.4f} | "
                f"{cdr_metrics['target_bin_mae'].get('mid_target', math.nan):.4f} | "
                f"{cdr_metrics['target_bin_mae'].get('high_target', math.nan):.4f} |"
            ),
        ]
    )
    if whole_metrics:
        lines.append(
            f"| `whole_sequence_best` | {whole_metrics['target_bin_mae'].get('low_target', math.nan):.4f} | "
            f"{whole_metrics['target_bin_mae'].get('mid_target', math.nan):.4f} | "
            f"{whole_metrics['target_bin_mae'].get('high_target', math.nan):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Regression-To-Mean Reading Guide",
            "",
            "- Better high-target MAE means the model is less weak on strong-binding/high target examples.",
            "- `prediction std / true std` closer to 1 means predictions spread more like the real targets.",
            "- `error vs true target Pearson` closer to 0 in absolute value means less systematic overestimate-low / underestimate-high behavior.",
            "- Spearman checks whether ranking ability is preserved or improved.",
            "",
            "## Files",
            "",
            f"- Predictions: `{Path(config['predictions_path']).as_posix()}`",
            f"- Checkpoint: `{Path(config['checkpoint_path']).as_posix()}`",
            "",
            "## Top CDR-Aware Test Errors",
            "",
        ]
    )
    top_columns = [
        "sample_id",
        TRUE_COLUMN,
        PRED_COLUMN,
        "error",
        "absolute_error",
        "fold_error",
        "source",
        "antigen_id",
    ]
    top_errors = cdr_predictions.sort_values("absolute_error", ascending=False).head(10)[top_columns]
    lines.append("| " + " | ".join(top_columns) + " |")
    lines.append("|" + "|".join(["---"] * len(top_columns)) + "|")
    for row in top_errors.to_dict("records"):
        cells = []
        for column_name in top_columns:
            value = row[column_name]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_metrics(name: str, metrics: dict) -> None:
    """Print compact comparison metrics to terminal."""

    print(
        f"{name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"Spearman={metrics['spearman']:.4f}, prediction_std={metrics['prediction_std']:.4f}, "
        f"pred_std/true_std={fmt(metrics['prediction_std_over_true_std'])}, "
        f"error_vs_true_Pearson={fmt(metrics['error_vs_true_target_pearson'])}"
    )
    print(f"  target-bin MAE: {metrics['target_bin_mae']}")


def main() -> None:
    """Evaluate checkpoint, save predictions, and write comparison markdown."""

    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    test_dataset = CDRAwareAffinityDataset(
        csv_path=config["test_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        input_cdr_fields=config.get("input_cdr_fields"),
    )
    print(
        f"Test CDR rows kept: {len(test_dataset)} / {test_dataset.raw_row_count} "
        f"(filtered extraction failures: {test_dataset.filtered_out_count})"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    model = SeqProFTCDRAwareAffinityRegressor(config).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, true_values, predicted_values = evaluate_cdr_affinity_model(model, test_dataloader, device)

    cdr_predictions = build_prediction_rows(test_dataset, true_values, predicted_values)
    predictions_path = Path(config["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    cdr_predictions.to_csv(predictions_path, index=False)

    cdr_metrics = metric_bundle(cdr_predictions)
    whole_predictions = load_whole_sequence_predictions(config, cdr_predictions)
    whole_metrics = metric_bundle(whole_predictions) if whole_predictions is not None else None

    print()
    print_metrics("CDR-aware", cdr_metrics)
    if whole_metrics:
        print_metrics("Whole-sequence best", whole_metrics)
    else:
        print("Whole-sequence comparison predictions not found; report will show CDR metrics only.")

    report_path = write_report(config, cdr_metrics, cdr_predictions, test_dataset, whole_metrics)
    print()
    print(f"Saved CDR-aware predictions to {predictions_path}")
    print(f"Saved CDR-aware report to {report_path}")


if __name__ == "__main__":
    main()
