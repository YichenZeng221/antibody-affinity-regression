"""Evaluate the all-CDR CDR-to-antigen cross-attention affinity baseline."""

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

from src.affinity_cross_attention_dataset import (
    CROSS_ATTENTION_CDR_FIELDS,
    CrossAttentionAffinityDataset,
)
from src.affinity_cross_attention_evaluate import (
    cross_attention_device,
    evaluate_cross_attention_affinity_model,
)
from src.affinity_cross_attention_model import SeqProFTCrossAttentionAffinityRegressor
from src.affinity_cross_attention_train import antigen_length_from_config
from src.affinity_evaluate import compute_regression_metrics
from src.utils import load_config


TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"
TARGET_BIN_LABELS = ["low_target", "mid_target", "high_target"]


def parse_args() -> argparse.Namespace:
    """Read cross-attention config path for test evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate all-CDR cross-attention affinity model.")
    parser.add_argument(
        "--config",
        default="config_affinity_unified_no_high_risk_cross_attention_all_cdrs_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def quantile_target_bins(true_values: pd.Series) -> pd.Series:
    """Create low/mid/high bins using test-target quantiles."""

    ranked = pd.to_numeric(true_values, errors="coerce").rank(method="first")
    return pd.qcut(ranked, q=3, labels=TARGET_BIN_LABELS)


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    """Return correlation or None when undefined."""

    value = pd.to_numeric(left, errors="coerce").corr(
        pd.to_numeric(right, errors="coerce"),
        method=method,
    )
    return None if pd.isna(value) else float(value)


def metric_bundle(predictions: pd.DataFrame) -> dict:
    """Compute core metrics and regression-to-mean diagnostics."""

    true_values = pd.to_numeric(predictions[TRUE_COLUMN], errors="raise")
    predicted_values = pd.to_numeric(predictions[PRED_COLUMN], errors="raise")
    error = predicted_values - true_values
    enriched = predictions.copy()
    enriched["target_bin"] = quantile_target_bins(true_values)
    enriched["metric_absolute_error"] = error.abs()
    true_std = float(true_values.std())
    predicted_std = float(predicted_values.std())
    return {
        **compute_regression_metrics(true_values.tolist(), predicted_values.tolist()),
        "rows": int(len(predictions)),
        "prediction_mean": float(predicted_values.mean()),
        "prediction_std": predicted_std,
        "true_mean": float(true_values.mean()),
        "true_std": true_std,
        "prediction_std_over_true_std": predicted_std / true_std if true_std else None,
        "error_vs_true_target_pearson": safe_corr(error, true_values, "pearson"),
        "target_bin_mae": {
            str(name): float(group["metric_absolute_error"].mean())
            for name, group in enriched.groupby("target_bin", observed=True)
        },
        "target_bin_mean_error": {
            str(name): float(
                (
                    pd.to_numeric(group[PRED_COLUMN], errors="raise")
                    - pd.to_numeric(group[TRUE_COLUMN], errors="raise")
                ).mean()
            )
            for name, group in enriched.groupby("target_bin", observed=True)
        },
    }


def build_prediction_rows(dataset: CrossAttentionAffinityDataset, true_values: list, predicted_values: list):
    """Save filtered test metadata beside model predictions."""

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
        for cdr_field in CROSS_ATTENTION_CDR_FIELDS:
            output_row[cdr_field] = row.get(cdr_field, "")
        rows.append(output_row)
    return pd.DataFrame(rows)


def load_prediction_metrics(path_value: str, current_predictions: pd.DataFrame) -> dict | None:
    """Load a previous prediction file on matching sample IDs if it exists."""

    path = Path(path_value)
    if not path.exists():
        return None
    predictions = pd.read_csv(path)
    required = {"sample_id", TRUE_COLUMN, PRED_COLUMN}
    if required - set(predictions.columns):
        raise ValueError(f"{path} is missing comparison prediction columns: {sorted(required)}")
    sample_ids = set(current_predictions["sample_id"].astype(str))
    matched = predictions[predictions["sample_id"].astype(str).isin(sample_ids)].copy()
    if len(matched) != len(current_predictions):
        raise ValueError(
            f"{path} has {len(matched)} matched rows, expected {len(current_predictions)}."
        )
    return metric_bundle(matched)


def fmt(value: float | None) -> str:
    """Format optional metrics for markdown."""

    return "NA" if value is None else f"{value:.4f}"


def metric_row(name: str, metrics: dict) -> str:
    """Render one metrics table row."""

    return (
        f"| `{name}` | {metrics['rows']} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
        f"{metrics['spearman']:.4f} | {metrics['prediction_std']:.4f} | "
        f"{fmt(metrics['prediction_std_over_true_std'])} | "
        f"{fmt(metrics['error_vs_true_target_pearson'])} |"
    )


def bin_row(name: str, metrics: dict) -> str:
    """Render low/mid/high target-bin MAE row."""

    return (
        f"| `{name}` | {metrics['target_bin_mae'].get('low_target', math.nan):.4f} | "
        f"{metrics['target_bin_mae'].get('mid_target', math.nan):.4f} | "
        f"{metrics['target_bin_mae'].get('high_target', math.nan):.4f} |"
    )


def baseline_reference_rows(config: dict) -> list[str]:
    """Keep requested fixed reference metrics visible even if CSVs move."""

    references = config.get("baseline_reference_metrics", {})
    rows = []
    for name, values in references.items():
        rows.append(
            f"| `{name}` | {float(values['mae']):.4f} | {float(values['rmse']):.4f} | "
            f"{float(values['spearman']):.4f} |"
        )
    return rows


def write_report(config: dict, metrics: dict, predictions: pd.DataFrame, comparison_metrics: dict):
    """Write evaluation report and comparison against saved baselines."""

    report_path = Path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Unified No High Risk All-CDR Cross-Attention Report",
        "",
        "## Experiment",
        "",
        "- Input: all six standard IMGT CDRs as CDR queries plus antigen tokens as key/value.",
        "- Encoder: shared ESM-2 8M + LoRA.",
        "- Interaction layer: learnable multi-head CDR-to-antigen cross-attention.",
        "- Pooling: attention-pooled attended CDR tokens, mean-pooled original CDR tokens, mean-pooled antigen tokens.",
        "- Head: `Linear -> GELU -> Dropout -> Linear -> scalar affinity prediction`.",
        "",
        "## Test Metrics",
        "",
        "| model | rows | MAE | RMSE | Spearman | prediction std | pred std / true std | error vs true Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("all_cdrs_cross_attention", metrics),
    ]
    for name, compared in comparison_metrics.items():
        lines.append(metric_row(name, compared))
    lines.extend(
        [
            "",
            "## Target-Bin MAE",
            "",
            "| model | low target MAE | mid target MAE | high target MAE |",
            "|---|---:|---:|---:|",
            bin_row("all_cdrs_cross_attention", metrics),
        ]
    )
    for name, compared in comparison_metrics.items():
        lines.append(bin_row(name, compared))
    lines.extend(
        [
            "",
            "## Requested Baseline Reference Metrics",
            "",
            "| baseline | MAE | RMSE | Spearman |",
            "|---|---:|---:|---:|",
            *baseline_reference_rows(config),
            "",
            "## Questions To Answer",
            "",
            "1. Does all-CDR cross-attention beat the all-CDR pooled baseline?",
            "2. Does prediction spread move closer to true spread and weaken regression-to-mean?",
            "3. Does high-target MAE fall?",
            "4. Does Spearman stay stable or improve?",
            "",
            "## Files",
            "",
            f"- Predictions: `{Path(config['predictions_path']).as_posix()}`",
            f"- Checkpoint: `{Path(config['checkpoint_path']).as_posix()}`",
            "",
            "## Top Test Errors",
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
    top_errors = predictions.sort_values("absolute_error", ascending=False).head(10)[top_columns]
    lines.append("| " + " | ".join(top_columns) + " |")
    lines.append("|" + "|".join(["---"] * len(top_columns)) + "|")
    for row in top_errors.to_dict("records"):
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row.values()]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    """Evaluate trained checkpoint and write independent report."""

    config = load_config(parse_args().config)
    device = cross_attention_device(config)
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = CrossAttentionAffinityDataset(
        csv_path=config["test_csv"],
        tokenizer=tokenizer,
        antigen_max_length=antigen_length_from_config(config),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    print(
        f"Test cross-attention rows kept: {len(dataset)} / {dataset.raw_row_count} "
        f"(filtered CDR extraction failures: {dataset.filtered_out_count})"
    )
    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    model = SeqProFTCrossAttentionAffinityRegressor(config).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, true_values, predicted_values = evaluate_cross_attention_affinity_model(
        model,
        dataloader,
        device,
    )

    predictions = build_prediction_rows(dataset, true_values, predicted_values)
    predictions_path = Path(config["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    metrics = metric_bundle(predictions)
    comparison_metrics = {}
    for name, path_value in config.get("baseline_prediction_paths", {}).items():
        loaded = load_prediction_metrics(path_value, predictions)
        if loaded is not None:
            comparison_metrics[name] = loaded
    report_path = write_report(config, metrics, predictions, comparison_metrics)
    print(
        f"Cross-attention: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"Spearman={metrics['spearman']:.4f}, pred_std/true_std="
        f"{fmt(metrics['prediction_std_over_true_std'])}, error_vs_true_Pearson="
        f"{fmt(metrics['error_vs_true_target_pearson'])}"
    )
    print(f"Target-bin MAE: {metrics['target_bin_mae']}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
