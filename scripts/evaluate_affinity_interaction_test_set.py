"""Evaluate the residue-level CDR-antigen interaction affinity baseline.

中文人话说明：
训练完成后，这个脚本会在 interaction-aware test set 上：
1. 加载 checkpoint。
2. 预测每个 test sample 的 `neg_log10_affinity`。
3. 保存 per-sample predictions。
4. 写 MAE/RMSE/Spearman 和 regression-to-mean 诊断报告。

这一步仍然是 sequence-only baseline：
它显式用 residue-level interaction matrix，但没有引入 3D structure/contact labels。
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

from src.affinity_interaction_dataset import InteractionAffinityDataset
from src.affinity_interaction_evaluate import evaluate_interaction_affinity_model
from src.affinity_interaction_model import SeqProFTInteractionAffinityRegressor
from src.affinity_evaluate import compute_regression_metrics
from src.utils import get_device, load_config


TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"
TARGET_BIN_LABELS = ["low_target", "mid_target", "high_target"]


def parse_args() -> argparse.Namespace:
    """Read interaction-aware config path."""

    parser = argparse.ArgumentParser(description="Evaluate interaction-aware affinity test set.")
    parser.add_argument(
        "--config",
        default="config_affinity_unified_no_high_risk_interaction_hcdr3_lcdr3_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def quantile_target_bins(true_values: pd.Series) -> pd.Series:
    """Create low/mid/high target bins with near-equal sample counts."""

    ranked = pd.to_numeric(true_values, errors="coerce").rank(method="first")
    return pd.qcut(ranked, q=3, labels=TARGET_BIN_LABELS)


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    """Return correlation or None when the value is undefined."""

    value = pd.to_numeric(left, errors="coerce").corr(
        pd.to_numeric(right, errors="coerce"),
        method=method,
    )
    return None if pd.isna(value) else float(value)


def metric_bundle(predictions: pd.DataFrame) -> dict:
    """Compute evaluation metrics and regression-to-mean diagnostics."""

    true_values = pd.to_numeric(predictions[TRUE_COLUMN], errors="raise")
    predicted_values = pd.to_numeric(predictions[PRED_COLUMN], errors="raise")
    error = predicted_values - true_values
    absolute_error = error.abs()
    metrics = compute_regression_metrics(true_values.tolist(), predicted_values.tolist())

    enriched = predictions.copy()
    enriched["target_bin"] = quantile_target_bins(true_values)
    enriched["absolute_error_for_metrics"] = absolute_error
    bin_mae = {
        str(bin_name): float(group["absolute_error_for_metrics"].mean())
        for bin_name, group in enriched.groupby("target_bin", observed=True)
    }
    bin_mean_error = {
        str(bin_name): float(
            (
                pd.to_numeric(group[PRED_COLUMN], errors="raise")
                - pd.to_numeric(group[TRUE_COLUMN], errors="raise")
            ).mean()
        )
        for bin_name, group in enriched.groupby("target_bin", observed=True)
    }

    true_std = float(true_values.std())
    prediction_std = float(predicted_values.std())
    return {
        **metrics,
        "rows": int(len(predictions)),
        "true_mean": float(true_values.mean()),
        "true_std": true_std,
        "prediction_mean": float(predicted_values.mean()),
        "prediction_std": prediction_std,
        "prediction_std_over_true_std": prediction_std / true_std if true_std else None,
        "error_vs_true_target_pearson": safe_corr(error, true_values, "pearson"),
        "target_bin_mae": bin_mae,
        "target_bin_mean_error": bin_mean_error,
    }


def build_prediction_rows(
    dataset: InteractionAffinityDataset,
    true_values: list[float],
    predicted_values: list[float],
) -> pd.DataFrame:
    """Pair filtered CSV metadata with predictions in DataLoader order."""

    rows = []
    for row, true_value, predicted_value in zip(
        dataset.data.to_dict("records"),
        true_values,
        predicted_values,
    ):
        error = float(predicted_value - true_value)
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                TRUE_COLUMN: float(true_value),
                PRED_COLUMN: float(predicted_value),
                "error": error,
                "absolute_error": abs(error),
                "fold_error": 10 ** abs(error),
                "source": row.get("source", ""),
                "pdb_or_antibody_id": row.get("pdb_or_antibody_id", ""),
                "antigen_id": row.get("antigen_id", ""),
                "HCDR3": row.get("HCDR3", ""),
                "LCDR3": row.get("LCDR3", ""),
                "antigen_sequence": row.get("antigen_sequence", ""),
            }
        )
    return pd.DataFrame(rows)


def load_matching_prediction_metrics(path_value: str, interaction_predictions: pd.DataFrame) -> dict | None:
    """Evaluate an optional prior prediction CSV on matching sample IDs."""

    path = Path(path_value)
    if not path.exists():
        return None
    previous = pd.read_csv(path)
    needed = {"sample_id", TRUE_COLUMN, PRED_COLUMN}
    if needed - set(previous.columns):
        raise ValueError(f"{path} is missing comparison prediction columns: {sorted(needed)}")
    sample_ids = set(interaction_predictions["sample_id"].astype(str))
    matched = previous[previous["sample_id"].astype(str).isin(sample_ids)].copy()
    if len(matched) != len(interaction_predictions):
        raise ValueError(
            f"{path} does not match interaction test rows: "
            f"expected {len(interaction_predictions)}, found {len(matched)}."
        )
    return metric_bundle(matched)


def fmt(value: float | None) -> str:
    """Format metric values for markdown and terminal."""

    return "NA" if value is None else f"{value:.4f}"


def metric_row(name: str, metrics: dict) -> str:
    """Build one markdown comparison-table row."""

    return (
        f"| `{name}` | {metrics['rows']} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
        f"{metrics['spearman']:.4f} | {metrics['prediction_std']:.4f} | "
        f"{metrics['true_std']:.4f} | {fmt(metrics['prediction_std_over_true_std'])} | "
        f"{fmt(metrics['error_vs_true_target_pearson'])} |"
    )


def target_bin_row(name: str, metrics: dict) -> str:
    """Build one low/mid/high MAE table row."""

    return (
        f"| `{name}` | {metrics['target_bin_mae'].get('low_target', math.nan):.4f} | "
        f"{metrics['target_bin_mae'].get('mid_target', math.nan):.4f} | "
        f"{metrics['target_bin_mae'].get('high_target', math.nan):.4f} |"
    )


def write_report(
    config: dict,
    interaction_metrics: dict,
    predictions: pd.DataFrame,
    dataset: InteractionAffinityDataset,
    comparison_metrics: dict[str, dict],
) -> Path:
    """Write report for the interaction baseline and optional prior baselines."""

    report_path = Path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Unified No High Risk Interaction-Aware Baseline Report",
        "",
        "## Experiment",
        "",
        "- Input mode: `HCDR3 + LCDR3 + antigen_sequence`.",
        "- Encoder: shared ESM-2 8M + LoRA.",
        "- Residue interaction: dot-product CDR token matrix x antigen token matrix.",
        (
            "- Interaction summaries: matrix mean, matrix max, top-k mean, "
            "row-wise max mean, column-wise max mean."
        ),
        f"- Interaction top-k: `{int(config.get('interaction_top_k', 5))}`.",
        f"- Test rows kept after CDR status filtering: `{len(dataset)}` / `{dataset.raw_row_count}`.",
        "",
        "## Test Metrics",
        "",
        "| model | rows | MAE | RMSE | Spearman | prediction std | true std | pred std / true std | error vs true Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("interaction_hcdr3_lcdr3_antigen", interaction_metrics),
    ]
    for name, metrics in comparison_metrics.items():
        lines.append(metric_row(name, metrics))

    lines.extend(
        [
            "",
            "## Target-Bin MAE",
            "",
            "| model | low target MAE | mid target MAE | high target MAE |",
            "|---|---:|---:|---:|",
            target_bin_row("interaction_hcdr3_lcdr3_antigen", interaction_metrics),
        ]
    )
    for name, metrics in comparison_metrics.items():
        lines.append(target_bin_row(name, metrics))

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- Lower MAE/RMSE means smaller absolute affinity-target error.",
            "- Spearman checks ranking quality.",
            "- `prediction std / true std` closer to 1 means prediction range is less compressed.",
            "- `error vs true Pearson` closer to 0 in absolute value means less regression-to-mean bias.",
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
        cells = []
        for column_name in top_columns:
            value = row[column_name]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_metrics(name: str, metrics: dict) -> None:
    """Print compact result line for one model."""

    print(
        f"{name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"Spearman={metrics['spearman']:.4f}, prediction_std={metrics['prediction_std']:.4f}, "
        f"pred_std/true_std={fmt(metrics['prediction_std_over_true_std'])}, "
        f"error_vs_true_Pearson={fmt(metrics['error_vs_true_target_pearson'])}"
    )
    print(f"  target-bin MAE: {metrics['target_bin_mae']}")


def main() -> None:
    """Evaluate interaction checkpoint and save report/predictions."""

    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    test_dataset = InteractionAffinityDataset(
        csv_path=config["test_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    print(
        f"Test interaction rows kept: {len(test_dataset)} / {test_dataset.raw_row_count} "
        f"(filtered CDR extraction failures: {test_dataset.filtered_out_count})"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    model = SeqProFTInteractionAffinityRegressor(config).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, true_values, predicted_values = evaluate_interaction_affinity_model(
        model,
        test_dataloader,
        device,
    )

    predictions = build_prediction_rows(test_dataset, true_values, predicted_values)
    predictions_path = Path(config["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    interaction_metrics = metric_bundle(predictions)

    comparison_metrics = {}
    comparison_paths = {
        "pooled_hcdr3_lcdr3": config.get("pooled_hcdr3_lcdr3_predictions_path", ""),
        "pooled_all_cdrs": config.get("pooled_all_cdrs_predictions_path", ""),
    }
    for name, path_value in comparison_paths.items():
        if path_value:
            metrics = load_matching_prediction_metrics(path_value, predictions)
            if metrics is not None:
                comparison_metrics[name] = metrics

    print()
    print_metrics("Interaction HCDR3+LCDR3", interaction_metrics)
    for name, metrics in comparison_metrics.items():
        print_metrics(name, metrics)

    report_path = write_report(
        config,
        interaction_metrics,
        predictions,
        test_dataset,
        comparison_metrics,
    )
    print()
    print(f"Saved interaction predictions to {predictions_path}")
    print(f"Saved interaction report to {report_path}")


if __name__ == "__main__":
    main()
