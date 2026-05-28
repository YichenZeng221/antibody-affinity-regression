"""Evaluate ANDD stratified all-CDR cross-attention after manual training.

:
 checkpoint  test ,
 cross-attention  stratified split  all-CDR pooled baseline
,, test set

 regression-to-the-mean:
- low/mid/high  train split  tertiles ;
- tail  train split  P10/P90 ;
-  test 
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.affinity_cross_attention_dataset import (  # noqa: E402
    CROSS_ATTENTION_CDR_FIELDS,
    CrossAttentionAffinityDataset,
    SUCCESS_STATUS_VALUES,
)
from src.affinity_cross_attention_evaluate import (  # noqa: E402
    cross_attention_device,
    evaluate_cross_attention_affinity_model,
)
from src.affinity_cross_attention_model import SeqProFTCrossAttentionAffinityRegressor  # noqa: E402
from src.affinity_cross_attention_train import antigen_length_from_config  # noqa: E402
from src.affinity_evaluate import compute_regression_metrics  # noqa: E402
from src.utils import load_config  # noqa: E402


TRUE_COL = "true_neg_log10_affinity"
PRED_COL = "predicted_neg_log10_affinity"


def parse_args() -> argparse.Namespace:
    """ ANDD stratified cross-attention config"""

    parser = argparse.ArgumentParser(
        description="Evaluate ANDD stratified all-CDR cross-attention checkpoint."
    )
    parser.add_argument(
        "--config",
        default="config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def successful_train_targets(config: dict) -> pd.Series:
    """ Dataset  CDR-success filter  train target """

    frame = pd.read_csv(config["train_csv"])
    heavy_ok = frame["heavy_cdr_status"].fillna("").astype(str).str.lower().isin(SUCCESS_STATUS_VALUES)
    light_ok = frame["light_cdr_status"].fillna("").astype(str).str.lower().isin(SUCCESS_STATUS_VALUES)
    return pd.to_numeric(
        frame.loc[heavy_ok & light_ok, config["target_column"]],
        errors="raise",
    )


def safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    """ Pearson undefined """

    value = left.corr(right, method="pearson")
    return None if pd.isna(value) else float(value)


def metric_bundle(predictions: pd.DataFrame, train_targets: pd.Series) -> dict:
    """train-defined bins  train-defined tail errors"""

    true_values = pd.to_numeric(predictions[TRUE_COL], errors="raise")
    predicted_values = pd.to_numeric(predictions[PRED_COL], errors="raise")
    errors = predicted_values - true_values
    absolute_errors = errors.abs()
    low_edge = float(train_targets.quantile(1 / 3))
    high_edge = float(train_targets.quantile(2 / 3))
    p10 = float(train_targets.quantile(0.10))
    p90 = float(train_targets.quantile(0.90))

    enriched = predictions.copy()
    enriched["absolute_error_for_metrics"] = absolute_errors
    enriched["target_bin"] = pd.cut(
        true_values,
        bins=[-math.inf, low_edge, high_edge, math.inf],
        labels=["low_target", "mid_target", "high_target"],
        include_lowest=True,
    )
    tail_masks = {
        "below_train_p10": true_values <= p10,
        "above_train_p90": true_values >= p90,
    }
    true_std = float(true_values.std())
    predicted_std = float(predicted_values.std())
    return {
        **compute_regression_metrics(true_values.tolist(), predicted_values.tolist()),
        "rows": int(len(predictions)),
        "prediction_mean": float(predicted_values.mean()),
        "prediction_std": predicted_std,
        "true_std": true_std,
        "pred_std_over_true_std": predicted_std / true_std if true_std else None,
        "error_vs_true_pearson": safe_corr(errors, true_values),
        "train_thresholds": {
            "low_upper": low_edge,
            "mid_upper": high_edge,
            "p10": p10,
            "p90": p90,
        },
        "target_bin_mae": {
            str(label): float(group["absolute_error_for_metrics"].mean())
            for label, group in enriched.groupby("target_bin", observed=True)
        },
        "target_bin_rows": {
            str(label): int(len(group))
            for label, group in enriched.groupby("target_bin", observed=True)
        },
        "tail_mae": {
            label: float(absolute_errors[mask].mean()) if int(mask.sum()) else None
            for label, mask in tail_masks.items()
        },
        "tail_rows": {label: int(mask.sum()) for label, mask in tail_masks.items()},
    }


def build_prediction_rows(
    dataset: CrossAttentionAffinityDataset,
    true_values: list[float],
    predicted_values: list[float],
) -> pd.DataFrame:
    """ prediction , pooled prediction """

    rows: list[dict] = []
    for row, true_value, predicted_value in zip(
        dataset.data.to_dict("records"),
        true_values,
        predicted_values,
    ):
        error = float(predicted_value - true_value)
        output = {
            "sample_id": row.get("sample_id", row.get("candidate_id", "")),
            "candidate_id": row.get("candidate_id", ""),
            "source": row.get("source", ""),
            "pdb_id": row.get("pdb_id", ""),
            "ag_name": row.get("ag_name", ""),
            "antigen_sequence": row.get("antigen_sequence", ""),
            TRUE_COL: float(true_value),
            PRED_COL: float(predicted_value),
            "error": error,
            "absolute_error": abs(error),
            "fold_error": 10 ** abs(error),
        }
        for cdr_field in CROSS_ATTENTION_CDR_FIELDS:
            output[cdr_field] = row.get(cdr_field, "")
        rows.append(output)
    return pd.DataFrame(rows)


def load_pooled_predictions(config: dict, cross_predictions: pd.DataFrame) -> pd.DataFrame | None:
    """ stratified test set  pooled predictions, sample IDs"""

    path = Path(config["pooled_baseline_predictions_path"])
    if not path.exists():
        return None
    pooled = pd.read_csv(path)
    cross_ids = set(cross_predictions["sample_id"].astype(str))
    pooled_ids = set(pooled["sample_id"].astype(str))
    if cross_ids != pooled_ids:
        raise ValueError("Pooled baseline and cross-attention predictions do not use identical test sample IDs.")
    return pooled


def fmt(value: float | None) -> str:
    """"""

    return "NA" if value is None or pd.isna(value) else f"{value:.4f}"


def metrics_row(name: str, metrics: dict) -> str:
    """"""

    return (
        f"| `{name}` | {metrics['rows']} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
        f"{metrics['spearman']:.4f} | {fmt(metrics['pred_std_over_true_std'])} | "
        f"{fmt(metrics['error_vs_true_pearson'])} |"
    )


def bin_row(name: str, metrics: dict) -> str:
    """ low/mid/high MAE """

    return (
        f"| `{name}` | {metrics['target_bin_rows'].get('low_target', 0)} / "
        f"{fmt(metrics['target_bin_mae'].get('low_target'))} | "
        f"{metrics['target_bin_rows'].get('mid_target', 0)} / "
        f"{fmt(metrics['target_bin_mae'].get('mid_target'))} | "
        f"{metrics['target_bin_rows'].get('high_target', 0)} / "
        f"{fmt(metrics['target_bin_mae'].get('high_target'))} |"
    )


def tail_row(name: str, metrics: dict) -> str:
    """ train-defined tails MAE """

    return (
        f"| `{name}` | {metrics['tail_rows']['below_train_p10']} / "
        f"{fmt(metrics['tail_mae']['below_train_p10'])} | "
        f"{metrics['tail_rows']['above_train_p90']} / "
        f"{fmt(metrics['tail_mae']['above_train_p90'])} |"
    )


def write_report(config: dict, cross_metrics: dict, pooled_metrics: dict | None) -> Path:
    """; stratified test set """

    report_path = Path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = cross_metrics["train_thresholds"]
    lines = [
        "# ANDD Antibody v2 Stratified Split: All-CDR Cross-Attention Report",
        "",
        "## Experiment",
        "",
        "- Dataset: ANDD antibody v2 stratified antigen-level split.",
        "- Input: six standard AbNumber/IMGT CDRs as query tokens; antigen sequence as key/value tokens.",
        "- Model: shared ESM2 + LoRA with learnable multi-head cross-attention.",
        "- Loss and training hyperparameters: unchanged MSE baseline settings.",
        "- Comparison: all-CDR pooled baseline trained on the same stratified split and test rows.",
        "",
        "## Train-Defined Thresholds",
        "",
        f"- Low target: `target <= {thresholds['low_upper']:.4f}`",
        f"- Mid target: `{thresholds['low_upper']:.4f} < target <= {thresholds['mid_upper']:.4f}`",
        f"- High target: `target > {thresholds['mid_upper']:.4f}`",
        f"- Lower tail: `target <= train P10 = {thresholds['p10']:.4f}`",
        f"- Upper tail: `target >= train P90 = {thresholds['p90']:.4f}`",
        "",
        "## Test Metrics",
        "",
        "| model | rows | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metrics_row("cross_attention_all_cdrs", cross_metrics),
    ]
    if pooled_metrics is not None:
        lines.append(metrics_row("all_cdrs_pooled", pooled_metrics))

    lines.extend(
        [
            "",
            "## Train-Defined Target-Bin MAE",
            "",
            "| model | low rows / MAE | mid rows / MAE | high rows / MAE |",
            "|---|---:|---:|---:|",
            bin_row("cross_attention_all_cdrs", cross_metrics),
        ]
    )
    if pooled_metrics is not None:
        lines.append(bin_row("all_cdrs_pooled", pooled_metrics))

    lines.extend(
        [
            "",
            "## Train-Defined Tail MAE",
            "",
            "| model | below train P10 rows / MAE | above train P90 rows / MAE |",
            "|---|---:|---:|",
            tail_row("cross_attention_all_cdrs", cross_metrics),
        ]
    )
    if pooled_metrics is not None:
        lines.append(tail_row("all_cdrs_pooled", pooled_metrics))

    reference = config.get("pooled_baseline_reference", {})
    lines.extend(
        [
            "",
            "## Pooled Baseline Reference",
            "",
            "These are the expected headline values supplied for the stratified pooled baseline:",
            "",
            f"- MAE/RMSE/Spearman: `{reference.get('mae')}` / `{reference.get('rmse')}` / `{reference.get('spearman')}`",
            f"- pred_std/true_std: `{reference.get('pred_std_over_true_std')}`",
            f"- error_vs_true_Pearson: `{reference.get('error_vs_true_pearson')}`",
            f"- low/mid/high target MAE: `{reference.get('low_target_mae')}` / `{reference.get('mid_target_mae')}` / `{reference.get('high_target_mae')}`",
            f"- below-P10 / above-P90 tail MAE: `{reference.get('below_train_p10_mae')}` / `{reference.get('above_train_p90_mae')}`",
            "",
            "## Questions To Answer",
            "",
            "1. Does cross-attention lower MAE/RMSE relative to pooled all-CDR on the same stratified test set?",
            "2. Does `pred std / true std` move closer to 1 and does `error vs true Pearson` move closer to 0?",
            "3. Does cross-attention reduce low/high target-bin MAE or P10/P90 tail MAE?",
            "4. Does ranking ability, measured by Spearman, improve?",
            "",
            "## Files",
            "",
            f"- Predictions: `{Path(config['predictions_path']).as_posix()}`",
            f"- Checkpoint: `{Path(config['checkpoint_path']).as_posix()}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    """ test inference;"""

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
        f"Test rows kept: {len(dataset)} / {dataset.raw_row_count} "
        f"(filtered CDR failures: {dataset.filtered_out_count})"
    )
    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    model = SeqProFTCrossAttentionAffinityRegressor(config).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, true_values, predicted_values = evaluate_cross_attention_affinity_model(model, dataloader, device)
    predictions = build_prediction_rows(dataset, true_values, predicted_values)

    output_path = Path(config["predictions_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    train_targets = successful_train_targets(config)
    cross_metrics = metric_bundle(predictions, train_targets)
    pooled_predictions = load_pooled_predictions(config, predictions)
    pooled_metrics = metric_bundle(pooled_predictions, train_targets) if pooled_predictions is not None else None
    report_path = write_report(config, cross_metrics, pooled_metrics)

    print(
        f"Cross-attention: MAE={cross_metrics['mae']:.4f}, RMSE={cross_metrics['rmse']:.4f}, "
        f"Spearman={cross_metrics['spearman']:.4f}, "
        f"pred_std/true_std={fmt(cross_metrics['pred_std_over_true_std'])}, "
        f"error_vs_true_Pearson={fmt(cross_metrics['error_vs_true_pearson'])}"
    )
    print(f"Target-bin MAE: {cross_metrics['target_bin_mae']}")
    print(f"Tail MAE: {cross_metrics['tail_mae']}")
    if pooled_metrics is not None:
        print(
            f"Pooled same split: MAE={pooled_metrics['mae']:.4f}, RMSE={pooled_metrics['rmse']:.4f}, "
            f"Spearman={pooled_metrics['spearman']:.4f}"
        )
    print(f"Saved predictions to {output_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
