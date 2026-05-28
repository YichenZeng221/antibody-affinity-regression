"""Evaluate the ANDD stratified all-CDR pooled run after manual training.

中文人话说明：
这个脚本不会训练模型。它只在你已经手动训练出的 checkpoint 上做 test 推理，
并把新 stratified split 的结果与旧 split baseline 放在一起比较。

和通用 CDR evaluator 不同的地方：
- low/mid/high 区间只用各自 train split 的 tertiles 定义；
- tail MAE 只用各自 train split 的 P10/P90 定义；
- 因而能更公平地检查 split coverage 对 regression-to-the-mean 的影响。
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

from src.affinity_cdr_dataset import CDRAwareAffinityDataset, SUCCESS_STATUS_VALUES
from src.affinity_cdr_evaluate import evaluate_cdr_affinity_model
from src.affinity_cdr_model import SeqProFTCDRAwareAffinityRegressor
from src.affinity_evaluate import compute_regression_metrics
from src.utils import get_device, load_config


TRUE_COL = "true_neg_log10_affinity"
PRED_COL = "predicted_neg_log10_affinity"
TARGET_COL = "neg_log10_affinity_candidate"
OLD_TRAIN_CSV = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2_cdr_annotated/train.csv"
OLD_PREDICTIONS_CSV = (
    ROOT
    / "outputs/andd_antibody_v2/all_cdr_pooled/"
    / "andd_antibody_v2_all_cdr_pooled_test_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    """读取新 stratified experiment 的配置路径。"""

    parser = argparse.ArgumentParser(description="Evaluate ANDD stratified CDR-aware checkpoint.")
    parser.add_argument(
        "--config",
        default="config_affinity_andd_antibody_v2_stratified_all_cdr_pooled_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def successful_target_values(csv_path: str | Path) -> pd.Series:
    """读取训练 target，并应用与 Dataset 相同的 CDR 成功过滤规则。"""

    frame = pd.read_csv(csv_path)
    heavy_ok = frame["heavy_cdr_status"].fillna("").astype(str).str.lower().isin(SUCCESS_STATUS_VALUES)
    light_ok = frame["light_cdr_status"].fillna("").astype(str).str.lower().isin(SUCCESS_STATUS_VALUES)
    return pd.to_numeric(frame.loc[heavy_ok & light_ok, TARGET_COL], errors="raise")


def metric_bundle(predictions: pd.DataFrame, train_targets: pd.Series) -> dict:
    """按照训练集定义的 bins/tails 计算指标。"""

    true_values = pd.to_numeric(predictions[TRUE_COL], errors="raise")
    predicted_values = pd.to_numeric(predictions[PRED_COL], errors="raise")
    error = predicted_values - true_values
    absolute_error = error.abs()
    train_low = float(train_targets.quantile(1 / 3))
    train_high = float(train_targets.quantile(2 / 3))
    train_p10 = float(train_targets.quantile(0.10))
    train_p90 = float(train_targets.quantile(0.90))

    frame = predictions.copy()
    frame["absolute_error_for_metrics"] = absolute_error
    frame["target_bin"] = pd.cut(
        true_values,
        bins=[-math.inf, train_low, train_high, math.inf],
        labels=["low_target", "mid_target", "high_target"],
        include_lowest=True,
    )
    tail_masks = {
        "below_train_p10": true_values <= train_p10,
        "above_train_p90": true_values >= train_p90,
    }
    regression_metrics = compute_regression_metrics(true_values.tolist(), predicted_values.tolist())
    true_std = float(true_values.std())
    prediction_std = float(predicted_values.std())
    return {
        **regression_metrics,
        "rows": int(len(frame)),
        "prediction_std": prediction_std,
        "true_std": true_std,
        "pred_std_over_true_std": prediction_std / true_std if true_std else None,
        "error_vs_true_pearson": float(error.corr(true_values, method="pearson")),
        "train_thresholds": {
            "tertile_low_upper": train_low,
            "tertile_mid_upper": train_high,
            "p10": train_p10,
            "p90": train_p90,
        },
        "target_bin_mae": {
            str(label): float(group["absolute_error_for_metrics"].mean())
            for label, group in frame.groupby("target_bin", observed=True)
        },
        "target_bin_rows": {
            str(label): int(len(group))
            for label, group in frame.groupby("target_bin", observed=True)
        },
        "tail_mae": {
            label: (float(absolute_error[mask].mean()) if int(mask.sum()) else None)
            for label, mask in tail_masks.items()
        },
        "tail_rows": {label: int(mask.sum()) for label, mask in tail_masks.items()},
    }


def prediction_rows(dataset: CDRAwareAffinityDataset, true_values: list[float], predicted_values: list[float]) -> pd.DataFrame:
    """按 DataLoader 顺序把 test metadata 与预测写在一起。"""

    rows: list[dict] = []
    for metadata, true_value, predicted_value in zip(
        dataset.data.to_dict("records"),
        true_values,
        predicted_values,
    ):
        error = float(predicted_value - true_value)
        rows.append(
            {
                "sample_id": metadata.get("sample_id", metadata.get("candidate_id", "")),
                "candidate_id": metadata.get("candidate_id", ""),
                "source": metadata.get("source", ""),
                "pdb_id": metadata.get("pdb_id", ""),
                "ag_name": metadata.get("ag_name", ""),
                "antigen_sequence": metadata.get("antigen_sequence", ""),
                TRUE_COL: float(true_value),
                PRED_COL: float(predicted_value),
                "error": error,
                "absolute_error": abs(error),
                "fold_error": 10 ** abs(error),
            }
        )
    return pd.DataFrame(rows)


def fmt(value: float | None) -> str:
    """在 Markdown 中清楚显示可空指标。"""

    return "NA" if value is None or pd.isna(value) else f"{value:.4f}"


def write_report(config: dict, new_metrics: dict, old_metrics: dict | None) -> Path:
    """输出新旧 split baseline 对比报告。"""

    path = Path(config["report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ANDD Antibody v2 Stratified Split: All-CDR Pooled Baseline",
        "",
        "## Question",
        "",
        "Does stratified antigen-level split with explicit validation/test tail coverage change the observed regression-to-the-mean behavior?",
        "",
        "## Controlled Setting",
        "",
        "- Model: existing all-CDR pooled shared ESM2 + LoRA regressor.",
        "- Loss: unchanged MSE loss.",
        "- Learning rate/epochs/seed: unchanged from the prior all-CDR pooled baseline.",
        "- Only the antigen-level split is changed.",
        "- Low/mid/high target bins and P10/P90 tails are defined from each experiment's train split.",
        "",
        "## Test Metrics",
        "",
        "| split version | MAE | RMSE | Spearman | pred std / true std | error vs true Pearson |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if old_metrics is not None:
        lines.append(
            f"| original antigen split | {old_metrics['mae']:.4f} | {old_metrics['rmse']:.4f} | "
            f"{old_metrics['spearman']:.4f} | {fmt(old_metrics['pred_std_over_true_std'])} | "
            f"{fmt(old_metrics['error_vs_true_pearson'])} |"
        )
    lines.append(
        f"| stratified antigen split | {new_metrics['mae']:.4f} | {new_metrics['rmse']:.4f} | "
        f"{new_metrics['spearman']:.4f} | {fmt(new_metrics['pred_std_over_true_std'])} | "
        f"{fmt(new_metrics['error_vs_true_pearson'])} |"
    )
    lines.extend(
        [
            "",
            "## Train-Defined Target-Bin MAE",
            "",
            "| split version | low rows / MAE | mid rows / MAE | high rows / MAE |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, metrics in [("original antigen split", old_metrics), ("stratified antigen split", new_metrics)]:
        if metrics is None:
            continue
        lines.append(
            f"| {label} | {metrics['target_bin_rows'].get('low_target', 0)} / "
            f"{fmt(metrics['target_bin_mae'].get('low_target'))} | "
            f"{metrics['target_bin_rows'].get('mid_target', 0)} / {fmt(metrics['target_bin_mae'].get('mid_target'))} | "
            f"{metrics['target_bin_rows'].get('high_target', 0)} / {fmt(metrics['target_bin_mae'].get('high_target'))} |"
        )
    lines.extend(
        [
            "",
            "## Train-Defined Tail MAE",
            "",
            "| split version | below train P10 rows / MAE | above train P90 rows / MAE |",
            "|---|---:|---:|",
        ]
    )
    for label, metrics in [("original antigen split", old_metrics), ("stratified antigen split", new_metrics)]:
        if metrics is None:
            continue
        lines.append(
            f"| {label} | {metrics['tail_rows']['below_train_p10']} / {fmt(metrics['tail_mae']['below_train_p10'])} | "
            f"{metrics['tail_rows']['above_train_p90']} / {fmt(metrics['tail_mae']['above_train_p90'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `pred std / true std` closer to 1 indicates less compressed prediction range.",
            "- `error vs true Pearson` closer to 0 indicates weaker systematic regression-to-the-mean.",
            "- Tail MAE now measures behavior in tails represented according to the training distribution.",
            "- If tail coverage improves but compression remains strong, split alone is unlikely to be the main bottleneck.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """加载手动训练好的 checkpoint，运行新 test inference 并保存报告。"""

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
    dataloader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False)
    model = SeqProFTCDRAwareAffinityRegressor(config).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, true_values, predicted_values = evaluate_cdr_affinity_model(model, dataloader, device)

    predictions = prediction_rows(test_dataset, true_values, predicted_values)
    predictions_path = Path(config["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    new_metrics = metric_bundle(predictions, successful_target_values(config["train_csv"]))

    old_metrics = None
    if OLD_PREDICTIONS_CSV.exists() and OLD_TRAIN_CSV.exists():
        old_metrics = metric_bundle(pd.read_csv(OLD_PREDICTIONS_CSV), successful_target_values(OLD_TRAIN_CSV))
    report_path = write_report(config, new_metrics, old_metrics)

    print(
        f"Stratified split: MAE={new_metrics['mae']:.4f}, RMSE={new_metrics['rmse']:.4f}, "
        f"Spearman={new_metrics['spearman']:.4f}, pred_std/true_std={new_metrics['pred_std_over_true_std']:.4f}, "
        f"error_vs_true_Pearson={new_metrics['error_vs_true_pearson']:.4f}"
    )
    print(f"Target-bin MAE: {new_metrics['target_bin_mae']}")
    print(f"Tail MAE: {new_metrics['tail_mae']}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
