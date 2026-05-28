"""Weighted training loop for CDR-aware affinity regression.

中文人话说明：
这个训练版本专门针对 regression-to-mean。

普通 MSE 会让模型倾向预测中间值，因为中间区域样本多、风险小。
这里把 train target 分成 low/mid/high 三个区间，然后给 low/high 更高 loss weight，
希望模型更认真学习 affinity extremes。

模型结构不变：
    six CDRs + antigen -> shared ESM2+LoRA -> mean pooling -> regression head
只改训练 loss 和 checkpoint 保存策略。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.affinity_cdr_dataset import CDRAwareAffinityDataset
from src.affinity_cdr_evaluate import cdr_model_inputs, evaluate_cdr_affinity_model
from src.affinity_cdr_model import SeqProFTCDRAwareAffinityRegressor
from src.train import count_trainable_parameters
from src.utils import ensure_output_dirs, get_device, set_seed


def target_bin_thresholds(targets: list[float]) -> tuple[float, float]:
    """Use train-set 33% and 67% quantiles as low/high boundaries."""

    series = pd.Series(targets, dtype=float)
    low_threshold = float(series.quantile(1 / 3))
    high_threshold = float(series.quantile(2 / 3))
    return low_threshold, high_threshold


def sample_weights(labels: torch.Tensor, low_threshold: float, high_threshold: float, config: dict) -> torch.Tensor:
    """Return per-sample weights based on target bin.

    low target 和 high target 是模型最容易被“拉回平均值”的区域，
    所以默认给它们更高权重。
    """

    low_weight = float(config.get("low_target_loss_weight", 2.0))
    mid_weight = float(config.get("mid_target_loss_weight", 1.0))
    high_weight = float(config.get("high_target_loss_weight", 2.0))

    weights = torch.full_like(labels, fill_value=mid_weight)
    weights = torch.where(labels <= low_threshold, torch.full_like(labels, low_weight), weights)
    weights = torch.where(labels >= high_threshold, torch.full_like(labels, high_weight), weights)
    return weights


def weighted_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted MSE: average of weight * squared error."""

    squared_error = (predictions - labels) ** 2
    return (weights * squared_error).mean()


def target_bin_metrics(true_values: list[float], predicted_values: list[float]) -> dict:
    """Compute MAE for low/mid/high target bins on validation or test predictions."""

    frame = pd.DataFrame({"true": true_values, "predicted": predicted_values})
    ranked = frame["true"].rank(method="first")
    frame["target_bin"] = pd.qcut(ranked, q=3, labels=["low_target", "mid_target", "high_target"])
    frame["absolute_error"] = (frame["predicted"] - frame["true"]).abs()
    return {
        str(bin_name): float(group["absolute_error"].mean())
        for bin_name, group in frame.groupby("target_bin", observed=True)
    }


def checkpoint_payload(model, config: dict, val_metrics: dict, epoch: int, train_dataset, val_dataset, thresholds: tuple[float, float]) -> dict:
    """Build a checkpoint dictionary with enough metadata to audit later."""

    return {
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_epoch": epoch,
        "best_val_metrics": val_metrics,
        "train_rows_after_cdr_filter": len(train_dataset),
        "val_rows_after_cdr_filter": len(val_dataset),
        "target_bin_thresholds": {
            "low_threshold": thresholds[0],
            "high_threshold": thresholds[1],
        },
    }


def save_history(history: list[dict], history_path: Path) -> None:
    """Save per-epoch metrics to CSV for later plotting/reporting."""

    history_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_weighted_cdr_affinity(config: dict) -> Path:
    """Train CDR-aware model with weighted MSE and best-val checkpoint selection."""

    set_seed(int(config["seed"]))
    ensure_output_dirs()
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_dataset = CDRAwareAffinityDataset(
        csv_path=config["train_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        input_cdr_fields=config.get("input_cdr_fields"),
    )
    val_dataset = CDRAwareAffinityDataset(
        csv_path=config["val_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        input_cdr_fields=config.get("input_cdr_fields"),
    )

    low_threshold, high_threshold = target_bin_thresholds(train_dataset.targets)
    print(f"Input CDR fields: {train_dataset.input_cdr_fields} + antigen_sequence")
    print(f"Train rows kept: {len(train_dataset)} / {train_dataset.raw_row_count}")
    print(f"Val rows kept: {len(val_dataset)} / {val_dataset.raw_row_count}")
    print(f"Weighted target thresholds: low <= {low_threshold:.4f}, high >= {high_threshold:.4f}")
    print(
        "Loss weights: "
        f"low={float(config.get('low_target_loss_weight', 2.0))}, "
        f"mid={float(config.get('mid_target_loss_weight', 1.0))}, "
        f"high={float(config.get('high_target_loss_weight', 2.0))}"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    model = SeqProFTCDRAwareAffinityRegressor(config).to(device)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    best_metric_name = config.get("best_checkpoint_metric", "mae")
    best_metric_value = None
    best_epoch = None
    history = []
    printed_prediction_shape = False
    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_unweighted_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Weighted CDR epoch {epoch}")

        for batch in progress_bar:
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(**cdr_model_inputs(batch, device, model.input_cdr_fields))
            predictions = outputs["predictions"]

            if not printed_prediction_shape:
                print(f"First batch prediction shape: {predictions.shape}")
                printed_prediction_shape = True

            weights = sample_weights(labels, low_threshold, high_threshold, config)
            loss = weighted_mse_loss(predictions, labels, weights)
            unweighted_loss = torch.mean((predictions - labels) ** 2)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_unweighted_loss += float(unweighted_loss.item())
            progress_bar.set_postfix(weighted_loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(train_dataloader)
        average_unweighted_loss = total_unweighted_loss / len(train_dataloader)
        val_metrics, val_true, val_predicted = evaluate_cdr_affinity_model(model, val_dataloader, device)
        val_bin_mae = target_bin_metrics(val_true, val_predicted)
        selected_metric = float(val_metrics[best_metric_name])

        history_row = {
            "epoch": epoch,
            "train_weighted_loss": average_loss,
            "train_unweighted_mse": average_unweighted_loss,
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_spearman": val_metrics["spearman"],
            "val_low_target_mae": val_bin_mae.get("low_target"),
            "val_mid_target_mae": val_bin_mae.get("mid_target"),
            "val_high_target_mae": val_bin_mae.get("high_target"),
        }
        history.append(history_row)

        print(
            f"Epoch {epoch}: train_weighted_loss={average_loss:.4f}, "
            f"train_unweighted_mse={average_unweighted_loss:.4f}, "
            f"val_MAE={val_metrics['mae']:.4f}, val_RMSE={val_metrics['rmse']:.4f}, "
            f"val_Spearman={val_metrics['spearman']:.4f}, "
            f"val_bins={val_bin_mae}"
        )

        if best_metric_value is None or selected_metric < best_metric_value:
            best_metric_value = selected_metric
            best_epoch = epoch
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    {**val_metrics, "target_bin_mae": val_bin_mae},
                    epoch,
                    train_dataset,
                    val_dataset,
                    (low_threshold, high_threshold),
                ),
                checkpoint_path,
            )
            print(f"Saved new best checkpoint at epoch {epoch}: {best_metric_name}={selected_metric:.4f}")

    history_path = Path(config.get("training_history_path", checkpoint_path.with_suffix(".history.csv")))
    save_history(history, history_path)

    print(f"Best epoch: {best_epoch}, best val {best_metric_name}: {best_metric_value:.4f}")
    print(f"Saved best weighted CDR checkpoint to {checkpoint_path}")
    print(f"Saved training history to {history_path}")
    return checkpoint_path
