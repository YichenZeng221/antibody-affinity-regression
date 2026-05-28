"""Tail-aware training loop for the existing all-CDR cross-attention model.

中文说明：
这个模块不改变模型结构，也不改数据集。它只改变训练目标：
使用 train split 的 P10/P90 定义两端 tail，并在 loss 中给 tail 样本更高权重。
这样可以检查模型预测范围压缩是否部分来自 objective 对极端样本关注不足。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.affinity_cross_attention_dataset import CrossAttentionAffinityDataset
from src.affinity_cross_attention_evaluate import (
    cross_attention_device,
    cross_attention_model_inputs,
    evaluate_cross_attention_affinity_model,
)
from src.affinity_cross_attention_model import SeqProFTCrossAttentionAffinityRegressor
from src.affinity_cross_attention_train import antigen_length_from_config, print_filter_summary
from src.affinity_evaluate import compute_regression_metrics
from src.train import count_trainable_parameters
from src.utils import set_seed


CHECKPOINT_POLICIES = {
    "best_val_mae": ("val_mae", "min"),
    "best_val_spearman": ("val_spearman", "max"),
    "best_val_spread": ("val_spread_distance_to_1", "min"),
    "best_val_tail_mae": ("val_tail_mae", "min"),
}


def tail_thresholds(train_targets: list[float]) -> tuple[float, float]:
    """只用 train target 计算 P10/P90，避免 validation/test 信息泄漏。"""

    targets = pd.Series(train_targets, dtype=float)
    return float(targets.quantile(0.10)), float(targets.quantile(0.90))


def tail_sample_weights(labels: torch.Tensor, lower_p10: float, upper_p90: float, config: dict) -> torch.Tensor:
    """给 train distribution 两端的样本更高 loss weight。"""

    regular_weight = float(config.get("regular_sample_weight", 1.0))
    tail_weight = float(config.get("tail_sample_weight", 3.0))
    weights = torch.full_like(labels, regular_weight)
    tail_mask = (labels <= lower_p10) | (labels >= upper_p90)
    return torch.where(tail_mask, torch.full_like(labels, tail_weight), weights)


def tail_weighted_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Conservative tail-weighted MSE: mean(weight * squared error)."""

    return (weights * (predictions - labels) ** 2).mean()


def validation_diagnostics(
    true_values: list[float],
    predicted_values: list[float],
    lower_p10: float,
    upper_p90: float,
) -> dict:
    """计算每 epoch 所需的 validation regression-to-the-mean 指标。"""

    true = pd.Series(true_values, dtype=float)
    predicted = pd.Series(predicted_values, dtype=float)
    error = predicted - true
    absolute_error = error.abs()
    lower_mask = true <= lower_p10
    upper_mask = true >= upper_p90
    core = compute_regression_metrics(true_values, predicted_values)
    true_std = float(true.std())
    prediction_std = float(predicted.std())
    spread_ratio = prediction_std / true_std if true_std else float("nan")
    lower_mae = float(absolute_error[lower_mask].mean()) if bool(lower_mask.any()) else float("nan")
    upper_mae = float(absolute_error[upper_mask].mean()) if bool(upper_mask.any()) else float("nan")
    tail_values = [value for value in [lower_mae, upper_mae] if not pd.isna(value)]
    return {
        "val_mae": float(core["mae"]),
        "val_rmse": float(core["rmse"]),
        "val_spearman": float(core["spearman"]),
        "val_pred_std_over_true_std": spread_ratio,
        "val_spread_distance_to_1": abs(spread_ratio - 1.0),
        "val_error_vs_true_pearson": float(error.corr(true, method="pearson")),
        "val_below_p10_mae": lower_mae,
        "val_above_p90_mae": upper_mae,
        "val_tail_mae": float(sum(tail_values) / len(tail_values)) if tail_values else float("nan"),
        "val_below_p10_rows": int(lower_mask.sum()),
        "val_above_p90_rows": int(upper_mask.sum()),
    }


def improved(new_value: float, current_value: float | None, direction: str) -> bool:
    """判断一个 validation selection metric 是否产生新的最佳 checkpoint。"""

    if pd.isna(new_value):
        return False
    if current_value is None:
        return True
    return new_value < current_value if direction == "min" else new_value > current_value


def checkpoint_payload(
    model,
    config: dict,
    policy_name: str,
    epoch: int,
    epoch_metrics: dict,
    lower_p10: float,
    upper_p90: float,
    train_dataset,
    val_dataset,
) -> dict:
    """保存可追溯的 checkpoint metadata，便于之后统一测试。"""

    return {
        "model_state_dict": model.state_dict(),
        "config": config,
        "checkpoint_policy": policy_name,
        "best_epoch": epoch,
        "best_val_metrics": epoch_metrics,
        "tail_thresholds_from_train": {"p10": lower_p10, "p90": upper_p90},
        "tail_sample_weight": float(config.get("tail_sample_weight", 3.0)),
        "regular_sample_weight": float(config.get("regular_sample_weight", 1.0)),
        "train_rows_after_cdr_filter": len(train_dataset),
        "val_rows_after_cdr_filter": len(val_dataset),
    }


def train_cross_attention_tailaware(config: dict) -> dict[str, Path]:
    """训练已有 cross-attention architecture，使用 tail-weighted MSE 与四种保存策略。"""

    set_seed(int(config["seed"]))
    device = cross_attention_device(config)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_dataset = CrossAttentionAffinityDataset(
        csv_path=config["train_csv"],
        tokenizer=tokenizer,
        antigen_max_length=antigen_length_from_config(config),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    val_dataset = CrossAttentionAffinityDataset(
        csv_path=config["val_csv"],
        tokenizer=tokenizer,
        antigen_max_length=antigen_length_from_config(config),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    lower_p10, upper_p90 = tail_thresholds(train_dataset.targets)
    print("Tail-aware input mode: all six CDRs -> antigen_sequence cross-attention")
    print_filter_summary("Train", train_dataset)
    print_filter_summary("Val", val_dataset)
    print(f"Train-defined tails: lower <= P10={lower_p10:.4f}, upper >= P90={upper_p90:.4f}")
    print(
        "Tail-weighted MSE: "
        f"tail_weight={float(config.get('tail_sample_weight', 3.0)):.1f}, "
        f"regular_weight={float(config.get('regular_sample_weight', 1.0)):.1f}"
    )

    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False)
    model = SeqProFTCrossAttentionAffinityRegressor(config).to(device)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    checkpoint_paths = {name: Path(path) for name, path in config["checkpoint_paths"].items()}
    for path in checkpoint_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    best_values: dict[str, float | None] = {policy: None for policy in CHECKPOINT_POLICIES}
    history: list[dict] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_weighted_loss = 0.0
        progress = tqdm(train_loader, desc=f"Tail-aware cross-attention epoch {epoch}")
        for batch in progress:
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(**cross_attention_model_inputs(batch, device))
            weights = tail_sample_weights(labels, lower_p10, upper_p90, config)
            loss = tail_weighted_mse_loss(outputs["predictions"], labels, weights)
            loss.backward()
            optimizer.step()
            total_weighted_loss += float(loss.item())
            progress.set_postfix(weighted_loss=f"{loss.item():.4f}")

        train_loss = total_weighted_loss / len(train_loader)
        _, val_true, val_predictions = evaluate_cross_attention_affinity_model(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            **validation_diagnostics(val_true, val_predictions, lower_p10, upper_p90),
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_MAE={epoch_metrics['val_mae']:.4f}, "
            f"val_RMSE={epoch_metrics['val_rmse']:.4f}, "
            f"val_Spearman={epoch_metrics['val_spearman']:.4f}, "
            f"val_pred_std/true_std={epoch_metrics['val_pred_std_over_true_std']:.4f}, "
            f"val_error_vs_true_Pearson={epoch_metrics['val_error_vs_true_pearson']:.4f}, "
            f"val_below_P10_MAE={epoch_metrics['val_below_p10_mae']:.4f}, "
            f"val_above_P90_MAE={epoch_metrics['val_above_p90_mae']:.4f}, "
            f"val_tail_MAE={epoch_metrics['val_tail_mae']:.4f}"
        )

        for policy_name, (metric_name, direction) in CHECKPOINT_POLICIES.items():
            metric_value = float(epoch_metrics[metric_name])
            if improved(metric_value, best_values[policy_name], direction):
                best_values[policy_name] = metric_value
                torch.save(
                    checkpoint_payload(
                        model,
                        config,
                        policy_name,
                        epoch,
                        epoch_metrics,
                        lower_p10,
                        upper_p90,
                        train_dataset,
                        val_dataset,
                    ),
                    checkpoint_paths[policy_name],
                )
                print(f"Saved {policy_name} at epoch {epoch}: {metric_name}={metric_value:.4f}")

    history_path = Path(config["training_history_path"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved tail-aware training history to {history_path}")
    for policy_name, path in checkpoint_paths.items():
        print(f"Saved {policy_name} checkpoint to {path}")
    return checkpoint_paths
