"""Training loop for the all-CDR CDR-to-antigen cross-attention baseline."""

from __future__ import annotations

from pathlib import Path

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
from src.train import count_trainable_parameters
from src.utils import ensure_output_dirs, set_seed


def antigen_length_from_config(config: dict) -> int:
    """Use branch-specific antigen length while allowing old `max_length` fallback."""

    return int(config.get("antigen_max_length", config.get("max_length", 512)))


def print_filter_summary(split_name: str, dataset: CrossAttentionAffinityDataset) -> None:
    """Print successful CDR row count before training."""

    print(
        f"{split_name} cross-attention rows kept: {len(dataset)} / {dataset.raw_row_count} "
        f"(filtered CDR extraction failures: {dataset.filtered_out_count})"
    )


def train_cross_attention_affinity(config: dict) -> Path:
    """Train all-six-CDR queries attending to antigen tokens."""

    set_seed(int(config["seed"]))
    ensure_output_dirs()
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
    print("Cross-attention input mode: all six CDRs -> antigen_sequence")
    print_filter_summary("Train", train_dataset)
    print_filter_summary("Val", val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    model = SeqProFTCrossAttentionAffinityRegressor(config).to(device)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    printed_prediction_shape = False
    final_val_metrics = None

    for epoch in range(int(config["epochs"])):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Cross-attention affinity epoch {epoch + 1}")
        for batch in progress_bar:
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(
                **cross_attention_model_inputs(batch, device),
                labels=labels,
            )
            if not printed_prediction_shape:
                print(f"First batch prediction shape: {outputs['predictions'].shape}")
                printed_prediction_shape = True
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_dataloader)
        val_metrics, _, _ = evaluate_cross_attention_affinity_model(
            model,
            val_dataloader,
            device,
        )
        final_val_metrics = val_metrics
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_MAE={val_metrics['mae']:.4f} log10 units, "
            f"val_MSE={val_metrics['mse']:.4f}, "
            f"val_RMSE={val_metrics['rmse']:.4f} log10 units, "
            f"val_Spearman={val_metrics['spearman']:.4f}, "
            f"val_MAE_fold_error~{val_metrics['approx_mae_fold_error']:.1f}x, "
            f"val_RMSE_fold_error~{val_metrics['approx_rmse_fold_error']:.1f}x"
        )

    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "final_val_metrics": final_val_metrics,
            "train_rows_after_cdr_filter": len(train_dataset),
            "val_rows_after_cdr_filter": len(val_dataset),
        },
        checkpoint_path,
    )
    print(f"Saved cross-attention affinity checkpoint to {checkpoint_path}")
    return checkpoint_path
