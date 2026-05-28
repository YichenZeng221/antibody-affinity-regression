"""Training loop for the first CDR-aware affinity regression baseline."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.affinity_cdr_dataset import CDRAwareAffinityDataset
from src.affinity_cdr_evaluate import cdr_model_inputs, evaluate_cdr_affinity_model
from src.affinity_cdr_model import SeqProFTCDRAwareAffinityRegressor
from src.train import count_trainable_parameters
from src.utils import ensure_output_dirs, get_device, set_seed


def print_dataset_filter_summary(split_name: str, dataset: CDRAwareAffinityDataset) -> None:
    """Show how many annotated rows remain after CDR success filtering."""

    print(
        f"{split_name} CDR rows kept: {len(dataset)} / {dataset.raw_row_count} "
        f"(filtered extraction failures: {dataset.filtered_out_count})"
    )


def train_cdr_affinity(config: dict) -> Path:
    """Train a config-selected CDR-subset plus antigen regression baseline."""

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
    print(f"Input CDR fields: {train_dataset.input_cdr_fields} + antigen_sequence")
    print_dataset_filter_summary("Train", train_dataset)
    print_dataset_filter_summary("Val", val_dataset)

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

    model = SeqProFTCDRAwareAffinityRegressor(config).to(device)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    printed_prediction_shape = False
    final_val_metrics = None

    for epoch in range(int(config["epochs"])):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"CDR affinity epoch {epoch + 1}")

        for batch in progress_bar:
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(
                **cdr_model_inputs(batch, device, model.input_cdr_fields),
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

        average_loss = total_loss / len(train_dataloader)
        val_metrics, _, _ = evaluate_cdr_affinity_model(model, val_dataloader, device)
        final_val_metrics = val_metrics
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={average_loss:.4f}, "
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
    print(f"Saved CDR-aware affinity checkpoint to {checkpoint_path}")
    return checkpoint_path
