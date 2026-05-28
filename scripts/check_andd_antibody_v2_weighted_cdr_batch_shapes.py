"""Dry-run shape check for weighted ANDD antibody v2 CDR training.

中文人话说明：
这个脚本只跑一个 batch 的 forward 和 weighted loss。
它不训练、不 optimizer.step、不保存 checkpoint。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.affinity_cdr_dataset import CDRAwareAffinityDataset
from src.affinity_cdr_evaluate import cdr_model_inputs
from src.affinity_cdr_model import SeqProFTCDRAwareAffinityRegressor
from src.affinity_cdr_weighted_train import sample_weights, target_bin_thresholds, weighted_mse_loss
from src.utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    """Read weighted config path."""

    parser = argparse.ArgumentParser(description="Check weighted CDR batch shapes.")
    parser.add_argument(
        "--config",
        default="config_affinity_andd_antibody_v2_all_cdr_pooled_weighted_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Run one model forward pass and print shapes/weights."""

    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = CDRAwareAffinityDataset(
        csv_path=config["train_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        input_cdr_fields=config.get("input_cdr_fields"),
    )
    low_threshold, high_threshold = target_bin_thresholds(dataset.targets)
    print(f"Rows kept: {len(dataset)} / {dataset.raw_row_count}")
    print(f"Weighted target thresholds: low <= {low_threshold:.4f}, high >= {high_threshold:.4f}")

    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    batch = next(iter(dataloader))
    labels = batch["labels"].to(device)

    model = SeqProFTCDRAwareAffinityRegressor(config).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**cdr_model_inputs(batch, device, model.input_cdr_fields))
        weights = sample_weights(labels, low_threshold, high_threshold, config)
        loss = weighted_mse_loss(outputs["predictions"], labels, weights)

    print(f"labels shape: {tuple(labels.shape)}")
    print(f"predictions shape: {tuple(outputs['predictions'].shape)}")
    print(f"sample weights shape: {tuple(weights.shape)}")
    print(f"sample weights values: {weights.detach().cpu().tolist()}")
    print(f"weighted loss value: {loss.item():.6f}")
    print("Weighted dry-run shape check passed.")


if __name__ == "__main__":
    main()
