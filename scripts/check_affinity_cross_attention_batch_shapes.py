"""Dry-run one all-CDR cross-attention batch without training.

:
 forward sanity check, checkpoint, evaluation
:
-  CDR  token matrix shape
- concat  all-CDR token shape
- antigen token shape
- cross-attention output shape
- pooled feature shape
- prediction shape
- MSE loss 
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.affinity_cross_attention_dataset import CrossAttentionAffinityDataset
from src.affinity_cross_attention_evaluate import (
    cross_attention_device,
    cross_attention_model_inputs,
)
from src.affinity_cross_attention_model import SeqProFTCrossAttentionAffinityRegressor
from src.affinity_cross_attention_train import antigen_length_from_config
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read config path for shape check."""

    parser = argparse.ArgumentParser(description="Check all-CDR cross-attention batch shapes.")
    parser.add_argument(
        "--config",
        default="config_affinity_unified_no_high_risk_cross_attention_all_cdrs_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Load one batch and run no-grad cross-attention forward."""

    config = load_config(parse_args().config)
    device = cross_attention_device(config)
    print(f"Config device request: {config.get('device', 'auto')}")
    print(f"Using device: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = CrossAttentionAffinityDataset(
        csv_path=config["train_csv"],
        tokenizer=tokenizer,
        antigen_max_length=antigen_length_from_config(config),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    batch = next(iter(DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)))
    model = SeqProFTCrossAttentionAffinityRegressor(config).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(
            **cross_attention_model_inputs(batch, device),
            labels=batch["labels"].to(device),
            return_debug_shapes=True,
        )

    print(f"Filtered train rows: {len(dataset)} / {dataset.raw_row_count}")
    for key in [
        "hcdr1_input_ids",
        "hcdr2_input_ids",
        "hcdr3_input_ids",
        "lcdr1_input_ids",
        "lcdr2_input_ids",
        "lcdr3_input_ids",
        "antigen_input_ids",
    ]:
        print(f"{key} shape: {tuple(batch[key].shape)}")
    print(f"Label shape: {tuple(batch['labels'].shape)}")
    for name, shape in outputs["debug_shapes"].items():
        print(f"{name}: {shape}")
    print(f"Loss scalar shape: {tuple(outputs['loss'].shape)}")
    print(f"Loss value is finite: {bool(torch.isfinite(outputs['loss']).item())}")
    print("Cross-attention dry-run complete. No model was trained.")


if __name__ == "__main__":
    main()
