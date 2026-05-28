"""Dry-run one interaction batch and print residue-matrix shapes.

:
 sanity check:
1.  interaction config
2.  filtered train set  batch
3.  forward
4.  deviceprediction shapeinteraction matrix shape

 checkpoint predictions
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

from src.affinity_interaction_dataset import InteractionAffinityDataset
from src.affinity_interaction_evaluate import interaction_model_inputs
from src.affinity_interaction_model import SeqProFTInteractionAffinityRegressor
from src.utils import get_device, load_config


def parse_args() -> argparse.Namespace:
    """Read interaction config path for the dry-run."""

    parser = argparse.ArgumentParser(description="Check interaction-aware batch shapes.")
    parser.add_argument(
        "--config",
        default="config_affinity_unified_no_high_risk_interaction_hcdr3_lcdr3_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Run one no-grad forward pass and print shape evidence."""

    config = load_config(parse_args().config)
    device = get_device()
    print(f"Using device: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = InteractionAffinityDataset(
        csv_path=config["train_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    batch = next(iter(dataloader))
    model = SeqProFTInteractionAffinityRegressor(config).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            **interaction_model_inputs(batch, device),
            labels=batch["labels"].to(device),
            return_debug_shapes=True,
        )

    print(f"Filtered train rows: {len(dataset)} / {dataset.raw_row_count}")
    print(f"HCDR3 input_ids shape: {tuple(batch['hcdr3_input_ids'].shape)}")
    print(f"LCDR3 input_ids shape: {tuple(batch['lcdr3_input_ids'].shape)}")
    print(f"Antigen input_ids shape: {tuple(batch['antigen_input_ids'].shape)}")
    print(f"Label shape: {tuple(batch['labels'].shape)}")
    print(f"Prediction shape: {tuple(outputs['predictions'].shape)}")
    for name, shape in outputs["debug_shapes"].items():
        print(f"{name}: {shape}")
    print("Interaction dry-run complete. No model was trained.")


if __name__ == "__main__":
    main()
