"""Single-batch dry-run for ANDD antibody v2 all-CDR pooled baseline.

:
 Dataset + Model forward
, checkpoint

:
-  CDR extraction  CSV 
-  tokenizer / tensor shape 
-  prediction shape  [batch_size]
-  loss 
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

from src.affinity_cdr_dataset import CDRAwareAffinityDataset, CDR_MODEL_SEQUENCE_KEYS
from src.affinity_cdr_evaluate import cdr_model_inputs
from src.affinity_cdr_model import SeqProFTCDRAwareAffinityRegressor
from src.utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    """Read config path for the dry-run."""

    parser = argparse.ArgumentParser(description="Check one ANDD antibody v2 CDR-aware batch.")
    parser.add_argument(
        "--config",
        default="config_affinity_andd_antibody_v2_all_cdr_pooled_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Load one train batch and run one forward pass without optimizer/training."""

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
    print(f"Rows kept after CDR status filtering: {len(dataset)} / {dataset.raw_row_count}")
    print(f"Input CDR fields: {dataset.input_cdr_fields}")

    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    batch = next(iter(dataloader))

    for cdr_field in dataset.input_cdr_fields:
        key = CDR_MODEL_SEQUENCE_KEYS[cdr_field]
        print(f"{cdr_field} input_ids shape: {tuple(batch[f'{key}_input_ids'].shape)}")
        print(f"{cdr_field} attention_mask shape: {tuple(batch[f'{key}_attention_mask'].shape)}")
    print(f"antigen input_ids shape: {tuple(batch['antigen_input_ids'].shape)}")
    print(f"antigen attention_mask shape: {tuple(batch['antigen_attention_mask'].shape)}")
    print(f"labels shape: {tuple(batch['labels'].shape)}")

    model = SeqProFTCDRAwareAffinityRegressor(config).to(device)
    model.eval()
    with torch.no_grad():
        labels = batch["labels"].to(device)
        outputs = model(**cdr_model_inputs(batch, device, model.input_cdr_fields), labels=labels)

    print(f"prediction shape: {tuple(outputs['predictions'].shape)}")
    print(f"loss value: {outputs['loss'].item():.6f}")
    print("Dry-run shape check passed.")


if __name__ == "__main__":
    main()
