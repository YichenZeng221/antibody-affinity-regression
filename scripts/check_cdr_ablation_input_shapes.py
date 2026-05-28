"""Dry-run the CDR ablation input shapes without training a model.

:
 mode  filtered training sample,
 config  CDR tensor padding  shape 
 forward ESM2 checkpoint,
"""

from __future__ import annotations

from pathlib import Path
import sys

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.affinity_cdr_dataset import CDRAwareAffinityDataset
from src.utils import load_config


CONFIGS = [
    "config_affinity_unified_no_high_risk_cdr_aware_lr3e-5_e10.yaml",
    "config_affinity_unified_no_high_risk_hcdr3_antigen_lr3e-5_e10.yaml",
    "config_affinity_unified_no_high_risk_hcdr3_lcdr3_antigen_lr3e-5_e10.yaml",
    "config_affinity_unified_no_high_risk_heavy_cdrs_antigen_lr3e-5_e10.yaml",
    "config_affinity_unified_no_high_risk_light_cdrs_antigen_lr3e-5_e10.yaml",
]


def main() -> None:
    """Print input keys and shapes for every configured CDR ablation mode."""

    tokenizer = None
    for config_path in CONFIGS:
        config = load_config(config_path)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        dataset = CDRAwareAffinityDataset(
            csv_path=config["train_csv"],
            tokenizer=tokenizer,
            max_length=int(config["max_length"]),
            cdr_max_length=int(config.get("cdr_max_length", 64)),
            target_column=config["target_column"],
            input_cdr_fields=config.get("input_cdr_fields"),
        )
        sample = dataset[0]
        tensor_shapes = {
            key: tuple(value.shape)
            for key, value in sample.items()
            if key.endswith("_input_ids")
        }
        print()
        print(f"Mode: {config['mode_name']}")
        print(f"  input_cdr_fields: {dataset.input_cdr_fields}")
        print(f"  filtered train rows: {len(dataset)} / {dataset.raw_row_count}")
        print(f"  input id shapes: {tensor_shapes}")
        print(f"  label shape: {tuple(sample['labels'].shape)}")
    print()
    print("CDR ablation dry-run shape check complete. No model was trained.")


if __name__ == "__main__":
    main()
