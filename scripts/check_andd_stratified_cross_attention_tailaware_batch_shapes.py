"""Single-batch dry-run for tail-aware cross-attention; no training is performed.

:
 train batch forward, tail-weighted MSE
 backwardoptimizer.step  checkpoint
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.affinity_cross_attention_dataset import CrossAttentionAffinityDataset  # noqa: E402
from src.affinity_cross_attention_evaluate import cross_attention_device, cross_attention_model_inputs  # noqa: E402
from src.affinity_cross_attention_model import SeqProFTCrossAttentionAffinityRegressor  # noqa: E402
from src.affinity_cross_attention_tailaware_train import (  # noqa: E402
    tail_sample_weights,
    tail_thresholds,
    tail_weighted_mse_loss,
)
from src.affinity_cross_attention_train import antigen_length_from_config  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read new tail-aware config path."""

    parser = argparse.ArgumentParser(description="Dry-run tail-aware all-CDR cross-attention.")
    parser.add_argument(
        "--config",
        default="config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_tailaware_lr3e-5_e30.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Run one no-grad batch to validate token shapes and weighted loss."""

    config = load_config(parse_args().config)
    set_seed(int(config["seed"]))
    device = cross_attention_device(config)
    print(f"Config device request: {config.get('device')}")
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
    lower_p10, upper_p90 = tail_thresholds(dataset.targets)
    target_series = pd.Series(dataset.targets, dtype=float)
    lower_rows = int((target_series <= lower_p10).sum())
    upper_rows = int((target_series >= upper_p90).sum())
    print(f"Rows kept: {len(dataset)} / {dataset.raw_row_count}")
    print(f"Train P10/P90 thresholds: {lower_p10:.4f} / {upper_p90:.4f}")
    print(f"Tail row counts: below_or_equal_P10={lower_rows}, above_or_equal_P90={upper_rows}")
    print(
        f"Loss weights: regular={float(config['regular_sample_weight']):.1f}, "
        f"tail={float(config['tail_sample_weight']):.1f}"
    )

    batch = next(iter(DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)))
    labels = batch["labels"].to(device)
    model = SeqProFTCrossAttentionAffinityRegressor(config).to(device)
    model.eval()
    with torch.no_grad():
        output = model(
            **cross_attention_model_inputs(batch, device),
            return_debug_shapes=True,
        )
        weights = tail_sample_weights(labels, lower_p10, upper_p90, config)
        loss = tail_weighted_mse_loss(output["predictions"], labels, weights)
        example_labels = torch.tensor(
            [lower_p10, (lower_p10 + upper_p90) / 2, upper_p90],
            dtype=torch.float32,
            device=device,
        )
        example_weights = tail_sample_weights(example_labels, lower_p10, upper_p90, config)

    for name, shape in output["debug_shapes"].items():
        print(f"{name}: {shape}")
    print(f"labels: {tuple(labels.shape)}")
    print(f"sample weights: {weights.detach().cpu().tolist()}")
    print(
        "Weight rule sanity check [P10, middle, P90]: "
        f"{example_weights.detach().cpu().tolist()}"
    )
    print(f"tail-weighted MSE finite: {bool(torch.isfinite(loss).item())}; value={loss.item():.6f}")
    print("Tail-aware cross-attention dry-run complete. No model was trained.")


if __name__ == "__main__":
    main()
