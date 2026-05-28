"""Train tail-aware all-CDR cross-attention affinity regression from YAML config."""

import argparse

from src.affinity_cross_attention_tailaware_train import train_cross_attention_tailaware
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read the independent tail-aware experiment config."""

    parser = argparse.ArgumentParser(description="Train tail-aware cross-attention affinity model.")
    parser.add_argument(
        "--config",
        default="config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_tailaware_lr3e-5_e30.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Start tail-aware training only when invoked manually."""

    train_cross_attention_tailaware(load_config(parse_args().config))


if __name__ == "__main__":
    main()
