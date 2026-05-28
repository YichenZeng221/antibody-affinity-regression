"""Train the all-CDR cross-attention affinity baseline from YAML config."""

import argparse

from src.affinity_cross_attention_train import train_cross_attention_affinity
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read explicit cross-attention config path."""

    parser = argparse.ArgumentParser(description="Train cross-attention affinity model.")
    parser.add_argument(
        "--config",
        default="config_affinity_unified_no_high_risk_cross_attention_all_cdrs_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and start the independent cross-attention training loop."""

    train_cross_attention_affinity(load_config(parse_args().config))


if __name__ == "__main__":
    main()
