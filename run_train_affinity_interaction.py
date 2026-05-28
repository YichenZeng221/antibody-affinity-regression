"""Train the independent interaction-aware affinity baseline from YAML config."""

import argparse

from src.affinity_interaction_train import train_interaction_affinity
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read interaction experiment config path."""

    parser = argparse.ArgumentParser(description="Train interaction-aware affinity model.")
    parser.add_argument(
        "--config",
        default="config_affinity_unified_no_high_risk_interaction_hcdr3_lcdr3_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and start the separate interaction training loop."""

    args = parse_args()
    train_interaction_affinity(load_config(args.config))


if __name__ == "__main__":
    main()
