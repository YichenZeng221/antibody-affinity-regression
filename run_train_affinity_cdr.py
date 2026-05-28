"""Train the CDR-aware affinity regression baseline from a YAML config."""

import argparse

from src.affinity_cdr_train import train_cdr_affinity
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read config path while keeping the CDR run explicitly opt-in."""

    parser = argparse.ArgumentParser(description="Train CDR-aware affinity regression model.")
    parser.add_argument(
        "--config",
        default="config_affinity_unified_no_high_risk_cdr_aware_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and start the separate CDR-aware training loop."""

    args = parse_args()
    train_cdr_affinity(load_config(args.config))


if __name__ == "__main__":
    main()
