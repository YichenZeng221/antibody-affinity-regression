"""Train weighted CDR-aware affinity regression from a YAML config."""

from __future__ import annotations

import argparse

from src.affinity_cdr_weighted_train import train_weighted_cdr_affinity
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read config path for the weighted CDR training run."""

    parser = argparse.ArgumentParser(description="Train weighted CDR-aware affinity regression model.")
    parser.add_argument(
        "--config",
        default="config_affinity_andd_antibody_v2_all_cdr_pooled_weighted_lr3e-5_e10.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Load config and start weighted CDR-aware training."""

    args = parse_args()
    train_weighted_cdr_affinity(load_config(args.config))


if __name__ == "__main__":
    main()
