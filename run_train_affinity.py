"""Train the affinity regression MVP.

:
 affinity regression 
,:
1.  config_affinity.yaml
2. tokenizer
3. 
4.  checkpoint

, src/affinity_train.py

:
    python run_train_affinity.py
"""

import argparse

from src.affinity_train import train_affinity
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    """Read command line arguments.

     config_affinity.yaml,
     clean_v2, --config 
    """

    parser = argparse.ArgumentParser(description="Train affinity regression model.")
    parser.add_argument("--config", default="config_affinity.yaml")
    return parser.parse_args()


def main() -> None:
    """Load config_affinity.yaml and start training."""

    args = parse_args()

    # config_affinity.yaml :
    # batch_sizelearning_ratecheckpoint 
    config = load_config(args.config)
    train_affinity(config)


if __name__ == "__main__":
    main()
