"""Check the affinity regression train/val/test CSV files.

:
,
:
-  split 
- target 
- split 
- sequence  train/val/test 
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config

SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    """Read command line arguments.

     config_affinity.yaml 
     --config  clean_v2  TDC v1 
    """

    parser = argparse.ArgumentParser(description="Check affinity regression CSV files.")
    parser.add_argument("--config", default="config_affinity.yaml")
    return parser.parse_args()


def load_split(split_name: str, csv_path: str, target_column: str) -> pd.DataFrame:
    """Load one split CSV and check required columns.

    :
    :
     CSV 
    , debug
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    dataframe = pd.read_csv(csv_path)
    required_columns = {
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        target_column,
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

    return dataframe


def print_length_stats(dataframe: pd.DataFrame, column_name: str) -> None:
    """Print min/max/mean length for one sequence column.

    :
    -  tokenizer truncation 
    - train/test ,
    """

    lengths = dataframe[column_name].astype(str).str.len()
    print(
        f"{column_name} length min/max/mean: "
        f"{int(lengths.min())}/{int(lengths.max())}/{lengths.mean():.2f}"
    )


def print_split_summary(split_name: str, dataframe: pd.DataFrame, target_column: str) -> None:
    """Print basic stats for one split.

    target min/max/mean/std  train/val/test 
     test target  train ,
    """

    print("=" * 80)
    print(f"Split: {split_name}")
    print(f"Rows: {len(dataframe)}")

    if dataframe.empty:
        print("WARNING: split is empty")
        return

    targets = dataframe[target_column].astype(float)
    if "pdb" in dataframe.columns:
        print(f"Unique PDB count: {dataframe['pdb'].astype(str).nunique()}")
    if "antigen_sequence" in dataframe.columns:
        print(f"Unique antigen_sequence count: {dataframe['antigen_sequence'].astype(str).nunique()}")
    print(
        f"Target {target_column} min/max/mean/std: "
        f"{targets.min():.4f}/{targets.max():.4f}/{targets.mean():.4f}/{targets.std():.4f}"
    )
    print_length_stats(dataframe, "heavy_sequence")
    print_length_stats(dataframe, "light_sequence")
    print_length_stats(dataframe, "antigen_sequence")
    print()
    print("First 5 rows:")
    print(dataframe.head(5).to_string(index=False))


def print_overlap(split_dataframes: dict[str, pd.DataFrame], column_name: str) -> None:
    """Check overlap between train/val/test for one column.

    :
    overlap 
     sequence  train  test,
    ,
    """

    print("=" * 80)
    print(f"Overlap check: {column_name}")

    split_sets = {
        split_name: set(dataframe[column_name].astype(str).tolist())
        for split_name, dataframe in split_dataframes.items()
    }

    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_sets[first] & split_sets[second]
        print(f"{first} vs {second}: {len(overlap)} overlaps")
        if overlap:
            print(f"  examples: {sorted(list(overlap))[:3]}")


def make_combined_key(dataframe: pd.DataFrame, columns: list[str]) -> set[str]:
    """Make combined keys, for example heavy+light+antigen triplet."""

    return set(dataframe[columns].astype(str).agg("||".join, axis=1))


def print_combined_overlap(split_dataframes: dict[str, pd.DataFrame], columns: list[str], name: str) -> None:
    """Check overlap for a combined multi-column key."""

    print("=" * 80)
    print(f"Overlap check: {name}")

    split_sets = {
        split_name: make_combined_key(dataframe, columns)
        for split_name, dataframe in split_dataframes.items()
    }

    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_sets[first] & split_sets[second]
        print(f"{first} vs {second}: {len(overlap)} overlaps")


def main() -> None:
    """Run all checks."""

    args = parse_args()
    config = load_config(args.config)
    target_column = config.get("target_column", "neg_log10_affinity")

    split_paths = {
        "train": config["train_csv"],
        "val": config["val_csv"],
        "test": config["test_csv"],
    }

    print(f"Checking config: {args.config}")
    print(f"Target column: {target_column}")
    print()

    split_dataframes = {}

    for split_name in SPLITS:
        dataframe = load_split(split_name, split_paths[split_name], target_column)
        split_dataframes[split_name] = dataframe
        print_split_summary(split_name, dataframe, target_column)
        print()

    for column_name in ["pdb", "antibody_id", "antigen_id", "heavy_sequence", "light_sequence", "antigen_sequence"]:
        if column_name in split_dataframes["train"].columns:
            print_overlap(split_dataframes, column_name)
            print()

    print_combined_overlap(
        split_dataframes,
        ["heavy_sequence", "light_sequence", "antigen_sequence"],
        "heavy+light+antigen triplet",
    )
    print()


if __name__ == "__main__":
    main()
