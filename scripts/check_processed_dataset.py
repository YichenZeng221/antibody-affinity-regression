"""Check the processed train/val/test CSV files.

:
, PDB
 CSV 

 CSV :
    sequence,label,pdb,chain,chain_type

:
    label = 0  light chain
    label = 1  heavy chain

 label ?
 split  heavy/light ,accuracy 

 PDB overlap?
 PDB-level split
 PDB  chain  train/val/test  split
 train  test  PDB, data leakage

 sequence?
 PDB , amino acid sequence
 sequence  train/test,

:
    python scripts/check_processed_dataset.py
"""

from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "val", "test"]


def load_split(split_name: str) -> pd.DataFrame:
    """ split  CSV"""

    csv_path = PROCESSED_DIR / f"{split_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    dataframe = pd.read_csv(csv_path)

    required_columns = {"sequence", "label", "pdb", "chain", "chain_type"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

    dataframe["sequence"] = dataframe["sequence"].astype(str)
    dataframe["pdb"] = dataframe["pdb"].astype(str)
    dataframe["chain"] = dataframe["chain"].astype(str)
    dataframe["chain_type"] = dataframe["chain_type"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)

    return dataframe


def print_split_summary(split_name: str, dataframe: pd.DataFrame) -> None:
    """ split """

    print("=" * 80)
    print(f"Split: {split_name}")
    print(f"Rows: {len(dataframe)}")

    if dataframe.empty:
        print("WARNING: this split is empty")
        return

    label_counts = dataframe["label"].value_counts().sort_index()
    sequence_lengths = dataframe["sequence"].str.len()

    print(f"Label 0 count (light chain): {int(label_counts.get(0, 0))}")
    print(f"Label 1 count (heavy chain): {int(label_counts.get(1, 0))}")
    print(f"Unique PDB count: {dataframe['pdb'].nunique()}")
    print(f"Sequence length min: {int(sequence_lengths.min())}")
    print(f"Sequence length max: {int(sequence_lengths.max())}")
    print(f"Sequence length mean: {sequence_lengths.mean():.2f}")
    print()
    print("First 5 rows:")
    print(dataframe.head(5).to_string(index=False))


def print_overlap_check(split_dataframes: dict[str, pd.DataFrame], column_name: str) -> None:
    """ train/val/test 

    column_name :
    - pdb: PDB
    - sequence: amino acid sequence
    """

    print("=" * 80)
    print(f"Overlap check for column: {column_name}")

    split_sets = {
        split_name: set(dataframe[column_name].astype(str).tolist())
        for split_name, dataframe in split_dataframes.items()
    }

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for first_split, second_split in pairs:
        overlap = split_sets[first_split] & split_sets[second_split]
        print(
            f"{first_split} vs {second_split}: "
            f"{len(overlap)} overlapping {column_name} values"
        )

        if overlap:
            examples = sorted(list(overlap))[:5]
            print(f"  examples: {examples}")


def main() -> None:
    """ train/val/test  CSV"""

    split_dataframes = {}

    for split_name in SPLITS:
        dataframe = load_split(split_name)
        split_dataframes[split_name] = dataframe
        print_split_summary(split_name, dataframe)
        print()

    print_overlap_check(split_dataframes, "pdb")
    print()
    print_overlap_check(split_dataframes, "sequence")


if __name__ == "__main__":
    main()
