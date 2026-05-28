"""Check the processed train/val/test CSV files.

中文人话说明：
这个脚本不训练模型，也不下载 PDB。
它只检查已经生成好的 CSV 数据集是否看起来正常。

现在我们的 CSV 应该包含：
    sequence,label,pdb,chain,chain_type

标签含义：
    label = 0 表示 light chain
    label = 1 表示 heavy chain

为什么要检查 label 分布？
如果某个 split 里 heavy/light 极度不平衡，accuracy 可能不好解释。

为什么要检查 PDB overlap？
我们现在使用 PDB-level split。
同一个 PDB 的所有 chain 应该只出现在 train/val/test 的其中一个 split。
如果 train 和 test 共享 PDB，说明可能有 data leakage。

为什么要检查重复 sequence？
即使 PDB 不重复，也可能出现完全相同的 amino acid sequence。
如果完全相同的 sequence 同时出现在 train/test，评估也可能偏乐观。

运行命令：
    python scripts/check_processed_dataset.py
"""

from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "val", "test"]


def load_split(split_name: str) -> pd.DataFrame:
    """读取一个 split 的 CSV。"""

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
    """打印单个 split 的基本统计。"""

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
    """检查 train/val/test 之间某一列是否有重叠。

    column_name 可以是：
    - pdb：检查是否共享 PDB
    - sequence：检查是否共享完全相同的 amino acid sequence
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
    """检查 train/val/test 三个 CSV。"""

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
