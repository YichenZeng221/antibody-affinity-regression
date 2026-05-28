"""Inspect the TDC Protein_SAbDab antibody-antigen affinity dataset.

中文人话说明：
这个脚本只做数据探索，不训练模型，也不合并到现有 clean_v2。

目标：
1. 从 TDC 读取 Protein_SAbDab antibody-antigen affinity dataset。
2. 看列名、样本数、target 格式、sequence 长度。
3. 检查默认 random split 是否有 antibody / antigen / pair overlap。
4. 保存 TDC 原始 split 和 inspection_report.json，方便之后再决定要不要接入项目。

如果你的环境没有安装 PyTDC，运行时会提示：
    python -m pip install PyTDC
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "tdc_antibody_affinity"


def import_tdc_antibody_affinity():
    """Import TDC AntibodyAff with a beginner-friendly error message."""

    try:
        from tdc.multi_pred import AntibodyAff
    except ImportError as error:
        print("PyTDC is not installed in this Python environment.")
        print("Install command:")
        print("  python -m pip install PyTDC")
        print()
        print(f"Original import error: {error}")
        return None

    return AntibodyAff


def normalize_split_name(split_name: str) -> str:
    """Normalize TDC split names for output filenames."""

    if split_name == "valid":
        return "val"
    return split_name


def find_target_column(dataframe: pd.DataFrame) -> str | None:
    """Guess target column name.

    TDC datasets often use "Y" as the target column, but we keep this robust
    in case the dataset uses a more descriptive name.
    """

    candidates = ["Y", "y", "target", "Target", "affinity", "Affinity", "Kd", "KD"]
    for column_name in candidates:
        if column_name in dataframe.columns:
            return column_name
    return None


def find_sequence_column(dataframe: pd.DataFrame, exact_candidates: list[str], keywords: list[str]) -> str | None:
    """Find a likely sequence column.

    中文人话说明：
    TDC 这个数据里有：
    - Antibody_ID：PDB ID，不是序列
    - Antibody：真正的 heavy/light sequence
    - Antigen_ID：抗原名字/ID，不是序列
    - Antigen：真正的 antigen sequence

    所以我们必须优先找精确列名 Antibody / Antigen，
    不能简单看到 "antibody" 就选 Antibody_ID。
    """

    for column_name in exact_candidates:
        if column_name in dataframe.columns:
            return column_name

    lower_to_original = {column.lower(): column for column in dataframe.columns}
    for lower_name, original_name in lower_to_original.items():
        if lower_name.endswith("_id") or lower_name.endswith("id"):
            continue
        if any(keyword in lower_name for keyword in keywords):
            return original_name
    return None


def describe_numeric(series: pd.Series) -> dict:
    """Return JSON-friendly numeric stats."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}

    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def describe_sequence_lengths(dataframe: pd.DataFrame, column_name: str | None) -> dict:
    """Return min/max/mean sequence length stats for one sequence column."""

    if column_name is None or column_name not in dataframe.columns:
        return {"column": column_name, "count": 0, "min": None, "max": None, "mean": None}

    lengths = dataframe[column_name].astype(str).str.len()
    return {
        "column": column_name,
        "count": int(len(lengths)),
        "min": int(lengths.min()) if len(lengths) else None,
        "max": int(lengths.max()) if len(lengths) else None,
        "mean": float(lengths.mean()) if len(lengths) else None,
    }


def overlap_count(first: pd.DataFrame, second: pd.DataFrame, columns: list[str]) -> int:
    """Count exact overlap for one column or a combined antibody+antigen key."""

    missing_columns = [column for column in columns if column not in first.columns or column not in second.columns]
    if missing_columns:
        return 0

    first_keys = set(first[columns].astype(str).agg("||".join, axis=1))
    second_keys = set(second[columns].astype(str).agg("||".join, axis=1))
    return len(first_keys & second_keys)


def guess_target_scale(target_stats: dict) -> str:
    """Give a simple guess: raw affinity or already log-scaled."""

    minimum = target_stats["min"]
    maximum = target_stats["max"]

    if minimum is None or maximum is None:
        return "Could not infer target scale because target values are missing/non-numeric."

    if minimum > 0 and maximum <= 1:
        return "Target looks like small positive raw affinity values. Consider -log10 transform."

    if minimum > 0 and maximum < 1e-3:
        return "Target looks like very small raw Kd/affinity values. Consider -log10 transform."

    if 3 <= minimum <= 12 and 3 <= maximum <= 15:
        return "Target looks like it may already be log-scale, roughly in a 4-12 style range."

    if minimum > 0 and maximum > 100:
        return "Target looks like raw affinity or mixed units, not log-scale. Inspect units before modeling."

    return "Target scale is ambiguous. Inspect TDC documentation and a few raw values before transforming."


def print_split_overview(split: dict[str, pd.DataFrame], target_column: str | None, antibody_column: str | None, antigen_column: str | None) -> None:
    """Print human-readable split summary."""

    print("Split keys:")
    print(f"  {list(split.keys())}")
    print()

    for split_name, dataframe in split.items():
        print("=" * 80)
        print(f"Split: {split_name}")
        print(f"Rows: {len(dataframe)}")
        print(f"Columns: {list(dataframe.columns)}")
        print()
        print("First 5 rows:")
        print(dataframe.head(5).to_string(index=False))
        print()

        if target_column is not None:
            print(f"Target column: {target_column}")
            print(f"Target stats: {describe_numeric(dataframe[target_column])}")

        antibody_lengths = describe_sequence_lengths(dataframe, antibody_column)
        antigen_lengths = describe_sequence_lengths(dataframe, antigen_column)
        print(f"Antibody sequence length stats: {antibody_lengths}")
        print(f"Antigen sequence length stats: {antigen_lengths}")
        print()


def save_raw_splits(split: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Save TDC raw split CSVs under data/external."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for split_name, dataframe in split.items():
        normalized_name = normalize_split_name(split_name)
        output_path = OUTPUT_DIR / f"raw_{normalized_name}.csv"
        dataframe.to_csv(output_path, index=False)
        saved_paths[split_name] = str(output_path.relative_to(PROJECT_ROOT))

    return saved_paths


def main() -> None:
    """Load TDC dataset, inspect it, and save raw split/report."""

    AntibodyAff = import_tdc_antibody_affinity()
    if AntibodyAff is None:
        sys.exit(1)

    print("Loading TDC AntibodyAff(name='Protein_SAbDab') ...")
    data = AntibodyAff(name="Protein_SAbDab")
    split = data.get_split()

    # TDC default uses "valid"; the user-facing project often says "val".
    train = split.get("train")
    valid = split.get("valid", split.get("val"))
    test = split.get("test")

    first_split = next(iter(split.values()))
    target_column = find_target_column(first_split)
    antibody_column = find_sequence_column(first_split, ["Antibody", "antibody"], ["antibody", "ab"])
    antigen_column = find_sequence_column(first_split, ["Antigen", "antigen"], ["antigen", "ag"])

    print()
    print(f"Guessed target column: {target_column}")
    print(f"Guessed antibody column: {antibody_column}")
    print(f"Guessed antigen column: {antigen_column}")
    print()

    print_split_overview(split, target_column, antibody_column, antigen_column)

    full_data = pd.concat(split.values(), ignore_index=True)
    target_stats = describe_numeric(full_data[target_column]) if target_column else {}
    target_scale_guess = guess_target_scale(target_stats) if target_column else "No target column found."

    overlap_report = {}
    if train is not None and test is not None and antibody_column and antigen_column:
        overlap_report = {
            "train_test_antibody_overlap": overlap_count(train, test, [antibody_column]),
            "train_test_antigen_overlap": overlap_count(train, test, [antigen_column]),
            "train_test_antibody_antigen_pair_overlap": overlap_count(
                train,
                test,
                [antibody_column, antigen_column],
            ),
        }

    saved_paths = save_raw_splits(split)

    report = {
        "dataset_name": "Protein_SAbDab",
        "split_keys": list(split.keys()),
        "split_sizes": {split_name: int(len(dataframe)) for split_name, dataframe in split.items()},
        "total_rows": int(sum(len(dataframe) for dataframe in split.values())),
        "columns_by_split": {split_name: list(dataframe.columns) for split_name, dataframe in split.items()},
        "target_column": target_column,
        "target_stats_all_splits": target_stats,
        "target_scale_guess": target_scale_guess,
        "antibody_column": antibody_column,
        "antigen_column": antigen_column,
        "antibody_length_stats_all_splits": describe_sequence_lengths(full_data, antibody_column),
        "antigen_length_stats_all_splits": describe_sequence_lengths(full_data, antigen_column),
        "overlap_report": overlap_report,
        "saved_raw_split_paths": saved_paths,
    }

    report_path = OUTPUT_DIR / "inspection_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("=" * 80)
    print("Overall report")
    print(f"Total rows: {report['total_rows']}")
    print(f"Split sizes: {report['split_sizes']}")
    print(f"Target stats all splits: {target_stats}")
    print(f"Target scale guess: {target_scale_guess}")
    print(f"Overlap report: {overlap_report}")
    print(f"Saved raw split CSVs: {saved_paths}")
    print(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
