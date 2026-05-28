"""Build a cleaner Stage 1 affinity regression dataset.

中文人话说明：
这个脚本不重新解析 PDB，也不重新训练模型。
它只从已经生成好的 sequence_only CSV 开始，做一版更干净的数据集：

    data/processed_affinity/sequence_only/
        train.csv
        val.csv
        test.csv

清洗后输出到新的目录：

    data/processed_affinity/clean_v2/

为什么要做 clean_v2？
我们发现当前模型基本在猜平均值。一个很常见的原因是：
1. 数据量小；
2. 重复样本多；
3. metadata 里有可疑值；
4. train/test split 虽然 PDB 不重叠，但 sequence 仍有少量重叠。

所以这一版先做简单、透明、适合初学者理解的数据清洗。
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sequence_only"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "clean_v2"
TARGET_COLUMN = "neg_log10_affinity"
SEED = 42


def normalize_affinity_method(value: object) -> tuple[str, bool]:
    """Normalize affinity_method and flag suspicious values.

    中文人话说明：
    正常 method 应该像 SPR / ITC / Other。
    如果 method 是纯数字，比如 18724939，它更像 PMID，不像实验方法。
    这种行先标记为 suspicious，clean_v2 暂时排除。
    """

    text = str(value).strip()
    upper_text = text.upper()

    if upper_text in {"", "NA", "NAN", "NONE"}:
        return "UNKNOWN", True

    # PMID 通常是一串数字。这里简单判断：只要全是数字，就当作可疑 method。
    if re.fullmatch(r"\d+", upper_text):
        return upper_text, True

    if upper_text == "OTHER":
        return "OTHER", False

    if upper_text in {"SPR", "ITC"}:
        return upper_text, False

    # 第一版只保留我们明确认识的 method。
    # 其他奇怪字符串先当作 suspicious，避免 silent data noise。
    return upper_text, True


def load_sequence_only_data() -> pd.DataFrame:
    """Load original train/val/test and remember original_split.

    中文人话说明：
    为什么先合并再清洗？
    因为重复样本可能分别出现在 train/val/test 中。
    如果只在每个 split 内部去重，就看不到跨 split 的重复。

    original_split 记录这条样本原来属于 train/val/test 哪一份。
    后面 clean_v2 会重新 split，但 original_split 可以帮助我们回溯来源。
    """

    frames = []
    for split_name in ["train", "val", "test"]:
        path = INPUT_DIR / f"{split_name}.csv"
        dataframe = pd.read_csv(path)
        dataframe["original_split"] = split_name
        frames.append(dataframe)

    return pd.concat(frames, ignore_index=True)


def add_excluded(excluded_records: list[pd.DataFrame], rows: pd.DataFrame, reason: str) -> None:
    """Add excluded rows with an exclusion_reason column.

    excluded_records.csv 的意义：
    不只是把数据扔掉，而是记录“为什么扔掉”。
    这样以后你可以检查清洗规则是否太严格。
    """

    if len(rows) == 0:
        return

    excluded = rows.copy()
    excluded["exclusion_reason"] = reason
    excluded_records.append(excluded)


def remove_invalid_rows(dataframe: pd.DataFrame, excluded_records: list[pd.DataFrame]) -> pd.DataFrame:
    """Remove suspicious method, bad target, and missing sequence rows.

    suspicious_affinity_method 是指 method 看起来不像实验方法。
    例如纯数字更像 PMID，可能说明 raw metadata 某些列错位或内容不干净。
    Stage 1 先保守排除，避免把明显可疑行混进训练。
    """

    cleaned = dataframe.copy()

    method_results = cleaned["affinity_method"].apply(normalize_affinity_method)
    cleaned["affinity_method_normalized"] = method_results.apply(lambda item: item[0])
    cleaned["suspicious_method"] = method_results.apply(lambda item: item[1])

    suspicious_rows = cleaned[cleaned["suspicious_method"]].copy()
    add_excluded(excluded_records, suspicious_rows, "suspicious_affinity_method")
    cleaned = cleaned[~cleaned["suspicious_method"]].copy()

    numeric_target = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    bad_target_rows = cleaned[numeric_target.isna()].copy()
    add_excluded(excluded_records, bad_target_rows, "missing_or_non_numeric_target")
    cleaned = cleaned[~numeric_target.isna()].copy()
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")

    required_sequence_columns = ["heavy_sequence", "light_sequence", "antigen_sequence"]
    missing_sequence_mask = pd.Series(False, index=cleaned.index)
    for column_name in required_sequence_columns:
        text = cleaned[column_name].astype(str).str.strip()
        missing_sequence_mask |= text.isin(["", "NA", "NaN", "nan", "None"])

    missing_sequence_rows = cleaned[missing_sequence_mask].copy()
    add_excluded(excluded_records, missing_sequence_rows, "missing_sequence")
    cleaned = cleaned[~missing_sequence_mask].copy()

    # 后续代码统一使用 normalized method，但保留原始 affinity_method 也有助于回溯。
    cleaned["affinity_method"] = cleaned["affinity_method_normalized"]
    cleaned = cleaned.drop(columns=["affinity_method_normalized", "suspicious_method"])

    return cleaned


def deduplicate_triplets(dataframe: pd.DataFrame, excluded_records: list[pd.DataFrame]) -> pd.DataFrame:
    """Deduplicate exact heavy+light+antigen triplets.

    中文人话说明：
    如果三条 sequence 完全一样，模型看到的信息也完全一样。
    重复保留会让数据集看起来比实际更大。

    如果 target 一致：保留第一条，其他重复行排除。
    如果 target 冲突：这组数据不可信，整组排除。
    """

    triplet_columns = ["heavy_sequence", "light_sequence", "antigen_sequence"]
    kept_rows = []

    for _, group in dataframe.groupby(triplet_columns, sort=False, dropna=False):
        targets = group[TARGET_COLUMN].astype(float)
        target_range = targets.max() - targets.min()

        if len(group) == 1:
            kept_rows.append(group.iloc[[0]])
            continue

        if target_range <= 1e-6:
            kept_rows.append(group.iloc[[0]])
            duplicate_rows = group.iloc[1:].copy()
            add_excluded(excluded_records, duplicate_rows, "duplicate_triplet_removed")
        else:
            add_excluded(excluded_records, group.copy(), "conflicting_duplicate_target")

    if not kept_rows:
        return dataframe.iloc[0:0].copy()

    return pd.concat(kept_rows, ignore_index=True)


def pdb_level_split(dataframe: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Split by PDB ID so one PDB cannot appear in multiple splits.

    clean_v2 重新 split，而不是沿用旧 split。
    因为我们合并、去重、排除了样本，旧 split 的比例和统计已经变了。
    """

    unique_pdbs = pd.Series(dataframe["pdb"].astype(str).unique()).sample(frac=1, random_state=seed).tolist()
    total_pdbs = len(unique_pdbs)

    train_count = int(total_pdbs * 0.8)
    val_count = int(total_pdbs * 0.1)

    # 如果数据很小，也尽量留一点 val/test。
    if total_pdbs >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        if train_count + val_count >= total_pdbs:
            val_count = 1
            train_count = total_pdbs - 2

    train_pdbs = set(unique_pdbs[:train_count])
    val_pdbs = set(unique_pdbs[train_count : train_count + val_count])
    test_pdbs = set(unique_pdbs[train_count + val_count :])

    return {
        "train": dataframe[dataframe["pdb"].astype(str).isin(train_pdbs)].copy(),
        "val": dataframe[dataframe["pdb"].astype(str).isin(val_pdbs)].copy(),
        "test": dataframe[dataframe["pdb"].astype(str).isin(test_pdbs)].copy(),
    }


def overlap_count(first: pd.DataFrame, second: pd.DataFrame, columns: list[str]) -> int:
    """Count overlap between two dataframes for one column or a combined key."""

    first_keys = set(first[columns].astype(str).agg("||".join, axis=1))
    second_keys = set(second[columns].astype(str).agg("||".join, axis=1))
    return len(first_keys & second_keys)


def split_overlap_report(splits: dict[str, pd.DataFrame]) -> dict:
    """Report train/val/test overlap checks."""

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    checks = {
        "pdb": ["pdb"],
        "heavy_sequence": ["heavy_sequence"],
        "light_sequence": ["light_sequence"],
        "antigen_sequence": ["antigen_sequence"],
        "heavy_light_pair": ["heavy_sequence", "light_sequence"],
        "heavy_light_antigen_triplet": ["heavy_sequence", "light_sequence", "antigen_sequence"],
    }

    report = {}
    for first_name, second_name in pairs:
        pair_key = f"{first_name}_vs_{second_name}"
        report[pair_key] = {}
        for check_name, columns in checks.items():
            report[pair_key][check_name] = overlap_count(splits[first_name], splits[second_name], columns)

    return report


def target_stats(dataframe: pd.DataFrame) -> dict:
    """Return JSON-friendly target stats."""

    values = dataframe[TARGET_COLUMN].astype(float)
    return {
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std()) if len(values) else None,
    }


def duplicate_triplet_count(dataframe: pd.DataFrame) -> int:
    """Count rows involved in duplicated exact triplets."""

    triplet_columns = ["heavy_sequence", "light_sequence", "antigen_sequence"]
    return int(dataframe.duplicated(subset=triplet_columns, keep=False).sum())


def build_dataset_report(name: str, dataframe: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> dict:
    """Build report for all_methods or spr_only.

    cleaning_report.json 的意义：
    让这次清洗过程可复查、可记录。
    里面包含剩余样本数、method 分布、target 分布、split overlap 等。
    """

    return {
        "name": name,
        "rows": int(len(dataframe)),
        "duplicate_triplet_rows": duplicate_triplet_count(dataframe),
        "affinity_method_counts": dataframe["affinity_method"].value_counts(dropna=False).to_dict(),
        "antigen_type_counts": dataframe["antigen_type"].value_counts(dropna=False).to_dict(),
        "target_stats": target_stats(dataframe),
        "split_sizes": {split: int(len(df)) for split, df in splits.items()},
        "unique_pdb_counts": {split: int(df["pdb"].nunique()) for split, df in splits.items()},
        "split_overlap_check": split_overlap_report(splits),
    }


def write_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write train/val/test CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, dataframe in splits.items():
        dataframe.to_csv(output_dir / f"{split_name}.csv", index=False)


def print_split_summary(name: str, report: dict) -> None:
    """Print the most important report fields to terminal."""

    print("=" * 80)
    print(f"{name}")
    print(f"Rows: {report['rows']}")
    print(f"Split sizes: {report['split_sizes']}")
    print(f"Unique PDB counts: {report['unique_pdb_counts']}")
    print("Train/test overlap:")
    print(report["split_overlap_check"]["train_vs_test"])
    print(f"Target stats: {report['target_stats']}")


def main() -> None:
    """Build clean_v2 all_methods and spr_only datasets."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded_records: list[pd.DataFrame] = []
    original = load_sequence_only_data()

    original_method_counts = original["affinity_method"].value_counts(dropna=False).to_dict()
    original_antigen_type_counts = original["antigen_type"].value_counts(dropna=False).to_dict()
    original_duplicate_triplet_rows = duplicate_triplet_count(original)

    cleaned = remove_invalid_rows(original, excluded_records)
    all_methods = deduplicate_triplets(cleaned, excluded_records)

    allowed_methods = {"SPR", "ITC", "OTHER"}
    unexpected_rows = all_methods[~all_methods["affinity_method"].isin(allowed_methods)].copy()
    add_excluded(excluded_records, unexpected_rows, "unsupported_affinity_method")
    all_methods = all_methods[all_methods["affinity_method"].isin(allowed_methods)].copy()

    # all_methods：保留 SPR / ITC / OTHER，样本稍多，但 assay method 更混杂。
    # spr_only：只保留 SPR，实验方法更一致，但样本更少。
    spr_only = all_methods[all_methods["affinity_method"] == "SPR"].copy()

    all_methods_splits = pdb_level_split(all_methods)
    spr_only_splits = pdb_level_split(spr_only)

    write_splits(all_methods_splits, OUTPUT_DIR / "all_methods")
    write_splits(spr_only_splits, OUTPUT_DIR / "spr_only")

    if excluded_records:
        excluded = pd.concat(excluded_records, ignore_index=True)
    else:
        excluded = original.iloc[0:0].copy()
        excluded["exclusion_reason"] = []
    excluded.to_csv(OUTPUT_DIR / "excluded_records.csv", index=False)

    report = {
        "original_total_rows": int(len(original)),
        "excluded_counts_by_reason": excluded["exclusion_reason"].value_counts(dropna=False).to_dict(),
        "duplicate_triplet_rows_before_cleaning": int(original_duplicate_triplet_rows),
        "duplicate_triplet_rows_after_all_methods_cleaning": duplicate_triplet_count(all_methods),
        "affinity_method_counts_before": original_method_counts,
        "affinity_method_counts_after_all_methods": all_methods["affinity_method"].value_counts(dropna=False).to_dict(),
        "antigen_type_counts_before": original_antigen_type_counts,
        "antigen_type_counts_after_all_methods": all_methods["antigen_type"].value_counts(dropna=False).to_dict(),
        "all_methods": build_dataset_report("all_methods", all_methods, all_methods_splits),
        "spr_only": build_dataset_report("spr_only", spr_only, spr_only_splits),
    }

    with open(OUTPUT_DIR / "cleaning_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Clean v2 dataset built successfully.")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Excluded records: {len(excluded)}")
    print(f"Excluded counts by reason: {report['excluded_counts_by_reason']}")
    print()
    print_split_summary("all_methods", report["all_methods"])
    print_split_summary("spr_only", report["spr_only"])
    print()
    print("Files written:")
    print(f"  {OUTPUT_DIR / 'all_methods' / 'train.csv'}")
    print(f"  {OUTPUT_DIR / 'all_methods' / 'val.csv'}")
    print(f"  {OUTPUT_DIR / 'all_methods' / 'test.csv'}")
    print(f"  {OUTPUT_DIR / 'spr_only' / 'train.csv'}")
    print(f"  {OUTPUT_DIR / 'spr_only' / 'val.csv'}")
    print(f"  {OUTPUT_DIR / 'spr_only' / 'test.csv'}")
    print(f"  {OUTPUT_DIR / 'cleaning_report.json'}")
    print(f"  {OUTPUT_DIR / 'excluded_records.csv'}")


if __name__ == "__main__":
    main()
