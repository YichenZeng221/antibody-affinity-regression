"""Build ablation datasets from unified_affinity_dataset_v1.

中文人话说明：
这个脚本只做数据版本切分，不训练模型。

我们从已经去重好的 ``all_unified_dedup.csv`` 出发，构建三个问题对照：
1. no_peptide：去掉 peptide antigen 风险标记；
2. no_high_risk：去掉 peptide / same H-L metadata / suspicious method；
3. no_less_strict：去掉 less-strict SAbDab 来源。

每个版本都重新按 antigen_sequence group split。
这样同一 antigen sequence 不会同时出现在 train/val/test，避免泄漏。
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "unified_affinity_dataset_v1"
    / "all_unified_dedup.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed_affinity" / "unified_ablation_datasets"
SPLITS = ["train", "val", "test"]
TRIPLET_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
SEED = 42

HIGH_RISK_FLAGS = {
    "peptide_antigen",
    "same_Hchain_Lchain_metadata",
    "suspicious_numeric_affinity_method",
}


def flag_set(value: object) -> set[str]:
    """Turn one pipe-separated risk_flags cell into a set."""

    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split("|") if part.strip()}


def has_flag(dataframe: pd.DataFrame, flag: str) -> pd.Series:
    """Return boolean mask for one risk flag."""

    return dataframe["risk_flags"].map(lambda value: flag in flag_set(value))


def has_any_flag(dataframe: pd.DataFrame, flags: set[str]) -> pd.Series:
    """Return boolean mask when any configured risk flag is present."""

    return dataframe["risk_flags"].map(lambda value: bool(flag_set(value) & flags))


def antigen_group_split(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Greedy 80/10/10 split that keeps each antigen sequence in one split."""

    antigen_sizes = dataframe.groupby("antigen_sequence").size().reset_index(name="count")
    antigen_sizes = antigen_sizes.sample(frac=1, random_state=SEED).reset_index(drop=True)
    target_sizes = {
        "train": len(dataframe) * 0.8,
        "val": len(dataframe) * 0.1,
        "test": len(dataframe) * 0.1,
    }
    split_antigens = {split_name: [] for split_name in SPLITS}
    split_sizes = {split_name: 0 for split_name in SPLITS}

    for _, row in antigen_sizes.iterrows():
        deficits = {
            split_name: target_sizes[split_name] - split_sizes[split_name]
            for split_name in SPLITS
        }
        split_name = max(deficits, key=deficits.get)
        split_antigens[split_name].append(row["antigen_sequence"])
        split_sizes[split_name] += int(row["count"])

    return {
        split_name: dataframe[dataframe["antigen_sequence"].isin(antigens)].copy()
        for split_name, antigens in split_antigens.items()
    }


def key_set(dataframe: pd.DataFrame, columns: list[str]) -> set[str]:
    """Build exact sequence keys for overlap checks."""

    if dataframe.empty:
        return set()
    return set(dataframe[columns].fillna("").astype(str).agg("||".join, axis=1))


def overlap_check(splits: dict[str, pd.DataFrame], columns: list[str]) -> dict[str, int]:
    """Count pairwise train/val/test overlaps for given key columns."""

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        f"{left}_vs_{right}": int(len(key_set(splits[left], columns) & key_set(splits[right], columns)))
        for left, right in pairs
    }


def numeric_stats(values: pd.Series) -> dict:
    """Summarize numeric target values for report."""

    series = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "count": int(len(series)),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std()),
    }


def split_target_stats(splits: dict[str, pd.DataFrame]) -> dict:
    """Summarize target distribution per split."""

    return {
        split_name: numeric_stats(frame["neg_log10_affinity"])
        for split_name, frame in splits.items()
    }


def source_counts(splits: dict[str, pd.DataFrame]) -> dict:
    """Show source composition after re-splitting."""

    return {
        split_name: {
            str(source): int(count)
            for source, count in frame["source"].value_counts(dropna=False).items()
        }
        for split_name, frame in splits.items()
    }


def risk_flag_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count rows for each risk flag remaining in a dataset version."""

    counts: dict[str, int] = {}
    all_flags = sorted({flag for value in dataframe["risk_flags"] for flag in flag_set(value)})
    for flag in all_flags:
        counts[flag] = int(has_flag(dataframe, flag).sum())
    counts["no_risk_flag"] = int(dataframe["risk_flags"].fillna("").astype(str).str.strip().eq("").sum())
    return counts


def dataset_versions(full: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str]]:
    """Create filtered ablation dataframes and explain each filter."""

    less_strict_mask = (
        full["source"].fillna("").astype(str).eq("SAbDab_Less_Strict")
        | has_flag(full, "less_strict_sabdab")
    )
    return {
        "unified_no_peptide": (
            full.loc[~has_flag(full, "peptide_antigen")].copy(),
            "Removed rows whose risk_flags contain peptide_antigen.",
        ),
        "unified_no_high_risk": (
            full.loc[~has_any_flag(full, HIGH_RISK_FLAGS)].copy(),
            "Removed peptide_antigen, same_Hchain_Lchain_metadata, and suspicious_numeric_affinity_method rows.",
        ),
        "unified_no_less_strict": (
            full.loc[~less_strict_mask].copy(),
            "Removed SAbDab_Less_Strict source rows and rows marked less_strict_sabdab.",
        ),
    }


def write_markdown(dataset_name: str, report: dict, output_dir: Path) -> None:
    """Write one readable dataset build report."""

    lines = [
        f"# {dataset_name} Ablation Dataset Report",
        "",
        "## Filter",
        "",
        f"- Input: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Rule: {report['filter_rule']}",
        f"- Rows kept: `{report['rows_after_filter']}` / `{report['input_rows']}`",
        f"- Rows removed: `{report['rows_removed']}`",
        "",
        "## Split",
        "",
        "- Split rule: antigen_sequence group split, target ratio 80/10/10, seed 42.",
        f"- Split sizes: `{report['split_sizes']}`",
        f"- antigen_sequence overlap: `{report['overlap_checks']['antigen_sequence']}`",
        f"- heavy+light+antigen triplet overlap: `{report['overlap_checks']['heavy_light_antigen_triplet']}`",
        f"- Target distribution: `{report['target_distribution_by_split']}`",
        f"- Source counts: `{report['source_counts_by_split']}`",
        "",
        "## Remaining Risk Flags",
        "",
        f"- `{report['remaining_risk_flags']}`",
        "",
        "## Output Files",
        "",
    ]
    lines.extend(f"- `{name}`: `{path}`" for name, path in report["output_files"].items())
    (output_dir / "processing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_one(dataset_name: str, dataframe: pd.DataFrame, filter_rule: str, input_rows: int) -> dict:
    """Build one filtered dataset directory."""

    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = dataframe.reset_index(drop=True)
    splits = antigen_group_split(dataframe)

    for split_name, frame in splits.items():
        frame.to_csv(output_dir / f"{split_name}.csv", index=False)

    antigen_overlap = overlap_check(splits, ["antigen_sequence"])
    triplet_overlap = overlap_check(splits, TRIPLET_COLUMNS)
    report = {
        "dataset_name": dataset_name,
        "input_rows": int(input_rows),
        "rows_after_filter": int(len(dataframe)),
        "rows_removed": int(input_rows - len(dataframe)),
        "filter_rule": filter_rule,
        "seed": SEED,
        "split_sizes": {split_name: int(len(frame)) for split_name, frame in splits.items()},
        "target_distribution_by_split": split_target_stats(splits),
        "source_counts_by_split": source_counts(splits),
        "remaining_risk_flags": risk_flag_counts(dataframe),
        "overlap_checks": {
            "antigen_sequence": antigen_overlap,
            "heavy_light_antigen_triplet": triplet_overlap,
        },
        "output_files": {
            split_name: str((output_dir / f"{split_name}.csv").relative_to(PROJECT_ROOT))
            for split_name in SPLITS
        },
    }
    (output_dir / "processing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(dataset_name, report, output_dir)
    return report


def main() -> None:
    """Build three new ablation dataset versions."""

    full = pd.read_csv(INPUT_PATH)
    reports = []
    for dataset_name, (dataframe, filter_rule) in dataset_versions(full).items():
        reports.append(build_one(dataset_name, dataframe, filter_rule, len(full)))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "ablation_dataset_build_summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Unified affinity ablation datasets built.")
    for report in reports:
        print(
            f"{report['dataset_name']}: rows={report['rows_after_filter']}, "
            f"splits={report['split_sizes']}, "
            f"antigen_overlap={report['overlap_checks']['antigen_sequence']}, "
            f"triplet_overlap={report['overlap_checks']['heavy_light_antigen_triplet']}"
        )
    print("No model training was run by this build script.")


if __name__ == "__main__":
    main()
