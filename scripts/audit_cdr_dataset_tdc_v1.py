"""Audit the TDC v1 CDR feature dataset before using it for experiments.

中文人话说明：
这个脚本只做数据检查，不训练模型。
它读取 CDR feature extraction 生成的 ``all_cdr.csv``，回答几个很实际的问题：

1. 这个 CSV 的列、split、target 和 sequence 长度是否还完整？
2. train/val/test 之间是否出现可能的 leakage？
3. 当前 CDR backend 到底是标准 annotation，还是 heuristic 粗切片？
4. heuristic 产生的 CDR 长度是否明显接近固定值？

CDR feature baseline 很容易被“看起来有 CDR 列”骗到。
所以在建模前先 audit 一遍，能避免把粗糙 fallback 当成正式特征。
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CDR_FEATURE_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "cdr_features"
INPUT_PATH = CDR_FEATURE_DIR / "all_cdr.csv"
JSON_REPORT_PATH = CDR_FEATURE_DIR / "cdr_dataset_audit.json"
MARKDOWN_REPORT_PATH = CDR_FEATURE_DIR / "cdr_dataset_audit.md"

EXPECTED_SPLITS = ["train", "val", "test"]
SEQUENCE_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
CDR_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]
CDR_LENGTH_COLUMNS = [f"{column}_len" for column in CDR_COLUMNS]
FOCUS_MISSING_COLUMNS = [
    *SEQUENCE_COLUMNS,
    "neg_log10_affinity",
    *CDR_COLUMNS,
    *CDR_LENGTH_COLUMNS,
]


def require_columns(dataframe: pd.DataFrame) -> None:
    """Stop early if the audit input is missing columns this script needs."""

    required_columns = {
        "split",
        "cdr_extract_status",
        "cdr_backend",
        "heavy_cdr_backend",
        "light_cdr_backend",
        "antibody_id",
        "antigen_id",
        "neg_log10_affinity",
        "heavy_len",
        "light_len",
        "antigen_len",
        *SEQUENCE_COLUMNS,
        *CDR_COLUMNS,
        *CDR_LENGTH_COLUMNS,
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{INPUT_PATH} is missing required columns: {sorted(missing_columns)}")


def is_missing(series: pd.Series) -> pd.Series:
    """Return a mask for real NaN values and blank string cells.

    Pandas 会把很多空白 CSV 单元格读成 NaN。
    有些字符串列也可能只是 ``""`` 或空格，所以这里一起算作 missing。
    """

    missing_mask = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        missing_mask = missing_mask | series.fillna("").astype(str).str.strip().eq("")
    return missing_mask


def count_values(series: pd.Series) -> dict[str, int]:
    """Return JSON-friendly value counts including missing cells."""

    display_series = series.copy()
    display_series = display_series.where(~is_missing(display_series), "<missing>")
    return {str(key): int(value) for key, value in display_series.value_counts(dropna=False).items()}


def numeric_summary(series: pd.Series) -> dict:
    """Return count, mean, spread, and range for one numeric column."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def split_numeric_summary(dataframe: pd.DataFrame) -> dict:
    """Summarize target and sequence lengths inside each split."""

    summaries = {}
    for split_name in EXPECTED_SPLITS:
        split_frame = dataframe[dataframe["split"] == split_name]
        summaries[split_name] = {
            "rows": int(len(split_frame)),
            "neg_log10_affinity": numeric_summary(split_frame["neg_log10_affinity"]),
            "antigen_len": numeric_summary(split_frame["antigen_len"]),
            "heavy_len": numeric_summary(split_frame["heavy_len"]),
            "light_len": numeric_summary(split_frame["light_len"]),
        }
    return summaries


def non_missing_set(series: pd.Series) -> set[str]:
    """Turn a column into a set while ignoring blank IDs/sequences."""

    usable = series[~is_missing(series)].astype(str)
    return set(usable.tolist())


def triplet_set(dataframe: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Build heavy+light+antigen sequence keys for overlap checks."""

    usable = dataframe.copy()
    for column_name in SEQUENCE_COLUMNS:
        usable = usable[~is_missing(usable[column_name])]
    return set(
        zip(
            usable["heavy_sequence"].astype(str),
            usable["light_sequence"].astype(str),
            usable["antigen_sequence"].astype(str),
        )
    )


def overlap_report(sets_by_split: dict[str, set]) -> dict:
    """Count overlap for train/val/test pairs.

    overlap count = 两个 split 里完全相同值的数量。
    对 antigen_sequence 和 triplet 来说，它能直接提醒我们是否破坏了 group split。
    """

    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        f"{left}_vs_{right}": int(len(sets_by_split[left] & sets_by_split[right]))
        for left, right in split_pairs
    }


def build_leakage_report(dataframe: pd.DataFrame) -> dict:
    """Check split overlap for IDs, antigen sequence, and full sequence triplet."""

    split_frames = {
        split_name: dataframe[dataframe["split"] == split_name]
        for split_name in EXPECTED_SPLITS
    }
    return {
        "antigen_sequence_overlap": overlap_report(
            {split_name: non_missing_set(frame["antigen_sequence"]) for split_name, frame in split_frames.items()}
        ),
        "heavy_light_antigen_triplet_overlap": overlap_report(
            {split_name: triplet_set(frame) for split_name, frame in split_frames.items()}
        ),
        "antibody_id_overlap": overlap_report(
            {split_name: non_missing_set(frame["antibody_id"]) for split_name, frame in split_frames.items()}
        ),
        "antigen_id_overlap": overlap_report(
            {split_name: non_missing_set(frame["antigen_id"]) for split_name, frame in split_frames.items()}
        ),
        "antigen_id_note": (
            "antigen_id overlap is reported for context. It is not automatically leakage "
            "when an ID/name can point to different antigen sequences."
        ),
    }


def cdr_length_audit(series: pd.Series) -> dict:
    """Describe one CDR length column and whether it looks nearly fixed."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {
            "unique_values": [],
            "unique_count": 0,
            "nonzero_unique_values": [],
            "nonzero_unique_count": 0,
            "std": None,
            "min": None,
            "max": None,
            "dominant_nonzero_value": None,
            "dominant_nonzero_share": None,
            "looks_nearly_fixed": False,
        }

    # Failed CDR extractions use length 0.  To inspect slicing behavior,
    # we also look at non-zero lengths from chains that produced a CDR.
    nonzero_values = values[values > 0]
    nonzero_counts = nonzero_values.value_counts()
    if len(nonzero_counts):
        dominant_value = float(nonzero_counts.index[0])
        dominant_share = float(nonzero_counts.iloc[0] / len(nonzero_values))
    else:
        dominant_value = None
        dominant_share = None

    nonzero_unique_values = sorted(float(value) for value in nonzero_values.unique())
    looks_nearly_fixed = bool(
        len(nonzero_values)
        and (len(nonzero_unique_values) <= 3 or (dominant_share is not None and dominant_share >= 0.9))
    )

    return {
        "unique_values": sorted(float(value) for value in values.unique()),
        "unique_count": int(values.nunique()),
        "nonzero_unique_values": nonzero_unique_values,
        "nonzero_unique_count": int(nonzero_values.nunique()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "dominant_nonzero_value": dominant_value,
        "dominant_nonzero_share": dominant_share,
        "looks_nearly_fixed": looks_nearly_fixed,
    }


def build_heuristic_report(dataframe: pd.DataFrame) -> dict:
    """Audit whether heuristic CDR lengths reveal fixed-index slicing."""

    lengths = {column_name: cdr_length_audit(dataframe[column_name]) for column_name in CDR_LENGTH_COLUMNS}
    backend_values = non_missing_set(dataframe["cdr_backend"])
    uses_heuristic = any("imgt_index_heuristic" in backend for backend in backend_values)
    cdr3_nearly_fixed = lengths["HCDR3_len"]["looks_nearly_fixed"] or lengths["LCDR3_len"]["looks_nearly_fixed"]
    heuristic_warning = bool(uses_heuristic and cdr3_nearly_fixed)

    warning_text = ""
    if heuristic_warning:
        warning_text = "heuristic likely fixed slicing, not valid standard CDR annotation"

    return {
        "uses_heuristic_backend": uses_heuristic,
        "cdr_length_summary": lengths,
        "hcdr3_len_looks_nearly_fixed": lengths["HCDR3_len"]["looks_nearly_fixed"],
        "lcdr3_len_looks_nearly_fixed": lengths["LCDR3_len"]["looks_nearly_fixed"],
        "heuristic_warning": heuristic_warning,
        "warning": warning_text,
    }


def missing_count_report(dataframe: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    """Count missing values for every column and important modeling columns."""

    all_missing = {column_name: int(is_missing(dataframe[column_name]).sum()) for column_name in dataframe.columns}
    focus_missing = {column_name: all_missing[column_name] for column_name in FOCUS_MISSING_COLUMNS}
    return all_missing, focus_missing


def base_dataset_is_complete(report: dict) -> bool:
    """Check columns needed before any model or CDR feature baseline."""

    essential_columns = [*SEQUENCE_COLUMNS, "neg_log10_affinity", "split"]
    return all(report["missing_values"]["focus_columns"].get(column_name, 0) == 0 for column_name in essential_columns)


def target_is_healthy(dataframe: pd.DataFrame) -> bool:
    """Return True when target values exist and are numeric."""

    target_values = pd.to_numeric(dataframe["neg_log10_affinity"], errors="coerce")
    return bool(target_values.notna().all() and len(target_values) > 0)


def expected_splits_present(dataframe: pd.DataFrame) -> bool:
    """Check that extraction kept train, val, and test split labels."""

    return set(EXPECTED_SPLITS).issubset(non_missing_set(dataframe["split"]))


def leakage_zero_for_group_split(leakage: dict) -> bool:
    """For this dataset, antigen sequence and full triplet overlap should be zero."""

    guarded_checks = [
        *leakage["antigen_sequence_overlap"].values(),
        *leakage["heavy_light_antigen_triplet_overlap"].values(),
    ]
    return all(count == 0 for count in guarded_checks)


def write_markdown_report(report: dict) -> None:
    """Write a short Markdown report that is easier to read than raw JSON."""

    split_lines = []
    for split_name, summary in report["split_health"].items():
        target = summary["neg_log10_affinity"]
        split_lines.append(
            f"| {split_name} | {summary['rows']} | {target['mean']:.4f} | {target['std']:.4f} | "
            f"{target['min']:.4f} | {target['max']:.4f} |"
        )

    leakage_lines = []
    for check_name in [
        "antigen_sequence_overlap",
        "heavy_light_antigen_triplet_overlap",
        "antibody_id_overlap",
        "antigen_id_overlap",
    ]:
        leakage_lines.append(f"- `{check_name}`: `{report['leakage'][check_name]}`")

    cdr_length_lines = []
    for column_name, summary in report["heuristic_cdr_audit"]["cdr_length_summary"].items():
        cdr_length_lines.append(
            f"| {column_name} | {summary['unique_count']} | {summary['nonzero_unique_values']} | "
            f"{summary['std']:.4f} | {summary['min']:.1f} | {summary['max']:.1f} | "
            f"{summary['looks_nearly_fixed']} |"
        )

    lines = [
        "# TDC v1 CDR Dataset Audit",
        "",
        "## Summary",
        "",
        f"- Input: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Shape: `{report['shape']}`",
        f"- Dataset base columns complete: `{report['conclusions']['dataset_complete']}`",
        f"- Split labels preserved: `{report['conclusions']['split_preserved']}`",
        f"- Target numeric and present: `{report['conclusions']['target_healthy']}`",
        f"- Antigen/triplet leakage guard is zero: `{report['conclusions']['group_split_leakage_zero']}`",
        f"- Heuristic CDR trusted for formal baseline: `{report['conclusions']['heuristic_cdr_trustworthy']}`",
        f"- Ready for simple CDR feature baseline: `{report['conclusions']['ready_for_simple_cdr_feature_baseline']}`",
        "",
        "## Basic Info",
        "",
        f"- Columns: `{report['columns']}`",
        f"- Split counts: `{report['split_counts']}`",
        f"- `cdr_extract_status` counts: `{report['cdr_extract_status_counts']}`",
        f"- `cdr_backend` counts: `{report['cdr_backend_counts']}`",
        f"- Heavy backend counts: `{report['heavy_cdr_backend_counts']}`",
        f"- Light backend counts: `{report['light_cdr_backend_counts']}`",
        "",
        "## Missing Values",
        "",
        "Important columns:",
        "",
    ]
    lines.extend(f"- `{column_name}`: {count}" for column_name, count in report["missing_values"]["focus_columns"].items())
    lines.extend(
        [
            "",
            "All-column missing counts are available in `cdr_dataset_audit.json`.",
            "",
            "## Split Health",
            "",
            "| split | rows | target mean | target std | target min | target max |",
            "|---|---:|---:|---:|---:|---:|",
            *split_lines,
            "",
            "Length summaries for `heavy_len`, `light_len`, and `antigen_len` are stored in JSON.",
            "",
            "## Leakage Checks",
            "",
            *leakage_lines,
            "",
            f"- Note: {report['leakage']['antigen_id_note']}",
            "",
            "## Heuristic CDR Audit",
            "",
            "The current heuristic fallback cuts raw sequence indices. If its non-zero CDR lengths have "
            "very few unique values, that is a warning sign that it is measuring the slice rule more than biology.",
            "",
            "| CDR length | unique count | non-zero unique values | std | min | max | nearly fixed |",
            "|---|---:|---|---:|---:|---:|---|",
            *cdr_length_lines,
            "",
        ]
    )
    if report["heuristic_cdr_audit"]["warning"]:
        lines.extend(
            [
                f"**Warning:** {report['heuristic_cdr_audit']['warning']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "- Split preservation and leakage checks can be healthy while heuristic CDR annotation is still not formal.",
            "- For a formal CDR-aware baseline, rerun extraction with standard `abnumber_anarci_imgt` coverage first.",
            "",
        ]
    )

    MARKDOWN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_report(dataframe: pd.DataFrame) -> dict:
    """Build the full audit dictionary before writing files."""

    all_missing, focus_missing = missing_count_report(dataframe)
    leakage = build_leakage_report(dataframe)
    heuristic_report = build_heuristic_report(dataframe)

    report = {
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "columns": list(dataframe.columns),
        "split_counts": count_values(dataframe["split"]),
        "cdr_extract_status_counts": count_values(dataframe["cdr_extract_status"]),
        "cdr_backend_counts": count_values(dataframe["cdr_backend"]),
        "heavy_cdr_backend_counts": count_values(dataframe["heavy_cdr_backend"]),
        "light_cdr_backend_counts": count_values(dataframe["light_cdr_backend"]),
        "missing_values": {
            "all_columns": all_missing,
            "focus_columns": focus_missing,
        },
        "split_health": split_numeric_summary(dataframe),
        "leakage": leakage,
        "heuristic_cdr_audit": heuristic_report,
    }

    dataset_complete = base_dataset_is_complete(report)
    split_preserved = expected_splits_present(dataframe)
    target_healthy = target_is_healthy(dataframe)
    group_split_leakage_zero = leakage_zero_for_group_split(leakage)
    heuristic_trustworthy = not heuristic_report["heuristic_warning"] and not heuristic_report["uses_heuristic_backend"]
    ready_for_baseline = all(
        [
            dataset_complete,
            split_preserved,
            target_healthy,
            group_split_leakage_zero,
            heuristic_trustworthy,
        ]
    )
    report["conclusions"] = {
        "dataset_complete": dataset_complete,
        "split_preserved": split_preserved,
        "target_healthy": target_healthy,
        "group_split_leakage_zero": group_split_leakage_zero,
        "heuristic_cdr_trustworthy": heuristic_trustworthy,
        "ready_for_simple_cdr_feature_baseline": ready_for_baseline,
    }
    return report


def print_human_summary(report: dict) -> None:
    """Print the audit conclusion in plain Chinese for terminal reading."""

    leakage = report["leakage"]
    conclusions = report["conclusions"]

    print("TDC v1 CDR feature dataset audit complete.")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(f"Dataset 基础字段是否完整: {conclusions['dataset_complete']}")
    print(f"Train/val/test split 是否仍然保留: {conclusions['split_preserved']}")
    print(f"Target neg_log10_affinity 是否都能正常读取: {conclusions['target_healthy']}")
    print("Leakage 检查:")
    print(f"  antigen_sequence overlap: {leakage['antigen_sequence_overlap']}")
    print(f"  heavy+light+antigen triplet overlap: {leakage['heavy_light_antigen_triplet_overlap']}")
    print(f"  antibody_id overlap: {leakage['antibody_id_overlap']}")
    print(f"  antigen_id overlap: {leakage['antigen_id_overlap']}")
    print(f"Heuristic CDR 是否可信: {conclusions['heuristic_cdr_trustworthy']}")
    if report["heuristic_cdr_audit"]["warning"]:
        print(f"  Warning: {report['heuristic_cdr_audit']['warning']}.")
    print(f"现在是否可以进入 simple CDR feature baseline: {conclusions['ready_for_simple_cdr_feature_baseline']}")
    if not conclusions["ready_for_simple_cdr_feature_baseline"]:
        print("建议：先拿标准 AbNumber/ANARCI IMGT CDR extraction 结果，再做正式 CDR baseline。")


def main() -> None:
    """Load all_cdr.csv, audit it, and save JSON + Markdown reports."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_PATH}. Run extract_cdr_features_tdc_v1.py first.")

    dataframe = pd.read_csv(INPUT_PATH)
    require_columns(dataframe)
    report = build_report(dataframe)

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_markdown_report(report)
    print_human_summary(report)


if __name__ == "__main__":
    main()
