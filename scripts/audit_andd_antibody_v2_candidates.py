"""Audit ANDD antibody-only expanded affinity v2 candidates.

:
,, train/val/test split

 ANDD  antibody candidates:
heavy_sequence + light_sequence + antigen_sequence + experimental Kd

 flags  CSV  summary , rows ,
 rows , rows  v2 dataset 
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_CSV = (
    ROOT
    / "data/processed_affinity/expanded_affinity_dataset_v2_candidates/"
    / "expanded_affinity_antibody_v2_candidates.csv"
)
OVERLAP_CSV = ROOT / "outputs/data_expansion/ANDD_v2_candidates/overlap_with_unified_no_high_risk.csv"
UNIFIED_DIR = ROOT / "data/processed_affinity/unified_ablation_datasets/unified_no_high_risk"

OUTPUT_DIR = ROOT / "outputs/data_expansion/ANDD_antibody_v2_audit"
PROCESSED_DIR = ROOT / "data/processed_affinity/expanded_affinity_dataset_v2_candidates"

AUDITED_CSV = "expanded_affinity_antibody_v2_audited_flags.csv"
REPORT_MD = "antibody_v2_quality_audit_report.md"
FLAG_SUMMARY_CSV = "antibody_v2_flag_summary.csv"
SOURCE_SUMMARY_CSV = "antibody_v2_source_summary.csv"
KD_SUMMARY_CSV = "antibody_v2_kd_distribution_summary.csv"

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
CDR_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]


def present(value) -> bool:
    """"""

    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"na", "n/a", "nan", "none", "\\", "unknown"}


def to_float(value):
    """ float; None"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def bool_value(value) -> bool:
    """CSV  True/False , bool"""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def sequence_issue_flags(sequence: str, kind: str) -> list[str]:
    """

    ,
    """

    flags = []
    if not present(sequence):
        return [f"missing_{kind}_sequence"]

    seq = str(sequence).strip().upper()
    length = len(seq)
    bad_chars = sorted(set(seq) - STANDARD_AA)
    if bad_chars:
        flags.append(f"nonstandard_{kind}_aa")

    if kind in {"heavy", "light"}:
        if length < 70:
            flags.append(f"short_{kind}_sequence")
        if length > 350:
            flags.append(f"long_{kind}_sequence")
    else:
        if length < 20:
            flags.append("short_antigen_sequence")
        if length > 1200:
            flags.append("long_antigen_sequence")
    return flags


def numeric_summary(values: pd.Series) -> dict:
    """"""

    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) == 0:
        return {"count": 0, "min": "", "q25": "", "median": "", "mean": "", "q75": "", "max": "", "std": ""}
    return {
        "count": int(len(clean)),
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
    }


def length_summary(df: pd.DataFrame, column: str) -> dict:
    """"""

    lengths = df[column].fillna("").astype(str).str.len()
    return numeric_summary(lengths)


def write_csv_to_outputs(filename: str, rows: list[dict], fieldnames: list[str]) -> None:
    """ outputs/  data/processed_affinity/,"""

    for directory in [OUTPUT_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def load_unified_splits() -> pd.DataFrame:
    """ unified_no_high_risk, overlap  test antigen overlap"""

    frames = []
    for split in ["train", "val", "test"]:
        path = UNIFIED_DIR / f"{split}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["unified_split"] = split
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def source_summary(df: pd.DataFrame) -> list[dict]:
    """ source  row count  Kd/target """

    rows = []
    for source, group in df.groupby("source", dropna=False):
        kd = numeric_summary(group["affinity_kd_m"])
        target = numeric_summary(group["neg_log10_affinity_candidate"])
        rows.append(
            {
                "source": source,
                "rows": len(group),
                "kd_min": kd["min"],
                "kd_median": kd["median"],
                "kd_mean": kd["mean"],
                "kd_max": kd["max"],
                "target_min": target["min"],
                "target_median": target["median"],
                "target_mean": target["mean"],
                "target_max": target["max"],
                "extreme_kd_rows": int(group["flag_extreme_kd"].sum()),
                "sequence_issue_rows": int(group["flag_sequence_issue"].sum()),
                "duplicate_rows": int(group["flag_duplicate"].sum()),
                "antigen_overlap_rows": int(group["flag_antigen_overlap"].sum()),
                "keep_safe_rows": int(group["keep_safe"].sum()),
            }
        )
    return sorted(rows, key=lambda item: item["rows"], reverse=True)


def kd_distribution_summary(df: pd.DataFrame) -> list[dict]:
    """ source  Kd  summary"""

    rows = []
    overall_kd = numeric_summary(df["affinity_kd_m"])
    overall_target = numeric_summary(df["neg_log10_affinity_candidate"])
    rows.append({"group": "all", **{f"kd_{k}": v for k, v in overall_kd.items()}, **{f"target_{k}": v for k, v in overall_target.items()}})
    for source, group in df.groupby("source", dropna=False):
        kd = numeric_summary(group["affinity_kd_m"])
        target = numeric_summary(group["neg_log10_affinity_candidate"])
        rows.append({"group": f"source:{source}", **{f"kd_{k}": v for k, v in kd.items()}, **{f"target_{k}": v for k, v in target.items()}})
    return rows


def cdr_length_summary(df: pd.DataFrame) -> list[str]:
    """ CDR  Markdown """

    lines = []
    for column in CDR_COLUMNS:
        if column not in df.columns:
            lines.append(f"- `{column}`: missing column")
            continue
        lengths = df[column].fillna("").astype(str).apply(lambda value: len(value.strip()) if present(value) else 0)
        unique_lengths = [int(value) for value in sorted(lengths.unique())[:20]]
        lines.append(
            f"- `{column}` length: min={lengths.min()}, median={lengths.median():.1f}, "
            f"mean={lengths.mean():.2f}, max={lengths.max()}, unique_lengths={unique_lengths}"
        )
    return lines


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CANDIDATE_CSV)
    unified = load_unified_splits()

    # unified  test antigen overlap : v2  test ,test antigen 
    unified_antigens = set(unified.get("antigen_sequence", pd.Series(dtype=str)).dropna().astype(str))
    unified_test_antigens = set(
        unified.loc[unified.get("unified_split", pd.Series(dtype=str)) == "test", "antigen_sequence"].dropna().astype(str)
    ) if not unified.empty and "antigen_sequence" in unified.columns else set()
    unified_triplets = set()
    if {"heavy_sequence", "light_sequence", "antigen_sequence"}.issubset(unified.columns):
        unified_triplets = set(
            unified["heavy_sequence"].fillna("").astype(str)
            + "||"
            + unified["light_sequence"].fillna("").astype(str)
            + "||"
            + unified["antigen_sequence"].fillna("").astype(str)
        )

    #  duplicate 
    triplet_keys = (
        df["heavy_sequence"].fillna("").astype(str)
        + "||"
        + df["light_sequence"].fillna("").astype(str)
        + "||"
        + df["antigen_sequence"].fillna("").astype(str)
    )
    df["audit_exact_triplet_key"] = triplet_keys
    df["audit_duplicate_exact_triplet_within_candidates"] = triplet_keys.duplicated(keep=False)
    df["audit_same_antigen_within_candidates"] = df["antigen_sequence"].fillna("").astype(str).duplicated(keep=False)

    audited_rows = []
    flag_counter = Counter()
    exclude_counter = Counter()

    for _, row in df.iterrows():
        kd = to_float(row.get("affinity_kd_m"))
        target = to_float(row.get("neg_log10_affinity_candidate"))

        flags = []
        exclude_reasons = []

        if kd is None:
            flags.append("invalid_kd")
            exclude_reasons.append("invalid_kd")
        elif kd <= 0:
            flags.append("nonpositive_kd")
            exclude_reasons.append("nonpositive_kd")
        else:
            if kd < 1e-12:
                flags.append("very_strong_or_tiny_kd")
            if kd > 1e-2:
                flags.append("very_weak_or_large_kd")

        if target is None:
            flags.append("invalid_neg_log10_affinity")
            exclude_reasons.append("invalid_neg_log10_affinity")

        heavy = str(row.get("heavy_sequence", "")).strip().upper()
        light = str(row.get("light_sequence", "")).strip().upper()
        antigen = str(row.get("antigen_sequence", "")).strip().upper()

        flags.extend(sequence_issue_flags(heavy, "heavy"))
        flags.extend(sequence_issue_flags(light, "light"))
        flags.extend(sequence_issue_flags(antigen, "antigen"))

        if heavy and light and heavy == light:
            flags.append("heavy_light_identical")

        if bool_value(row.get("duplicate_exact_triplet_within_ANDD")) or bool_value(row.get("audit_duplicate_exact_triplet_within_candidates")):
            flags.append("duplicate_exact_triplet_within_ANDD")
        if bool_value(row.get("overlap_exact_triplet")) or row["audit_exact_triplet_key"] in unified_triplets:
            flags.append("overlap_exact_triplet_with_unified")
        if antigen in unified_antigens or bool_value(row.get("overlap_antigen_sequence")):
            flags.append("overlap_antigen_sequence_with_unified")
        if antigen in unified_test_antigens:
            flags.append("overlap_current_test_antigen")
        if bool_value(row.get("overlap_source_id")):
            flags.append("overlap_source_id_with_unified")

        # CDR  ANDD , IMGT
        missing_cdrs = [column for column in CDR_COLUMNS if not present(row.get(column))]
        if missing_cdrs:
            flags.append("missing_cdr_field")

        flag_extreme_kd = any(flag in flags for flag in ["very_strong_or_tiny_kd", "very_weak_or_large_kd", "invalid_kd", "nonpositive_kd"])
        flag_sequence_issue = any(
            flag.startswith(("missing_", "short_", "long_", "nonstandard_")) or flag == "heavy_light_identical"
            for flag in flags
        )
        flag_duplicate = any("duplicate" in flag or "overlap_exact_triplet" in flag for flag in flags)
        flag_antigen_overlap = any(flag in flags for flag in ["overlap_antigen_sequence_with_unified", "overlap_current_test_antigen"])

        # conservative keep_safe:,/ split 
        keep_safe = not any([flag_extreme_kd, flag_sequence_issue, flag_duplicate, flag_antigen_overlap])

        if flag_extreme_kd:
            exclude_reasons.append("flag_extreme_kd")
        if flag_sequence_issue:
            exclude_reasons.append("flag_sequence_issue")
        if flag_duplicate:
            exclude_reasons.append("flag_duplicate")
        if flag_antigen_overlap:
            exclude_reasons.append("flag_antigen_overlap")

        for flag in flags:
            flag_counter[flag] += 1
        for reason in sorted(set(exclude_reasons)):
            exclude_counter[reason] += 1

        audited = row.to_dict()
        audited.update(
            {
                "heavy_len": len(heavy),
                "light_len": len(light),
                "antigen_len": len(antigen),
                "flag_extreme_kd": flag_extreme_kd,
                "flag_sequence_issue": flag_sequence_issue,
                "flag_duplicate": flag_duplicate,
                "flag_antigen_overlap": flag_antigen_overlap,
                "overlap_current_test_antigen": "overlap_current_test_antigen" in flags,
                "keep_safe": keep_safe,
                "audit_flags": ";".join(sorted(set(flags))),
                "exclude_reason": ";".join(sorted(set(exclude_reasons))),
                "needs_standard_imgt_cdr_extraction": True,
            }
        )
        audited_rows.append(audited)

    audited_df = pd.DataFrame(audited_rows)

    # Summary CSVs
    flag_summary_rows = []
    for flag, count in flag_counter.most_common():
        flag_summary_rows.append({"flag": flag, "rows": count})
    for reason, count in exclude_counter.most_common():
        flag_summary_rows.append({"flag": f"exclude_reason:{reason}", "rows": count})

    source_rows = source_summary(audited_df)
    kd_rows = kd_distribution_summary(audited_df)

    # Write CSV outputs.
    write_csv_to_outputs(AUDITED_CSV, audited_df.to_dict("records"), list(audited_df.columns))
    write_csv_to_outputs(FLAG_SUMMARY_CSV, flag_summary_rows, ["flag", "rows"])
    write_csv_to_outputs(
        SOURCE_SUMMARY_CSV,
        source_rows,
        [
            "source",
            "rows",
            "kd_min",
            "kd_median",
            "kd_mean",
            "kd_max",
            "target_min",
            "target_median",
            "target_mean",
            "target_max",
            "extreme_kd_rows",
            "sequence_issue_rows",
            "duplicate_rows",
            "antigen_overlap_rows",
            "keep_safe_rows",
        ],
    )
    write_csv_to_outputs(KD_SUMMARY_CSV, kd_rows, list(kd_rows[0].keys()))

    # Markdown report.
    total = len(audited_df)
    keep_safe_count = int(audited_df["keep_safe"].sum())
    cdr_cols_exist = all(column in audited_df.columns for column in CDR_COLUMNS)

    kd_summary = numeric_summary(audited_df["affinity_kd_m"])
    target_summary = numeric_summary(audited_df["neg_log10_affinity_candidate"])
    heavy_len = length_summary(audited_df, "heavy_sequence")
    light_len = length_summary(audited_df, "light_sequence")
    antigen_len = length_summary(audited_df, "antigen_sequence")

    report = [
        "# ANDD Antibody-only v2 Candidate Quality Audit",
        "",
        "## Scope",
        "",
        "This audit prepares antibody-only `expanded_affinity_dataset_v2` candidates for manual review.",
        "",
        "- No model was trained.",
        "- No final train/val/test split was created.",
        "- Existing `unified_no_high_risk` data was not modified.",
        "- Original candidate rows were not deleted; quality issues are stored as flags.",
        "",
        "## 1. Overall Counts",
        "",
        f"- Total antibody candidate rows: `{total}`",
        f"- Conservative `keep_safe` rows: `{keep_safe_count}`",
        f"- Rows with extreme Kd flags: `{int(audited_df['flag_extreme_kd'].sum())}`",
        f"- Rows with sequence issue flags: `{int(audited_df['flag_sequence_issue'].sum())}`",
        f"- Rows with duplicate / exact-triplet overlap flags: `{int(audited_df['flag_duplicate'].sum())}`",
        f"- Rows with antigen overlap flags: `{int(audited_df['flag_antigen_overlap'].sum())}`",
        f"- Rows overlapping current unified test antigens: `{int(audited_df['overlap_current_test_antigen'].sum())}`",
        "",
        "## 2. Kd Value Audit",
        "",
        f"- `affinity_kd_m`: count={kd_summary['count']}, min={kd_summary['min']:.4g}, median={kd_summary['median']:.4g}, mean={kd_summary['mean']:.4g}, max={kd_summary['max']:.4g}, std={kd_summary['std']:.4g}",
        f"- `neg_log10_affinity_candidate`: count={target_summary['count']}, min={target_summary['min']:.4g}, median={target_summary['median']:.4g}, mean={target_summary['mean']:.4g}, max={target_summary['max']:.4g}, std={target_summary['std']:.4g}",
        "",
        "Extreme Kd rows are not removed here. They are flagged for manual review because very large or very tiny Kd values can dominate regression loss and may reflect unit/provenance problems.",
        "",
        "## 3. Source / Provenance Audit",
        "",
        "| Source | Rows | Extreme Kd | Sequence Issue | Duplicate | Antigen Overlap | Keep Safe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_rows:
        report.append(
            f"| `{row['source']}` | {row['rows']} | {row['extreme_kd_rows']} | {row['sequence_issue_rows']} | "
            f"{row['duplicate_rows']} | {row['antigen_overlap_rows']} | {row['keep_safe_rows']} |"
        )

    report.extend(
        [
            "",
            "## 4. Sequence Quality Audit",
            "",
            f"- Heavy length: min={heavy_len['min']}, median={heavy_len['median']}, mean={heavy_len['mean']:.2f}, max={heavy_len['max']}",
            f"- Light length: min={light_len['min']}, median={light_len['median']}, mean={light_len['mean']:.2f}, max={light_len['max']}",
            f"- Antigen length: min={antigen_len['min']}, median={antigen_len['median']}, mean={antigen_len['mean']:.2f}, max={antigen_len['max']}",
            f"- Nonstandard heavy AA rows: `{flag_counter.get('nonstandard_heavy_aa', 0)}`",
            f"- Nonstandard light AA rows: `{flag_counter.get('nonstandard_light_aa', 0)}`",
            f"- Nonstandard antigen AA rows: `{flag_counter.get('nonstandard_antigen_aa', 0)}`",
            f"- Heavy/light identical rows: `{flag_counter.get('heavy_light_identical', 0)}`",
            "",
            "## 5. CDR Field Audit",
            "",
            f"- CDR columns present: `{cdr_cols_exist}`",
            "- Important: ANDD CDR fields should still be treated as source-provided annotations. For a formal benchmark, CDRs should be regenerated with one consistent method, preferably AbNumber + IMGT, before CDR-aware modeling.",
            "",
        ]
    )
    report.extend(cdr_length_summary(audited_df))
    report.extend(
        [
            "",
            "## 6. Duplicate / Leakage Audit",
            "",
            f"- Exact triplet duplicates within ANDD candidates: `{flag_counter.get('duplicate_exact_triplet_within_ANDD', 0)}`",
            f"- Exact triplet overlap with current unified_no_high_risk: `{flag_counter.get('overlap_exact_triplet_with_unified', 0)}`",
            f"- Antigen sequence overlap with current unified_no_high_risk: `{flag_counter.get('overlap_antigen_sequence_with_unified', 0)}`",
            f"- Antigen sequence overlap with current unified_no_high_risk test split: `{flag_counter.get('overlap_current_test_antigen', 0)}`",
            f"- Source ID overlap with current unified_no_high_risk: `{flag_counter.get('overlap_source_id_with_unified', 0)}`",
            "",
            "Antigen overlap rows are flagged but not deleted, because final antigen-group split design needs this information.",
            "",
            "## 7. Conservative Filtering Proposal",
            "",
            "`keep_safe=True` means the row currently has no extreme Kd, sequence issue, exact duplicate/triplet overlap, or antigen overlap flag.",
            "",
            "Suggested exclusions before building a formal antibody-only v2 training dataset:",
            "",
            "1. Exclude invalid/nonpositive Kd rows if any appear.",
            "2. Exclude or manually inspect extreme Kd rows (`Kd < 1e-12 M` or `Kd > 1e-2 M`).",
            "3. Exclude sequence issue rows with nonstandard amino acids, abnormal lengths, or identical heavy/light chains.",
            "4. Remove exact heavy+light+antigen duplicates and exact overlaps with current unified data.",
            "5. Keep antigen overlap information for split design; do not mix overlapping antigens across train/val/test.",
            "6. Re-run standard CDR extraction with AbNumber + IMGT before CDR-aware v2 experiments.",
            "",
            "## 8. Recommendation",
            "",
        ]
    )
    if keep_safe_count > 0:
        report.append(
            f"Yes, it is worth building a conservative antibody-only v2 next. Start from the `{keep_safe_count}` `keep_safe` rows, then do manual review of high-value flagged rows if more data is needed."
        )
    else:
        report.append("Do not build v2 yet; no rows passed the conservative `keep_safe` filter.")

    report.extend(
        [
            "",
            "Do not mix antibody and nanobody in this antibody-only v2. Nanobody should be a separate task because the input structure is different.",
            "",
            "## 9. Output Files",
            "",
            f"- `{OUTPUT_DIR / AUDITED_CSV}`",
            f"- `{OUTPUT_DIR / FLAG_SUMMARY_CSV}`",
            f"- `{OUTPUT_DIR / SOURCE_SUMMARY_CSV}`",
            f"- `{OUTPUT_DIR / KD_SUMMARY_CSV}`",
            f"- `{OUTPUT_DIR / REPORT_MD}`",
            "",
        ]
    )

    for directory in [OUTPUT_DIR, PROCESSED_DIR]:
        (directory / REPORT_MD).write_text("\n".join(report), encoding="utf-8")

    print(f"Total antibody candidates: {total}")
    print(f"keep_safe rows: {keep_safe_count}")
    print(f"extreme Kd rows: {int(audited_df['flag_extreme_kd'].sum())}")
    print(f"sequence issue rows: {int(audited_df['flag_sequence_issue'].sum())}")
    print(f"duplicate rows: {int(audited_df['flag_duplicate'].sum())}")
    print(f"antigen overlap rows: {int(audited_df['flag_antigen_overlap'].sum())}")
    print(f"Report saved to: {OUTPUT_DIR / REPORT_MD}")


if __name__ == "__main__":
    main()
