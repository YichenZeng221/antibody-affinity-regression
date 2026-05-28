"""Build conservative ANDD expanded affinity v2 candidate tables.

:
, train/val/test split
 ANDD  Tier 1 experimental rows :

1. antibody candidates: heavy + light + antigen + experimental Kd
2. nanobody candidates: VHH/nanobody sequence + antigen + experimental Kd

 overlap/risk/provenance, v2 dataset 
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import pandas as pd

#  `python scripts/xxx.py` ,Python  scripts/  import path,
# , audit 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_andd_data_source import (
    ANDD_XLSX,
    CURRENT_UNIFIED_SPLIT_DIR,
    classify_format,
    find_column,
    is_predicted_row,
    is_present,
    is_sequence,
    normalize_text,
    quality_tier,
    read_xlsx_first_sheet,
    source_id,
    to_float,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs/data_expansion/ANDD_v2_candidates"
PROCESSED_DIR = PROJECT_ROOT / "data/processed_affinity/expanded_affinity_dataset_v2_candidates"

ANTIBODY_OUTPUT = "expanded_affinity_antibody_v2_candidates.csv"
NANOBODY_OUTPUT = "expanded_affinity_nanobody_v2_candidates.csv"
EXCLUDED_OUTPUT = "excluded_or_flagged_rows.csv"
OVERLAP_OUTPUT = "overlap_with_unified_no_high_risk.csv"
SUMMARY_OUTPUT = "expanded_affinity_v2_candidate_summary.md"

MISSING_VALUES = {"", "na", "n/a", "nan", "none", "\\", "unknown", "not_reported", "not reported"}


def load_current_unified() -> pd.DataFrame:
    """Load current unified_no_high_risk split files for overlap checks."""

    frames = []
    for split in ["train", "val", "test"]:
        path = CURRENT_UNIFIED_SPLIT_DIR / f"{split}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["split"] = split
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean_sequence(value: str) -> str:
    """Normalize amino-acid sequence text for candidate tables."""

    return normalize_text(value).replace(" ", "").replace("-", "")


def is_nanobody_format(molecule_format: str) -> bool:
    """Return True for the single-domain VHH/nanobody task."""

    return molecule_format == "VHH / nanobody"


def is_antibody_format(molecule_format: str) -> bool:
    """Return True for heavy+light antibody task.

    scFv and BJ/light-chain dimers are intentionally excluded from the main
    antibody candidate table because their input schema differs from normal
    paired heavy/light antibody examples.
    """

    return molecule_format == "antibody"


def neg_log10(value: float) -> float | None:
    """Convert Kd in M to -log10(Kd)."""

    if value is None or value <= 0 or pd.isna(value):
        return None
    return -math.log10(value)


def kd_risk_flags(kd_value: float | None) -> list[str]:
    """Flag extreme Kd values for human review."""

    flags = []
    if kd_value is None or kd_value <= 0 or pd.isna(kd_value):
        flags.append("invalid_kd")
        return flags
    target = neg_log10(kd_value)
    if target is not None and target < 3:
        flags.append("very_weak_or_large_kd")
    if target is not None and target > 12:
        flags.append("very_strong_or_tiny_kd")
    if kd_value > 1e-2:
        flags.append("kd_greater_than_1e-2_M")
    if kd_value < 1e-12:
        flags.append("kd_less_than_1e-12_M")
    return flags


def triplet_key(heavy: str, light: str, antigen: str) -> str:
    """Build exact heavy+light+antigen key."""

    return "||".join([heavy, light, antigen])


def nanobody_key(sequence: str, antigen: str) -> str:
    """Build exact nanobody+antigen key."""

    return "||".join([sequence, antigen])


def build_unified_overlap_sets(unified: pd.DataFrame) -> dict:
    """Collect current unified exact keys for overlap flagging."""

    if unified.empty:
        return {
            "antigens": set(),
            "antibody_triplets": set(),
            "ids": set(),
        }
    antigens = set(unified.get("antigen_sequence", pd.Series(dtype=str)).dropna().astype(str))
    antibody_triplets = set(
        unified.get("heavy_sequence", pd.Series(dtype=str)).fillna("").astype(str)
        + "||"
        + unified.get("light_sequence", pd.Series(dtype=str)).fillna("").astype(str)
        + "||"
        + unified.get("antigen_sequence", pd.Series(dtype=str)).fillna("").astype(str)
    )
    ids = set()
    for possible_id in ["pdb_or_antibody_id", "antibody_id", "PDB_ID", "pdb"]:
        if possible_id in unified.columns:
            ids |= set(unified[possible_id].dropna().astype(str).str.upper())
    return {"antigens": antigens, "antibody_triplets": antibody_triplets, "ids": ids}


def write_csv_both_dirs(filename: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Write identical candidate output to outputs/ and data/processed_affinity/."""

    for directory in [OUTPUT_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
            #  candidate 
            #  writer , provenance 
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def numeric_summary(values: list[float]) -> dict:
    """Small numeric summary for report tables."""

    clean = [value for value in values if value is not None and not pd.isna(value)]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def format_summary(summary: dict) -> str:
    """Format numeric summary in compact markdown form."""

    if summary["count"] == 0:
        return "count=0"
    return (
        f"count={summary['count']}, min={summary['min']:.4g}, "
        f"median={summary['median']:.4g}, mean={summary['mean']:.4g}, "
        f"max={summary['max']:.4g}, std={summary['std']:.4g}"
    )


def make_base_candidate_row(row: pd.Series, kd_column: str, delta_column: str | None) -> dict:
    """Create metadata fields shared by antibody and nanobody candidates."""

    kd_value = to_float(row.get(kd_column, ""))
    neg_log10_affinity = neg_log10(kd_value)
    risk_flags = kd_risk_flags(kd_value)
    if is_predicted_row(row):
        risk_flags.append("predicted_label")
    if row.get("overlap_antigen_sequence", False):
        risk_flags.append("overlap_antigen_sequence")
    if row.get("overlap_source_id", False):
        risk_flags.append("overlap_source_id")
    return {
        "source": row.get("Source", ""),
        "pdb_id": row.get("PDB_ID", ""),
        "source_id": row.get("source_id", ""),
        "molecule_format": row.get("molecule_format", ""),
        "ag_name": row.get("Ag_Name", ""),
        "antigen_sequence": clean_sequence(row.get("Ag_Seq", "")),
        "affinity_kd_m": kd_value,
        "neg_log10_affinity_candidate": neg_log10_affinity,
        "delta_g_binding": to_float(row.get(delta_column, "")) if delta_column else None,
        "affinity_unit": "M",
        "affinity_method": row.get("Affinity_Method", ""),
        "predicted_or_not": row.get("Predicted_or_Not", ""),
        "provenance": row.get("Provenance", ""),
        "reason_code": row.get("Reason_Code", ""),
        "overlap_antigen_sequence": bool(row.get("overlap_antigen_sequence", False)),
        "overlap_source_id": bool(row.get("overlap_source_id", False)),
        "risk_flags": ";".join(sorted(set(risk_flags))),
    }


def main() -> None:
    """Build conservative ANDD candidate CSVs and summary reports."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    andd = read_xlsx_first_sheet(ANDD_XLSX)
    columns = list(andd.columns)
    kd_column = find_column(columns, "Affinity_Kd")
    delta_column = find_column(columns, "Gbinding") or find_column(columns, "Gbinding")
    if kd_column is None:
        raise ValueError("ANDD Kd column not found.")

    andd["molecule_format"] = andd.apply(classify_format, axis=1)
    andd["source_id"] = andd.apply(source_id, axis=1)
    andd["quality_tier"] = andd.apply(lambda row: quality_tier(row, kd_column, delta_column), axis=1)
    andd["is_predicted_affinity"] = andd.apply(is_predicted_row, axis=1)
    andd["is_antipasti_predicted"] = andd.apply(
        lambda row: "antipasti"
        in " ".join(
            normalize_text(row.get(column, ""))
            for column in ["Predicted_or_Not", "Provenance", "Affinity_Method", "Source"]
        ).lower(),
        axis=1,
    )
    andd["kd_value"] = andd[kd_column].apply(to_float)
    andd["kd_positive"] = andd["kd_value"].apply(lambda value: value is not None and value > 0)

    unified_sets = build_unified_overlap_sets(load_current_unified())
    andd["overlap_antigen_sequence"] = andd["Ag_Seq"].astype(str).isin(unified_sets["antigens"])
    andd["overlap_source_id"] = andd["source_id"].astype(str).str.upper().isin(unified_sets["ids"])

    antibody_rows = []
    nanobody_rows = []
    excluded_rows = []
    overlap_rows = []

    tier1_rows = andd[andd["quality_tier"] == "Tier 1"].copy()
    tier1_after_predicted_filter = tier1_rows[
        ~(tier1_rows["is_predicted_affinity"] | tier1_rows["is_antipasti_predicted"])
    ].copy()

    seen_antibody_keys = Counter()
    seen_nanobody_keys = Counter()

    for original_index, row in tier1_after_predicted_filter.iterrows():
        molecule_format = row["molecule_format"]
        heavy = clean_sequence(row.get("Ab/Nano H_Chain AA", ""))
        light = clean_sequence(row.get("Ab/Nano L_Chain AA", ""))
        antigen = clean_sequence(row.get("Ag_Seq", ""))
        kd_value = row["kd_value"]
        exclusion_reasons = []

        if not row["kd_positive"]:
            exclusion_reasons.append("invalid_or_missing_positive_kd")
        if not is_sequence(antigen):
            exclusion_reasons.append("missing_or_invalid_antigen_sequence")
        if is_predicted_row(row) or row["is_antipasti_predicted"]:
            exclusion_reasons.append("predicted_affinity_label")

        base = make_base_candidate_row(row, kd_column, delta_column)
        if is_antibody_format(molecule_format):
            if not is_sequence(heavy):
                exclusion_reasons.append("missing_or_invalid_heavy_sequence")
            if not is_sequence(light):
                exclusion_reasons.append("missing_or_invalid_light_sequence")
            exact_key = triplet_key(heavy, light, antigen)
            overlap_triplet = exact_key in unified_sets["antibody_triplets"]
            duplicate_within_andd = seen_antibody_keys[exact_key] > 0
            seen_antibody_keys[exact_key] += 1
            if overlap_triplet:
                base["risk_flags"] = ";".join(
                    sorted(set(filter(None, base["risk_flags"].split(";") + ["overlap_exact_triplet"])))
                )
            if duplicate_within_andd:
                base["risk_flags"] = ";".join(
                    sorted(set(filter(None, base["risk_flags"].split(";") + ["duplicate_exact_triplet_within_ANDD"])))
                )
            if exclusion_reasons:
                excluded_rows.append(
                    {
                        "original_row_index": int(original_index),
                        "source": row.get("Source", ""),
                        "pdb_id": row.get("PDB_ID", ""),
                        "molecule_format": molecule_format,
                        "exclusion_reason": ";".join(sorted(set(exclusion_reasons))),
                        "quality_tier": row["quality_tier"],
                    }
                )
                continue
            candidate = {
                "candidate_id": f"ANDD_AB_{len(antibody_rows) + 1:06d}",
                **base,
                "heavy_sequence": heavy,
                "light_sequence": light,
                "nanobody_sequence": "",
                "exact_triplet_key": exact_key,
                "overlap_exact_triplet": overlap_triplet,
                "duplicate_exact_triplet_within_ANDD": duplicate_within_andd,
                "HCDR1": row.get("Ab/Nano_CDR H1", ""),
                "HCDR2": row.get("Ab/Nano_CDR H2", ""),
                "HCDR3": row.get("Ab/Nano_CDR H3", ""),
                "LCDR1": row.get("Ab/Nano_CDR L1", ""),
                "LCDR2": row.get("Ab/Nano_CDR L2", ""),
                "LCDR3": row.get("Ab/Nano_CDR L3", ""),
                "cdr_nomenclature": row.get("CDR Nomenclature", ""),
            }
            antibody_rows.append(candidate)
            if overlap_triplet or candidate["overlap_antigen_sequence"] or candidate["overlap_source_id"]:
                overlap_rows.append({**candidate, "candidate_type": "antibody"})
        elif is_nanobody_format(molecule_format):
            if not is_sequence(heavy):
                exclusion_reasons.append("missing_or_invalid_nanobody_sequence")
            exact_key = nanobody_key(heavy, antigen)
            duplicate_within_andd = seen_nanobody_keys[exact_key] > 0
            seen_nanobody_keys[exact_key] += 1
            if duplicate_within_andd:
                base["risk_flags"] = ";".join(
                    sorted(set(filter(None, base["risk_flags"].split(";") + ["duplicate_nanobody_antigen_within_ANDD"])))
                )
            if exclusion_reasons:
                excluded_rows.append(
                    {
                        "original_row_index": int(original_index),
                        "source": row.get("Source", ""),
                        "pdb_id": row.get("PDB_ID", ""),
                        "molecule_format": molecule_format,
                        "exclusion_reason": ";".join(sorted(set(exclusion_reasons))),
                        "quality_tier": row["quality_tier"],
                    }
                )
                continue
            candidate = {
                "candidate_id": f"ANDD_NB_{len(nanobody_rows) + 1:06d}",
                **base,
                "heavy_sequence": "",
                "light_sequence": "",
                "nanobody_sequence": heavy,
                "exact_nanobody_antigen_key": exact_key,
                "duplicate_nanobody_antigen_within_ANDD": duplicate_within_andd,
                "NCDR1_or_HCDR1": row.get("Ab/Nano_CDR H1", ""),
                "NCDR2_or_HCDR2": row.get("Ab/Nano_CDR H2", ""),
                "NCDR3_or_HCDR3": row.get("Ab/Nano_CDR H3", ""),
                "cdr_nomenclature": row.get("CDR Nomenclature", ""),
            }
            nanobody_rows.append(candidate)
            if candidate["overlap_antigen_sequence"] or candidate["overlap_source_id"]:
                overlap_rows.append({**candidate, "candidate_type": "nanobody"})
        else:
            excluded_rows.append(
                {
                    "original_row_index": int(original_index),
                    "source": row.get("Source", ""),
                    "pdb_id": row.get("PDB_ID", ""),
                    "molecule_format": molecule_format,
                    "exclusion_reason": "unsupported_molecule_format_for_main_candidates",
                    "quality_tier": row["quality_tier"],
                }
            )

    antibody_fields = [
        "candidate_id",
        "source",
        "pdb_id",
        "source_id",
        "molecule_format",
        "ag_name",
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        "affinity_kd_m",
        "affinity_unit",
        "neg_log10_affinity_candidate",
        "delta_g_binding",
        "affinity_method",
        "predicted_or_not",
        "provenance",
        "reason_code",
        "HCDR1",
        "HCDR2",
        "HCDR3",
        "LCDR1",
        "LCDR2",
        "LCDR3",
        "cdr_nomenclature",
        "overlap_antigen_sequence",
        "overlap_source_id",
        "overlap_exact_triplet",
        "duplicate_exact_triplet_within_ANDD",
        "risk_flags",
        "exact_triplet_key",
    ]
    nanobody_fields = [
        "candidate_id",
        "source",
        "pdb_id",
        "source_id",
        "molecule_format",
        "ag_name",
        "nanobody_sequence",
        "antigen_sequence",
        "affinity_kd_m",
        "affinity_unit",
        "neg_log10_affinity_candidate",
        "delta_g_binding",
        "affinity_method",
        "predicted_or_not",
        "provenance",
        "reason_code",
        "NCDR1_or_HCDR1",
        "NCDR2_or_HCDR2",
        "NCDR3_or_HCDR3",
        "cdr_nomenclature",
        "overlap_antigen_sequence",
        "overlap_source_id",
        "duplicate_nanobody_antigen_within_ANDD",
        "risk_flags",
        "exact_nanobody_antigen_key",
    ]
    excluded_fields = [
        "original_row_index",
        "source",
        "pdb_id",
        "molecule_format",
        "quality_tier",
        "exclusion_reason",
    ]
    overlap_fields = sorted({key for row in overlap_rows for key in row.keys()})

    write_csv_both_dirs(ANTIBODY_OUTPUT, antibody_rows, antibody_fields)
    write_csv_both_dirs(NANOBODY_OUTPUT, nanobody_rows, nanobody_fields)
    write_csv_both_dirs(EXCLUDED_OUTPUT, excluded_rows, excluded_fields)
    write_csv_both_dirs(OVERLAP_OUTPUT, overlap_rows, overlap_fields)

    antibody_targets = [row["neg_log10_affinity_candidate"] for row in antibody_rows]
    nanobody_targets = [row["neg_log10_affinity_candidate"] for row in nanobody_rows]
    antibody_kds = [row["affinity_kd_m"] for row in antibody_rows]
    nanobody_kds = [row["affinity_kd_m"] for row in nanobody_rows]
    antibody_source_counts = Counter(row["source"] for row in antibody_rows)
    nanobody_source_counts = Counter(row["source"] for row in nanobody_rows)
    antibody_risk_counts = Counter(
        flag
        for row in antibody_rows
        for flag in str(row["risk_flags"]).split(";")
        if flag
    )
    nanobody_risk_counts = Counter(
        flag
        for row in nanobody_rows
        for flag in str(row["risk_flags"]).split(";")
        if flag
    )
    summary = {
        "tier1_rows": int(len(tier1_rows)),
        "tier1_after_predicted_filter": int(len(tier1_after_predicted_filter)),
        "antibody_candidate_rows": len(antibody_rows),
        "nanobody_candidate_rows": len(nanobody_rows),
        "excluded_or_unsupported_rows": len(excluded_rows),
        "overlap_rows": len(overlap_rows),
        "antibody_exact_triplet_overlap_rows": sum(row["overlap_exact_triplet"] for row in antibody_rows),
        "antibody_antigen_overlap_rows": sum(row["overlap_antigen_sequence"] for row in antibody_rows),
        "nanobody_antigen_overlap_rows": sum(row["overlap_antigen_sequence"] for row in nanobody_rows),
        "antibody_duplicate_within_andd_rows": sum(
            row["duplicate_exact_triplet_within_ANDD"] for row in antibody_rows
        ),
        "nanobody_duplicate_within_andd_rows": sum(
            row["duplicate_nanobody_antigen_within_ANDD"] for row in nanobody_rows
        ),
        "antibody_kd_summary": numeric_summary(antibody_kds),
        "nanobody_kd_summary": numeric_summary(nanobody_kds),
        "antibody_target_summary": numeric_summary(antibody_targets),
        "nanobody_target_summary": numeric_summary(nanobody_targets),
        "antibody_source_counts": dict(antibody_source_counts.most_common(20)),
        "nanobody_source_counts": dict(nanobody_source_counts.most_common(20)),
        "antibody_risk_counts": dict(antibody_risk_counts),
        "nanobody_risk_counts": dict(nanobody_risk_counts),
    }
    for directory in [OUTPUT_DIR, PROCESSED_DIR]:
        (directory / "expanded_affinity_v2_candidate_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    lines = [
        "# ANDD expanded_affinity_dataset_v2 Candidate Summary",
        "",
        "## Scope",
        "",
        "This is a conservative candidate dataset, not a final train/val/test dataset.",
        "",
        "- No model was trained.",
        "- No final split was created.",
        "- Current `unified_no_high_risk` was not modified.",
        "- Antibody and nanobody candidates are intentionally separated.",
        "",
        "## 1. Candidate Counts",
        "",
        f"- Tier 1 rows from ANDD audit: `{len(tier1_rows)}`",
        f"- Tier 1 rows after predicted/ANTIPASTI filter: `{len(tier1_after_predicted_filter)}`",
        f"- Antibody candidate rows: `{len(antibody_rows)}`",
        f"- Nanobody candidate rows: `{len(nanobody_rows)}`",
        f"- Excluded or unsupported Tier 1 rows: `{len(excluded_rows)}`",
        "",
        "## 2. Overlap And Duplicate Checks",
        "",
        f"- Antibody exact heavy+light+antigen triplet overlaps with unified_no_high_risk: `{summary['antibody_exact_triplet_overlap_rows']}`",
        f"- Antibody antigen_sequence overlap rows: `{summary['antibody_antigen_overlap_rows']}`",
        f"- Nanobody antigen_sequence overlap rows: `{summary['nanobody_antigen_overlap_rows']}`",
        f"- Antibody exact triplet duplicates within ANDD candidates: `{summary['antibody_duplicate_within_andd_rows']}`",
        f"- Nanobody+antigen duplicates within ANDD candidates: `{summary['nanobody_duplicate_within_andd_rows']}`",
        f"- Total rows written to overlap file: `{len(overlap_rows)}`",
        "",
        "Duplicate antigen_sequence rows are flagged but not removed, because future antigen-group split design needs this information.",
        "",
        "## 3. Kd And Target Distribution",
        "",
        "| Candidate group | Kd(M) summary | neg_log10 target summary |",
        "|---|---|---|",
        f"| antibody | {format_summary(summary['antibody_kd_summary'])} | {format_summary(summary['antibody_target_summary'])} |",
        f"| nanobody | {format_summary(summary['nanobody_kd_summary'])} | {format_summary(summary['nanobody_target_summary'])} |",
        "",
        "Kd is retained in M as `affinity_kd_m`, and `neg_log10_affinity_candidate` is computed as `-log10(Kd)`.",
        "Rows with very tiny or very large Kd are not removed here; they are flagged in `risk_flags` for human review.",
        "",
        "## 4. Top Sources",
        "",
        "### Antibody candidates",
        "",
        "| Source | Rows |",
        "|---|---:|",
    ]
    for source, count in antibody_source_counts.most_common(10):
        lines.append(f"| `{source}` | {count} |")
    lines.extend(["", "### Nanobody candidates", "", "| Source | Rows |", "|---|---:|"])
    for source, count in nanobody_source_counts.most_common(10):
        lines.append(f"| `{source}` | {count} |")
    lines.extend(
        [
            "",
            "## 5. Risk Flags",
            "",
            "### Antibody risk flags",
            "",
            "| Flag | Rows |",
            "|---|---:|",
        ]
    )
    for flag, count in antibody_risk_counts.most_common():
        lines.append(f"| `{flag}` | {count} |")
    lines.extend(["", "### Nanobody risk flags", "", "| Flag | Rows |", "|---|---:|"])
    for flag, count in nanobody_risk_counts.most_common():
        lines.append(f"| `{flag}` | {count} |")
    lines.extend(
        [
            "",
            "## 6. Recommendation",
            "",
            "Do not mix antibody and nanobody into one main task yet.",
            "",
            "Recommended next step:",
            "",
            "1. Build `expanded_affinity_antibody_v2` first if the next model is heavy+light+antigen.",
            "2. Build `expanded_affinity_nanobody_v2` separately for VHH/nanobody input mode.",
            "3. Keep predicted labels out of the primary supervised benchmark.",
            "4. Review extreme Kd flags and source/provenance before final split.",
            "5. Only after manual review, create antigen-sequence group split for each task.",
            "",
            "## 7. Output Files",
            "",
            f"- `outputs/data_expansion/ANDD_v2_candidates/{ANTIBODY_OUTPUT}`",
            f"- `outputs/data_expansion/ANDD_v2_candidates/{NANOBODY_OUTPUT}`",
            f"- `outputs/data_expansion/ANDD_v2_candidates/{EXCLUDED_OUTPUT}`",
            f"- `outputs/data_expansion/ANDD_v2_candidates/{OVERLAP_OUTPUT}`",
            f"- `data/processed_affinity/expanded_affinity_dataset_v2_candidates/{ANTIBODY_OUTPUT}`",
            f"- `data/processed_affinity/expanded_affinity_dataset_v2_candidates/{NANOBODY_OUTPUT}`",
        ]
    )
    for directory in [OUTPUT_DIR, PROCESSED_DIR]:
        (directory / SUMMARY_OUTPUT).write_text("\n".join(lines), encoding="utf-8")

    print(f"Tier 1 rows: {len(tier1_rows)}")
    print(f"Tier 1 after predicted filter: {len(tier1_after_predicted_filter)}")
    print(f"Antibody candidates: {len(antibody_rows)}")
    print(f"Nanobody candidates: {len(nanobody_rows)}")
    print(f"Overlap rows: {len(overlap_rows)}")
    print(f"Saved summary to {OUTPUT_DIR / SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
