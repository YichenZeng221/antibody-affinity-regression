"""Audit Zenodo Protein_SAbDab raw CSV against processed TDC v1.

:
Zenodo CSV  raw antibody-antigen affinity pair table
 TDC v1  raw rows :

1. target ;
2. heavy/light sequence parsing;
3. exact heavy+light+antigen triplet ;
4. antigen-group split;

 clean processed dataset

 raw 493 rows  clean 466 rows ,,
 processed dataset
"""

from __future__ import annotations

from pathlib import Path
import ast
import json
import math
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZENODO_PATH = PROJECT_ROOT / "data" / "raw" / "zenodo_antibody_affinity_protein_sabdab.csv"
TDC_SPLIT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
PROCESSING_REPORT_PATH = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "processing_report.json"
EXCLUDED_RECORDS_PATH = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "excluded_records.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "zenodo_tdc_overlap_audit"
JSON_REPORT_PATH = OUTPUT_DIR / "zenodo_vs_tdc_v1.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "zenodo_vs_tdc_v1.md"
NOT_IN_CLEAN_PATH = OUTPUT_DIR / "zenodo_rows_not_in_tdc_clean.csv"

SPLITS = ["train", "val", "test"]
SEQUENCE_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
RAW_ID_COLUMNS = ["Antibody_ID", "Antibody", "Antigen_ID", "Antigen", "Y"]
AMINO_ACID_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZUO]+$")


def is_missing(value: object) -> bool:
    """Treat blank cells and common NA spellings as missing."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def normalize_id(value: object) -> str:
    """Normalize PDB-like Antibody_ID values for overlap checks."""

    return str(value).strip().lower()


def clean_sequence(sequence: object) -> str:
    """Normalize TDC sequence text in the same style as the TDC v1 builder."""

    return re.sub(r"[\s,;|]+", "", str(sequence).strip().upper())


def parse_antibody_chains(value: object) -> tuple[str | None, str | None]:
    """Parse Zenodo/TDC Antibody list string into heavy and light sequences."""

    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return clean_sequence(parsed[0]), clean_sequence(parsed[1])
    except (SyntaxError, ValueError):
        pass

    simplified = text.strip("[]()").replace('"', "").replace("'", "")
    for delimiter in ["|", ";", ","]:
        parts = [clean_sequence(part) for part in simplified.split(delimiter)]
        parts = [part for part in parts if part]
        if len(parts) >= 2:
            return parts[0], parts[1]

    parts = [clean_sequence(part) for part in simplified.split()]
    parts = [part for part in parts if part]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def valid_sequence(value: object) -> bool:
    """Check whether one sequence looks like amino acid letters."""

    sequence = clean_sequence(value)
    return bool(sequence) and bool(AMINO_ACID_PATTERN.fullmatch(sequence))


def load_zenodo() -> pd.DataFrame:
    """Load Zenodo raw CSV and validate the raw TDC-like schema."""

    if not ZENODO_PATH.exists():
        raise FileNotFoundError(f"Cannot find {ZENODO_PATH}")
    data = pd.read_csv(ZENODO_PATH)
    missing_columns = set(RAW_ID_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Zenodo CSV missing columns: {sorted(missing_columns)}")
    return data


def load_tdc_clean() -> pd.DataFrame:
    """Load processed TDC v1 train/val/test rows."""

    frames = []
    for split_name in SPLITS:
        path = TDC_SPLIT_DIR / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find TDC clean split: {path}")
        frame = pd.read_csv(path)
        frame["split"] = split_name
        frames.append(frame)
    clean = pd.concat(frames, ignore_index=True)
    required_columns = {"Antibody_ID", "Y", "affinity", "neg_log10_affinity", *SEQUENCE_COLUMNS}
    missing_columns = required_columns - set(clean.columns)
    if missing_columns:
        raise ValueError(f"TDC v1 clean missing columns: {sorted(missing_columns)}")
    return clean


def load_processing_report() -> dict:
    """Read existing TDC v1 processing report."""

    if not PROCESSING_REPORT_PATH.exists():
        raise FileNotFoundError(f"Cannot find {PROCESSING_REPORT_PATH}")
    return json.loads(PROCESSING_REPORT_PATH.read_text(encoding="utf-8"))


def load_excluded() -> pd.DataFrame:
    """Read rows excluded by the existing TDC v1 builder."""

    if not EXCLUDED_RECORDS_PATH.exists():
        raise FileNotFoundError(f"Cannot find {EXCLUDED_RECORDS_PATH}")
    excluded = pd.read_csv(EXCLUDED_RECORDS_PATH)
    if "exclusion_reason" not in excluded.columns:
        raise ValueError("excluded_records.csv needs exclusion_reason column.")
    return excluded


def add_parsed_raw_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Attach cleaned sequences and target transform to Zenodo raw rows."""

    parsed = raw.copy()
    chain_pairs = parsed["Antibody"].apply(parse_antibody_chains)
    parsed["heavy_sequence"] = chain_pairs.apply(lambda pair: pair[0])
    parsed["light_sequence"] = chain_pairs.apply(lambda pair: pair[1])
    parsed["antigen_sequence"] = parsed["Antigen"].map(clean_sequence)
    parsed["Y_numeric"] = pd.to_numeric(parsed["Y"], errors="coerce")
    parsed["Y_positive"] = parsed["Y_numeric"] > 0
    parsed["neg_log10_affinity"] = parsed["Y_numeric"].map(
        lambda value: -math.log10(float(value)) if pd.notna(value) and value > 0 else float("nan")
    )
    parsed["parsed_sequences_valid"] = (
        parsed["heavy_sequence"].map(valid_sequence)
        & parsed["light_sequence"].map(valid_sequence)
        & parsed["antigen_sequence"].map(valid_sequence)
    )
    parsed["triplet_key"] = parsed[SEQUENCE_COLUMNS].fillna("").astype(str).agg("||".join, axis=1)
    return parsed


def numeric_stats(series: pd.Series) -> dict:
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


def triplet_keys(data: pd.DataFrame) -> set[str]:
    """Build exact sequence-triplet keys."""

    usable = data.dropna(subset=SEQUENCE_COLUMNS).copy()
    return set(usable[SEQUENCE_COLUMNS].astype(str).agg("||".join, axis=1))


def raw_row_key(data: pd.DataFrame) -> pd.Series:
    """Build exact raw-row identity key using the five Zenodo columns."""

    key_data = data[RAW_ID_COLUMNS].copy()
    for column_name in RAW_ID_COLUMNS:
        key_data[column_name] = key_data[column_name].astype(str).str.strip()
    return key_data.astype(str).agg("||".join, axis=1)


def excluded_reason_counts(excluded: pd.DataFrame) -> dict:
    """Count known and other exclusion reasons."""

    raw_counts = excluded["exclusion_reason"].value_counts(dropna=False).to_dict()
    return {
        "conflicting_duplicate_target": int(raw_counts.get("conflicting_duplicate_target", 0)),
        "duplicate_triplet_removed": int(raw_counts.get("duplicate_triplet_removed", 0)),
        "other_reasons": {
            str(reason): int(count)
            for reason, count in raw_counts.items()
            if reason not in {"conflicting_duplicate_target", "duplicate_triplet_removed"}
        },
        "all_reasons": {str(reason): int(count) for reason, count in raw_counts.items()},
    }


def annotate_not_in_clean(parsed_raw: pd.DataFrame, clean: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    """Collect Zenodo raw rows not retained by TDC v1 clean."""

    # Zenodo raw  Antibody_ID  493  ID
    #  Antibody_ID  raw row  clean,:
    #  row  Y  CSV round-trip 
    # ``8.6e-10`` vs ``8.599999999999996e-10``, row
    clean_ids = {normalize_id(value) for value in clean["Antibody_ID"]}
    excluded_small = excluded[["Antibody_ID", "exclusion_reason"]].copy()
    excluded_small["antibody_id_key"] = excluded_small["Antibody_ID"].map(normalize_id)
    reason_lookup = (
        excluded_small.groupby("antibody_id_key")["exclusion_reason"]
        .apply(lambda values: "|".join(sorted(set(values.astype(str)))))
        .to_dict()
    )

    raw = parsed_raw.copy()
    raw["antibody_id_key"] = raw["Antibody_ID"].map(normalize_id)
    raw["raw_key"] = raw_row_key(raw)
    raw["present_in_tdc_clean_by_antibody_id"] = raw["antibody_id_key"].isin(clean_ids)
    not_in_clean = raw[~raw["present_in_tdc_clean_by_antibody_id"]].copy()
    not_in_clean["excluded_reason_from_tdc_v1"] = not_in_clean["antibody_id_key"].map(reason_lookup).fillna(
        "not_found_in_excluded_records"
    )
    return not_in_clean


def build_report(
    parsed_raw: pd.DataFrame,
    clean: pd.DataFrame,
    excluded: pd.DataFrame,
    processing_report: dict,
    not_in_clean: pd.DataFrame,
) -> dict:
    """Build the full Zenodo-vs-clean comparison report."""

    raw_ids = {normalize_id(value) for value in parsed_raw["Antibody_ID"] if not is_missing(value)}
    clean_ids = {normalize_id(value) for value in clean["Antibody_ID"] if not is_missing(value)}
    raw_triplets = triplet_keys(parsed_raw)
    clean_triplets = triplet_keys(clean)
    raw_y = numeric_stats(parsed_raw["Y_numeric"])
    clean_y = numeric_stats(clean["Y"])
    raw_target = numeric_stats(parsed_raw["neg_log10_affinity"])
    clean_target = numeric_stats(clean["neg_log10_affinity"])
    excluded_counts = excluded_reason_counts(excluded)

    unique_not_in_clean_triplets = len(set(not_in_clean["triplet_key"]))
    clean_triplet_overlap_not_in_clean = len(set(not_in_clean["triplet_key"]) & clean_triplets)
    usable_unseen_triplets_not_in_clean = int(
        (
            not_in_clean["Y_positive"]
            & not_in_clean["parsed_sequences_valid"]
            & ~not_in_clean["triplet_key"].isin(clean_triplets)
            & (not_in_clean["excluded_reason_from_tdc_v1"] != "conflicting_duplicate_target")
        ).sum()
    )

    return {
        "inputs": {
            "zenodo_raw_csv": str(ZENODO_PATH.relative_to(PROJECT_ROOT)),
            "tdc_v1_split_dir": str(TDC_SPLIT_DIR.relative_to(PROJECT_ROOT)),
            "tdc_v1_processing_report": str(PROCESSING_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "tdc_v1_excluded_records": str(EXCLUDED_RECORDS_PATH.relative_to(PROJECT_ROOT)),
        },
        "zenodo_raw": {
            "rows": int(len(parsed_raw)),
            "unique_antibody_ids": int(len(raw_ids)),
            "y_numeric_rows": int(parsed_raw["Y_numeric"].notna().sum()),
            "y_gt_0_rows": int(parsed_raw["Y_positive"].sum()),
            "parsed_valid_sequence_rows": int(parsed_raw["parsed_sequences_valid"].sum()),
            "raw_y_stats": raw_y,
            "neg_log10_target_stats": raw_target,
        },
        "tdc_v1_clean": {
            "rows": int(len(clean)),
            "unique_antibody_ids": int(len(clean_ids)),
            "raw_y_stats": clean_y,
            "neg_log10_target_stats": clean_target,
            "processing_report_headline": {
                "raw_total_rows": processing_report.get("raw_total_rows"),
                "rows_after_target_filtering": processing_report.get("rows_after_target_filtering"),
                "rows_after_antibody_parsing": processing_report.get("rows_after_antibody_parsing"),
                "rows_after_dedup": processing_report.get("rows_after_dedup"),
                "conflicting_duplicate_target_groups": processing_report.get(
                    "conflicting_duplicate_target_groups"
                ),
            },
        },
        "overlap": {
            "antibody_id": {
                "zenodo_unique": int(len(raw_ids)),
                "tdc_clean_unique": int(len(clean_ids)),
                "overlap": int(len(raw_ids & clean_ids)),
                "zenodo_not_in_tdc_clean": int(len(raw_ids - clean_ids)),
                "tdc_clean_not_in_zenodo": int(len(clean_ids - raw_ids)),
            },
            "sequence_triplet": {
                "zenodo_unique_triplets": int(len(raw_triplets)),
                "tdc_clean_unique_triplets": int(len(clean_triplets)),
                "triplet_overlap": int(len(raw_triplets & clean_triplets)),
                "zenodo_triplets_not_in_tdc_clean": int(len(raw_triplets - clean_triplets)),
            },
            "target_range_consistency": {
                "raw_y_min_equal": raw_y["min"] == clean_y["min"],
                "raw_y_max_equal": raw_y["max"] == clean_y["max"],
                "zenodo_raw_y_range": {"min": raw_y["min"], "max": raw_y["max"]},
                "tdc_clean_raw_y_range": {"min": clean_y["min"], "max": clean_y["max"]},
                "zenodo_neg_log10_range": {"min": raw_target["min"], "max": raw_target["max"]},
                "tdc_clean_neg_log10_range": {"min": clean_target["min"], "max": clean_target["max"]},
            },
        },
        "zenodo_rows_not_in_tdc_clean": {
            "rows": int(len(not_in_clean)),
            "unique_antibody_ids": int(not_in_clean["Antibody_ID"].map(normalize_id).nunique()),
            "unique_triplets": int(unique_not_in_clean_triplets),
            "triplets_already_represented_in_clean": int(clean_triplet_overlap_not_in_clean),
            "reason_counts": {
                str(reason): int(count)
                for reason, count in not_in_clean["excluded_reason_from_tdc_v1"]
                .value_counts(dropna=False)
                .items()
            },
            "csv": str(NOT_IN_CLEAN_PATH.relative_to(PROJECT_ROOT)),
            "row_membership_key": (
                "Antibody_ID. Zenodo has unique Antibody_ID values; using all raw CSV text fields can "
                "misclassify rows when Y float formatting differs after CSV round-trip."
            ),
        },
        "excluded_records": excluded_counts,
        "answers": {
            "same_source_answer": (
                "Yes at file-schema and row-lineage level: Zenodo contains the same 493-row "
                "Protein_SAbDab raw table shape used before TDC v1 cleaning, and the TDC v1 report "
                "also starts from 493 raw rows."
            ),
            "why_clean_is_466": (
                f"TDC v1 kept all 493 rows through target filtering and antibody parsing, then removed "
                f"{len(excluded)} duplicate-triplet rows during dedup: "
                f"{excluded_counts['conflicting_duplicate_target']} conflicting-target rows and "
                f"{excluded_counts['duplicate_triplet_removed']} same-target duplicate rows."
            ),
            "unused_clean_rows_answer": (
                "No obvious clean add-back rows remain in these 27 excluded Zenodo rows. "
                "Same-target duplicate rows already have their triplet represented in clean data, and "
                "conflicting-target duplicate rows were deliberately excluded because one input triplet "
                "maps to inconsistent labels in the beginner clean policy."
            ),
            "usable_unseen_triplet_rows_if_ignoring_policy": int(usable_unseen_triplets_not_in_clean),
        },
    }


def write_markdown(report: dict) -> None:
    """Write a readable Markdown report."""

    raw = report["zenodo_raw"]
    clean = report["tdc_v1_clean"]
    overlap = report["overlap"]
    not_in = report["zenodo_rows_not_in_tdc_clean"]
    excluded = report["excluded_records"]
    answers = report["answers"]
    lines = [
        "# Zenodo vs TDC v1 Audit",
        "",
        "## Scope",
        "",
        f"- Zenodo raw CSV: `{report['inputs']['zenodo_raw_csv']}`",
        f"- TDC v1 clean splits: `{report['inputs']['tdc_v1_split_dir']}`",
        f"- Existing TDC v1 processing report: `{report['inputs']['tdc_v1_processing_report']}`",
        f"- Existing TDC v1 excluded records: `{report['inputs']['tdc_v1_excluded_records']}`",
        "- This audit does not train models or modify datasets.",
        "",
        "## Headline Counts",
        "",
        "| source | rows | unique Antibody_ID | Y numeric rows | Y > 0 rows |",
        "|---|---:|---:|---:|---:|",
        f"| Zenodo raw | {raw['rows']} | {raw['unique_antibody_ids']} | {raw['y_numeric_rows']} | {raw['y_gt_0_rows']} |",
        f"| TDC v1 clean | {clean['rows']} | {clean['unique_antibody_ids']} | {clean['raw_y_stats']['count']} | {clean['raw_y_stats']['count']} |",
        "",
        "## Raw To Clean Processing",
        "",
        f"- Existing processing headline: `{clean['processing_report_headline']}`",
        f"- Excluded reason counts: `{excluded}`",
        "",
        "## Overlap",
        "",
        f"- Antibody_ID overlap: `{overlap['antibody_id']}`",
        f"- Sequence triplet overlap: `{overlap['sequence_triplet']}`",
        f"- Target range consistency: `{overlap['target_range_consistency']}`",
        "",
        "## Zenodo Rows Not In TDC Clean",
        "",
        f"- Summary: `{not_in}`",
        f"- CSV: `{not_in['csv']}`",
        "",
        "## Answers",
        "",
        f"- Are Zenodo 493 and TDC raw the same source? {answers['same_source_answer']}",
        f"- Why is TDC v1 clean 466 rows? {answers['why_clean_is_466']}",
        f"- Are there unused clean rows to add back now? {answers['unused_clean_rows_answer']}",
    ]
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: dict) -> None:
    """Print the takeaway in terminal."""

    print("Zenodo vs TDC v1 audit complete.")
    print(
        f"Zenodo raw rows: {report['zenodo_raw']['rows']} | "
        f"TDC v1 clean rows: {report['tdc_v1_clean']['rows']}"
    )
    print(f"Excluded reasons: {report['excluded_records']['all_reasons']}")
    print(f"Zenodo rows not in TDC clean: {report['zenodo_rows_not_in_tdc_clean']['rows']}")
    print(report["answers"]["why_clean_is_466"])
    print(report["answers"]["unused_clean_rows_answer"])
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Not-in-clean CSV: {NOT_IN_CLEAN_PATH.relative_to(PROJECT_ROOT)}")
    print("No model training was run.")


def main() -> None:
    """Run Zenodo raw vs TDC clean audit."""

    raw = load_zenodo()
    parsed_raw = add_parsed_raw_columns(raw)
    clean = load_tdc_clean()
    excluded = load_excluded()
    processing_report = load_processing_report()
    not_in_clean = annotate_not_in_clean(parsed_raw, clean, excluded)

    report = build_report(parsed_raw, clean, excluded, processing_report, not_in_clean)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    not_in_clean.to_csv(NOT_IN_CLEAN_PATH, index=False)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print_summary(report)


if __name__ == "__main__":
    main()
