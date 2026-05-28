"""Audit overlap between local SAbDab summary affinity rows and TDC v1.

:
, dataset
:

1.  SAbDab summary TSV:
    metadata , PDB  row,
    antibody/antigen chain 
2. TDC Protein_SAbDab v1:
    antibody-antigen affinity pair 

 overlap  PDB ID:
SAbDab  ``pdb``  TDC  ``Antibody_ID`` / ``antibody_id``
row-level  sequence/chain ,
 PDB overlap 
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "sabdab_summary.tsv"
TDC_SPLIT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "sabdab_overlap_audit"
JSON_REPORT_PATH = OUTPUT_DIR / "sabdab_tdc_overlap_audit.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "sabdab_tdc_overlap_audit.md"
MISSING_ROWS_PATH = OUTPUT_DIR / "sabdab_affinity_positive_not_in_tdc.csv"
SUPPLEMENT_CANDIDATES_PATH = OUTPUT_DIR / "sabdab_possible_supplement_candidates.csv"

TDC_SPLITS = ["train", "val", "test"]
SUMMARY_COLUMNS_TO_SAVE = [
    "pdb",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "antigen_type",
    "antigen_name",
    "affinity",
    "delta_g",
    "affinity_method",
    "temperature",
    "pmid",
]


def load_summary() -> pd.DataFrame:
    """Load local SAbDab summary with all fields kept as text first."""

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Cannot find {SUMMARY_PATH}")

    summary = pd.read_csv(SUMMARY_PATH, sep="\t", dtype=str, keep_default_na=False)
    required_columns = set(SUMMARY_COLUMNS_TO_SAVE)
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(f"SAbDab summary is missing required columns: {sorted(missing_columns)}")
    return summary


def load_tdc_v1() -> pd.DataFrame:
    """Load processed TDC train/val/test and remember split label."""

    frames = []
    for split_name in TDC_SPLITS:
        path = TDC_SPLIT_DIR / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}. Build TDC v1 first.")
        dataframe = pd.read_csv(path)
        dataframe["split"] = split_name
        frames.append(dataframe)

    tdc = pd.concat(frames, ignore_index=True)
    if "Antibody_ID" not in tdc.columns and "antibody_id" not in tdc.columns:
        raise ValueError("TDC v1 needs Antibody_ID or antibody_id for PDB overlap audit.")
    return tdc


def normalize_id(value: object) -> str:
    """Normalize PDB-like IDs to lowercase text for overlap checks."""

    return str(value).strip().lower()


def is_missing(value: object) -> bool:
    """Treat blank and common NA spellings as missing metadata."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def is_valid_antigen_type(value: object) -> bool:
    """Mirror the beginner sequence-only filter used in SAbDab builder."""

    text = str(value).strip().lower()
    has_sequence_type = "protein" in text or "peptide" in text
    is_hapten = "hapten" in text
    return has_sequence_type and not is_hapten


def looks_suspicious_affinity_method(value: object) -> bool:
    """Flag PMID-like numeric affinity_method cells.

     clean_v2  metadata :
    affinity_method  SPR / ITC / Other,
     PMID 
    """

    text = str(value).strip()
    return bool(text) and text.replace(".", "", 1).isdigit()


def numeric_summary(series: pd.Series) -> dict:
    """Return simple numeric stats for a report."""

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


def prepare_summary_affinity_rows(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mark non-empty/numeric/positive affinity rows.

    affinity , ``-log10(affinity)`` regression target
     non-emptynumericpositive , summary metadata 
    """

    prepared = summary.copy()
    prepared["pdb_norm"] = prepared["pdb"].map(normalize_id)
    prepared["affinity_text_present"] = ~prepared["affinity"].map(is_missing)
    prepared["affinity_numeric"] = pd.to_numeric(prepared["affinity"], errors="coerce")
    prepared["affinity_is_numeric"] = prepared["affinity_numeric"].notna()
    prepared["affinity_is_positive"] = prepared["affinity_numeric"] > 0

    positive_rows = prepared[prepared["affinity_is_positive"]].copy()
    stats = {
        "summary_total_rows": int(len(prepared)),
        "affinity_nonempty_rows": int(prepared["affinity_text_present"].sum()),
        "affinity_numeric_rows": int(prepared["affinity_is_numeric"].sum()),
        "affinity_positive_rows": int(prepared["affinity_is_positive"].sum()),
        "affinity_positive_unique_pdbs": int(positive_rows["pdb_norm"].nunique()),
        "positive_affinity_stats": numeric_summary(positive_rows["affinity_numeric"]),
    }
    return positive_rows, stats


def add_supplement_flags(rows: pd.DataFrame) -> pd.DataFrame:
    """Judge whether missing TDC rows are plausible SAbDab supplement candidates.

     metadata-level ,
     row  PDB  heavy/light/antigen sequences,
    label  split 
    """

    flagged = rows.copy()
    flagged["has_Hchain"] = ~flagged["Hchain"].map(is_missing)
    flagged["has_Lchain"] = ~flagged["Lchain"].map(is_missing)
    flagged["has_antigen_chain"] = ~flagged["antigen_chain"].map(is_missing)
    flagged["antigen_type_sequence_ok"] = flagged["antigen_type"].map(is_valid_antigen_type)
    flagged["same_heavy_light_chain_id"] = (
        flagged["Hchain"].astype(str).str.strip().str.upper()
        == flagged["Lchain"].astype(str).str.strip().str.upper()
    )
    flagged["suspicious_affinity_method"] = flagged["affinity_method"].map(looks_suspicious_affinity_method)
    flagged["metadata_stage1_eligible"] = (
        flagged["has_Hchain"]
        & flagged["has_Lchain"]
        & flagged["has_antigen_chain"]
        & flagged["antigen_type_sequence_ok"]
    )
    # :
    # 1. Hchain == Lchain, heavy/light pair;
    # 2. affinity_method , metadata 
    flagged["conservative_supplement_candidate"] = (
        flagged["metadata_stage1_eligible"]
        & ~flagged["same_heavy_light_chain_id"]
        & ~flagged["suspicious_affinity_method"]
    )

    reasons = []
    for _, row in flagged.iterrows():
        row_reasons = []
        if not row["has_Hchain"]:
            row_reasons.append("missing_Hchain")
        if not row["has_Lchain"]:
            row_reasons.append("missing_Lchain")
        if not row["has_antigen_chain"]:
            row_reasons.append("missing_antigen_chain")
        if not row["antigen_type_sequence_ok"]:
            if "hapten" in str(row["antigen_type"]).lower():
                row_reasons.append("hapten_or_contains_hapten")
            else:
                row_reasons.append("antigen_type_not_protein_or_peptide")
        if not row_reasons:
            row_reasons.append("possible_candidate_needs_sequence_extraction")
        reasons.append("|".join(row_reasons))

    flagged["supplement_judgement"] = reasons
    caution_text = []
    for _, row in flagged.iterrows():
        cautions = []
        if row["same_heavy_light_chain_id"]:
            cautions.append("same_Hchain_Lchain_id")
        if row["suspicious_affinity_method"]:
            cautions.append("numeric_affinity_method_maybe_metadata_shift")
        if not cautions:
            cautions.append("no_extra_metadata_caution")
        caution_text.append("|".join(cautions))
    flagged["supplement_cautions"] = caution_text
    return flagged


def count_by_reason(flagged_rows: pd.DataFrame) -> dict[str, int]:
    """Count supplement judgement strings for a quick terminal/report view."""

    return {
        str(reason): int(count)
        for reason, count in flagged_rows["supplement_judgement"].value_counts(dropna=False).items()
    }


def overlap_stats(positive_rows: pd.DataFrame, tdc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute PDB coverage of positive-affinity SAbDab rows by TDC v1."""

    tdc_pdb_column = "Antibody_ID" if "Antibody_ID" in tdc.columns else "antibody_id"
    tdc_pdbs = {normalize_id(value) for value in tdc[tdc_pdb_column] if not is_missing(value)}
    tdc_with_norm = tdc.copy()
    tdc_with_norm["tdc_pdb_norm"] = tdc_with_norm[tdc_pdb_column].map(normalize_id)

    overlap_mask = positive_rows["pdb_norm"].isin(tdc_pdbs)
    overlap_rows = positive_rows[overlap_mask]
    missing_rows = positive_rows[~overlap_mask].copy()

    positive_pdbs = set(positive_rows["pdb_norm"])
    overlap_pdbs = positive_pdbs & tdc_pdbs
    missing_pdbs = positive_pdbs - tdc_pdbs

    stats = {
        "tdc_rows": int(len(tdc)),
        "tdc_unique_antibody_ids_as_pdbs": int(len(tdc_pdbs)),
        "tdc_pdb_column_used": tdc_pdb_column,
        "positive_summary_rows_with_tdc_pdb_overlap": int(len(overlap_rows)),
        "positive_summary_rows_without_tdc_pdb_overlap": int(len(missing_rows)),
        "positive_summary_unique_pdb_overlap_with_tdc": int(len(overlap_pdbs)),
        "positive_summary_unique_pdb_missing_from_tdc": int(len(missing_pdbs)),
        "positive_summary_row_coverage_by_tdc_pdb": float(len(overlap_rows) / len(positive_rows))
        if len(positive_rows)
        else None,
        "positive_summary_pdb_coverage_by_tdc": float(len(overlap_pdbs) / len(positive_pdbs))
        if positive_pdbs
        else None,
        "tdc_pdbs_not_in_positive_summary": int(len(tdc_pdbs - positive_pdbs)),
    }
    return missing_rows, stats


def save_missing_rows(flagged_missing_rows: pd.DataFrame) -> None:
    """Save missing rows and a narrower candidate table for manual review."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        *[column for column in SUMMARY_COLUMNS_TO_SAVE if column in flagged_missing_rows.columns],
        "pdb_norm",
        "affinity_numeric",
        "has_Hchain",
        "has_Lchain",
        "has_antigen_chain",
        "antigen_type_sequence_ok",
        "same_heavy_light_chain_id",
        "suspicious_affinity_method",
        "metadata_stage1_eligible",
        "conservative_supplement_candidate",
        "supplement_judgement",
        "supplement_cautions",
    ]
    flagged_missing_rows[output_columns].to_csv(MISSING_ROWS_PATH, index=False)
    flagged_missing_rows[flagged_missing_rows["metadata_stage1_eligible"]][output_columns].to_csv(
        SUPPLEMENT_CANDIDATES_PATH,
        index=False,
    )


def build_report(summary_stats: dict, overlap: dict, flagged_missing_rows: pd.DataFrame) -> dict:
    """Build JSON-friendly report."""

    candidates = flagged_missing_rows[flagged_missing_rows["metadata_stage1_eligible"]]
    conservative_candidates = flagged_missing_rows[flagged_missing_rows["conservative_supplement_candidate"]]
    unique_candidate_pdbs = candidates["pdb_norm"].nunique()
    unique_conservative_pdbs = conservative_candidates["pdb_norm"].nunique()
    report = {
        **summary_stats,
        **overlap,
        "missing_positive_summary_rows_antigen_type_counts": {
            str(key): int(value)
            for key, value in flagged_missing_rows["antigen_type"].replace("", "<missing>").value_counts().items()
        },
        "missing_positive_summary_rows_supplement_judgement_counts": count_by_reason(flagged_missing_rows),
        "metadata_possible_supplement_rows": int(len(candidates)),
        "metadata_possible_supplement_unique_pdbs": int(unique_candidate_pdbs),
        "metadata_possible_candidates_with_same_heavy_light_chain_id": int(
            candidates["same_heavy_light_chain_id"].sum()
        ),
        "metadata_possible_candidates_with_suspicious_affinity_method": int(
            candidates["suspicious_affinity_method"].sum()
        ),
        "conservative_supplement_rows": int(len(conservative_candidates)),
        "conservative_supplement_unique_pdbs": int(unique_conservative_pdbs),
        "supplement_note": (
            "metadata_stage1_eligible means the row has positive affinity, heavy/light/antigen chain IDs, "
            "and protein/peptide antigen_type outside Hapten. It still needs PDB sequence extraction, "
            "deduplication, label review, and split design before it can be merged."
        ),
        "output_files": {
            "json_report": str(JSON_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "markdown_report": str(MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "missing_rows_csv": str(MISSING_ROWS_PATH.relative_to(PROJECT_ROOT)),
            "possible_candidates_csv": str(SUPPLEMENT_CANDIDATES_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    return report


def write_markdown(report: dict) -> None:
    """Write a concise human-readable overlap audit."""

    coverage_rows = report["positive_summary_row_coverage_by_tdc_pdb"]
    coverage_pdbs = report["positive_summary_pdb_coverage_by_tdc"]
    lines = [
        "# SAbDab Summary vs TDC v1 Affinity Overlap Audit",
        "",
        "## Scope",
        "",
        "- SAbDab input: `data/raw/sabdab_summary.tsv`",
        "- TDC v1 input: processed antigen-group split train/val/test CSVs",
        "- Primary overlap key: SAbDab `pdb` vs TDC `Antibody_ID` treated as PDB-like IDs",
        "- Caution: PDB overlap is a structure-level coverage check; it is not a full chain/sequence pair match.",
        "",
        "## Affinity rows in SAbDab summary",
        "",
        f"- Summary total rows: {report['summary_total_rows']}",
        f"- Affinity non-empty rows: {report['affinity_nonempty_rows']}",
        f"- Affinity numeric rows: {report['affinity_numeric_rows']}",
        f"- Affinity positive rows: {report['affinity_positive_rows']}",
        f"- Positive-affinity unique PDBs: {report['affinity_positive_unique_pdbs']}",
        "",
        "## TDC overlap",
        "",
        f"- TDC processed rows: {report['tdc_rows']}",
        f"- TDC unique `{report['tdc_pdb_column_used']}` IDs used as PDBs: {report['tdc_unique_antibody_ids_as_pdbs']}",
        f"- Positive SAbDab rows with TDC PDB overlap: {report['positive_summary_rows_with_tdc_pdb_overlap']}",
        f"- Positive SAbDab rows without TDC PDB overlap: {report['positive_summary_rows_without_tdc_pdb_overlap']}",
        f"- Positive SAbDab unique PDB overlap with TDC: {report['positive_summary_unique_pdb_overlap_with_tdc']}",
        f"- Positive SAbDab unique PDBs missing from TDC: {report['positive_summary_unique_pdb_missing_from_tdc']}",
        f"- Positive SAbDab row coverage by TDC PDB: {coverage_rows:.2%}",
        f"- Positive SAbDab PDB coverage by TDC: {coverage_pdbs:.2%}",
        f"- TDC PDB IDs not found among positive-affinity SAbDab summary PDBs: {report['tdc_pdbs_not_in_positive_summary']}",
        "",
        "## Missing-from-TDC supplement screen",
        "",
        f"- Positive SAbDab rows missing from TDC: {report['positive_summary_rows_without_tdc_pdb_overlap']}",
        f"- Metadata-level possible supplement rows: {report['metadata_possible_supplement_rows']}",
        f"- Metadata-level possible supplement unique PDBs: {report['metadata_possible_supplement_unique_pdbs']}",
        f"- Metadata possible rows with Hchain == Lchain caution: {report['metadata_possible_candidates_with_same_heavy_light_chain_id']}",
        f"- Metadata possible rows with numeric/suspicious affinity_method: {report['metadata_possible_candidates_with_suspicious_affinity_method']}",
        f"- Conservative supplement rows after those cautions: {report['conservative_supplement_rows']}",
        f"- Conservative supplement unique PDBs: {report['conservative_supplement_unique_pdbs']}",
        "",
        "Judgement counts:",
        "",
    ]
    lines.extend(
        f"- `{reason}`: {count}"
        for reason, count in report["missing_positive_summary_rows_supplement_judgement_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Rows marked as possible candidates have enough summary metadata to try the existing PDB sequence extraction path.",
            "- They are not merge-ready yet. Sequence extraction can fail, rows can duplicate TDC sequences, "
            "and a new merged dataset needs a leakage-aware split policy.",
            "- Missing Hapten/non-sequence antigen rows should not be added to the current sequence-only affinity task.",
            "",
            "## Output files",
            "",
            f"- Missing rows: `{report['output_files']['missing_rows_csv']}`",
            f"- Possible supplement candidates: `{report['output_files']['possible_candidates_csv']}`",
            f"- JSON report: `{report['output_files']['json_report']}`",
            "",
        ]
    )
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: dict) -> None:
    """Print key audit answers in terminal-friendly language."""

    print("SAbDab summary vs TDC v1 affinity overlap audit complete.")
    print(f"Affinity non-empty rows in summary.tsv: {report['affinity_nonempty_rows']}")
    print(f"Positive numeric affinity rows in summary.tsv: {report['affinity_positive_rows']}")
    print(f"Positive-affinity unique PDBs in summary.tsv: {report['affinity_positive_unique_pdbs']}")
    print(
        "PDB overlap with TDC v1: "
        f"{report['positive_summary_unique_pdb_overlap_with_tdc']} / "
        f"{report['affinity_positive_unique_pdbs']} "
        f"({report['positive_summary_pdb_coverage_by_tdc']:.2%})"
    )
    print(
        "Row coverage by TDC PDB: "
        f"{report['positive_summary_rows_with_tdc_pdb_overlap']} / "
        f"{report['affinity_positive_rows']} "
        f"({report['positive_summary_row_coverage_by_tdc_pdb']:.2%})"
    )
    print(f"Positive-affinity SAbDab rows missing from TDC v1: {report['positive_summary_rows_without_tdc_pdb_overlap']}")
    print(f"Metadata-level possible supplement rows: {report['metadata_possible_supplement_rows']}")
    print(f"Possible supplement unique PDBs: {report['metadata_possible_supplement_unique_pdbs']}")
    print(
        "Possible rows with extra metadata cautions: "
        f"Hchain==Lchain {report['metadata_possible_candidates_with_same_heavy_light_chain_id']}, "
        f"suspicious method {report['metadata_possible_candidates_with_suspicious_affinity_method']}"
    )
    print(
        "Conservative supplement screen after cautions: "
        f"{report['conservative_supplement_rows']} rows / "
        f"{report['conservative_supplement_unique_pdbs']} PDBs"
    )
    print("Important caution: possible candidates still need sequence extraction, dedup, and split review.")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Missing rows CSV: {MISSING_ROWS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Possible candidates CSV: {SUPPLEMENT_CANDIDATES_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Run the overlap audit and save report artifacts."""

    summary = load_summary()
    tdc = load_tdc_v1()
    positive_rows, summary_stats = prepare_summary_affinity_rows(summary)
    missing_rows, overlap = overlap_stats(positive_rows, tdc)
    flagged_missing_rows = add_supplement_flags(missing_rows)

    save_missing_rows(flagged_missing_rows)
    report = build_report(summary_stats, overlap, flagged_missing_rows)

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_markdown(report)
    print_summary(report)


if __name__ == "__main__":
    main()
