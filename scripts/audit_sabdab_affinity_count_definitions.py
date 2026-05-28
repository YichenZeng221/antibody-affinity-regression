"""Audit several SAbDab affinity-count definitions against local TDC v1.

:
SAbDab  700+ usable affinity samples,,
 ``usable`` :

1. summary.tsv  affinity metadata ?
2. delta_g ?
3.  heavy/light/antigen chain ID ?
4.  antigen  sequence-based protein antigen ?

 metadata audit, PDB, dataset,
 sequence regression
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "sabdab_summary.tsv"
TDC_SPLIT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sabdab_overlap_audit"
JSON_REPORT_PATH = OUTPUT_DIR / "sabdab_affinity_count_definitions.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "sabdab_affinity_count_definitions.md"

TDC_SPLITS = ["train", "val", "test"]
CHAIN_COLUMNS = ["Hchain", "Lchain", "antigen_chain"]
COMBO_COLUMNS = ["pdb", *CHAIN_COLUMNS]


def is_missing(value: object) -> bool:
    """Treat blank cells and common NA spellings as missing metadata."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def normalize_id(value: object) -> str:
    """Normalize PDB-like identifiers before comparing sources."""

    return str(value).strip().lower()


def normalize_chain_value(value: object) -> str:
    """Normalize chain cells for unique PDB+chain-combo counting."""

    return "<missing>" if is_missing(value) else str(value).strip().upper()


def numeric_mask(series: pd.Series) -> pd.Series:
    """Return True where a text column can be converted to a number."""

    return pd.to_numeric(series, errors="coerce").notna()


def looks_suspicious_affinity_method(value: object) -> bool:
    """Flag PMID-like numeric affinity_method cells.

    ``affinity_method``  SPR / ITC / ELISA
    , clean_v2 audit  metadata ,
     strict count 
    """

    text = str(value).strip()
    return bool(text) and bool(re.fullmatch(r"\d+(?:\.\d+)?", text))


def load_summary() -> pd.DataFrame:
    """Read SAbDab summary TSV as text before making numeric flags."""

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Cannot find {SUMMARY_PATH}")

    summary = pd.read_csv(SUMMARY_PATH, sep="\t", dtype=str, keep_default_na=False)
    required_columns = {
        "pdb",
        "Hchain",
        "Lchain",
        "antigen_chain",
        "antigen_type",
        "affinity",
        "delta_g",
        "affinity_method",
    }
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(f"SAbDab summary missing columns: {sorted(missing_columns)}")
    return summary


def load_tdc_v1_pdbs() -> tuple[pd.DataFrame, set[str], str]:
    """Read TDC v1 splits and collect PDB-like antibody IDs.

    TDC v1 uses ``antibody_id`` and also keeps original ``Antibody_ID``.
    In Protein_SAbDab those IDs are PDB-like, so they are the conservative
    structure-level key available for this metadata overlap check.
    """

    frames = []
    for split_name in TDC_SPLITS:
        path = TDC_SPLIT_DIR / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find TDC v1 split: {path}")
        frame = pd.read_csv(path)
        frame["split"] = split_name
        frames.append(frame)

    tdc = pd.concat(frames, ignore_index=True)
    id_column = "antibody_id" if "antibody_id" in tdc.columns else "Antibody_ID"
    if id_column not in tdc.columns:
        raise ValueError("TDC v1 needs antibody_id or Antibody_ID for PDB coverage audit.")

    tdc_pdbs = {
        normalize_id(value)
        for value in tdc[id_column]
        if not is_missing(value)
    }
    return tdc, tdc_pdbs, id_column


def prepare_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Add reusable metadata-quality flags to the SAbDab summary rows."""

    prepared = summary.copy()
    prepared["pdb_norm"] = prepared["pdb"].map(normalize_id)
    prepared["affinity_present"] = ~prepared["affinity"].map(is_missing)
    prepared["affinity_numeric"] = pd.to_numeric(prepared["affinity"], errors="coerce")
    prepared["affinity_is_numeric"] = prepared["affinity_numeric"].notna()
    prepared["affinity_positive"] = prepared["affinity_numeric"] > 0
    prepared["delta_g_present"] = ~prepared["delta_g"].map(is_missing)
    prepared["delta_g_numeric"] = pd.to_numeric(prepared["delta_g"], errors="coerce")
    prepared["delta_g_is_numeric"] = prepared["delta_g_numeric"].notna()

    antigen_text = prepared["antigen_type"].astype(str).str.lower()
    prepared["protein_antigen"] = antigen_text.str.contains("protein", na=False)
    prepared["hapten_antigen"] = antigen_text.str.contains("hapten", na=False)
    prepared["protein_not_hapten_antigen"] = prepared["protein_antigen"] & ~prepared["hapten_antigen"]

    for chain_column in CHAIN_COLUMNS:
        prepared[f"{chain_column}_present"] = ~prepared[chain_column].map(is_missing)

    prepared["all_chain_ids_complete"] = (
        prepared["Hchain_present"]
        & prepared["Lchain_present"]
        & prepared["antigen_chain_present"]
    )
    prepared["heavy_light_chain_ids_different"] = (
        prepared["Hchain"].astype(str).str.strip().str.upper()
        != prepared["Lchain"].astype(str).str.strip().str.upper()
    )
    prepared["suspicious_numeric_affinity_method"] = prepared["affinity_method"].map(
        looks_suspicious_affinity_method
    )
    return prepared


def build_definition_masks(prepared: pd.DataFrame) -> OrderedDict[str, pd.Series]:
    """Build every row-selection rule requested by the audit."""

    #  affinity regression  affinity  0, -log10
    # delta_g  target,affinity  delta_g 
    #  positive affinity OR numeric delta_g
    masks: OrderedDict[str, pd.Series] = OrderedDict()
    masks["total_rows"] = pd.Series(True, index=prepared.index)
    masks["affinity_nonempty"] = prepared["affinity_present"]
    masks["affinity_numeric"] = prepared["affinity_is_numeric"]
    masks["affinity_gt_0"] = prepared["affinity_positive"]
    masks["delta_g_nonempty"] = prepared["delta_g_present"]
    masks["delta_g_numeric"] = prepared["delta_g_is_numeric"]
    masks["affinity_gt_0_or_delta_g_numeric"] = (
        prepared["affinity_positive"] | prepared["delta_g_is_numeric"]
    )
    masks["protein_antigen_and_affinity_gt_0"] = (
        prepared["protein_not_hapten_antigen"] & prepared["affinity_positive"]
    )
    masks["complete_chain_ids_and_affinity_gt_0"] = (
        prepared["all_chain_ids_complete"] & prepared["affinity_positive"]
    )
    masks["strict_regression_ready_metadata"] = (
        prepared["affinity_positive"]
        & prepared["all_chain_ids_complete"]
        & prepared["protein_not_hapten_antigen"]
        & prepared["heavy_light_chain_ids_different"]
        & ~prepared["suspicious_numeric_affinity_method"]
    )
    return masks


def pdb_chain_combo_count(rows: pd.DataFrame) -> int:
    """Count unique PDB + Hchain + Lchain + antigen_chain metadata combos."""

    if len(rows) == 0:
        return 0

    normalized = pd.DataFrame(index=rows.index)
    normalized["pdb"] = rows["pdb"].map(normalize_id)
    for chain_column in CHAIN_COLUMNS:
        normalized[chain_column] = rows[chain_column].map(normalize_chain_value)
    return int(normalized.drop_duplicates(COMBO_COLUMNS).shape[0])


def summarize_definition(name: str, rows: pd.DataFrame, tdc_pdbs: set[str]) -> dict:
    """Count one definition and its structure-level TDC PDB coverage."""

    pdbs = {value for value in rows["pdb_norm"] if not is_missing(value)}
    covered_pdbs = pdbs & tdc_pdbs
    uncovered_pdbs = pdbs - tdc_pdbs
    rows_covered = rows["pdb_norm"].isin(tdc_pdbs)

    return {
        "definition": name,
        "row_count": int(len(rows)),
        "unique_pdb_count": int(len(pdbs)),
        "unique_pdb_hchain_lchain_antigen_chain_count": pdb_chain_combo_count(rows),
        "rows_with_tdc_pdb_coverage": int(rows_covered.sum()),
        "rows_without_tdc_pdb_coverage": int((~rows_covered).sum()),
        "pdbs_covered_by_tdc": int(len(covered_pdbs)),
        "pdbs_not_covered_by_tdc": int(len(uncovered_pdbs)),
        "tdc_pdb_coverage_fraction": float(len(covered_pdbs) / len(pdbs)) if pdbs else None,
    }


def definition_notes() -> dict[str, str]:
    """Human-readable definition notes saved in JSON and Markdown."""

    return {
        "total_rows": "All rows in the local SAbDab summary TSV.",
        "affinity_nonempty": "The affinity cell is not blank/NA text.",
        "affinity_numeric": "The affinity cell converts to a number; zero still counts here.",
        "affinity_gt_0": "Affinity is numeric and > 0, so -log10(affinity) is defined.",
        "delta_g_nonempty": "The delta_g cell is not blank/NA text.",
        "delta_g_numeric": "The delta_g cell converts to a number.",
        "affinity_gt_0_or_delta_g_numeric": "Positive affinity OR numeric delta_g. delta_g is an alternative continuous target.",
        "protein_antigen_and_affinity_gt_0": "Positive affinity and antigen_type contains protein but not hapten.",
        "complete_chain_ids_and_affinity_gt_0": "Positive affinity plus Hchain, Lchain, and antigen_chain metadata present.",
        "strict_regression_ready_metadata": (
            "Positive affinity, complete H/L/antigen chain IDs, protein antigen outside hapten, "
            "Hchain != Lchain, and affinity_method is not numeric/suspicious."
        ),
    }


def nearest_700_definition(definitions: list[dict]) -> dict:
    """Find the non-total definition whose row count is nearest to 700."""

    candidates = [row for row in definitions if row["definition"] != "total_rows"]
    return min(candidates, key=lambda row: abs(row["row_count"] - 700))


def build_interpretation(definitions: list[dict]) -> dict:
    """Answer the user-facing questions from the computed audit counts."""

    by_name = {row["definition"]: row for row in definitions}
    nearest = nearest_700_definition(definitions)
    strict = by_name["strict_regression_ready_metadata"]
    strict_uncovered = strict["pdbs_not_covered_by_tdc"]
    affinity_or_delta = by_name["affinity_gt_0_or_delta_g_numeric"]

    return {
        "definition_closest_to_700_by_row_count": {
            "definition": nearest["definition"],
            "row_count": nearest["row_count"],
            "distance_from_700": int(abs(nearest["row_count"] - 700)),
        },
        "closest_700_answer": (
            f"No local metadata definition reaches 700 rows. The closest non-total definition here is "
            f"{nearest['definition']} with {nearest['row_count']} rows."
        ),
        "affinity_or_delta_g_answer": (
            f"In this local summary, positive affinity OR numeric delta_g gives "
            f"{affinity_or_delta['row_count']} rows; delta_g does not expand the positive-affinity set."
        ),
        "direct_sequence_regression_answer": (
            "No. A broad affinity/delta_g metadata count is not automatically sequence-regression-ready. "
            "A sequence-only sample still needs usable target metadata, heavy/light/antigen chain IDs, "
            "a sequence antigen policy, sequence extraction, deduplication, and split checks."
        ),
        "strict_missing_from_tdc_answer": (
            f"The strict metadata screen leaves {strict['row_count']} rows / {strict['unique_pdb_count']} PDBs. "
            f"At PDB level, {strict_uncovered} of those PDBs are not covered by current TDC v1."
        ),
        "build_supplement_v2_recommendation": (
            "Do not expect a large supplement_v2 from this local summary alone. "
            "A small follow-up is reasonable only for strict rows not covered by TDC PDB IDs, "
            "with offline sequence extraction and duplicate review before merge."
        ),
    }


def build_report(prepared: pd.DataFrame, tdc: pd.DataFrame, tdc_id_column: str, tdc_pdbs: set[str]) -> dict:
    """Build the complete JSON-serializable report."""

    definition_rows = []
    masks = build_definition_masks(prepared)
    for name, mask in masks.items():
        definition_rows.append(summarize_definition(name, prepared[mask].copy(), tdc_pdbs))

    strict_rows = prepared[masks["strict_regression_ready_metadata"]].copy()
    strict_missing_pdbs = sorted(set(strict_rows["pdb_norm"]) - tdc_pdbs)
    return {
        "inputs": {
            "sabdab_summary": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "tdc_v1_split_dir": str(TDC_SPLIT_DIR.relative_to(PROJECT_ROOT)),
        },
        "count_definition_notes": definition_notes(),
        "tdc_v1": {
            "rows": int(len(tdc)),
            "pdb_like_id_column_used": tdc_id_column,
            "unique_pdb_like_ids": int(len(tdc_pdbs)),
        },
        "definitions": definition_rows,
        "strict_metadata_rows_not_covered_by_tdc_pdb_ids": {
            "pdb_count": int(len(strict_missing_pdbs)),
            "pdb_ids": strict_missing_pdbs,
        },
        "interpretation": build_interpretation(definition_rows),
    }


def percent(value: float | None) -> str:
    """Format optional fraction as report-friendly percentage."""

    return "NA" if value is None else f"{100 * value:.2f}%"


def write_markdown(report: dict) -> None:
    """Write a short readable Markdown companion for the JSON audit."""

    notes = report["count_definition_notes"]
    interpretation = report["interpretation"]
    lines = [
        "# SAbDab Affinity Count Definitions Audit",
        "",
        "## Scope",
        "",
        f"- SAbDab metadata input: `{report['inputs']['sabdab_summary']}`",
        f"- Current TDC v1 input: `{report['inputs']['tdc_v1_split_dir']}`",
        "- This is a metadata-only audit. It does not parse PDB structures, change datasets, or train models.",
        "- TDC coverage below is PDB-level coverage: SAbDab `pdb` vs TDC PDB-like `antibody_id`.",
        "",
        "## Count Definitions",
        "",
        "| definition | row count | unique PDB | unique PDB+H+L+antigen chain | rows covered by TDC PDB | PDBs not covered by TDC | TDC PDB coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in report["definitions"]:
        lines.append(
            f"| `{row['definition']}` | {row['row_count']} | {row['unique_pdb_count']} | "
            f"{row['unique_pdb_hchain_lchain_antigen_chain_count']} | "
            f"{row['rows_with_tdc_pdb_coverage']} | {row['pdbs_not_covered_by_tdc']} | "
            f"{percent(row['tdc_pdb_coverage_fraction'])} |"
        )

    lines.extend(["", "## Definition Notes", ""])
    for definition, note in notes.items():
        lines.append(f"- `{definition}`: {note}")

    strict_missing = report["strict_metadata_rows_not_covered_by_tdc_pdb_ids"]
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Closest local tracked definition to 700 rows: {interpretation['closest_700_answer']}",
            f"- Delta-g check: {interpretation['affinity_or_delta_g_answer']}",
            f"- Can 700+ be used directly for sequence regression? {interpretation['direct_sequence_regression_answer']}",
            f"- Are more strict clean metadata candidates missing from TDC? {interpretation['strict_missing_from_tdc_answer']}",
            f"- Next build decision: {interpretation['build_supplement_v2_recommendation']}",
            "",
            "## Strict Rows Missing From TDC PDB IDs",
            "",
            f"- Count: {strict_missing['pdb_count']} PDB IDs",
            f"- PDB IDs: `{', '.join(strict_missing['pdb_ids']) if strict_missing['pdb_ids'] else 'none'}`",
            "",
            "## Reading Guide",
            "",
            "- The broad affinity/delta_g rows answer what metadata exists.",
            "- The strict screen is closer to what might become sequence-only regression data after sequence extraction.",
            "- PDB-level TDC coverage is conservative but incomplete: it does not prove an exact chain/sequence match.",
        ]
    )
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_terminal_summary(report: dict) -> None:
    """Print the most important audit answer in the terminal."""

    interpretation = report["interpretation"]
    strict = next(
        row for row in report["definitions"] if row["definition"] == "strict_regression_ready_metadata"
    )
    print("SAbDab affinity count-definition audit complete.")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(interpretation["closest_700_answer"])
    print(interpretation["affinity_or_delta_g_answer"])
    print(
        "Strict metadata screen: "
        f"{strict['row_count']} rows / {strict['unique_pdb_count']} unique PDBs / "
        f"{strict['pdbs_not_covered_by_tdc']} PDBs not covered by TDC v1."
    )
    print(f"Direct sequence-regression answer: {interpretation['direct_sequence_regression_answer']}")
    print(f"Next decision: {interpretation['build_supplement_v2_recommendation']}")


def main() -> None:
    """Load metadata, count definitions, and save reports."""

    summary = load_summary()
    prepared = prepare_summary(summary)
    tdc, tdc_pdbs, tdc_id_column = load_tdc_v1_pdbs()
    report = build_report(prepared, tdc, tdc_id_column, tdc_pdbs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print_terminal_summary(report)


if __name__ == "__main__":
    main()
