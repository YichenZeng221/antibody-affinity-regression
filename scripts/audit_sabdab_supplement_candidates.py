"""Audit conservative SAbDab supplement candidates without downloading files.

:
:

    overlap audit  conservative supplement rows,
     processed SAbDab CSV  sequence?

 audit?
- summary TSV  PDB ID  chain ID, sequence
- sequence-only SAbDab processed CSV  PDB chain extraction,
   exact match,
-  processed files  sequence,,
  , PDB

``supplement_ready_candidates.csv``  merged dataset
 split
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERLAP_AUDIT_CANDIDATE_PATHS = [
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "tdc_v1"
    / "sabdab_overlap_audit"
    / "sabdab_possible_supplement_candidates.csv",
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "sabdab_overlap"
    / "sabdab_possible_supplement_candidates.csv",
]
SEQUENCE_ONLY_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sequence_only"
TDC_SPLIT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sabdab_supplement"
JSON_REPORT_PATH = OUTPUT_DIR / "supplement_candidate_audit.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "supplement_candidate_audit.md"
READY_CANDIDATES_PATH = OUTPUT_DIR / "supplement_ready_candidates.csv"

SPLITS = ["train", "val", "test"]
CHAIN_KEY_COLUMNS = ["pdb", "Hchain", "Lchain", "antigen_chain"]
SEQUENCE_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
TARGET_COLUMN = "neg_log10_affinity"


def find_candidate_path() -> Path:
    """Find the candidate CSV path saved by the previous overlap audit."""

    for path in OVERLAP_AUDIT_CANDIDATE_PATHS:
        if path.exists():
            return path
    readable_paths = "\n".join(f"  - {path}" for path in OVERLAP_AUDIT_CANDIDATE_PATHS)
    raise FileNotFoundError(f"Cannot find supplement candidate CSV. Checked:\n{readable_paths}")


def normalize_text(value: object) -> str:
    """Normalize PDB/chain metadata for exact local lookup."""

    return str(value).strip().upper()


def is_missing(value: object) -> bool:
    """Treat blank and common NA spellings as missing."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def load_conservative_candidates(candidate_path: Path) -> pd.DataFrame:
    """Read only candidates already marked conservative by overlap audit."""

    candidates = pd.read_csv(candidate_path)
    required_columns = {
        *CHAIN_KEY_COLUMNS,
        "affinity",
        "affinity_method",
        "conservative_supplement_candidate",
    }
    missing_columns = required_columns - set(candidates.columns)
    if missing_columns:
        raise ValueError(f"{candidate_path} is missing columns: {sorted(missing_columns)}")

    conservative_mask = candidates["conservative_supplement_candidate"].astype(bool)
    conservative = candidates[conservative_mask].copy().reset_index(drop=True)
    conservative["candidate_row_id"] = [f"SABDAB_SUPP_{index + 1:04d}" for index in range(len(conservative))]
    return conservative


def load_local_sequence_only_rows() -> pd.DataFrame:
    """Load local SAbDab sequence-only processed rows as offline sequence source."""

    frames = []
    for split_name in SPLITS:
        path = SEQUENCE_ONLY_DIR / f"{split_name}.csv"
        if not path.exists():
            continue
        dataframe = pd.read_csv(path)
        dataframe["sequence_source_split"] = split_name
        dataframe["sequence_source_file"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(dataframe)

    if not frames:
        raise FileNotFoundError(
            f"No local sequence-only CSV files found under {SEQUENCE_ONLY_DIR}. "
            "This audit will not download PDB files."
        )

    source = pd.concat(frames, ignore_index=True)
    required_columns = {*CHAIN_KEY_COLUMNS, *SEQUENCE_COLUMNS, "affinity", TARGET_COLUMN}
    missing_columns = required_columns - set(source.columns)
    if missing_columns:
        raise ValueError(f"Local sequence source is missing columns: {sorted(missing_columns)}")
    return source


def load_tdc_v1() -> pd.DataFrame:
    """Load TDC v1 train/val/test for overlap checks."""

    frames = []
    for split_name in SPLITS:
        path = TDC_SPLIT_DIR / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find TDC v1 split {path}")
        dataframe = pd.read_csv(path)
        dataframe["tdc_split"] = split_name
        frames.append(dataframe)
    tdc = pd.concat(frames, ignore_index=True)
    required_columns = {"antibody_id", *SEQUENCE_COLUMNS, TARGET_COLUMN}
    missing_columns = required_columns - set(tdc.columns)
    if missing_columns:
        raise ValueError(f"TDC v1 data is missing columns: {sorted(missing_columns)}")
    return tdc


def add_lookup_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add normalized exact keys for PDB + chain metadata."""

    keyed = dataframe.copy()
    for column_name in CHAIN_KEY_COLUMNS:
        keyed[f"{column_name}_key"] = keyed[column_name].map(normalize_text)
    return keyed


def local_sequence_lookup(candidates: pd.DataFrame, sequence_source: pd.DataFrame) -> pd.DataFrame:
    """Exact-match candidates to already extracted SAbDab sequences.

     fuzzy match
    PDB  chain ID , biological pair
     exact match PDB + Hchain + Lchain + antigen_chain
    """

    keyed_candidates = add_lookup_keys(candidates)
    keyed_source = add_lookup_keys(sequence_source)
    join_keys = [f"{column_name}_key" for column_name in CHAIN_KEY_COLUMNS]

    source_columns = [
        *join_keys,
        *SEQUENCE_COLUMNS,
        "affinity",
        TARGET_COLUMN,
        "sequence_source_split",
        "sequence_source_file",
    ]
    source_for_join = keyed_source[source_columns].drop_duplicates(join_keys, keep="first")
    audited = keyed_candidates.merge(
        source_for_join,
        on=join_keys,
        how="left",
        suffixes=("_candidate", "_from_local_sequence"),
    )

    audited["has_local_sequences"] = audited["heavy_sequence"].notna()
    audited["sequence_lookup_status"] = audited["has_local_sequences"].map(
        {True: "found_exact_local_sequence_only_match", False: "missing_exact_local_processed_sequence"}
    )
    return audited


def add_target_checks(dataframe: pd.DataFrame, tdc: pd.DataFrame) -> pd.DataFrame:
    """Check raw affinity and -log10 target sanity."""

    audited = dataframe.copy()
    audited["affinity_numeric"] = pd.to_numeric(audited["affinity_candidate"], errors="coerce")
    audited["affinity_positive"] = audited["affinity_numeric"] > 0

    audited["computed_neg_log10_affinity"] = audited["affinity_numeric"].map(
        lambda value: -math.log10(value) if pd.notna(value) and value > 0 else float("nan")
    )
    audited["local_target_numeric"] = pd.to_numeric(audited[TARGET_COLUMN], errors="coerce")
    audited["target_matches_local_sequence_source"] = (
        (audited["computed_neg_log10_affinity"] - audited["local_target_numeric"]).abs() < 1e-8
    )

    tdc_target = pd.to_numeric(tdc[TARGET_COLUMN], errors="coerce").dropna()
    target_min = float(tdc_target.min())
    target_max = float(tdc_target.max())
    audited["target_in_tdc_v1_range"] = audited["computed_neg_log10_affinity"].between(target_min, target_max)
    audited["target_check_status"] = "ok"
    audited.loc[~audited["affinity_positive"], "target_check_status"] = "invalid_non_positive_affinity"
    audited.loc[
        audited["affinity_positive"] & audited["computed_neg_log10_affinity"].isna(),
        "target_check_status",
    ] = "cannot_compute_neg_log10_affinity"
    return audited


def key_set(dataframe: pd.DataFrame, columns: list[str]) -> set[str]:
    """Create set keys for overlap checks from one or more columns."""

    usable = dataframe.copy()
    for column_name in columns:
        usable = usable[~usable[column_name].map(is_missing)]
    if len(usable) == 0:
        return set()
    return set(usable[columns].astype(str).agg("||".join, axis=1))


def add_tdc_overlap_checks(dataframe: pd.DataFrame, tdc: pd.DataFrame) -> pd.DataFrame:
    """Flag candidate overlap with all TDC v1 splits."""

    audited = dataframe.copy()
    tdc_pdbs = {normalize_text(value) for value in tdc["antibody_id"] if not is_missing(value)}
    tdc_antigens = key_set(tdc, ["antigen_sequence"])
    tdc_triplets = key_set(tdc, SEQUENCE_COLUMNS)

    audited["tdc_pdb_overlap"] = audited["pdb"].map(normalize_text).isin(tdc_pdbs)
    audited["tdc_antigen_sequence_overlap"] = audited["antigen_sequence"].map(
        lambda value: str(value) in tdc_antigens if not is_missing(value) else False
    )
    audited["tdc_triplet_overlap"] = audited.apply(
        lambda row: "||".join(str(row[column]) for column in SEQUENCE_COLUMNS) in tdc_triplets
        if row["has_local_sequences"]
        else False,
        axis=1,
    )
    return audited


def add_conflict_checks(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Flag conflicting affinity values for the same PDB/chain combo."""

    audited = dataframe.copy()
    join_keys = [f"{column_name}_key" for column_name in CHAIN_KEY_COLUMNS]
    candidate_target_range = audited.groupby(join_keys, dropna=False)["computed_neg_log10_affinity"].transform(
        lambda values: values.max() - values.min()
    )
    audited["same_pdb_chain_combo_conflicting_affinity"] = candidate_target_range.fillna(0) > 1e-8
    return audited


def add_ready_decision(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Mark candidates that can move to a future supplement build step."""

    audited = dataframe.copy()
    audited["supplement_ready"] = (
        audited["has_local_sequences"]
        & audited["affinity_positive"]
        & audited["computed_neg_log10_affinity"].notna()
        & ~audited["same_pdb_chain_combo_conflicting_affinity"]
        & ~audited["tdc_triplet_overlap"]
    )

    reasons = []
    for _, row in audited.iterrows():
        row_reasons = []
        if not row["has_local_sequences"]:
            row_reasons.append("no_exact_local_processed_sequence")
        if not row["affinity_positive"]:
            row_reasons.append("invalid_non_positive_affinity")
        if pd.isna(row["computed_neg_log10_affinity"]):
            row_reasons.append("missing_neg_log10_affinity")
        if row["same_pdb_chain_combo_conflicting_affinity"]:
            row_reasons.append("conflicting_affinity_same_pdb_chain_combo")
        if row["tdc_triplet_overlap"]:
            row_reasons.append("duplicate_triplet_with_tdc_v1")
        if not row_reasons:
            row_reasons.append("ready_for_future_supplement_build")
        reasons.append("|".join(row_reasons))
    audited["ready_decision_reason"] = reasons
    return audited


def json_safe_counts(series: pd.Series) -> dict[str, int]:
    """Return value counts with JSON-safe keys."""

    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def numeric_summary(series: pd.Series) -> dict:
    """Return compact numeric summary."""

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


def build_report(audited: pd.DataFrame, candidate_path: Path, tdc: pd.DataFrame) -> dict:
    """Build structured report for JSON and Markdown."""

    ready = audited[audited["supplement_ready"]]
    tdc_target_stats = numeric_summary(tdc[TARGET_COLUMN])
    report = {
        "candidate_input": str(candidate_path.relative_to(PROJECT_ROOT)),
        "local_sequence_source": str(SEQUENCE_ONLY_DIR.relative_to(PROJECT_ROOT)),
        "conservative_candidate_rows": int(len(audited)),
        "conservative_candidate_unique_pdbs": int(audited["pdb"].map(normalize_text).nunique()),
        "rows_with_local_sequences": int(audited["has_local_sequences"].sum()),
        "rows_without_local_sequences": int((~audited["has_local_sequences"]).sum()),
        "sequence_lookup_status_counts": json_safe_counts(audited["sequence_lookup_status"]),
        "target_check_status_counts": json_safe_counts(audited["target_check_status"]),
        "candidate_neg_log10_affinity_stats": numeric_summary(audited["computed_neg_log10_affinity"]),
        "tdc_v1_neg_log10_affinity_stats": tdc_target_stats,
        "target_outside_tdc_v1_range_rows": int((~audited["target_in_tdc_v1_range"]).sum()),
        "conflicting_same_pdb_chain_combo_rows": int(
            audited["same_pdb_chain_combo_conflicting_affinity"].sum()
        ),
        "tdc_overlap": {
            "pdb_or_antibody_id_overlap_rows": int(audited["tdc_pdb_overlap"].sum()),
            "antigen_sequence_overlap_rows": int(audited["tdc_antigen_sequence_overlap"].sum()),
            "heavy_light_antigen_triplet_overlap_rows": int(audited["tdc_triplet_overlap"].sum()),
        },
        "supplement_ready_rows": int(len(ready)),
        "supplement_ready_unique_pdbs": int(ready["pdb"].map(normalize_text).nunique()),
        "ready_decision_reason_counts": json_safe_counts(audited["ready_decision_reason"]),
        "notes": {
            "sequence_source_note": (
                "Only exact matches from local data/processed_affinity/sequence_only CSVs are reused. "
                "No PDB download or online sequence extraction is performed."
            ),
            "antigen_overlap_note": (
                "Antigen sequence overlap with TDC is flagged. It is not automatically an exact duplicate "
                "unless the heavy+light+antigen triplet also overlaps."
            ),
            "ready_note": (
                "Ready means local sequences exist, target is computable, no same PDB/chain target conflict "
                "was seen, and the exact sequence triplet is not already in TDC v1."
            ),
        },
    }
    return report


def save_ready_candidates(audited: pd.DataFrame) -> None:
    """Write only rows marked ready for a future supplement builder."""

    ready = audited[audited["supplement_ready"]].copy()
    output_columns = [
        "candidate_row_id",
        "pdb",
        "Hchain",
        "Lchain",
        "antigen_chain",
        "antigen_type",
        "antigen_name",
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        "affinity_numeric",
        "computed_neg_log10_affinity",
        "affinity_method",
        "sequence_source_split",
        "sequence_source_file",
        "tdc_pdb_overlap",
        "tdc_antigen_sequence_overlap",
        "tdc_triplet_overlap",
        "target_in_tdc_v1_range",
        "ready_decision_reason",
    ]
    available_columns = [column_name for column_name in output_columns if column_name in ready.columns]
    ready[available_columns].to_csv(READY_CANDIDATES_PATH, index=False)


def write_markdown(report: dict) -> None:
    """Write a readable audit summary."""

    lines = [
        "# SAbDab Supplement Candidate Extraction Audit",
        "",
        "## Scope",
        "",
        f"- Conservative candidate input: `{report['candidate_input']}`",
        f"- Offline sequence source: `{report['local_sequence_source']}`",
        "- No network download and no new PDB parsing were used.",
        "",
        "## Conservative candidate extraction",
        "",
        f"- Conservative rows audited: {report['conservative_candidate_rows']}",
        f"- Conservative unique PDBs: {report['conservative_candidate_unique_pdbs']}",
        f"- Rows with exact local three-sequence lookup: {report['rows_with_local_sequences']}",
        f"- Rows missing exact local processed sequence: {report['rows_without_local_sequences']}",
        f"- Lookup status counts: `{report['sequence_lookup_status_counts']}`",
        "",
        "## Target checks",
        "",
        f"- Target check status counts: `{report['target_check_status_counts']}`",
        f"- Candidate neg_log10_affinity stats: `{report['candidate_neg_log10_affinity_stats']}`",
        f"- TDC v1 neg_log10_affinity stats: `{report['tdc_v1_neg_log10_affinity_stats']}`",
        f"- Candidate rows outside TDC v1 target range: {report['target_outside_tdc_v1_range_rows']}",
        f"- Conflicting target rows for same PDB/chain combo: {report['conflicting_same_pdb_chain_combo_rows']}",
        "",
        "## TDC v1 overlap",
        "",
        f"- PDB / antibody_id overlap rows: {report['tdc_overlap']['pdb_or_antibody_id_overlap_rows']}",
        f"- antigen_sequence overlap rows: {report['tdc_overlap']['antigen_sequence_overlap_rows']}",
        f"- heavy+light+antigen triplet overlap rows: {report['tdc_overlap']['heavy_light_antigen_triplet_overlap_rows']}",
        "",
        "## Ready candidates",
        "",
        f"- Supplement-ready rows: {report['supplement_ready_rows']}",
        f"- Supplement-ready unique PDBs: {report['supplement_ready_unique_pdbs']}",
        f"- Ready decision counts: `{report['ready_decision_reason_counts']}`",
        "",
        "Ready here means the row can move into a future supplement dataset builder. "
        "It still needs merged-dataset dedup and a fresh leakage-aware split.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {text}" for text in report["notes"].values())
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_human_summary(report: dict) -> None:
    """Print plain-language audit result."""

    duplicate_rows = report["tdc_overlap"]["heavy_light_antigen_triplet_overlap_rows"]
    print("SAbDab conservative supplement candidate audit complete.")
    print(
        f"{report['conservative_candidate_rows']} conservative rows ,"
        f"{report['rows_with_local_sequences']}  processed SAbDab CSV  sequence"
    )
    print(
        " TDC v1  overlap:"
        f"PDB/antibody_id {report['tdc_overlap']['pdb_or_antibody_id_overlap_rows']} rows,"
        f"antigen_sequence {report['tdc_overlap']['antigen_sequence_overlap_rows']} rows,"
        f"exact triplet {duplicate_rows} rows"
    )
    print(
        f" tdc_plus_sabdab_supplement_v1 :"
        f"{report['supplement_ready_rows']} rows / {report['supplement_ready_unique_pdbs']} PDBs"
    )
    if report["supplement_ready_rows"] > 0:
        print(" supplement dataset  build , merged split")
    else:
        print(" ready , sequence extraction  supplement dataset")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Ready candidates CSV: {READY_CANDIDATES_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Run the offline supplement audit."""

    candidate_path = find_candidate_path()
    candidates = load_conservative_candidates(candidate_path)
    local_sequences = load_local_sequence_only_rows()
    tdc = load_tdc_v1()

    audited = local_sequence_lookup(candidates, local_sequences)
    audited = add_target_checks(audited, tdc)
    audited = add_tdc_overlap_checks(audited, tdc)
    audited = add_conflict_checks(audited)
    audited = add_ready_decision(audited)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(audited, candidate_path, tdc)
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_markdown(report)
    save_ready_candidates(audited)
    print_human_summary(report)


if __name__ == "__main__":
    main()
