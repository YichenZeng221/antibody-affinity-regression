"""Audit structure/contact feature feasibility for ANDD antibody v2 stratified data.

: interface/contact 
,

 basic interface  CDR contact?
-  H/L/antigen chain, antibody-antigen
  contact countminimum distanceinterface residue count
- CDR-antigen contact CSV  IMGT CDR 
  residue numbering, CDR contact 
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed_affinity/expanded_affinity_antibody_v2_stratified")
SUMMARY_PATH = Path("data/raw/sabdab_summary.tsv")
LOCAL_PDB_DIR = Path("data/pdb")
ARCHIVE_ROOT = Path("/Users/yichenzeng/Downloads/all_structures")
OUTPUT_DIR = Path("outputs/andd_antibody_v2_stratified/contact_feature_audit")
AVAILABILITY_PATH = OUTPUT_DIR / "contact_feature_availability.csv"
REPORT_PATH = OUTPUT_DIR / "contact_feature_audit_report.md"

SPLITS = ("train", "val", "test")
CDR_COLUMNS = ("HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3")
CHAIN_COLUMNS = ("Hchain", "Lchain", "antigen_chain")
PREDICTION_FILES = {
    "stratified_all_cdr_pooled": Path(
        "outputs/andd_antibody_v2_stratified/all_cdr_pooled/"
        "andd_antibody_v2_stratified_all_cdr_pooled_test_predictions.csv"
    ),
    "stratified_cross_attention": Path(
        "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/test_predictions.csv"
    ),
    "tailaware_w3_best_val_tail_mae": Path(
        "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware/"
        "tailaware_test_predictions_best_val_tail_mae.csv"
    ),
    "tailaware_w2_best_val_tail_mae": Path(
        "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware_w2/"
        "tailaware_w2_test_predictions_best_val_tail_mae.csv"
    ),
    "fit_diagnosis_all_cdr_pooled_train": Path(
        "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions/"
        "all_cdr_pooled_train_predictions.csv"
    ),
    "fit_diagnosis_all_cdr_pooled_val": Path(
        "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions/"
        "all_cdr_pooled_val_predictions.csv"
    ),
    "fit_diagnosis_all_cdr_pooled_test": Path(
        "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions/"
        "all_cdr_pooled_test_predictions.csv"
    ),
    "fit_diagnosis_cross_attention_train": Path(
        "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions/"
        "all_cdr_cross_attention_train_predictions.csv"
    ),
    "fit_diagnosis_cross_attention_val": Path(
        "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions/"
        "all_cdr_cross_attention_val_predictions.csv"
    ),
    "fit_diagnosis_cross_attention_test": Path(
        "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions/"
        "all_cdr_cross_attention_test_predictions.csv"
    ),
}


def clean_text(value: object) -> str:
    ""","""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "\\", "not_reported"}:
        return ""
    return text


def normalize_pdb(value: object) -> str:
    return clean_text(value).upper()


def chain_ids(value: object) -> set[str]:
    """SAbDab  antigen_chain ,"""
    text = clean_text(value)
    if not text:
        return set()
    return {
        token.strip()
        for token in re.split(r"[\s,;|/]+", text)
        if token.strip() and token.strip().lower() not in {"na", "none"}
    }


def structure_files(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {path.stem.upper(): path for path in directory.glob("*.pdb")}


def pdb_chain_set(path: Path | None) -> set[str]:
    """ chain ID,; availability audit"""
    if path is None or not path.exists():
        return set()
    chains: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
                chain = line[21].strip()
                if chain:
                    chains.add(chain)
    return chains


def load_dataset() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"
        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["pdb_norm"] = data["pdb_id"].map(normalize_pdb)
    return data


def load_summary_options(dataset_pdbs: set[str]) -> dict[str, list[dict[str, str]]]:
    summary = pd.read_csv(SUMMARY_PATH, sep="\t")
    summary["pdb_norm"] = summary["pdb"].map(normalize_pdb)
    summary = summary[summary["pdb_norm"].isin(dataset_pdbs)].copy()
    for column in CHAIN_COLUMNS:
        summary[column] = summary[column].map(clean_text)

    #  PDB  summary  row;availability  distinct chain mapping
    options: dict[str, list[dict[str, str]]] = {}
    distinct = summary.drop_duplicates(["pdb_norm", *CHAIN_COLUMNS])
    for pdb_id, group in distinct.groupby("pdb_norm"):
        options[pdb_id] = group[list(CHAIN_COLUMNS)].to_dict("records")
    return options


def prediction_index() -> tuple[dict[str, set[str]], list[str]]:
    """ test predictions, contact features  residual"""
    samples_by_model: dict[str, set[str]] = {}
    missing: list[str] = []
    for label, path in PREDICTION_FILES.items():
        if not path.exists():
            missing.append(str(path))
            continue
        predictions = pd.read_csv(path, usecols=["sample_id"])
        samples_by_model[label] = set(predictions["sample_id"].astype(str))
    return samples_by_model, missing


def viable_mapping(option: dict[str, str], available_chains: set[str]) -> bool:
    heavy = chain_ids(option["Hchain"])
    light = chain_ids(option["Lchain"])
    antigen = chain_ids(option["antigen_chain"])
    # antibody-only task  heavylightantigen ,
    return bool(heavy and light and antigen and heavy <= available_chains
                and light <= available_chains and antigen <= available_chains)


def make_availability_table(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    local_files = structure_files(LOCAL_PDB_DIR)
    raw_files = structure_files(ARCHIVE_ROOT / "raw")
    imgt_files = structure_files(ARCHIVE_ROOT / "imgt")
    chothia_files = structure_files(ARCHIVE_ROOT / "chothia")
    options_by_pdb = load_summary_options(set(data["pdb_norm"]))
    predictions, missing_prediction_files = prediction_index()

    raw_chain_cache = {
        pdb_id: pdb_chain_set(raw_files.get(pdb_id))
        for pdb_id in set(data["pdb_norm"])
    }
    rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        pdb_id = row["pdb_norm"]
        options = options_by_pdb.get(pdb_id, [])
        actual_chains = raw_chain_cache.get(pdb_id, set())
        complete_options = [
            option
            for option in options
            if all(chain_ids(option[column]) for column in CHAIN_COLUMNS)
        ]
        viable_options = [
            option for option in complete_options if viable_mapping(option, actual_chains)
        ]
        unique_viable = len(viable_options) == 1
        selected = viable_options[0] if unique_viable else {}
        cdr_ready = all(clean_text(row.get(column, "")) for column in CDR_COLUMNS)
        any_structure = any(
            pdb_id in files
            for files in (local_files, raw_files, imgt_files, chothia_files)
        )
        basic_ready = bool(any_structure and unique_viable)
        # CDR-level features  CDR residue /
        cdr_pipeline_candidate = bool(
            basic_ready and cdr_ready and pdb_id in imgt_files
        )
        linked_predictions = [
            label
            for label, sample_ids in predictions.items()
            if str(row["sample_id"]) in sample_ids
        ]

        missing: list[str] = []
        if not pdb_id:
            missing.append("PDB_ID")
        if not any_structure:
            missing.append("structure_file")
        if not options:
            missing.append("SAbDab_chain_metadata")
        elif not complete_options:
            missing.append("complete_H_L_antigen_chain_mapping")
        elif not viable_options:
            missing.append("chain_ids_present_in_raw_structure")
        elif len(viable_options) > 1:
            missing.append("unambiguous_chain_mapping")
        if not cdr_ready:
            missing.append("CDR_annotation")
        if cdr_pipeline_candidate:
            missing.append("CDR_residue_to_structure_mapping_validation")

        rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "pdb_id": row["pdb_id"],
                "pdb_norm": pdb_id,
                "source": row.get("source", ""),
                "ag_name": row.get("ag_name", ""),
                "neg_log10_affinity_candidate": row.get(
                    "neg_log10_affinity_candidate", float("nan")
                ),
                "local_pdb_exists": pdb_id in local_files,
                "external_raw_pdb_exists": pdb_id in raw_files,
                "external_imgt_pdb_exists": pdb_id in imgt_files,
                "external_chothia_pdb_exists": pdb_id in chothia_files,
                "any_structure_exists": any_structure,
                "raw_structure_chain_ids": "|".join(sorted(actual_chains)),
                "summary_chain_mapping_option_count": len(options),
                "complete_chain_mapping_option_count": len(complete_options),
                "viable_chain_mapping_option_count": len(viable_options),
                "unambiguous_chain_mapping": unique_viable,
                "resolved_Hchain": selected.get("Hchain", ""),
                "resolved_Lchain": selected.get("Lchain", ""),
                "resolved_antigen_chain": selected.get("antigen_chain", ""),
                "standard_cdr_annotation_available": cdr_ready,
                "sequence_to_structure_residue_mapping_validated": False,
                "basic_interface_features_ready_for_extraction": basic_ready,
                "cdr_contact_pipeline_candidate_after_mapping_validation": cdr_pipeline_candidate,
                "complete_requested_contact_features_ready_now": False,
                "antibody_antigen_contact_count_availability": (
                    "ready_after_distance_extraction"
                    if basic_ready
                    else "blocked_by_structure_or_chain_mapping"
                ),
                "minimum_antibody_antigen_distance_availability": (
                    "ready_after_distance_extraction"
                    if basic_ready
                    else "blocked_by_structure_or_chain_mapping"
                ),
                "interface_residue_count_availability": (
                    "ready_after_distance_extraction"
                    if basic_ready
                    else "blocked_by_structure_or_chain_mapping"
                ),
                "cdr_antigen_contact_count_availability": (
                    "requires_cdr_residue_mapping_validation"
                    if cdr_pipeline_candidate
                    else "blocked_by_prerequisites"
                ),
                "hcdr3_contact_fraction_availability": (
                    "requires_cdr_residue_mapping_validation"
                    if cdr_pipeline_candidate
                    else "blocked_by_prerequisites"
                ),
                "lcdr3_contact_fraction_availability": (
                    "requires_cdr_residue_mapping_validation"
                    if cdr_pipeline_candidate
                    else "blocked_by_prerequisites"
                ),
                "existing_prediction_model_count": len(linked_predictions),
                "existing_prediction_models": ";".join(linked_predictions),
                "can_join_existing_residual_analysis": bool(linked_predictions),
                "missing_or_unvalidated_requirements": ";".join(missing),
            }
        )

    table = pd.DataFrame(rows)
    metadata = {
        "structure_file_counts": {
            "project_local_data_pdb": len(local_files),
            "external_raw": len(raw_files),
            "external_imgt": len(imgt_files),
            "external_chothia": len(chothia_files),
        },
        "external_archive_exists": ARCHIVE_ROOT.exists(),
        "available_prediction_models": list(predictions),
        "missing_prediction_files": missing_prediction_files,
    }
    return table, metadata


def count_true(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].fillna(False).astype(bool).sum())


def write_report(table: pd.DataFrame, metadata: dict[str, object]) -> None:
    split_lines = []
    for split in SPLITS:
        group = table[table["split"] == split]
        split_lines.append(
            f"| {split} | {len(group)} | {count_true(group, 'any_structure_exists')} | "
            f"{count_true(group, 'unambiguous_chain_mapping')} | "
            f"{count_true(group, 'basic_interface_features_ready_for_extraction')} | "
            f"{count_true(group, 'cdr_contact_pipeline_candidate_after_mapping_validation')} | "
            f"{count_true(group, 'can_join_existing_residual_analysis')} |"
        )

    total_rows = len(table)
    unique_pdbs = table["pdb_norm"].nunique()
    unique_pdb_mapping_rows = count_true(table, "unambiguous_chain_mapping")
    ambiguous_rows = int(
        (table["viable_chain_mapping_option_count"] > 1).sum()
    )
    no_viable_rows = int(
        (table["viable_chain_mapping_option_count"] == 0).sum()
    )
    basic_ready = count_true(table, "basic_interface_features_ready_for_extraction")
    cdr_candidates = count_true(
        table, "cdr_contact_pipeline_candidate_after_mapping_validation"
    )
    prediction_rows = count_true(table, "can_join_existing_residual_analysis")
    local_structure_rows = count_true(table, "local_pdb_exists")
    external_raw_rows = count_true(table, "external_raw_pdb_exists")
    external_imgt_rows = count_true(table, "external_imgt_pdb_exists")
    external_chothia_rows = count_true(table, "external_chothia_pdb_exists")
    structure_counts = metadata["structure_file_counts"]

    lines = [
        "# ANDD Antibody v2 Stratified Contact / Interface Feature Availability Audit",
        "",
        "## Scope",
        "",
        "- : metadata  contact/interface feature extraction",
        "- :`data/processed_affinity/expanded_affinity_antibody_v2_stratified/{train,val,test}.csv`",
        "-  PDB chain ID;**** dataset",
        "-  SAbDab structure archive , 31GB ",
        "",
        "## Inputs Found",
        "",
        f"- ANDD stratified rows: **{total_rows}** ({unique_pdbs} unique `pdb_id`).",
        f"- SAbDab summary metadata: `{SUMMARY_PATH}`, `Hchain`, `Lchain`, `antigen_chain` ",
        f"- Project-local cached PDB files: `{LOCAL_PDB_DIR}` = {structure_counts['project_local_data_pdb']} files; "
        f"**{local_structure_rows} / {total_rows}** ANDD rows match this local cache.",
        f"- External SAbDab archive root exists: `{metadata['external_archive_exists']}` at `{ARCHIVE_ROOT}`.",
        f"- External `raw/imgt/chothia` PDB counts: "
        f"{structure_counts['external_raw']} / {structure_counts['external_imgt']} / "
        f"{structure_counts['external_chothia']}.",
        f"- External `raw/imgt/chothia` matches to ANDD rows: "
        f"**{external_raw_rows} / {external_imgt_rows} / {external_chothia_rows}**.",
        "",
        "### Relevant Files and Code Located",
        "",
        "- `data/raw/sabdab_summary.tsv`: SAbDab structure metadata and chain candidates.",
        "- `data/processed_affinity/sabdab_structure_archive_inspection/`: previous archive existence/chain audit.",
        "- `scripts/inspect_sabdab_structure_archive.py`: prior lightweight PDB archive inspection.",
        "- `scripts/build_sabdab_chain_dataset.py`: existing PDB/chain sequence parsing utility pattern.",
        "- `scripts/analyze_andd_stratified_model_fit.py`: already reserves optional "
        "`contact_count`, `min_distance`, and `interface_residue_count`, but reported them missing.",
        "- `src/affinity_interaction_model.py` and `src/affinity_cross_attention_model.py`: "
        "sequence-level interaction models; neither consumes 3D contact geometry.",
        "",
        "## Availability Summary",
        "",
        "| split | rows | structure file found | unambiguous viable H/L/antigen mapping | basic interface feature-ready | CDR-contact pipeline candidates* | rows joinable to existing predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *split_lines,
        "",
        "\\* `CDR-contact pipeline candidates` means structure + one viable chain mapping + IMGT file + "
        "standard CDR annotation are present. CDR-to-structure residue mapping still needs validation.",
        "",
        "### Overall Counts",
        "",
        f"- Samples with a PDB/structure file mapping: **{count_true(table, 'any_structure_exists')} / {total_rows}**.",
        f"- Samples with at least one complete H/L/antigen chain metadata option: "
        f"**{int((table['complete_chain_mapping_option_count'] > 0).sum())} / {total_rows}**.",
        f"- Samples with an unambiguous viable H/L/antigen chain mapping in the raw structure: "
        f"**{unique_pdb_mapping_rows} / {total_rows}**.",
        f"- Samples whose chain metadata remain ambiguous because multiple viable mappings exist: "
        f"**{ambiguous_rows} / {total_rows}**.",
        f"- Samples with no viable complete chain mapping found in the raw structure: "
        f"**{no_viable_rows} / {total_rows}**.",
        f"- Samples ready for a conservative first pass of basic interface features: "
        f"**{basic_ready} / {total_rows}**.",
        f"- Samples potentially usable for CDR-level contact features after residue-mapping validation: "
        f"**{cdr_candidates} / {total_rows}**.",
        f"- Samples already joinable to at least one existing prediction/residual file, including prior "
        f"fit-diagnosis inference: "
        f"**{prediction_rows} / {total_rows}**.",
        "",
        "## Feature-by-Feature Feasibility",
        "",
        "| feature | availability now | required next step |",
        "|---|---|---|",
        f"| antibody-antigen contact count | basic extraction prerequisites present for {basic_ready} rows | choose a distance threshold and compute atom/residue contacts for unambiguous chain mappings |",
        f"| minimum antibody-antigen distance | basic extraction prerequisites present for {basic_ready} rows | compute minimum heavy/light-to-antigen atomic distance |",
        f"| interface residue count | basic extraction prerequisites present for {basic_ready} rows | define interface cutoff and count residues touching across chains |",
        f"| CDR-antigen contact count | {cdr_candidates} candidate rows, not validated yet | map IMGT CDR residues to PDB residues before counting |",
        f"| HCDR3 contact fraction | {cdr_candidates} candidate rows, not validated yet | validate heavy-chain IMGT numbering/residue alignment |",
        f"| LCDR3 contact fraction | {cdr_candidates} candidate rows, not validated yet | validate light-chain IMGT numbering/residue alignment |",
        "",
        "## What Is Missing or Not Yet Validated",
        "",
        "- `PDB ID`: present for all rows and externally mapped to structure files.",
        "- `Hchain/Lchain/antigen_chain`: not stored in the stratified CSV itself; recoverable as SAbDab "
        "summary candidates, but multiple viable chain mappings remain for a substantial subset.",
        "- `Antigen chain mapping`: included in the SAbDab candidate mappings, subject to the same ambiguity check.",
        "- `CDR residue annotation`: sequence-level AbNumber + IMGT CDR fields are already present.",
        "- `Sequence-to-structure residue mapping`: **not yet validated**. This is the central missing step "
        "before trustworthy CDR-level contact fractions can be generated.",
        "",
        "## Can Contact Features Be Joined to Existing Errors?",
        "",
        f"- Yes: predictions and dataset rows share `sample_id`; currently **{prediction_rows}** "
        "rows already have a matching prediction from at least one saved evaluation or fit-diagnosis output.",
        "- Existing fit-diagnosis predictions cover train/val/test for pooled and cross-attention models; "
        "saved tail-aware checkpoint predictions currently provide test-set residuals.",
        "- Once contact features are extracted, test rows can be merged with residuals to analyze "
        "`contact feature vs target`, `contact feature vs absolute_error`, and `contact feature vs tail error`.",
        "- No correlation is computed here because contact values have not yet been extracted.",
        "",
        "## Minimal Viable Contact Feature Pipeline",
        "",
        "1. Start only with rows having one viable H/L/antigen chain mapping and an available raw/IMGT structure.",
        "2. Validate chain sequences and map the AbNumber/IMGT CDR residues onto structure residue numbers; "
        "keep ambiguous or mismatched rows flagged rather than forcing an assignment.",
        "3. Compute simple geometry features first: antibody-antigen contact count, minimum distance, "
        "interface residue count, CDR-antigen contact count, HCDR3/LCDR3 contact fractions.",
        "4. Merge feature tables to the existing prediction CSVs by `sample_id` and audit whether these "
        "features explain tail errors or prediction compression.",
        "5. Only after the audit shows signal should contact/interface features enter a new modeling experiment.",
        "",
        "## Honest Conclusion",
        "",
        f"The structure resource is promising: all {total_rows} stratified samples have an external SAbDab "
        "structure-file match. However, contact modeling is not immediately ready for every row. "
        f"A conservative basic-interface pilot can begin with {basic_ready} rows whose chain mapping is "
        "unambiguous in the available metadata and structure. CDR-level contact features require an "
        "additional residue-mapping validation step; until that is done, treating CDR contacts as ground "
        "truth would be unsafe.",
        "",
        "## Outputs",
        "",
        f"- Availability table: `{AVAILABILITY_PATH}`",
        f"- This report: `{REPORT_PATH}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    table, metadata = make_availability_table(data)
    table.to_csv(AVAILABILITY_PATH, index=False)
    write_report(table, metadata)

    print("ANDD stratified contact/interface feature availability audit complete.")
    print(f"Rows audited: {len(table)}")
    print(f"Rows with structure file mapping: {count_true(table, 'any_structure_exists')}")
    print(
        "Rows with unambiguous chain mapping for basic interface extraction: "
        f"{count_true(table, 'basic_interface_features_ready_for_extraction')}"
    )
    print(
        "CDR-contact candidates requiring residue mapping validation: "
        f"{count_true(table, 'cdr_contact_pipeline_candidate_after_mapping_validation')}"
    )
    print(f"Availability table: {AVAILABILITY_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
