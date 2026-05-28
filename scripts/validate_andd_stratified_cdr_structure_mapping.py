"""Validate CDR-to-structure residue mapping for the ANDD stratified pilot.

 basic interface geometry pilot , chain mapping
 472 ,

 mapping :
1. CDR sequence  heavy/light sequence 
2.  sequence  raw PDB chain sequence
3. CDR  residue  amino-acid residue
4.  alignment  CDR residue mapping ,, contact

, preliminary CDR-antigen contact features pilot ,

"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from Bio import Align
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


DATA_DIR = Path("data/processed_affinity/expanded_affinity_antibody_v2_stratified")
FEATURE_DIR = Path("outputs/andd_antibody_v2_stratified/contact_feature_audit")
BASIC_FEATURE_PATH = FEATURE_DIR / "basic_interface_features.csv"
AVAILABILITY_INPUT_PATH = FEATURE_DIR / "contact_feature_availability.csv"
ARCHIVE_RAW_DIR = Path("/Users/yichenzeng/Downloads/all_structures/raw")
MAPPING_OUTPUT_PATH = FEATURE_DIR / "cdr_mapping_availability.csv"
CONTACT_OUTPUT_PATH = FEATURE_DIR / "preliminary_cdr_contact_features.csv"
REPORT_PATH = FEATURE_DIR / "cdr_mapping_validation_report.md"

TARGET_COLUMN = "neg_log10_affinity_candidate"
CDR_FIELDS = ("HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3")
HEAVY_CDR_FIELDS = ("HCDR1", "HCDR2", "HCDR3")
LIGHT_CDR_FIELDS = ("LCDR1", "LCDR2", "LCDR3")
PREDICTION_ERROR_COLUMNS = {
    "unweighted_cross_attention": (
        "unweighted_cross_attention_error",
        "unweighted_cross_attention_absolute_error",
    ),
    "tailaware_w2_best_val_tail_mae": (
        "tailaware_w2_best_val_tail_mae_error",
        "tailaware_w2_best_val_tail_mae_absolute_error",
    ),
}


def clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Za-z]", "", str(value)).upper()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "na", "none", "\\"} else text


def split_chain_ids(value: object) -> list[str]:
    text = clean_text(value)
    for separator in ("|", ",", ";", "/"):
        text = text.replace(separator, " ")
    return [token for token in text.split() if token]


def load_pilot() -> pd.DataFrame:
    basic = pd.read_csv(BASIC_FEATURE_PATH)
    basic = basic[basic["geometry_extraction_status"].eq("success")].copy()
    availability = pd.read_csv(AVAILABILITY_INPUT_PATH)[
        ["sample_id", "basic_interface_features_ready_for_extraction"]
    ]
    basic = basic.merge(availability, on="sample_id", how="left", suffixes=("", "_audit"))
    basic = basic[
        basic["basic_interface_features_ready_for_extraction_audit"].fillna(False).astype(bool)
    ].copy()
    full = pd.concat(
        [pd.read_csv(DATA_DIR / f"{split}.csv").assign(split=split) for split in ("train", "val", "test")],
        ignore_index=True,
    )
    annotations = [
        "sample_id",
        "heavy_sequence",
        "light_sequence",
        *CDR_FIELDS,
        "heavy_cdr_status",
        "light_cdr_status",
    ]
    return basic.merge(full[annotations], on="sample_id", how="left", suffixes=("", "_annotation"))


def structure_residues(chain) -> tuple[str, list[object]]:
    residues = [residue for residue in chain if is_aa(residue, standard=False)]
    sequence = "".join(
        seq1(residue.resname, custom_map={"MSE": "M"}, undef_code="X") for residue in residues
    )
    return sequence.upper(), residues


def make_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -3.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -1.0
    aligner.end_insertion_score = -1.0
    aligner.end_deletion_score = -1.0
    return aligner


def occurrence_indices(sequence: str, subsequence: str) -> list[int]:
    if not subsequence:
        return []
    starts: list[int] = []
    start = sequence.find(subsequence)
    while start >= 0:
        starts.append(start)
        start = sequence.find(subsequence, start + 1)
    return starts


def alignment_position_map(alignment) -> dict[int, int]:
    mapping: dict[int, int] = {}
    target_blocks, query_blocks = alignment.aligned
    for (target_start, target_end), (query_start, query_end) in zip(target_blocks, query_blocks):
        for offset in range(int(target_end - target_start)):
            mapping[int(target_start + offset)] = int(query_start + offset)
    return mapping


def chain_mapping_validation(
    input_sequence: str,
    cdr_sequences: dict[str, str],
    structure_sequence: str,
    structure_residue_list: list[object],
) -> dict[str, object]:
    """Return strict CDR residue mappings from input positions to PDB residues."""
    fields = list(cdr_sequences)

    def fatal_failure(reason: str, **extra: object) -> dict[str, object]:
        return {
            "status": "failed",
            "reason": reason,
            "cdr_statuses": {field: "failed" for field in fields},
            "cdr_reasons": {field: reason for field in fields},
            **extra,
        }

    if not input_sequence:
        return fatal_failure("chain_sequence_mismatch")

    ranges: dict[str, list[int]] = {}
    cdr_statuses: dict[str, str] = {}
    cdr_reasons: dict[str, str] = {}
    for field, cdr_sequence in cdr_sequences.items():
        if not cdr_sequence:
            cdr_statuses[field] = "failed"
            cdr_reasons[field] = "missing_CDR_annotation"
            continue
        starts = occurrence_indices(input_sequence, cdr_sequence)
        if len(starts) != 1:
            cdr_statuses[field] = "failed"
            cdr_reasons[field] = "insertion_deletion_ambiguity"
            continue
        ranges[field] = list(range(starts[0], starts[0] + len(cdr_sequence)))

    alignments = make_aligner().align(input_sequence, structure_sequence)
    first = alignments[0]
    input_to_structure = alignment_position_map(first)
    aligned_positions = list(input_to_structure)
    if not aligned_positions:
        return fatal_failure("chain_sequence_mismatch")
    exact_matches = sum(
        input_sequence[position] == structure_sequence[structure_position]
        for position, structure_position in input_to_structure.items()
    )
    identity = exact_matches / len(aligned_positions)
    coverage = len(aligned_positions) / len(input_sequence)
    if identity < 0.95 or coverage < 0.80:
        return fatal_failure(
            "chain_sequence_mismatch",
            alignment_identity=identity,
            alignment_coverage=coverage,
        )

    mapped_indices: dict[str, list[int]] = {}
    for field, positions in ranges.items():
        if any(position not in input_to_structure for position in positions):
            cdr_statuses[field] = "failed"
            cdr_reasons[field] = "unresolved_residues"
            continue
        structure_positions = [input_to_structure[position] for position in positions]
        if any(
            input_sequence[input_position] != structure_sequence[structure_position]
            for input_position, structure_position in zip(positions, structure_positions)
        ):
            cdr_statuses[field] = "failed"
            cdr_reasons[field] = "chain_sequence_mismatch"
            continue
        mapped_indices[field] = structure_positions
        cdr_statuses[field] = "success"
        cdr_reasons[field] = ""

    # , alignment  CDR  structure residues
    try:
        second = alignments[1]
    except IndexError:
        second = None
    if second is not None:
        second_map = alignment_position_map(second)
        for field, positions in ranges.items():
            if cdr_statuses.get(field) == "success" and all(position in second_map for position in positions):
                alternative = [second_map[position] for position in positions]
                if alternative != mapped_indices[field]:
                    cdr_statuses[field] = "failed"
                    cdr_reasons[field] = "insertion_deletion_ambiguity"
                    mapped_indices.pop(field, None)

    residue_ids = {
        field: [
            f"{structure_residue_list[position].get_parent().id}:{structure_residue_list[position].id[1]}"
            f"{str(structure_residue_list[position].id[2]).strip()}"
            for position in indices
        ]
        for field, indices in mapped_indices.items()
    }
    reasons = sorted({reason for reason in cdr_reasons.values() if reason})
    all_success = all(cdr_statuses.get(field) == "success" for field in fields)
    return {
        "status": "success" if all_success else "failed",
        "reason": ";".join(reasons),
        "alignment_identity": identity,
        "alignment_coverage": coverage,
        "mapped_indices": mapped_indices,
        "mapped_residue_ids": residue_ids,
        "cdr_statuses": cdr_statuses,
        "cdr_reasons": cdr_reasons,
    }


def residue_atom_coordinates(residue) -> list[np.ndarray]:
    return [
        atom.coord.astype(float)
        for atom in residue.get_atoms()
        if str(getattr(atom, "element", "")).strip().upper() != "H"
    ]


def cdr_contact_set(cdr_residues: list[object], antigen_residues: list[object], cutoff: float = 5.0):
    contacts: set[tuple[str, str]] = set()
    cutoff_squared = cutoff * cutoff
    for cdr_residue in cdr_residues:
        cdr_id = f"{cdr_residue.get_parent().id}:{cdr_residue.id[1]}{str(cdr_residue.id[2]).strip()}"
        cdr_coords = np.asarray(residue_atom_coordinates(cdr_residue))
        if cdr_coords.size == 0:
            continue
        for antigen_residue in antigen_residues:
            ag_id = (
                f"{antigen_residue.get_parent().id}:{antigen_residue.id[1]}"
                f"{str(antigen_residue.id[2]).strip()}"
            )
            ag_coords = np.asarray(residue_atom_coordinates(antigen_residue))
            if ag_coords.size == 0:
                continue
            squared = np.sum((cdr_coords[:, None, :] - ag_coords[None, :, :]) ** 2, axis=2)
            if bool(np.any(squared <= cutoff_squared)):
                contacts.add((cdr_id, ag_id))
    return contacts


def minimum_cdr_distance(cdr_residues: list[object], antigen_residues: list[object]) -> float:
    cdr_coords = np.asarray(
        [coord for residue in cdr_residues for coord in residue_atom_coordinates(residue)]
    )
    antigen_coords = np.asarray(
        [coord for residue in antigen_residues for coord in residue_atom_coordinates(residue)]
    )
    if cdr_coords.size == 0 or antigen_coords.size == 0:
        return float("nan")
    minimum = float("inf")
    for start in range(0, len(cdr_coords), 128):
        distances = np.linalg.norm(
            cdr_coords[start : start + 128, None, :] - antigen_coords[None, :, :], axis=2
        )
        minimum = min(minimum, float(distances.min()))
    return minimum


def build_contact_features(
    row: pd.Series,
    chain_residues: dict[str, list[object]],
    mapped_indices: dict[str, list[int]],
    antigen_residues: list[object],
) -> dict[str, float]:
    mapped_residues = {
        field: [chain_residues[field[0]][index] for index in indices]
        for field, indices in mapped_indices.items()
    }
    output = {feature: float("nan") for feature in feature_names()}
    if "HCDR3" in mapped_residues:
        h3_contacts = cdr_contact_set(mapped_residues["HCDR3"], antigen_residues)
        contacted_h3_residues = {cdr_id for cdr_id, _ in h3_contacts}
        output["hcdr3_contact_count_5A"] = float(len(h3_contacts))
        output["hcdr3_contact_fraction_5A"] = float(
            len(contacted_h3_residues) / len(mapped_residues["HCDR3"])
        )
    if "LCDR3" in mapped_residues:
        l3_contacts = cdr_contact_set(mapped_residues["LCDR3"], antigen_residues)
        contacted_l3_residues = {cdr_id for cdr_id, _ in l3_contacts}
        output["lcdr3_contact_count_5A"] = float(len(l3_contacts))
        output["lcdr3_contact_fraction_5A"] = float(
            len(contacted_l3_residues) / len(mapped_residues["LCDR3"])
        )
    if all(field in mapped_residues for field in CDR_FIELDS):
        all_cdr_residues = [
            residue for field in CDR_FIELDS for residue in mapped_residues[field]
        ]
        all_contacts = cdr_contact_set(all_cdr_residues, antigen_residues)
        contacted_all_cdr_residues = {cdr_id for cdr_id, _ in all_contacts}
        output["all_cdr_contact_count_5A"] = float(len(all_contacts))
        output["cdr_interface_residue_count_5A"] = float(len(contacted_all_cdr_residues))
        output["cdr_min_distance"] = minimum_cdr_distance(all_cdr_residues, antigen_residues)
    return output


def feature_names() -> list[str]:
    return [
        "all_cdr_contact_count_5A",
        "hcdr3_contact_count_5A",
        "lcdr3_contact_count_5A",
        "hcdr3_contact_fraction_5A",
        "lcdr3_contact_fraction_5A",
        "cdr_interface_residue_count_5A",
        "cdr_min_distance",
    ]


def validate_rows(pilot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parser = PDBParser(QUIET=True)
    pdb_cache: dict[str, object] = {}
    mapping_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    for _, row in pilot.iterrows():
        pdb_id = str(row["pdb_norm"]).upper()
        status_record: dict[str, object] = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "pdb_id": row["pdb_id"],
            "pdb_norm": pdb_id,
            TARGET_COLUMN: row[TARGET_COLUMN],
            "resolved_Hchain": row["resolved_Hchain"],
            "resolved_Lchain": row["resolved_Lchain"],
            "resolved_antigen_chain": row["resolved_antigen_chain"],
        }
        try:
            path = ARCHIVE_RAW_DIR / f"{pdb_id.lower()}.pdb"
            if not path.exists():
                raise FileNotFoundError("missing_residue_numbering")
            if pdb_id not in pdb_cache:
                pdb_cache[pdb_id] = next(parser.get_structure(pdb_id, str(path)).get_models())
            model = pdb_cache[pdb_id]
            h_ids = split_chain_ids(row["resolved_Hchain"])
            l_ids = split_chain_ids(row["resolved_Lchain"])
            antigen_ids = split_chain_ids(row["resolved_antigen_chain"])
            if len(h_ids) != 1 or len(l_ids) != 1:
                raise ValueError("insertion_deletion_ambiguity")
            if any(chain_id not in model for chain_id in antigen_ids):
                raise ValueError("antigen_chain_mismatch")
            if h_ids[0] not in model or l_ids[0] not in model:
                raise ValueError("chain_sequence_mismatch")

            heavy_structure_sequence, heavy_residues = structure_residues(model[h_ids[0]])
            light_structure_sequence, light_residues = structure_residues(model[l_ids[0]])
            antigen_residues = [
                residue
                for chain_id in antigen_ids
                for residue in model[chain_id]
                if is_aa(residue, standard=False)
            ]
            if not antigen_residues:
                raise ValueError("antigen_chain_mismatch")
            heavy_map = chain_mapping_validation(
                clean_sequence(row["heavy_sequence"]),
                {field: clean_sequence(row[field]) for field in HEAVY_CDR_FIELDS},
                heavy_structure_sequence,
                heavy_residues,
            )
            light_map = chain_mapping_validation(
                clean_sequence(row["light_sequence"]),
                {field: clean_sequence(row[field]) for field in LIGHT_CDR_FIELDS},
                light_structure_sequence,
                light_residues,
            )
            status_record.update(
                {
                    "heavy_mapping_status": heavy_map["status"],
                    "heavy_mapping_error": heavy_map.get("reason", ""),
                    "heavy_alignment_identity": heavy_map.get("alignment_identity", float("nan")),
                    "heavy_alignment_coverage": heavy_map.get("alignment_coverage", float("nan")),
                    "light_mapping_status": light_map["status"],
                    "light_mapping_error": light_map.get("reason", ""),
                    "light_alignment_identity": light_map.get("alignment_identity", float("nan")),
                    "light_alignment_coverage": light_map.get("alignment_coverage", float("nan")),
                    "antigen_chain_mapping_status": "success",
                }
            )
            all_mapping_success = (
                heavy_map["status"] == "success" and light_map["status"] == "success"
            )
            for field in CDR_FIELDS:
                result = heavy_map if field.startswith("H") else light_map
                success = (
                    result.get("cdr_statuses", {}).get(field) == "success"
                    and field in result.get("mapped_residue_ids", {})
                )
                status_record[f"{field}_mapping_status"] = "success" if success else "failed"
                status_record[f"{field}_mapping_error"] = result.get("cdr_reasons", {}).get(field, "")
                status_record[f"{field}_mapped_residue_ids"] = (
                    "|".join(result["mapped_residue_ids"][field]) if success else ""
                )
            status_record["cdr_contact_feature_eligible"] = all_mapping_success
            status_record["hcdr3_contact_feature_eligible"] = (
                status_record["HCDR3_mapping_status"] == "success"
            )
            status_record["lcdr3_contact_feature_eligible"] = (
                status_record["LCDR3_mapping_status"] == "success"
            )
            status_record["hcdr3_lcdr3_contact_feature_eligible"] = (
                status_record["HCDR3_mapping_status"] == "success"
                and status_record["LCDR3_mapping_status"] == "success"
            )
            errors = [
                error for error in (heavy_map.get("reason", ""), light_map.get("reason", "")) if error
            ]
            status_record["mapping_failure_reasons"] = ";".join(sorted(set(errors)))
            if (
                status_record["hcdr3_contact_feature_eligible"]
                or status_record["lcdr3_contact_feature_eligible"]
            ):
                mapped_indices = {
                    **heavy_map["mapped_indices"],
                    **light_map["mapped_indices"],
                }
                geometry = build_contact_features(
                    row,
                    {"H": heavy_residues, "L": light_residues},
                    mapped_indices,
                    antigen_residues,
                )
                contact_rows.append({**row.to_dict(), **status_record, **geometry})
        except Exception as error:
            reason = str(error)
            status_record.update(
                {
                    "heavy_mapping_status": "failed",
                    "heavy_mapping_error": reason,
                    "light_mapping_status": "failed",
                    "light_mapping_error": reason,
                    "antigen_chain_mapping_status": (
                        "failed" if reason == "antigen_chain_mismatch" else "unknown"
                    ),
                    "cdr_contact_feature_eligible": False,
                    "hcdr3_contact_feature_eligible": False,
                    "lcdr3_contact_feature_eligible": False,
                    "hcdr3_lcdr3_contact_feature_eligible": False,
                    "mapping_failure_reasons": reason,
                }
            )
            for field in CDR_FIELDS:
                status_record[f"{field}_mapping_status"] = "failed"
                status_record[f"{field}_mapped_residue_ids"] = ""
        mapping_rows.append(status_record)
    return pd.DataFrame(mapping_rows), pd.DataFrame(contact_rows)


def safe_corr(frame: pd.DataFrame, x: str, y: str, method: str) -> float:
    valid = frame[[x, y]].dropna()
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return float("nan")
    return float(valid[x].corr(valid[y], method=method))


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def format_markdown(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        values = [
            fmt(row[column]) if isinstance(row[column], (float, np.floating)) else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(mapping: pd.DataFrame, contacts: pd.DataFrame) -> None:
    eligible = mapping[mapping["cdr_contact_feature_eligible"].fillna(False).astype(bool)]
    failures = mapping[~mapping["cdr_contact_feature_eligible"].fillna(False).astype(bool)]
    cdr_success_rows = []
    for field in CDR_FIELDS:
        success = int(mapping[f"{field}_mapping_status"].eq("success").sum())
        cdr_success_rows.append(
            {"CDR": field, "success_rows": success, "total_rows": len(mapping), "success_rate": success / len(mapping)}
        )
    cdr_success = pd.DataFrame(cdr_success_rows).sort_values(
        ["success_rate", "CDR"], ascending=[False, True]
    )
    failure_patterns = (
        failures["mapping_failure_reasons"].fillna("unknown").replace("", "unknown")
        .value_counts().rename_axis("reason").reset_index(name="rows")
    )
    requested_reason_labels = [
        "chain_sequence_mismatch",
        "missing_residue_numbering",
        "insertion_deletion_ambiguity",
        "unresolved_residues",
        "antigen_chain_mismatch",
        "missing_CDR_annotation",
    ]
    failure_reason_counts = pd.DataFrame(
        {
            "reason": requested_reason_labels,
            "rows": [
                int(
                    failures["mapping_failure_reasons"]
                    .fillna("")
                    .str.split(";")
                    .map(lambda reasons: label in reasons)
                    .sum()
                )
                for label in requested_reason_labels
            ],
        }
    )
    correlations: list[dict[str, object]] = []
    for feature in feature_names():
        correlations.append(
            {
                "feature": feature,
                "outcome": "target_affinity",
                "n": int(contacts[[feature, TARGET_COLUMN]].dropna().shape[0]) if not contacts.empty else 0,
                "pearson": safe_corr(contacts, feature, TARGET_COLUMN, "pearson") if not contacts.empty else float("nan"),
                "spearman": safe_corr(contacts, feature, TARGET_COLUMN, "spearman") if not contacts.empty else float("nan"),
            }
        )
        for model, (_, abs_error) in PREDICTION_ERROR_COLUMNS.items():
            if not contacts.empty and abs_error in contacts.columns:
                correlations.append(
                    {
                        "feature": feature,
                        "outcome": f"{model}_absolute_error",
                        "n": int(contacts[[feature, abs_error]].dropna().shape[0]),
                        "pearson": safe_corr(contacts, feature, abs_error, "pearson"),
                        "spearman": safe_corr(contacts, feature, abs_error, "spearman"),
                    }
                )
    corr = pd.DataFrame(correlations)
    focus_corr = corr[
        corr["feature"].isin(
            ["all_cdr_contact_count_5A", "hcdr3_contact_count_5A", "lcdr3_contact_count_5A", "cdr_min_distance"]
        )
    ].copy()
    for column in ("pearson", "spearman"):
        focus_corr[column] = focus_corr[column].map(fmt)

    train = pd.read_csv(DATA_DIR / "train.csv")
    lower = float(train[TARGET_COLUMN].quantile(0.10))
    upper = float(train[TARGET_COLUMN].quantile(0.90))
    all_cdr_contacts = contacts[
        contacts["cdr_contact_feature_eligible"].fillna(False).astype(bool)
    ] if not contacts.empty else contacts
    if all_cdr_contacts.empty:
        tail = pd.DataFrame()
    else:
        all_cdr_contacts = all_cdr_contacts.copy()
        all_cdr_contacts["target_tail"] = np.select(
            [all_cdr_contacts[TARGET_COLUMN] <= lower, all_cdr_contacts[TARGET_COLUMN] >= upper],
            ["below_train_P10", "above_train_P90"],
            default="middle_P10_to_P90",
        )
        tail = all_cdr_contacts.groupby("target_tail").agg(
            n=("sample_id", "size"),
            all_cdr_contact_count_5A_mean=("all_cdr_contact_count_5A", "mean"),
            hcdr3_contact_count_5A_mean=("hcdr3_contact_count_5A", "mean"),
            lcdr3_contact_count_5A_mean=("lcdr3_contact_count_5A", "mean"),
            cdr_min_distance_mean=("cdr_min_distance", "mean"),
        ).reset_index()

    lines = [
        "# ANDD Stratified CDR-to-Structure Mapping Validation",
        "",
        "## Scope and Safety Rules",
        "",
        "-  basic interface pilot  chain mapping  472 rows",
        "- , dataset, 695  ambiguous chain mappings",
        "- CDR annotations  AbNumber + IMGT extraction; SAbDab raw PDB",
        "- CDR contact  full heavy/light sequence  chain  strict mapping  preliminary pilot ",
        "",
        "## Mapping Validation Rule",
        "",
        "1.  CDR sequence  full chain sequence ",
        "2. Full chain sequence  chain ,alignment identity  95%,coverage  80%",
        "3.  CDR residue  amino acid  coordinate-bearing residue",
        "4.  alignment  CDR  mapping, insertion/deletion ambiguity, contact",
        "",
        "## Overall Result",
        "",
        f"- Pilot rows validated: **{len(mapping)}**.",
        f"- Rows safe for preliminary CDR-antigen contact features: **{len(eligible)} / {len(mapping)}** "
        f"({len(eligible) / len(mapping):.2%}).",
        f"- Failed mapping rows: **{len(failures)} / {len(mapping)}**.",
        f"- Eligible rows by split: `{eligible.groupby('split').size().to_dict()}`.",
        f"- HCDR3-only contact-safe rows: **{int(mapping['hcdr3_contact_feature_eligible'].sum())} / {len(mapping)}**.",
        f"- LCDR3-only contact-safe rows: **{int(mapping['lcdr3_contact_feature_eligible'].sum())} / {len(mapping)}**.",
        f"- HCDR3+LCDR3 jointly contact-safe rows: "
        f"**{int(mapping['hcdr3_lcdr3_contact_feature_eligible'].sum())} / {len(mapping)}**.",
        "",
        "## CDR Mapping Success Rate",
        "",
    ]
    display_success = cdr_success.copy()
    display_success["success_rate"] = display_success["success_rate"].map(lambda x: f"{x:.2%}")
    lines.extend(format_markdown(display_success, ["CDR", "success_rows", "total_rows", "success_rate"]))
    lines.extend(["", "## Failure Reasons", ""])
    if failures.empty:
        lines.append("- None. All pilot rows passed strict mapping validation.")
    else:
        lines.extend(format_markdown(failure_reason_counts, ["reason", "rows"]))
        lines.extend(["", "### Failure Reason Co-occurrence Patterns", ""])
        lines.extend(format_markdown(failure_patterns, ["reason", "rows"]))
    lines.extend(
        [
            "",
            "## Preliminary CDR Contact Features",
            "",
            "- `all_cdr_contact_count_5A`: all six mapped CDRs  antigen  5 A residue-pair contact ",
            "- `hcdr3_contact_count_5A`, `lcdr3_contact_count_5A`: CDR3 loops  5 A residue-pair contact ",
            "- `hcdr3_contact_fraction_5A`, `lcdr3_contact_fraction_5A`:  antigen  CDR3 residues ",
            "- `cdr_interface_residue_count_5A`:  antigen  CDR residues ",
            "- `cdr_min_distance`: all mapped CDR residues  antigen ",
            "- All-CDR aggregate features  CDR ;HCDR3/LCDR3 "
            "features  loop , CDR3-only ",
            f"- Preliminary contact table rows: **{len(contacts)}**( CDR3 loop );"
            f" all-six features **{int(contacts['all_cdr_contact_count_5A'].notna().sum())}** rows,"
            f"HCDR3 features **{int(contacts['hcdr3_contact_count_5A'].notna().sum())}** rows,"
            f"LCDR3 features **{int(contacts['lcdr3_contact_count_5A'].notna().sum())}** rows",
            "",
            "### Exploratory Correlations",
            "",
        ]
    )
    if focus_corr.empty:
        lines.append("- No preliminary contact rows are available for correlations.")
    else:
        lines.extend(format_markdown(focus_corr, ["feature", "outcome", "n", "pearson", "spearman"]))
    lines.extend(
        [
            "",
            "### Tail Pattern",
            "",
            f"- Train-defined tail thresholds: P10 = **{lower:.4f}**, P90 = **{upper:.4f}**.",
        ]
    )
    if tail.empty:
        lines.append("- No mapped contact rows available.")
    else:
        tail_display = tail.copy()
        for column in tail_display.columns:
            if column not in {"target_tail", "n"}:
                tail_display[column] = tail_display[column].map(fmt)
        lines.extend(
            format_markdown(
                tail_display,
                [
                    "target_tail",
                    "n",
                    "all_cdr_contact_count_5A_mean",
                    "hcdr3_contact_count_5A_mean",
                    "lcdr3_contact_count_5A_mean",
                    "cdr_min_distance_mean",
                ],
            )
        )
    best_cdrs = ", ".join(
        cdr_success.loc[cdr_success["success_rate"] == cdr_success["success_rate"].max(), "CDR"].tolist()
    )
    h3_ok = int(mapping["HCDR3_mapping_status"].eq("success").sum())
    l3_ok = int(mapping["LCDR3_mapping_status"].eq("success").sum())
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Reliable CDR-contact rows: **{len(eligible)} / {len(mapping)}** pilot samples.",
            f"2. Most stable CDR mapping: **{best_cdrs}** based on strict mapping success rate.",
            f"3. HCDR3/LCDR3 availability: HCDR3 **{h3_ok} / {len(mapping)}**, "
            f"LCDR3 **{l3_ok} / {len(mapping)}**.",
            f"4. Main failure reasons: "
            f"`{failure_reason_counts.to_dict('records') if not failures.empty else 'none'}`.",
            "5. Modeling recommendation: CDR-contact-aware modeling is technically feasible. A conservative "
            "next experiment can prioritize HCDR3/LCDR3 contact features because their mapping coverage is "
            "higher than all-six-CDR coverage. Preliminary weak correlations mean this should remain an "
            "incremental, controlled baseline rather than a claimed solution.",
            "6. If coverage is inadequate, the missing ingredient is reliable residue-level mapping for affected "
            "structures, including unresolved residues, indels, and chain assignment confirmation.",
            "",
            "## Outputs",
            "",
            f"- Mapping availability: `{MAPPING_OUTPUT_PATH}`",
            f"- Preliminary CDR contacts: `{CONTACT_OUTPUT_PATH}`",
            f"- This report: `{REPORT_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    pilot = load_pilot()
    mapping, contacts = validate_rows(pilot)
    mapping.to_csv(MAPPING_OUTPUT_PATH, index=False)
    contacts.to_csv(CONTACT_OUTPUT_PATH, index=False)
    write_report(mapping, contacts)
    eligible = int(mapping["cdr_contact_feature_eligible"].fillna(False).astype(bool).sum())
    print("CDR-to-structure mapping validation complete.")
    print(f"Pilot rows validated: {len(mapping)}")
    print(f"Rows eligible for preliminary CDR contacts: {eligible}")
    print(f"Failed mapping rows: {len(mapping) - eligible}")
    print(f"Mapping table: {MAPPING_OUTPUT_PATH}")
    print(f"Preliminary contact table: {CONTACT_OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
