"""Audit Patrick's SAbDab ``affinity=True`` search result summary.

中文人话说明：
SAbDab 网页上显示的 ``735 structures`` 和 “735 个可训练 affinity 样本”
不是一回事：

- 网页搜索结果先按 structure/PDB 选中结构；
- summary TSV 是 chain-row metadata，一个 PDB 可以展开成多行；
- 很多被搜索命中的结构行本身 affinity cell 仍然可能是空的。

这个脚本只做 metadata 与现有 dataset overlap audit：
1. 不训练模型；
2. 不覆盖已有 dataset；
3. 不解析全部结构；
4. 对 ``all_structures`` 只检查 raw/imgt/chothia PDB 文件是否存在。
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "sabdab_affinity_search_735_summary.tsv"
FULL_SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "sabdab_summary.tsv"
TDC_V1_SPLIT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
SUPPLEMENT_V1_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "tdc_plus_sabdab_supplement_v1"
    / "antigen_group_split"
)
# sequence_only 是之前已经从 SAbDab/PDB 提取成功的本地 sequence cache。
# search summary 本身没有 sequence，所以只有 exact chain metadata 能匹配到这里时，
# 我们才能做真正的 sequence-triplet overlap check。
SEQUENCE_ONLY_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sequence_only"
ALL_STRUCTURES_ROOT = Path("/Users/yichenzeng/Downloads/all_structures")
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sabdab_affinity_search_735_audit"
JSON_REPORT_PATH = OUTPUT_DIR / "search_735_audit.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "search_735_audit.md"
NEW_CANDIDATE_ROWS_PATH = OUTPUT_DIR / "new_candidate_rows.csv"

SUMMARY_KEY_COLUMNS = [
    "pdb",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "affinity",
    "delta_g",
    "affinity_method",
]
CHAIN_COMBO_COLUMNS = ["pdb", "Hchain", "Lchain", "antigen_chain"]
SEQUENCE_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
SPLITS = ["train", "val", "test"]
ARCHIVE_FOLDERS = ["raw", "imgt", "chothia"]


def is_missing(value: object) -> bool:
    """Treat blank cells and common NA spellings as missing."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def normalize_text(value: object) -> str:
    """Normalize row-key metadata before exact comparisons."""

    return "<missing>" if is_missing(value) else str(value).strip().upper()


def normalize_pdb(value: object) -> str:
    """Normalize PDB-like IDs to lowercase file/overlap style."""

    return str(value).strip().lower()


def read_summary(path: Path) -> pd.DataFrame:
    """Read one SAbDab summary TSV and validate needed columns."""

    if not path.exists():
        raise FileNotFoundError(f"Cannot find summary TSV: {path}")
    summary = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required_columns = {
        "pdb",
        "Hchain",
        "Lchain",
        "antigen_chain",
        "antigen_type",
        "affinity",
        "delta_g",
    }
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")
    summary["pdb_norm"] = summary["pdb"].map(normalize_pdb)
    return summary


def prepare_flags(summary: pd.DataFrame) -> pd.DataFrame:
    """Add reusable target, antigen, and chain-completeness flags."""

    prepared = summary.copy()
    prepared["affinity_present"] = ~prepared["affinity"].map(is_missing)
    prepared["affinity_numeric"] = pd.to_numeric(prepared["affinity"], errors="coerce")
    prepared["affinity_is_numeric"] = prepared["affinity_numeric"].notna()
    prepared["affinity_positive"] = prepared["affinity_numeric"] > 0
    prepared["delta_g_numeric"] = pd.to_numeric(prepared["delta_g"], errors="coerce")
    prepared["delta_g_is_numeric"] = prepared["delta_g_numeric"].notna()

    antigen_type = prepared["antigen_type"].astype(str).str.lower()
    prepared["contains_hapten"] = antigen_type.str.contains("hapten", na=False)
    prepared["protein_antigen"] = antigen_type.str.contains("protein", na=False) & ~prepared[
        "contains_hapten"
    ]
    prepared["protein_or_peptide_antigen"] = (
        antigen_type.str.contains("protein|peptide", regex=True, na=False)
        & ~prepared["contains_hapten"]
    )
    for column_name in ["Hchain", "Lchain", "antigen_chain"]:
        prepared[f"{column_name}_present"] = ~prepared[column_name].map(is_missing)
    prepared["complete_h_l_antigen_chain"] = (
        prepared["Hchain_present"]
        & prepared["Lchain_present"]
        & prepared["antigen_chain_present"]
    )
    return prepared


def count_masks(prepared: pd.DataFrame) -> OrderedDict[str, pd.Series]:
    """Build the requested count definitions for Patrick's summary."""

    masks: OrderedDict[str, pd.Series] = OrderedDict()
    masks["total_rows"] = pd.Series(True, index=prepared.index)
    masks["affinity_nonempty"] = prepared["affinity_present"]
    masks["affinity_numeric"] = prepared["affinity_is_numeric"]
    masks["affinity_gt_0"] = prepared["affinity_positive"]
    masks["delta_g_numeric"] = prepared["delta_g_is_numeric"]
    masks["affinity_gt_0_or_delta_g_numeric"] = (
        prepared["affinity_positive"] | prepared["delta_g_is_numeric"]
    )
    masks["complete_h_l_antigen_chain_and_affinity_gt_0"] = (
        prepared["complete_h_l_antigen_chain"] & prepared["affinity_positive"]
    )
    masks["protein_antigen_and_affinity_gt_0"] = prepared["protein_antigen"] & prepared[
        "affinity_positive"
    ]
    masks["protein_or_peptide_antigen_and_affinity_gt_0"] = (
        prepared["protein_or_peptide_antigen"] & prepared["affinity_positive"]
    )
    masks["complete_protein_antigen_and_affinity_gt_0"] = (
        prepared["complete_h_l_antigen_chain"]
        & prepared["protein_antigen"]
        & prepared["affinity_positive"]
    )
    masks["complete_protein_or_peptide_antigen_and_affinity_gt_0"] = (
        prepared["complete_h_l_antigen_chain"]
        & prepared["protein_or_peptide_antigen"]
        & prepared["affinity_positive"]
    )
    return masks


def normalized_key_series(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Build exact metadata keys from selected columns."""

    normalized = pd.DataFrame(index=data.index)
    for column_name in columns:
        normalized[column_name] = data[column_name].map(normalize_text)
    return normalized.astype(str).agg("||".join, axis=1)


def unique_combo_count(data: pd.DataFrame) -> int:
    """Count unique PDB+Hchain+Lchain+antigen_chain metadata rows."""

    return int(normalized_key_series(data, CHAIN_COMBO_COLUMNS).nunique())


def count_definition(name: str, rows: pd.DataFrame) -> dict:
    """Count one trainability definition in row/PDB/chain-combo units."""

    return {
        "definition": name,
        "row_count": int(len(rows)),
        "unique_pdb_count": int(rows["pdb_norm"].nunique()),
        "unique_pdb_hchain_lchain_antigen_chain_count": unique_combo_count(rows),
    }


def load_processed_splits(split_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load train/val/test rows for one processed affinity dataset."""

    frames = []
    for split_name in SPLITS:
        path = split_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {dataset_name} split: {path}")
        frame = pd.read_csv(path)
        frame["split"] = split_name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    missing_sequence_columns = set(SEQUENCE_COLUMNS) - set(data.columns)
    if missing_sequence_columns:
        raise ValueError(f"{dataset_name} missing sequence columns: {sorted(missing_sequence_columns)}")
    return data


def dataset_pdb_ids(data: pd.DataFrame) -> tuple[set[str], str]:
    """Collect PDB-like IDs from processed datasets."""

    id_column = "pdb" if "pdb" in data.columns else "antibody_id"
    if id_column not in data.columns:
        raise ValueError("Processed dataset needs pdb or antibody_id for PDB overlap.")
    ids = {normalize_pdb(value) for value in data[id_column] if not is_missing(value)}
    return ids, id_column


def sequence_triplet_set(data: pd.DataFrame) -> set[str]:
    """Build exact heavy+light+antigen sequence keys."""

    usable = data.dropna(subset=SEQUENCE_COLUMNS).copy()
    for column_name in SEQUENCE_COLUMNS:
        usable = usable[~usable[column_name].map(is_missing)]
    return set(usable[SEQUENCE_COLUMNS].astype(str).agg("||".join, axis=1))


def load_sequence_only_cache() -> pd.DataFrame | None:
    """Load local extracted SAbDab sequence rows when available."""

    frames = []
    for split_name in SPLITS:
        path = SEQUENCE_ONLY_DIR / f"{split_name}.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        frame["sequence_only_split"] = split_name
        frames.append(frame)
    cache = pd.concat(frames, ignore_index=True)
    required_columns = {*CHAIN_COMBO_COLUMNS, *SEQUENCE_COLUMNS}
    if required_columns - set(cache.columns):
        return None
    cache["chain_combo_key"] = normalized_key_series(cache, CHAIN_COMBO_COLUMNS)
    return cache


def attach_cached_sequences(search: pd.DataFrame, cache: pd.DataFrame | None) -> pd.DataFrame:
    """Attach sequences to search rows only by exact PDB+chain metadata match."""

    attached = search.copy()
    attached["chain_combo_key"] = normalized_key_series(attached, CHAIN_COMBO_COLUMNS)
    if cache is None:
        attached["has_local_sequence_only_triplet"] = False
        for column_name in SEQUENCE_COLUMNS:
            attached[column_name] = ""
        return attached

    cache_columns = ["chain_combo_key", *SEQUENCE_COLUMNS]
    dedup_cache = cache[cache_columns].drop_duplicates("chain_combo_key", keep="first")
    attached = attached.merge(dedup_cache, on="chain_combo_key", how="left")
    attached["has_local_sequence_only_triplet"] = attached["heavy_sequence"].notna()
    return attached


def triplet_overlap_flags(rows: pd.DataFrame, reference_triplets: set[str], prefix: str) -> pd.DataFrame:
    """Flag overlap only where search row has a cached sequence triplet."""

    flagged = rows.copy()
    keys = flagged[SEQUENCE_COLUMNS].fillna("").astype(str).agg("||".join, axis=1)
    flagged[f"{prefix}_triplet_overlap"] = flagged["has_local_sequence_only_triplet"] & keys.isin(
        reference_triplets
    )
    return flagged


def pdb_overlap_report(search: pd.DataFrame, reference: pd.DataFrame, dataset_name: str) -> dict:
    """Report PDB-level overlap between search summary and one dataset."""

    reference_pdbs, id_column = dataset_pdb_ids(reference)
    search_pdbs = set(search["pdb_norm"])
    return {
        "dataset": dataset_name,
        "dataset_pdb_column_used": id_column,
        "dataset_rows": int(len(reference)),
        "dataset_unique_pdb_like_ids": int(len(reference_pdbs)),
        "search_unique_pdbs": int(len(search_pdbs)),
        "pdb_overlap_count": int(len(search_pdbs & reference_pdbs)),
        "search_pdbs_not_in_dataset": int(len(search_pdbs - reference_pdbs)),
    }


def triplet_overlap_report(attached: pd.DataFrame, reference: pd.DataFrame, dataset_name: str) -> dict:
    """Report sequence triplet overlap for search rows with cached sequences."""

    reference_triplets = sequence_triplet_set(reference)
    keys = attached[SEQUENCE_COLUMNS].fillna("").astype(str).agg("||".join, axis=1)
    available = attached["has_local_sequence_only_triplet"]
    overlap = available & keys.isin(reference_triplets)
    return {
        "dataset": dataset_name,
        "reference_unique_triplets": int(len(reference_triplets)),
        "search_rows_with_local_cached_triplets": int(available.sum()),
        "search_rows_with_triplet_overlap": int(overlap.sum()),
        "search_unique_cached_triplets": int(keys[available].nunique()),
        "search_unique_cached_triplets_overlapping_dataset": int(keys[overlap].nunique()),
        "limitation": (
            "SAbDab search summary has no sequence columns. Search triplet overlap is checked only for "
            "rows that exact-match local sequence_only PDB+chain metadata."
        ),
    }


def full_summary_comparison(search: pd.DataFrame, full: pd.DataFrame) -> dict:
    """Compare Patrick search result against the currently downloaded full summary."""

    search_pdbs = set(search["pdb_norm"])
    full_pdbs = set(full["pdb_norm"])
    search_row_keys = set(normalized_key_series(search, SUMMARY_KEY_COLUMNS))
    full_row_keys = set(normalized_key_series(full, SUMMARY_KEY_COLUMNS))
    return {
        "search_unique_pdbs": int(len(search_pdbs)),
        "full_unique_pdbs": int(len(full_pdbs)),
        "search_pdbs_are_full_summary_subset": bool(search_pdbs <= full_pdbs),
        "search_row_keys_are_full_summary_subset": bool(search_row_keys <= full_row_keys),
        "pdbs_new_in_search_vs_full": sorted(search_pdbs - full_pdbs),
        "pdbs_in_full_not_in_search_count": int(len(full_pdbs - search_pdbs)),
        "search_row_keys_new_vs_full_count": int(len(search_row_keys - full_row_keys)),
        "full_row_keys_not_in_search_count": int(len(full_row_keys - search_row_keys)),
        "row_key_definition": SUMMARY_KEY_COLUMNS,
    }


def archive_file_existence(search: pd.DataFrame) -> dict:
    """Check all_structures file existence without parsing PDB contents."""

    unique_pdbs = sorted(set(search["pdb_norm"]))
    report = {
        "archive_root": str(ALL_STRUCTURES_ROOT),
        "folders": {},
    }
    for folder_name in ARCHIVE_FOLDERS:
        folder_path = ALL_STRUCTURES_ROOT / folder_name
        present = []
        missing = []
        for pdb_id in unique_pdbs:
            path = folder_path / f"{pdb_id}.pdb"
            (present if path.exists() else missing).append(pdb_id)
        report["folders"][folder_name] = {
            "folder_exists": bool(folder_path.exists() and folder_path.is_dir()),
            "search_pdb_files_present": int(len(present)),
            "search_pdb_files_missing": int(len(missing)),
            "missing_pdb_ids": missing,
        }
    return report


def annotate_new_candidates(
    prepared_search: pd.DataFrame,
    tdc_v1: pd.DataFrame,
    supplement_v1: pd.DataFrame,
) -> pd.DataFrame:
    """Create a metadata candidate CSV for rows not PDB-covered by supplement v1.

    这里的 ``new_candidate`` 仍然只是 possible candidate：
    summary row 有可计算 affinity target 和完整 sequence-antigen chain metadata，
    但 search summary 本身不提供 sequences。若本地 sequence_only 没有 exact match，
    后面仍需单独做 sequence extraction。
    """

    candidate_mask = (
        prepared_search["complete_h_l_antigen_chain"]
        & prepared_search["protein_or_peptide_antigen"]
        & prepared_search["affinity_positive"]
    )
    candidates = prepared_search[candidate_mask].copy()
    tdc_pdbs, _ = dataset_pdb_ids(tdc_v1)
    supplement_pdbs, _ = dataset_pdb_ids(supplement_v1)
    candidates["tdc_v1_pdb_overlap"] = candidates["pdb_norm"].isin(tdc_pdbs)
    candidates["supplement_v1_pdb_overlap"] = candidates["pdb_norm"].isin(supplement_pdbs)
    candidates = candidates[~candidates["supplement_v1_pdb_overlap"]].copy()
    candidates["candidate_status"] = candidates["has_local_sequence_only_triplet"].map(
        {
            True: "possible_new_candidate_with_local_cached_triplet",
            False: "possible_new_candidate_needs_sequence_extraction",
        }
    )
    columns_to_keep = [
        column_name
        for column_name in [
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
            "affinity_numeric",
            "tdc_v1_pdb_overlap",
            "supplement_v1_pdb_overlap",
            "has_local_sequence_only_triplet",
            "tdc_v1_triplet_overlap",
            "supplement_v1_triplet_overlap",
            "candidate_status",
            *SEQUENCE_COLUMNS,
        ]
        if column_name in candidates.columns
    ]
    return candidates[columns_to_keep].copy()


def candidate_summary(candidates: pd.DataFrame) -> dict:
    """Summarize candidate CSV content for the report."""

    if len(candidates) == 0:
        return {
            "rows": 0,
            "unique_pdbs": 0,
            "status_counts": {},
            "rows_with_cached_triplets": 0,
        }
    return {
        "rows": int(len(candidates)),
        "unique_pdbs": int(candidates["pdb"].map(normalize_pdb).nunique()),
        "status_counts": {
            str(key): int(value)
            for key, value in candidates["candidate_status"].value_counts(dropna=False).items()
        },
        "rows_with_cached_triplets": int(candidates["has_local_sequence_only_triplet"].sum()),
    }


def build_report(
    prepared_search: pd.DataFrame,
    full: pd.DataFrame,
    definitions: list[dict],
    tdc_v1: pd.DataFrame,
    supplement_v1: pd.DataFrame,
    attached_search: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict:
    """Build JSON report for all requested audit dimensions."""

    total = next(row for row in definitions if row["definition"] == "total_rows")
    positive = next(row for row in definitions if row["definition"] == "affinity_gt_0")
    return {
        "inputs": {
            "search_summary": str(SEARCH_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "full_summary": str(FULL_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "tdc_v1_split_dir": str(TDC_V1_SPLIT_DIR.relative_to(PROJECT_ROOT)),
            "supplement_v1_split_dir": str(SUPPLEMENT_V1_SPLIT_DIR.relative_to(PROJECT_ROOT)),
            "all_structures_root": str(ALL_STRUCTURES_ROOT),
        },
        "search_summary_counts": {
            "total_rows": total["row_count"],
            "unique_pdbs": total["unique_pdb_count"],
            "unique_pdb_hchain_lchain_antigen_chain": total[
                "unique_pdb_hchain_lchain_antigen_chain_count"
            ],
        },
        "trainability_count_definitions": definitions,
        "full_summary_comparison": full_summary_comparison(prepared_search, full),
        "tdc_v1_overlap": {
            "pdb": pdb_overlap_report(prepared_search, tdc_v1, "tdc_v1"),
            "sequence_triplet": triplet_overlap_report(attached_search, tdc_v1, "tdc_v1"),
        },
        "supplement_v1_overlap": {
            "pdb": pdb_overlap_report(prepared_search, supplement_v1, "tdc_plus_sabdab_supplement_v1"),
            "sequence_triplet": triplet_overlap_report(
                attached_search, supplement_v1, "tdc_plus_sabdab_supplement_v1"
            ),
        },
        "possible_new_candidates_not_pdb_covered_by_supplement_v1": candidate_summary(candidates),
        "archive_file_existence": archive_file_existence(prepared_search),
        "answers": {
            "why_web_page_shows_735_structures": (
                f"The search summary has {total['unique_pdb_count']} unique PDB structures but "
                f"{total['row_count']} summary chain rows. The web structure count is PDB-level, "
                "while the TSV can contain multiple rows per structure."
            ),
            "why_positive_numeric_affinity_is_smaller": (
                f"Only {positive['row_count']} search-summary rows have numeric affinity > 0. "
                "A structure can be returned by the affinity=True search while some expanded summary "
                "chain rows still have empty affinity metadata."
            ),
            "is_735_trainable_samples": (
                "No. Trainable sequence affinity regression rows need a usable target, complete "
                "heavy/light/antigen chain metadata, sequence-antigen policy, sequence extraction, "
                "deduplication, and leakage-aware split checks."
            ),
            "build_new_dataset_recommendation": (
                "Use this search result as a candidate discovery input, not as a 735-row ready dataset. "
                "Build a new dataset only after auditing possible_new_candidates and extracting/validating "
                "their heavy, light, and antigen sequences."
            ),
        },
    }


def percent(numerator: int, denominator: int) -> str:
    """Format a simple percentage for Markdown tables."""

    return "NA" if denominator == 0 else f"{100 * numerator / denominator:.2f}%"


def markdown_definition_table(definitions: list[dict]) -> list[str]:
    """Render count definitions as Markdown table lines."""

    lines = [
        "| definition | rows | unique PDBs | unique PDB+H+L+antigen chain |",
        "|---|---:|---:|---:|",
    ]
    for row in definitions:
        lines.append(
            f"| `{row['definition']}` | {row['row_count']} | {row['unique_pdb_count']} | "
            f"{row['unique_pdb_hchain_lchain_antigen_chain_count']} |"
        )
    return lines


def write_markdown(report: dict) -> None:
    """Write a readable audit report for discussion."""

    summary = report["search_summary_counts"]
    comparison = report["full_summary_comparison"]
    candidates = report["possible_new_candidates_not_pdb_covered_by_supplement_v1"]
    archive = report["archive_file_existence"]
    answers = report["answers"]
    lines = [
        "# SAbDab Affinity Search 735 Audit",
        "",
        "## Scope",
        "",
        f"- Patrick search summary: `{report['inputs']['search_summary']}`",
        f"- Full local summary: `{report['inputs']['full_summary']}`",
        f"- TDC v1 split dir: `{report['inputs']['tdc_v1_split_dir']}`",
        f"- Supplement v1 split dir: `{report['inputs']['supplement_v1_split_dir']}`",
        f"- all_structures checked by absolute path: `{report['inputs']['all_structures_root']}`",
        "- This audit does not train, overwrite datasets, or parse all PDB contacts.",
        "",
        "## Search Summary Headline Counts",
        "",
        f"- Total summary rows: {summary['total_rows']}",
        f"- Unique PDB structures: {summary['unique_pdbs']}",
        f"- Unique PDB+Hchain+Lchain+antigen_chain combos: {summary['unique_pdb_hchain_lchain_antigen_chain']}",
        "",
        "## Trainability Count Definitions",
        "",
    ]
    lines.extend(markdown_definition_table(report["trainability_count_definitions"]))
    lines.extend(
        [
            "",
            "## Full Summary Comparison",
            "",
            f"- Search PDBs are full-summary subset: `{comparison['search_pdbs_are_full_summary_subset']}`",
            f"- Search row keys are full-summary subset: `{comparison['search_row_keys_are_full_summary_subset']}`",
            f"- PDBs new in search vs full summary: {len(comparison['pdbs_new_in_search_vs_full'])}",
            f"- PDBs in full summary but not search: {comparison['pdbs_in_full_not_in_search_count']}",
            f"- Search row keys new vs full: {comparison['search_row_keys_new_vs_full_count']}",
            f"- Full row keys not in search: {comparison['full_row_keys_not_in_search_count']}",
            f"- Row key definition: `{comparison['row_key_definition']}`",
            "",
            "## Existing Dataset Overlap",
            "",
            f"- TDC v1 PDB overlap: `{report['tdc_v1_overlap']['pdb']}`",
            f"- TDC v1 sequence-triplet overlap: `{report['tdc_v1_overlap']['sequence_triplet']}`",
            f"- Supplement v1 PDB overlap: `{report['supplement_v1_overlap']['pdb']}`",
            f"- Supplement v1 sequence-triplet overlap: `{report['supplement_v1_overlap']['sequence_triplet']}`",
            "",
            "Triplet note: the search summary itself has no sequences. Sequence triplet overlap is only checked ",
            "for search rows that exact-match a local `sequence_only` PDB+chain combo.",
            "",
            "## Possible New Candidates",
            "",
            "- Candidate policy here: complete H/L/antigen chain metadata, positive affinity, protein or peptide antigen, and no supplement-v1 PDB overlap.",
            f"- Candidate summary: `{candidates}`",
            f"- Candidate CSV: `{NEW_CANDIDATE_ROWS_PATH.relative_to(PROJECT_ROOT)}`",
            "",
            "## all_structures File Existence",
            "",
            "| folder | folder exists | search PDB files present | missing |",
            "|---|---|---:|---:|",
        ]
    )
    for folder_name, folder_report in archive["folders"].items():
        lines.append(
            f"| `{folder_name}` | `{folder_report['folder_exists']}` | "
            f"{folder_report['search_pdb_files_present']} | {folder_report['search_pdb_files_missing']} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Why does the webpage show 735 structures? {answers['why_web_page_shows_735_structures']}",
            f"- Why can positive numeric affinity rows be only about 358? {answers['why_positive_numeric_affinity_is_smaller']}",
            f"- Is 735 equal to 735 trainable affinity regression samples? {answers['is_735_trainable_samples']}",
            f"- Should we build a new dataset from this search result? {answers['build_new_dataset_recommendation']}",
        ]
    )
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_terminal_summary(report: dict) -> None:
    """Print the main takeaway after report generation."""

    headline = report["search_summary_counts"]
    positive = next(
        row for row in report["trainability_count_definitions"] if row["definition"] == "affinity_gt_0"
    )
    candidate = report["possible_new_candidates_not_pdb_covered_by_supplement_v1"]
    print("SAbDab affinity search 735 audit complete.")
    print(
        f"Search summary: {headline['total_rows']} rows / "
        f"{headline['unique_pdbs']} unique PDB structures / "
        f"{headline['unique_pdb_hchain_lchain_antigen_chain']} unique PDB+chain combos."
    )
    print(f"Positive numeric affinity rows: {positive['row_count']}")
    print(f"Possible new candidate rows not PDB-covered by supplement v1: {candidate['rows']}")
    print(report["answers"]["why_web_page_shows_735_structures"])
    print(report["answers"]["is_735_trainable_samples"])
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Candidate CSV: {NEW_CANDIDATE_ROWS_PATH.relative_to(PROJECT_ROOT)}")
    print("No model training was run.")


def main() -> None:
    """Run the complete audit."""

    search = prepare_flags(read_summary(SEARCH_SUMMARY_PATH))
    full = prepare_flags(read_summary(FULL_SUMMARY_PATH))
    definitions = [
        count_definition(name, search[mask].copy())
        for name, mask in count_masks(search).items()
    ]

    tdc_v1 = load_processed_splits(TDC_V1_SPLIT_DIR, "tdc_v1")
    supplement_v1 = load_processed_splits(SUPPLEMENT_V1_SPLIT_DIR, "supplement_v1")
    attached_search = attach_cached_sequences(search, load_sequence_only_cache())
    attached_search = triplet_overlap_flags(attached_search, sequence_triplet_set(tdc_v1), "tdc_v1")
    attached_search = triplet_overlap_flags(
        attached_search, sequence_triplet_set(supplement_v1), "supplement_v1"
    )
    candidates = annotate_new_candidates(attached_search, tdc_v1, supplement_v1)

    report = build_report(search, full, definitions, tdc_v1, supplement_v1, attached_search, candidates)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(NEW_CANDIDATE_ROWS_PATH, index=False)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print_terminal_summary(report)


if __name__ == "__main__":
    main()
