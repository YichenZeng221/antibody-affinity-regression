"""Inspect a local SAbDab all_structures archive without copying it.

:
 archive:

    /Users/yichenzeng/Downloads/all_structures

 31GB , contact matrix
:
1. raw / imgt / chothia  PDB 
2. summary.tsv  chain ID 
3.  archive  CDR extraction  interaction/contact 
"""

from __future__ import annotations

from pathlib import Path
import json
import random
import re

import pandas as pd
from Bio.PDB import PDBParser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path("/Users/yichenzeng/Downloads/all_structures")
SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "sabdab_summary.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sabdab_structure_archive_inspection"
JSON_REPORT_PATH = OUTPUT_DIR / "structure_archive_inspection.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "structure_archive_inspection.md"

ARCHIVE_FOLDERS = ["raw", "imgt", "chothia"]
RANDOM_SAMPLE_SIZE = 10
RANDOM_SEED = 42
SUPPLEMENT_PDBS = ["3eoa", "3h42", "2wub", "2wuc", "3sdy", "1yyl"]
ERROR_EXAMPLE_PDBS = ["2oqj", "5c0n", "4idj", "5f3b", "2ny3", "2nxy", "6cdm", "4u6v"]
FOCUS_PDBS = SUPPLEMENT_PDBS + ERROR_EXAMPLE_PDBS


def normalize_pdb_id(value: object) -> str:
    """Normalize PDB IDs to lowercase file-name style."""

    return str(value).strip().lower()


def is_missing(value: object) -> bool:
    """Treat blank and common NA spellings as missing metadata."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def archive_folder(folder_name: str) -> Path:
    """Return one archive subfolder path."""

    return ARCHIVE_ROOT / folder_name


def pdb_path(folder_name: str, pdb_id: str) -> Path:
    """Build absolute path to one archive PDB file."""

    return archive_folder(folder_name) / f"{normalize_pdb_id(pdb_id)}.pdb"


def check_archive_folders() -> dict:
    """Check raw/imgt/chothia folder existence and count PDB files."""

    folder_report = {}
    for folder_name in ARCHIVE_FOLDERS:
        folder_path = archive_folder(folder_name)
        exists = folder_path.exists() and folder_path.is_dir()
        pdb_files = list(folder_path.glob("*.pdb")) if exists else []
        folder_report[folder_name] = {
            "path": str(folder_path),
            "exists": bool(exists),
            "pdb_file_count": int(len(pdb_files)),
        }
    return folder_report


def raw_pdb_ids() -> list[str]:
    """List PDB IDs from raw archive filenames for deterministic sampling."""

    raw_dir = archive_folder("raw")
    if not raw_dir.exists():
        return []
    return sorted(path.stem.lower() for path in raw_dir.glob("*.pdb"))


def random_correspondence_sample() -> list[dict]:
    """Sample raw PDB IDs and check matching files in all archive folders."""

    pdb_ids = raw_pdb_ids()
    rng = random.Random(RANDOM_SEED)
    sample_ids = rng.sample(pdb_ids, k=min(RANDOM_SAMPLE_SIZE, len(pdb_ids)))

    rows = []
    for pdb_id in sample_ids:
        row = {"pdb": pdb_id}
        for folder_name in ARCHIVE_FOLDERS:
            row[f"{folder_name}_exists"] = pdb_path(folder_name, pdb_id).exists()
        row["all_three_exist"] = all(row[f"{folder_name}_exists"] for folder_name in ARCHIVE_FOLDERS)
        rows.append(row)
    return rows


def load_summary() -> pd.DataFrame:
    """Read local summary.tsv and normalize PDB IDs for lookup."""

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Cannot find {SUMMARY_PATH}")
    summary = pd.read_csv(SUMMARY_PATH, sep="\t", dtype=str, keep_default_na=False)
    required_columns = {"pdb", "Hchain", "Lchain", "antigen_chain"}
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(f"summary.tsv missing columns: {sorted(missing_columns)}")
    summary["pdb_norm"] = summary["pdb"].map(normalize_pdb_id)
    return summary


def split_chain_ids(value: object) -> list[str]:
    """Split summary chain fields like 'A | B' into chain ID list."""

    if is_missing(value):
        return []
    parts = re.split(r"\s*\|\s*", str(value).strip())
    return [part.strip() for part in parts if part.strip() and not is_missing(part)]


def summary_row_chain_ids(row: pd.Series) -> dict:
    """Return heavy/light/antigen chain IDs for one summary row."""

    return {
        "Hchain": split_chain_ids(row["Hchain"]),
        "Lchain": split_chain_ids(row["Lchain"]),
        "antigen_chain": split_chain_ids(row["antigen_chain"]),
    }


def extract_pdb_chain_ids(path: Path) -> dict:
    """Parse one PDB and list chain IDs in the first model.

    :
    contact matrix  chain  residues
     PDB  summary  chain
    """

    if not path.exists():
        return {"parse_status": "missing_file", "chain_ids": [], "parse_error": ""}

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(path.stem, path)
        first_model = next(structure.get_models())
        chain_ids = sorted(str(chain.id).strip() for chain in first_model.get_chains())
        return {"parse_status": "ok", "chain_ids": chain_ids, "parse_error": ""}
    except Exception as error:
        return {
            "parse_status": "parse_failed",
            "chain_ids": [],
            "parse_error": str(error),
        }


def row_chain_presence(row: pd.Series, archive_chain_ids: list[str]) -> dict:
    """Check whether one summary row's H/L/antigen chains appear in a PDB."""

    chain_ids = summary_row_chain_ids(row)
    archive_chain_set = set(archive_chain_ids)
    presence = {}
    for field_name, requested_ids in chain_ids.items():
        found_ids = [chain_id for chain_id in requested_ids if chain_id in archive_chain_set]
        missing_ids = [chain_id for chain_id in requested_ids if chain_id not in archive_chain_set]
        presence[field_name] = {
            "requested": requested_ids,
            "found": found_ids,
            "missing": missing_ids,
            "all_requested_found": bool(requested_ids) and not missing_ids,
        }
    return presence


def summary_rows_for_pdb(summary: pd.DataFrame, pdb_id: str) -> list[dict]:
    """Return compact summary metadata rows for one PDB."""

    rows = summary[summary["pdb_norm"] == normalize_pdb_id(pdb_id)]
    compact_rows = []
    for row_index, row in rows.iterrows():
        compact_rows.append(
            {
                "summary_index": int(row_index),
                "Hchain": str(row["Hchain"]),
                "Lchain": str(row["Lchain"]),
                "antigen_chain": str(row["antigen_chain"]),
                "antigen_type": str(row.get("antigen_type", "")),
                "affinity": str(row.get("affinity", "")),
            }
        )
    return compact_rows


def inspect_focus_pdb(summary: pd.DataFrame, pdb_id: str) -> dict:
    """Inspect one focused PDB across raw/imgt/chothia archive files."""

    focus_rows = summary[summary["pdb_norm"] == normalize_pdb_id(pdb_id)]
    archive_files = {}
    for folder_name in ARCHIVE_FOLDERS:
        path = pdb_path(folder_name, pdb_id)
        chain_report = extract_pdb_chain_ids(path)
        archive_files[folder_name] = {
            "path": str(path),
            "exists": path.exists(),
            **chain_report,
        }

    row_checks = []
    for row_index, row in focus_rows.iterrows():
        row_check = {
            "summary_index": int(row_index),
            "Hchain": str(row["Hchain"]),
            "Lchain": str(row["Lchain"]),
            "antigen_chain": str(row["antigen_chain"]),
            "archive_chain_presence": {},
        }
        for folder_name in ARCHIVE_FOLDERS:
            row_check["archive_chain_presence"][folder_name] = row_chain_presence(
                row,
                archive_files[folder_name]["chain_ids"],
            )
        row_checks.append(row_check)

    return {
        "pdb": normalize_pdb_id(pdb_id),
        "focus_group": "supplement" if normalize_pdb_id(pdb_id) in SUPPLEMENT_PDBS else "error_example",
        "archive_files": archive_files,
        "summary_rows": summary_rows_for_pdb(summary, pdb_id),
        "summary_chain_checks": row_checks,
    }


def inspect_focus_pdbs(summary: pd.DataFrame) -> list[dict]:
    """Inspect all user-requested PDB IDs."""

    return [inspect_focus_pdb(summary, pdb_id) for pdb_id in FOCUS_PDBS]


def focus_row_line(pdb_report: dict, row_check: dict, folder_name: str) -> str:
    """Render one Markdown row for summary-chain presence."""

    presence = row_check["archive_chain_presence"][folder_name]
    missing_parts = []
    for field_name in ["Hchain", "Lchain", "antigen_chain"]:
        missing_ids = presence[field_name]["missing"]
        if missing_ids:
            missing_parts.append(f"{field_name}:{missing_ids}")
    missing_text = "; ".join(missing_parts) if missing_parts else "none"
    all_requested = all(
        presence[field_name]["all_requested_found"]
        for field_name in ["Hchain", "Lchain", "antigen_chain"]
    )
    return (
        f"| `{pdb_report['pdb']}` | `{row_check['summary_index']}` | `{folder_name}` | "
        f"`{row_check['Hchain']}` | `{row_check['Lchain']}` | `{row_check['antigen_chain']}` | "
        f"`{all_requested}` | `{missing_text}` |"
    )


def build_report() -> dict:
    """Build structure archive inspection report."""

    summary = load_summary()
    report = {
        "archive_root": str(ARCHIVE_ROOT),
        "summary_path": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        "archive_folders": check_archive_folders(),
        "random_sample": {
            "seed": RANDOM_SEED,
            "requested_size": RANDOM_SAMPLE_SIZE,
            "rows": random_correspondence_sample(),
        },
        "focus_pdbs": inspect_focus_pdbs(summary),
        "notes": {
            "raw_note": "raw PDB files preserve archive structural chains for later residue/contact inspection.",
            "numbered_note": "imgt/chothia files are checked for availability and chain presence before standard numbering/CDR work.",
            "scope_note": "This inspection parses only focused PDBs and does not compute contact matrices.",
        },
    }
    return report


def write_markdown(report: dict) -> None:
    """Write human-readable Markdown report."""

    folder_lines = [
        "| folder | exists | .pdb count | path |",
        "|---|---|---:|---|",
    ]
    for folder_name, folder_report in report["archive_folders"].items():
        folder_lines.append(
            f"| `{folder_name}` | `{folder_report['exists']}` | {folder_report['pdb_file_count']} | "
            f"`{folder_report['path']}` |"
        )

    random_lines = [
        "| pdb | raw | imgt | chothia | all three |",
        "|---|---|---|---|---|",
    ]
    for row in report["random_sample"]["rows"]:
        random_lines.append(
            f"| `{row['pdb']}` | `{row['raw_exists']}` | `{row['imgt_exists']}` | "
            f"`{row['chothia_exists']}` | `{row['all_three_exist']}` |"
        )

    focus_file_lines = [
        "| pdb | group | raw exists/chains | imgt exists/chains | chothia exists/chains | summary rows |",
        "|---|---|---|---|---|---:|",
    ]
    chain_presence_lines = [
        "| pdb | summary index | archive folder | Hchain | Lchain | antigen_chain | all requested found | missing IDs |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for pdb_report in report["focus_pdbs"]:
        cell_text = {}
        for folder_name in ARCHIVE_FOLDERS:
            folder = pdb_report["archive_files"][folder_name]
            cell_text[folder_name] = f"{folder['exists']} / {folder['chain_ids']}"
        focus_file_lines.append(
            f"| `{pdb_report['pdb']}` | `{pdb_report['focus_group']}` | `{cell_text['raw']}` | "
            f"`{cell_text['imgt']}` | `{cell_text['chothia']}` | {len(pdb_report['summary_rows'])} |"
        )
        for row_check in pdb_report["summary_chain_checks"]:
            for folder_name in ARCHIVE_FOLDERS:
                chain_presence_lines.append(focus_row_line(pdb_report, row_check, folder_name))

    lines = [
        "# SAbDab all_structures Archive Inspection",
        "",
        "## Scope",
        "",
        f"- Archive root read by absolute path: `{report['archive_root']}`",
        f"- Summary metadata: `{report['summary_path']}`",
        "- This report does not copy the archive, train models, or compute contact matrices.",
        "",
        "## Archive Folders",
        "",
        *folder_lines,
        "",
        "## Random Raw-PDB Correspondence Sample",
        "",
        f"- Seed: {report['random_sample']['seed']}",
        f"- Sample size: {len(report['random_sample']['rows'])}",
        "",
        *random_lines,
        "",
        "## Focus PDB File And Chain Sets",
        "",
        "Each chain list comes from the first model parsed from that archive PDB file.",
        "",
        *focus_file_lines,
        "",
        "## Summary Chain Presence In Focus PDBs",
        "",
        "For multi-chain `antigen_chain` fields such as `B | A`, all requested antigen IDs must be present "
        "for `all requested found` to be true.",
        "",
        *chain_presence_lines,
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {text}" for text in report["notes"].values())
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: dict) -> None:
    """Print concise terminal summary."""

    print("SAbDab all_structures archive inspection complete.")
    for folder_name, folder_report in report["archive_folders"].items():
        print(
            f"{folder_name}: exists={folder_report['exists']}, "
            f"pdb_count={folder_report['pdb_file_count']}"
        )
    all_three = sum(row["all_three_exist"] for row in report["random_sample"]["rows"])
    print(f"Random sample with all three files: {all_three}/{len(report['random_sample']['rows'])}")
    print("Focused PDBs inspected:")
    for pdb_report in report["focus_pdbs"]:
        existence = {
            folder_name: pdb_report["archive_files"][folder_name]["exists"]
            for folder_name in ARCHIVE_FOLDERS
        }
        print(f"  {pdb_report['pdb']}: summary_rows={len(pdb_report['summary_rows'])}, files={existence}")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Run archive inspection and write JSON/Markdown reports."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_markdown(report)
    print_summary(report)


if __name__ == "__main__":
    main()
