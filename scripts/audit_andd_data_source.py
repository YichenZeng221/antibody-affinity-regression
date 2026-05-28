"""Audit ANDD as a possible data expansion source.

中文人话说明：
这个脚本只做数据源审计，不创建训练集，不训练模型，也不修改现有 unified dataset。

为什么不用 pandas.read_excel？
当前项目 .venv 里没有 openpyxl。为了避免临时安装依赖，这里用 Python 标准库
直接读取 xlsx 内部的 XML 表格。这样 audit 可复现，也不会改变环境。
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANDD_DIR = PROJECT_ROOT / "data/external/ANDD"
OUTPUT_DIR = PROJECT_ROOT / "outputs/data_expansion/ANDD_audit"
ANDD_XLSX = ANDD_DIR / "ANDD_v2.xlsx"
DATA_DICTIONARY = ANDD_DIR / "Data_dictionary.csv"

CURRENT_UNIFIED_SPLIT_DIR = (
    PROJECT_ROOT / "data/processed_affinity/unified_ablation_datasets/unified_no_high_risk"
)

MISSING_VALUES = {"", "na", "n/a", "nan", "none", "\\", "unknown", "not_reported", "not reported"}
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$", re.IGNORECASE)
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def normalize_text(value) -> str:
    """Turn spreadsheet values into comparable strings."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def is_present(value) -> bool:
    """Return True when a field contains real information."""

    return normalize_text(value).lower() not in MISSING_VALUES


def is_sequence(value) -> bool:
    """Check whether a string looks like an amino-acid sequence."""

    text = normalize_text(value).replace(" ", "").replace("-", "")
    return len(text) >= 5 and bool(AA_PATTERN.fullmatch(text))


def to_float(value):
    """Parse numeric fields such as Kd or delta-G."""

    text = normalize_text(value).replace(",", "")
    if not is_present(text):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def xlsx_column_index(cell_ref: str) -> int:
    """Convert Excel cell reference like AA4 into zero-based column index."""

    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for char in letters.upper():
        index = index * 26 + ord(char) - 64
    return index - 1


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    """Read the first worksheet of a simple xlsx file using only stdlib."""

    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for string_item in shared_root.findall(f"{XLSX_NS}si"):
                pieces = [node.text or "" for node in string_item.iter(f"{XLSX_NS}t")]
                shared_strings.append("".join(pieces))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        parsed_rows = []
        max_columns = 0
        for row in sheet_root.iter(f"{XLSX_NS}row"):
            values = []
            for cell in row.findall(f"{XLSX_NS}c"):
                column_index = xlsx_column_index(cell.attrib.get("r", "A1"))
                while len(values) <= column_index:
                    values.append("")
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{XLSX_NS}v")
                inline_node = cell.find(f"{XLSX_NS}is")
                value = ""
                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(node.text or "" for node in inline_node.iter(f"{XLSX_NS}t"))
                elif value_node is not None:
                    value = value_node.text or ""
                values[column_index] = normalize_text(value)
            max_columns = max(max_columns, len(values))
            parsed_rows.append(values)

    if not parsed_rows:
        raise ValueError(f"No rows found in {path}")
    header = parsed_rows[0]
    rows = []
    for raw_row in parsed_rows[1:]:
        padded = raw_row + [""] * (len(header) - len(raw_row))
        rows.append(padded[: len(header)])
    return pd.DataFrame(rows, columns=header)


def read_data_dictionary(path: Path) -> pd.DataFrame:
    """Read data dictionary with encoding fallback."""

    for encoding in ["utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def find_column(columns: list[str], startswith: str) -> str | None:
    """Find first column whose name starts with a prefix."""

    for column in columns:
        if column.lower().startswith(startswith.lower()):
            return column
    return None


def classify_format(row: pd.Series) -> str:
    """Classify molecule format into broad audit buckets."""

    ab_or_nano = normalize_text(row.get("Ab_or_Nano", "")).lower()
    joined_text = " ".join(
        normalize_text(row.get(column, ""))
        for column in [
            "Ab_or_Nano",
            "Structure_Title",
            "H_Chain Macromolecule Name",
            "L_Chain Macromolecule Name",
            "Provenance",
        ]
    ).lower()
    heavy_present = is_sequence(row.get("Ab/Nano H_Chain AA", ""))
    light_present = is_sequence(row.get("Ab/Nano L_Chain AA", ""))

    if "scfv" in joined_text or "single-chain" in joined_text or "single chain" in joined_text:
        return "scFv"
    if "bj" in joined_text or "light-chain dimer" in joined_text or "light chain dimer" in joined_text:
        return "BJ / light-chain dimer"
    if "nano" in ab_or_nano or "vhh" in joined_text or "nanobody" in joined_text:
        return "VHH / nanobody"
    if "antibody" in ab_or_nano or (heavy_present and light_present):
        return "antibody"
    return "other / unknown"


def is_predicted_row(row: pd.Series) -> bool:
    """Detect predicted affinity labels, including ANTIPASTI provenance."""

    text = " ".join(
        normalize_text(row.get(column, ""))
        for column in ["Predicted_or_Not", "Provenance", "Affinity_Method", "Reason_Code", "Source"]
    ).lower()
    return "predict" in text or "antipasti" in text


def usable_antibody_sequence(row: pd.Series, molecule_format: str) -> bool:
    """Check whether the antibody-side sequence is usable for this row type."""

    heavy = is_sequence(row.get("Ab/Nano H_Chain AA", ""))
    light = is_sequence(row.get("Ab/Nano L_Chain AA", ""))
    if molecule_format == "VHH / nanobody":
        return heavy
    if molecule_format == "scFv":
        return heavy or light
    if molecule_format == "BJ / light-chain dimer":
        return light
    return heavy and light


def quality_tier(row: pd.Series, kd_column: str, delta_column: str | None) -> str:
    """Assign broad label-quality tier for possible future dataset expansion."""

    molecule_format = row["molecule_format"]
    has_antigen = is_sequence(row.get("Ag_Seq", ""))
    has_antibody = usable_antibody_sequence(row, molecule_format)
    kd = to_float(row.get(kd_column, ""))
    delta_g = to_float(row.get(delta_column, "")) if delta_column else None
    kd_positive = kd is not None and kd > 0
    predicted = is_predicted_row(row)
    mutation_text = normalize_text(row.get("Ab/Nano_Mutation", "")).lower()
    has_mutation = is_present(mutation_text) and mutation_text not in {"no", "false"}

    if not has_antigen or not has_antibody:
        return "Exclude"
    if kd_positive and not predicted and not has_mutation:
        return "Tier 1"
    if (kd_positive or delta_g is not None) and predicted:
        return "Tier 2"
    if kd_positive or delta_g is not None:
        return "Tier 3"
    return "Tier 3"


def source_id(row: pd.Series) -> str:
    """Build a loose source/PDB identifier for overlap checks."""

    pdb = normalize_text(row.get("PDB_ID", "")).upper()
    source = normalize_text(row.get("Source", ""))
    provenance = normalize_text(row.get("Provenance", ""))
    return pdb or provenance or source


def triplet_key(row: pd.Series) -> str:
    """Build heavy+light+antigen exact sequence key for antibody rows."""

    heavy = normalize_text(row.get("Ab/Nano H_Chain AA", ""))
    light = normalize_text(row.get("Ab/Nano L_Chain AA", ""))
    antigen = normalize_text(row.get("Ag_Seq", ""))
    return "||".join([heavy, light, antigen])


def load_current_unified() -> pd.DataFrame:
    """Load current unified_no_high_risk splits for overlap checks."""

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


def describe_bool(count: int, total: int) -> str:
    """Format count plus percent."""

    pct = (count / total * 100) if total else 0.0
    return f"{count} ({pct:.1f}%)"


def numeric_summary(values: list[float]) -> dict:
    """Return small numeric summary for audit report."""

    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def is_real_number(value) -> bool:
    """Return True for non-NaN numeric values."""

    return value is not None and not pd.isna(value)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write list of dictionaries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run ANDD audit and save Markdown/CSV summaries."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    andd = read_xlsx_first_sheet(ANDD_XLSX)
    dictionary = read_data_dictionary(DATA_DICTIONARY)
    columns = list(andd.columns)
    kd_column = find_column(columns, "Affinity_Kd")
    delta_column = find_column(columns, "∆Gbinding") or find_column(columns, "ΔGbinding")
    if kd_column is None:
        raise ValueError("Could not find Affinity_Kd column in ANDD.")

    andd["molecule_format"] = andd.apply(classify_format, axis=1)
    andd["has_heavy_sequence"] = andd["Ab/Nano H_Chain AA"].apply(is_sequence)
    andd["has_light_sequence"] = andd["Ab/Nano L_Chain AA"].apply(is_sequence)
    andd["has_antigen_sequence"] = andd["Ag_Seq"].apply(is_sequence)
    andd["has_vhh_or_nanobody_sequence"] = andd.apply(
        lambda row: row["molecule_format"] == "VHH / nanobody"
        and is_sequence(row.get("Ab/Nano H_Chain AA", "")),
        axis=1,
    )
    cdr_columns = [col for col in columns if col.startswith("Ab/Nano_CDR")]
    andd["has_any_cdr_field"] = andd[cdr_columns].apply(
        lambda row: any(is_present(value) for value in row),
        axis=1,
    )
    andd["has_all_six_cdr_fields"] = andd[cdr_columns].apply(
        lambda row: all(is_present(value) for value in row),
        axis=1,
    )
    andd["kd_value"] = andd[kd_column].apply(to_float)
    andd["kd_positive"] = andd["kd_value"].apply(lambda value: value is not None and value > 0)
    if delta_column:
        andd["delta_g_value"] = andd[delta_column].apply(to_float)
    else:
        andd["delta_g_value"] = None
    andd["has_delta_g"] = andd["delta_g_value"].apply(is_real_number)
    andd["is_predicted_affinity"] = andd.apply(is_predicted_row, axis=1)
    andd["is_antipasti_predicted"] = andd.apply(
        lambda row: "antipasti"
        in " ".join(
            normalize_text(row.get(col, ""))
            for col in ["Predicted_or_Not", "Provenance", "Affinity_Method", "Source"]
        ).lower(),
        axis=1,
    )
    andd["quality_tier"] = andd.apply(lambda row: quality_tier(row, kd_column, delta_column), axis=1)
    andd["source_id"] = andd.apply(source_id, axis=1)
    andd["andd_triplet_key"] = andd.apply(triplet_key, axis=1)

    unified = load_current_unified()
    unified_antigens = set()
    unified_triplets = set()
    unified_ids = set()
    if not unified.empty:
        unified_antigens = set(unified.get("antigen_sequence", pd.Series(dtype=str)).dropna().astype(str))
        unified_triplets = set(
            (
                unified.get("heavy_sequence", pd.Series(dtype=str)).fillna("").astype(str)
                + "||"
                + unified.get("light_sequence", pd.Series(dtype=str)).fillna("").astype(str)
                + "||"
                + unified.get("antigen_sequence", pd.Series(dtype=str)).fillna("").astype(str)
            )
        )
        for possible_id in ["pdb_or_antibody_id", "antibody_id", "PDB_ID", "pdb"]:
            if possible_id in unified.columns:
                unified_ids |= set(unified[possible_id].dropna().astype(str).str.upper())

    andd["overlap_antigen_sequence"] = andd["Ag_Seq"].astype(str).isin(unified_antigens)
    andd["overlap_triplet"] = andd["andd_triplet_key"].isin(unified_triplets)
    andd["overlap_source_id"] = andd["source_id"].astype(str).str.upper().isin(unified_ids)

    total_rows = len(andd)
    format_counts = andd["molecule_format"].value_counts(dropna=False).to_dict()
    tier_counts = andd["quality_tier"].value_counts(dropna=False).to_dict()
    method_counts = andd["Affinity_Method"].fillna("").astype(str).replace("", "missing").value_counts().head(25)
    predicted_counts = andd["is_predicted_affinity"].value_counts().to_dict()
    antipasti_count = int(andd["is_antipasti_predicted"].sum())

    column_summary_rows = []
    dictionary_by_column = {
        normalize_text(row.get("Column_name", "")): row.to_dict()
        for _, row in dictionary.iterrows()
    }
    for column in columns:
        present_count = int(andd[column].apply(is_present).sum())
        dict_row = dictionary_by_column.get(column, {})
        column_summary_rows.append(
            {
                "column_name": column,
                "present_count": present_count,
                "missing_count": total_rows - present_count,
                "present_percent": round(present_count / total_rows * 100, 2) if total_rows else 0,
                "description": normalize_text(dict_row.get("Description", "")),
                "data_type": normalize_text(dict_row.get("Data_type", "")),
                "allowed_values": normalize_text(dict_row.get("Controlled_terms / Allowed_values", "")),
                "unit": normalize_text(dict_row.get("Unit", "")),
            }
        )
    write_csv(
        OUTPUT_DIR / "ANDD_column_summary.csv",
        column_summary_rows,
        [
            "column_name",
            "present_count",
            "missing_count",
            "present_percent",
            "description",
            "data_type",
            "allowed_values",
            "unit",
        ],
    )

    candidate_rows = []
    for _, row in andd.iterrows():
        if row["quality_tier"] == "Exclude":
            continue
        candidate_rows.append(
            {
                "quality_tier": row["quality_tier"],
                "source": row.get("Source", ""),
                "pdb_id": row.get("PDB_ID", ""),
                "molecule_format": row["molecule_format"],
                "has_heavy_sequence": row["has_heavy_sequence"],
                "has_light_sequence": row["has_light_sequence"],
                "has_antigen_sequence": row["has_antigen_sequence"],
                "has_all_six_cdr_fields": row["has_all_six_cdr_fields"],
                "kd_value": row["kd_value"],
                "delta_g_value": row["delta_g_value"],
                "affinity_method": row.get("Affinity_Method", ""),
                "predicted_or_not": row.get("Predicted_or_Not", ""),
                "is_predicted_affinity": row["is_predicted_affinity"],
                "is_antipasti_predicted": row["is_antipasti_predicted"],
                "overlap_antigen_sequence": row["overlap_antigen_sequence"],
                "overlap_triplet": row["overlap_triplet"],
                "overlap_source_id": row["overlap_source_id"],
                "ag_name": row.get("Ag_Name", ""),
            }
        )
    write_csv(
        OUTPUT_DIR / "ANDD_candidate_rows_summary.csv",
        candidate_rows,
        [
            "quality_tier",
            "source",
            "pdb_id",
            "molecule_format",
            "has_heavy_sequence",
            "has_light_sequence",
            "has_antigen_sequence",
            "has_all_six_cdr_fields",
            "kd_value",
            "delta_g_value",
            "affinity_method",
            "predicted_or_not",
            "is_predicted_affinity",
            "is_antipasti_predicted",
            "overlap_antigen_sequence",
            "overlap_triplet",
            "overlap_source_id",
            "ag_name",
        ],
    )

    report_json = {
        "total_rows": total_rows,
        "columns": columns,
        "molecule_format_counts": format_counts,
        "sequence_availability": {
            "heavy_sequence": int(andd["has_heavy_sequence"].sum()),
            "light_sequence": int(andd["has_light_sequence"].sum()),
            "nanobody_or_vhh_sequence": int(andd["has_vhh_or_nanobody_sequence"].sum()),
            "antigen_sequence": int(andd["has_antigen_sequence"].sum()),
            "any_cdr_field": int(andd["has_any_cdr_field"].sum()),
            "all_six_cdr_fields": int(andd["has_all_six_cdr_fields"].sum()),
        },
        "affinity_availability": {
            "kd_column": kd_column,
            "kd_present": int(andd["kd_value"].notna().sum()),
            "kd_positive": int(andd["kd_positive"].sum()),
            "delta_g_column": delta_column,
            "delta_g_present": int(andd["has_delta_g"].sum()),
            "predicted_rows": int(andd["is_predicted_affinity"].sum()),
            "antipasti_predicted_rows": antipasti_count,
            "kd_numeric_summary": numeric_summary(andd["kd_value"].tolist()),
            "delta_g_numeric_summary": numeric_summary(andd["delta_g_value"].tolist()),
        },
        "quality_tier_counts": tier_counts,
        "overlap_with_unified_no_high_risk": {
            "antigen_sequence_overlap_rows": int(andd["overlap_antigen_sequence"].sum()),
            "triplet_overlap_rows": int(andd["overlap_triplet"].sum()),
            "source_id_overlap_rows": int(andd["overlap_source_id"].sum()),
        },
    }
    (OUTPUT_DIR / "ANDD_audit_summary.json").write_text(
        json.dumps(report_json, indent=2),
        encoding="utf-8",
    )

    key_columns = [
        "Source",
        "PDB_ID",
        "Ab_or_Nano",
        "Ab/Nano H_Chain AA",
        "Ab/Nano L_Chain AA",
        "Ag_Seq",
        kd_column,
        delta_column or "",
        "Affinity_Method",
        "Predicted_or_Not",
        "Provenance",
    ]
    key_columns = [col for col in key_columns if col and col in columns]

    lines = [
        "# ANDD Data Source Audit",
        "",
        "## Scope",
        "",
        "- Input workbook: `data/external/ANDD/ANDD_v2.xlsx`",
        "- Data dictionary: `data/external/ANDD/Data_dictionary.csv`",
        "- QC report copied for reference: `data/external/ANDD/Data_quality_control_report.pdf`",
        "- This audit does not create a training dataset and does not modify existing unified datasets.",
        "",
        "## 1. Row And Column Overview",
        "",
        f"- Total rows: `{total_rows}`",
        f"- Total columns: `{len(columns)}`",
        "",
        "### Columns",
        "",
        ", ".join(f"`{column}`" for column in columns),
        "",
        "### Key Fields From Data Dictionary",
        "",
        "| Column | Dictionary description | Unit / allowed values |",
        "|---|---|---|",
    ]
    for column in key_columns:
        dict_row = dictionary_by_column.get(column, {})
        lines.append(
            f"| `{column}` | {normalize_text(dict_row.get('Description', '')) or 'Not described'} | "
            f"{normalize_text(dict_row.get('Unit', '')) or normalize_text(dict_row.get('Controlled_terms / Allowed_values', '')) or 'NA'} |"
        )

    lines.extend(
        [
            "",
            "## 2. Molecule Format Distribution",
            "",
            "| Format bucket | Rows |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(format_counts.items(), key=lambda item: item[0]):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## 3. Sequence Availability",
            "",
            f"- Heavy sequence present: `{describe_bool(int(andd['has_heavy_sequence'].sum()), total_rows)}`",
            f"- Light sequence present: `{describe_bool(int(andd['has_light_sequence'].sum()), total_rows)}`",
            f"- Nanobody/VHH sequence present: `{describe_bool(int(andd['has_vhh_or_nanobody_sequence'].sum()), total_rows)}`",
            f"- Antigen sequence present: `{describe_bool(int(andd['has_antigen_sequence'].sum()), total_rows)}`",
            f"- Any CDR field present: `{describe_bool(int(andd['has_any_cdr_field'].sum()), total_rows)}`",
            f"- All six CDR fields present: `{describe_bool(int(andd['has_all_six_cdr_fields'].sum()), total_rows)}`",
            "",
            "## 4. Affinity Availability",
            "",
            f"- Kd column: `{kd_column}`",
            f"- Numeric Kd rows: `{describe_bool(int(andd['kd_value'].notna().sum()), total_rows)}`",
            f"- Positive Kd rows: `{describe_bool(int(andd['kd_positive'].sum()), total_rows)}`",
            f"- Delta-G column: `{delta_column or 'not found'}`",
            f"- Numeric delta-G rows: `{describe_bool(int(andd['has_delta_g'].sum()), total_rows)}`",
            f"- Predicted affinity rows: `{describe_bool(int(andd['is_predicted_affinity'].sum()), total_rows)}`",
            f"- ANTIPASTI predicted rows: `{describe_bool(antipasti_count, total_rows)}`",
            "",
            "### Top Affinity_Method Values",
            "",
            "| Affinity_Method | Rows |",
            "|---|---:|",
        ]
    )
    for method, count in method_counts.items():
        lines.append(f"| `{method}` | {count} |")

    lines.extend(
        [
            "",
            "## 5. Label Quality Tiers",
            "",
            "Tier definitions used in this audit:",
            "",
            "- Tier 1: experimental quantitative positive Kd-like affinity + usable antibody/nanobody sequence + antigen sequence, with no mutation flag.",
            "- Tier 2: predicted affinity or mixed quantitative labels with usable sequences.",
            "- Tier 3: quantitative but unclear/mutation/incomplete-label cases that still have usable sequences.",
            "- Exclude: missing usable antibody/nanobody sequence or missing antigen sequence.",
            "",
            "| Tier | Rows |",
            "|---|---:|",
        ]
    )
    for tier in ["Tier 1", "Tier 2", "Tier 3", "Exclude"]:
        lines.append(f"| `{tier}` | {tier_counts.get(tier, 0)} |")

    lines.extend(
        [
            "",
            "## 6. Overlap With Current unified_no_high_risk",
            "",
            f"- Duplicate antigen_sequence rows: `{int(andd['overlap_antigen_sequence'].sum())}`",
            f"- Duplicate heavy+light+antigen triplet rows: `{int(andd['overlap_triplet'].sum())}`",
            f"- Duplicate PDB/source ID rows: `{int(andd['overlap_source_id'].sum())}`",
            "",
            "Interpretation: overlap rows should not be blindly merged. Future dataset construction should deduplicate by exact sequence triplet and preserve source/provenance fields.",
            "",
            "## 7. Answering The Main Questions",
            "",
            "### 1. Is ANDD suitable for expanding the current antibody-antigen affinity regression task?",
            "",
            "Potentially yes, but it should be expanded carefully rather than merged wholesale. ANDD contains antibody, nanobody/VHH, antigen sequence, CDR, Kd, delta-G, method, predicted-label, and provenance fields. That makes it valuable, but also heterogeneous.",
            "",
            "### 2. How many Tier 1 candidate rows are available?",
            "",
            f"- Tier 1 candidate rows by this first audit: `{tier_counts.get('Tier 1', 0)}`",
            "",
            "This number should be treated as a candidate count, not a final trainable count. The next build step should deduplicate exact sequence triplets, separate antibodies from nanobodies, and verify target/source consistency.",
            "",
            "### 3. Should antibody and nanobody be separate tasks?",
            "",
            "Yes. Antibodies usually have heavy+light chain pairing, while VHH/nanobody samples are single-domain binders. Mixing them directly can confuse the input schema and biological interpretation. They can share infrastructure, but should be separate dataset versions or explicit input modes.",
            "",
            "### 4. Should predicted affinity be excluded from main training?",
            "",
            "Yes for the main supervised benchmark. Predicted labels, including ANTIPASTI-like rows, should be excluded from the primary experimental Kd training set or placed in a separate Tier 2/pretraining/auxiliary experiment.",
            "",
            "### 5. Is it worth building expanded_affinity_dataset_v2?",
            "",
            "Yes, if the build is conservative. The recommended next step is to create an ANDD-derived candidate table with Tier 1 experimental rows only, deduplicate against current unified_no_high_risk, then decide whether to build an antibody-only v2, a nanobody-only v2, or both.",
            "",
            "## 8. Output Files",
            "",
            "- Column summary: `outputs/data_expansion/ANDD_audit/ANDD_column_summary.csv`",
            "- Candidate row summary: `outputs/data_expansion/ANDD_audit/ANDD_candidate_rows_summary.csv`",
            "- Machine-readable audit summary: `outputs/data_expansion/ANDD_audit/ANDD_audit_summary.json`",
        ]
    )
    (OUTPUT_DIR / "ANDD_data_source_audit.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"ANDD rows: {total_rows}")
    print(f"Tier counts: {tier_counts}")
    print(f"Molecule formats: {format_counts}")
    print(f"Positive Kd rows: {int(andd['kd_positive'].sum())}")
    print(f"ANTIPASTI predicted rows: {antipasti_count}")
    print(f"Saved audit report to {OUTPUT_DIR / 'ANDD_data_source_audit.md'}")


if __name__ == "__main__":
    main()
