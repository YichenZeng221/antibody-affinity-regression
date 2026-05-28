"""Extract standard IMGT CDRs for ANDD-only antibody v2.

:
 split  ANDD antibody v2  CDR
, split 

 ANDD  CDR ?
 CDR / CDR-aware baseline
 AbNumber + IMGT

:
 `abnumber-cdr` ,:
    conda run -n abnumber-cdr python scripts/extract_standard_cdrs_andd_antibody_v2.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2"
OUTPUT_DIR = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2_cdr_annotated"

SPLITS = ["train", "val", "test"]
BACKEND_NAME = "abnumber_anarci_imgt"
FAILED_BACKEND = "failed"
REQUIRED_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
CDR_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]


def load_abnumber_chain():
    """ AbNumber, hmmscan  PATH """

    current_python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = current_python_bin + os.pathsep + os.environ.get("PATH", "")
    try:
        from abnumber import Chain
    except ImportError as error:
        raise ImportError(
            "Cannot import AbNumber. Please run with the verified `abnumber-cdr` environment."
        ) from error
    return Chain


def short_error_message(error: Exception) -> str:
    """ AbNumber , CSV"""

    message = " ".join(str(error).split())
    return message or error.__class__.__name__


def extract_chain_cdrs(sequence: str, expected_chain: str, Chain) -> dict:
    """ IMGT  heavy/light chain  CDR1/2/3"""

    if pd.isna(sequence) or not str(sequence).strip():
        return {
            "cdrs": {"CDR1": "", "CDR2": "", "CDR3": ""},
            "backend": FAILED_BACKEND,
            "status": "failed",
            "error": "missing_sequence",
        }

    try:
        chain = Chain(str(sequence).strip(), scheme="imgt")
    except Exception as error:
        return {
            "cdrs": {"CDR1": "", "CDR2": "", "CDR3": ""},
            "backend": FAILED_BACKEND,
            "status": "failed",
            "error": f"abnumber_numbering_failed: {short_error_message(error)}",
        }

    if expected_chain == "heavy" and not chain.is_heavy_chain():
        return {
            "cdrs": {"CDR1": "", "CDR2": "", "CDR3": ""},
            "backend": FAILED_BACKEND,
            "status": "failed",
            "error": f"chain_type_mismatch: expected heavy, got {chain.chain_type}",
        }
    if expected_chain == "light" and not chain.is_light_chain():
        return {
            "cdrs": {"CDR1": "", "CDR2": "", "CDR3": ""},
            "backend": FAILED_BACKEND,
            "status": "failed",
            "error": f"chain_type_mismatch: expected light, got {chain.chain_type}",
        }

    cdrs = {"CDR1": str(chain.cdr1_seq), "CDR2": str(chain.cdr2_seq), "CDR3": str(chain.cdr3_seq)}
    if any(not value for value in cdrs.values()):
        return {
            "cdrs": cdrs,
            "backend": FAILED_BACKEND,
            "status": "failed",
            "error": "empty_cdr_returned_by_abnumber",
        }
    return {"cdrs": cdrs, "backend": BACKEND_NAME, "status": "success", "error": ""}


def annotate_row(row: dict, Chain) -> dict:
    """, CDR  status/error"""

    heavy = extract_chain_cdrs(row["heavy_sequence"], "heavy", Chain)
    light = extract_chain_cdrs(row["light_sequence"], "light", Chain)

    annotated = dict(row)
    annotated["HCDR1"] = heavy["cdrs"]["CDR1"]
    annotated["HCDR2"] = heavy["cdrs"]["CDR2"]
    annotated["HCDR3"] = heavy["cdrs"]["CDR3"]
    annotated["LCDR1"] = light["cdrs"]["CDR1"]
    annotated["LCDR2"] = light["cdrs"]["CDR2"]
    annotated["LCDR3"] = light["cdrs"]["CDR3"]
    annotated["heavy_cdr_backend"] = heavy["backend"]
    annotated["light_cdr_backend"] = light["backend"]
    annotated["heavy_cdr_status"] = heavy["status"]
    annotated["light_cdr_status"] = light["status"]
    annotated["heavy_cdr_error"] = heavy["error"]
    annotated["light_cdr_error"] = light["error"]
    return annotated


def annotate_split(split_name: str, Chain) -> pd.DataFrame:
    """ split"""

    path = INPUT_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    rows = []
    print(f"Annotating {split_name}: {len(df)} rows")
    for index, row in enumerate(df.to_dict("records"), start=1):
        rows.append(annotate_row(row, Chain))
        if index == 1 or index % 50 == 0 or index == len(df):
            print(f"  {split_name}: processed {index}/{len(df)}")
    return pd.DataFrame(rows)


def cdr_length_summary(df: pd.DataFrame) -> list[str]:
    """ CDR  summary,"""

    lines = []
    for column in CDR_COLUMNS:
        lengths = df[column].fillna("").astype(str).str.len()
        unique_lengths = [int(value) for value in sorted(lengths.unique())[:25]]
        lines.append(
            f"- `{column}`: nonempty={(lengths > 0).sum()}, min={lengths.min()}, "
            f"mean={lengths.mean():.2f}, max={lengths.max()}, unique_lengths={unique_lengths}"
        )
    return lines


def grouped_error(error: str) -> str:
    """ error """

    if not error:
        return ""
    if "Found 2 antibody domains in sequence" in error:
        return "abnumber_numbering_failed: found_2_antibody_domains_in_sequence"
    if error.startswith("chain_type_mismatch"):
        return "chain_type_mismatch"
    if error.startswith("abnumber_numbering_failed"):
        return "abnumber_numbering_failed: other"
    return error


def build_report(split_frames: dict[str, pd.DataFrame], failures: pd.DataFrame) -> str:
    """ Markdown summary"""

    all_df = pd.concat(split_frames.values(), ignore_index=True)
    lines = [
        "# ANDD-only Antibody v2 Standard CDR Extraction Summary",
        "",
        "- Backend: `AbNumber + IMGT`.",
        "- No model was trained by this script.",
        "- Original split files were not overwritten.",
        "",
        "## Split Success Counts",
        "",
        "| Split | Rows | heavy success | light success | both success |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, df in split_frames.items():
        heavy_ok = df["heavy_cdr_status"].eq("success")
        light_ok = df["light_cdr_status"].eq("success")
        lines.append(
            f"| {split} | {len(df)} | {int(heavy_ok.sum())} | {int(light_ok.sum())} | {int((heavy_ok & light_ok).sum())} |"
        )

    lines.extend(["", "## Overall CDR Length Distribution", ""])
    lines.extend(cdr_length_summary(all_df))

    lines.extend(["", "## Failure Reason Counts", ""])
    if failures.empty:
        lines.append("- None.")
    else:
        counter = Counter()
        for _, row in failures.iterrows():
            if row.get("heavy_cdr_error", ""):
                counter[f"heavy:{grouped_error(str(row['heavy_cdr_error']))}"] += 1
            if row.get("light_cdr_error", ""):
                counter[f"light:{grouped_error(str(row['light_cdr_error']))}"] += 1
        for reason, count in counter.most_common():
            lines.append(f"- `{reason}`: {count}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Chain = load_abnumber_chain()

    split_frames = {}
    for split in SPLITS:
        annotated = annotate_split(split, Chain)
        annotated.to_csv(OUTPUT_DIR / f"{split}.csv", index=False)
        split_frames[split] = annotated

    failure_frames = []
    for split, df in split_frames.items():
        failed = df[(df["heavy_cdr_status"] != "success") | (df["light_cdr_status"] != "success")].copy()
        if not failed.empty:
            failed.insert(0, "split_name", split)
            failure_frames.append(failed)
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame()
    failures.to_csv(OUTPUT_DIR / "cdr_extraction_failures.csv", index=False)

    report = build_report(split_frames, failures)
    (OUTPUT_DIR / "cdr_extraction_summary.md").write_text(report, encoding="utf-8")
    print(f"Saved CDR-annotated split files to {OUTPUT_DIR}")
    print(f"Failure rows: {len(failures)}")


if __name__ == "__main__":
    main()
