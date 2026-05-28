"""Annotate unified_no_high_risk with standard AbNumber IMGT CDR sequences.

:
,, ablation dataset

 whole-sequence :
    heavy_sequence, light_sequence, antigen_sequence

, CDR :
    HCDR1/HCDR2/HCDR3
    LCDR1/LCDR2/LCDR3

 AbNumber  IMGT numbering:
- AbNumber  antibody numbering  CDR 
-  fixed index slicing, CDR 
-  heavy/light sequence , row ,
   status/error ,
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import os
import sys
import textwrap

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "unified_ablation_datasets"
    / "unified_no_high_risk"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "unified_ablation_datasets"
    / "unified_no_high_risk_cdr_annotated"
)

SPLITS = ["train", "val", "test"]
BACKEND_NAME = "abnumber_anarci_imgt"
FAILED_BACKEND = "failed"
REQUIRED_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
CDR_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]


def load_abnumber_chain():
    """Import AbNumber only when the script starts running.

    :
     `abnumber-cdr` ,
     AbNumber,
    """

    # ANARCI  `hmmscan` conda env  Python
    # , `conda activate`,shell PATH 
    #  hmmscan Python  bin/  PATH,
    current_python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = current_python_bin + os.pathsep + os.environ.get("PATH", "")

    try:
        from abnumber import Chain
    except ImportError as error:
        raise ImportError(
            "Cannot import AbNumber. Run this script with the verified "
            "`abnumber-cdr` environment."
        ) from error
    return Chain


def load_split(split_name: str) -> pd.DataFrame:
    """Read one original unified_no_high_risk split without changing it."""

    csv_path = INPUT_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")
    return dataframe


def short_error_message(error: Exception) -> str:
    """Keep library errors readable inside CSV and markdown reports."""

    message = " ".join(str(error).split())
    return message or error.__class__.__name__


def extract_chain_cdrs(sequence: str, expected_chain: str, Chain) -> dict:
    """Extract CDR1/CDR2/CDR3 from one antibody chain with IMGT numbering.

    expected_chain is either ``heavy`` or ``light``.
    We check the detected chain type because a parsed sequence is not enough:
    a heavy sequence accidentally passed as light would give misleading features.
    """

    if pd.isna(sequence) or not str(sequence).strip():
        return {
            "cdrs": {"CDR1": "", "CDR2": "", "CDR3": ""},
            "backend": FAILED_BACKEND,
            "status": "failed",
            "error": "missing_sequence",
        }

    try:
        chain = Chain(str(sequence).strip(), scheme="imgt")
    except Exception as error:  # AbNumber may raise several numbering/backend errors.
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

    cdrs = {
        "CDR1": str(chain.cdr1_seq),
        "CDR2": str(chain.cdr2_seq),
        "CDR3": str(chain.cdr3_seq),
    }
    if any(not cdr for cdr in cdrs.values()):
        return {
            "cdrs": cdrs,
            "backend": FAILED_BACKEND,
            "status": "failed",
            "error": "empty_cdr_returned_by_abnumber",
        }

    return {
        "cdrs": cdrs,
        "backend": BACKEND_NAME,
        "status": "success",
        "error": "",
    }


def annotate_row(row: dict, Chain) -> dict:
    """Copy one row and add heavy/light standard CDR annotation columns."""

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


def annotate_split(split_name: str, dataframe: pd.DataFrame, Chain) -> pd.DataFrame:
    """Annotate one split and print gentle progress for longer runs."""

    rows = dataframe.to_dict("records")
    annotated_rows = []
    total_rows = len(rows)

    print(f"Annotating {split_name}: {total_rows} rows")
    for index, row in enumerate(rows, start=1):
        annotated_rows.append(annotate_row(row, Chain))
        if index == 1 or index % 25 == 0 or index == total_rows:
            print(f"  {split_name}: processed {index}/{total_rows}")

    return pd.DataFrame(annotated_rows)


def length_summary(dataframe: pd.DataFrame, cdr_column: str) -> dict:
    """Summarize CDR sequence lengths for markdown."""

    lengths = dataframe[cdr_column].fillna("").astype(str).str.len()
    return {
        "count": int(len(lengths)),
        "nonempty_count": int((lengths > 0).sum()),
        "min": int(lengths.min()) if len(lengths) else None,
        "max": int(lengths.max()) if len(lengths) else None,
        "mean": float(lengths.mean()) if len(lengths) else None,
        "std": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
    }


def failure_reason_counts(dataframe: pd.DataFrame, error_column: str) -> Counter:
    """Count non-empty extraction error messages for one chain side."""

    errors = dataframe[error_column].fillna("").astype(str)
    return Counter(summarize_failure_reason(error) for error in errors if error)


def summarize_failure_reason(error: str) -> str:
    """Group verbose AbNumber messages into readable summary categories.

    Full error text stays in `cdr_extraction_failures.csv`.
    Markdown only needs the failure family so we can see the main pattern quickly.
    """

    if "Found 2 antibody domains in sequence" in error:
        return "abnumber_numbering_failed: found_2_antibody_domains_in_sequence"
    if error.startswith("chain_type_mismatch"):
        return error.split(":", maxsplit=1)[0]
    if error.startswith("abnumber_numbering_failed"):
        return "abnumber_numbering_failed: other"
    return error


def markdown_counter(counter: Counter) -> list[str]:
    """Render reason counts as short markdown bullets."""

    if not counter:
        return ["- None."]
    return [f"- `{reason}`: {count}" for reason, count in counter.most_common()]


def build_failures_table(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Collect rows where heavy or light standard extraction failed."""

    failure_frames = []
    for split_name, dataframe in split_frames.items():
        failed = dataframe[
            (dataframe["heavy_cdr_status"] != "success")
            | (dataframe["light_cdr_status"] != "success")
        ].copy()
        if not failed.empty:
            failed.insert(0, "split", split_name)
            failure_frames.append(failed)

    if not failure_frames:
        columns = ["split"] + list(next(iter(split_frames.values())).columns)
        return pd.DataFrame(columns=columns)
    return pd.concat(failure_frames, ignore_index=True)


def write_summary(split_frames: dict[str, pd.DataFrame], failures: pd.DataFrame) -> None:
    """Write a compact markdown report for this annotation run."""

    all_rows = pd.concat(split_frames.values(), ignore_index=True)
    heavy_errors = failure_reason_counts(all_rows, "heavy_cdr_error")
    light_errors = failure_reason_counts(all_rows, "light_cdr_error")

    lines = [
        "# Unified No High Risk Standard CDR Extraction Summary",
        "",
        "## Scope",
        "",
        f"- Input dataset: `{INPUT_DIR.relative_to(PROJECT_ROOT)}`",
        f"- Output dataset: `{OUTPUT_DIR.relative_to(PROJECT_ROOT)}`",
        "- Backend: `AbNumber` with `scheme='imgt'`.",
        "- This annotation run does not use fixed-index CDR slicing.",
        "- Failed extraction rows are kept in split CSVs and also copied to `cdr_extraction_failures.csv`.",
        "",
        "## Rows And Status",
        "",
        "| split | rows | heavy success | heavy failure | light success | light failure |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for split_name in SPLITS:
        dataframe = split_frames[split_name]
        heavy_success = int((dataframe["heavy_cdr_status"] == "success").sum())
        light_success = int((dataframe["light_cdr_status"] == "success").sum())
        lines.append(
            f"| `{split_name}` | {len(dataframe)} | {heavy_success} | "
            f"{len(dataframe) - heavy_success} | {light_success} | "
            f"{len(dataframe) - light_success} |"
        )

    lines.extend(
        [
            "",
            f"- Total rows: `{len(all_rows)}`",
            f"- Rows with any heavy/light extraction failure: `{len(failures)}`",
            f"- Heavy backend counts: `{all_rows['heavy_cdr_backend'].value_counts(dropna=False).to_dict()}`",
            f"- Light backend counts: `{all_rows['light_cdr_backend'].value_counts(dropna=False).to_dict()}`",
            "",
            "## CDR Length Distribution",
            "",
            "Lengths below are amino-acid counts from the extracted CDR strings.",
            "",
            "| CDR | non-empty rows | min | max | mean | std |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for cdr_column in CDR_COLUMNS:
        summary = length_summary(all_rows, cdr_column)
        lines.append(
            f"| `{cdr_column}` | {summary['nonempty_count']}/{summary['count']} | "
            f"{summary['min']} | {summary['max']} | {summary['mean']:.3f} | "
            f"{summary['std']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Failure Reason Counts",
            "",
            "Heavy-chain extraction errors:",
            "",
            *markdown_counter(heavy_errors),
            "",
            "Light-chain extraction errors:",
            "",
            *markdown_counter(light_errors),
            "",
            "## Next Check",
            "",
            textwrap.dedent(
                """\
                Inspect `cdr_extraction_failures.csv` before creating a CDR-aware training dataset.
                A later training script should choose an explicit policy for failed rows:
                either filter rows with missing CDRs or define a documented fallback.
                """
            ).strip(),
            "",
        ]
    )

    (OUTPUT_DIR / "cdr_extraction_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_outputs(split_frames: dict[str, pd.DataFrame]) -> None:
    """Save annotated splits, failure table, and markdown summary."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, dataframe in split_frames.items():
        dataframe.to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False)

    failures = build_failures_table(split_frames)
    failures.to_csv(OUTPUT_DIR / "cdr_extraction_failures.csv", index=False)
    write_summary(split_frames, failures)

    print()
    print("Standard CDR extraction finished.")
    print(f"Annotated CSVs: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Failure rows copied to: {(OUTPUT_DIR / 'cdr_extraction_failures.csv').relative_to(PROJECT_ROOT)}")
    print(f"Summary: {(OUTPUT_DIR / 'cdr_extraction_summary.md').relative_to(PROJECT_ROOT)}")
    print(f"Rows with any heavy/light extraction failure: {len(failures)}")


def main() -> None:
    """Run standard CDR annotation for train/val/test splits."""

    Chain = load_abnumber_chain()
    print("Using standard AbNumber IMGT CDR extraction.")
    print(f"Input directory: {INPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print()

    split_frames = {}
    for split_name in SPLITS:
        split_frames[split_name] = annotate_split(split_name, load_split(split_name), Chain)
    write_outputs(split_frames)


if __name__ == "__main__":
    main()
