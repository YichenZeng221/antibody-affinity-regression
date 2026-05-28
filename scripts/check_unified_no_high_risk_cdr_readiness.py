"""Check whether unified_no_high_risk is ready for a CDR-aware baseline.

:

 train/val/test  CDR , readiness report,
 heavy/light sequence  index  CDR
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "unified_ablation_datasets"
    / "unified_no_high_risk"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cdr_aware" / "unified_no_high_risk"
REPORT_PATH = OUTPUT_DIR / "cdr_aware_report.md"
TDC_CDR_AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "tdc_v1"
    / "cdr_features"
    / "cdr_dataset_audit.md"
)
TDC_BACKEND_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "tdc_v1"
    / "cdr_features"
    / "cdr_backend_comparison.md"
)

SPLITS = ["train", "val", "test"]
CDR_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]
CDR_STATUS_COLUMNS = [
    "heavy_cdr_backend",
    "light_cdr_backend",
    "cdr_backend",
    "cdr_extract_status",
    "cdr_extract_error",
]


def inspect_splits() -> list[dict]:
    """Record whether each split contains required CDR columns."""

    results = []
    for split_name in SPLITS:
        path = DATASET_DIR / f"{split_name}.csv"
        dataframe = pd.read_csv(path, nrows=5)
        columns = list(dataframe.columns)
        results.append(
            {
                "split": split_name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "rows": int(len(pd.read_csv(path, usecols=["sample_id"]))),
                "cdr_columns_present": [column for column in CDR_COLUMNS if column in columns],
                "missing_cdr_columns": [column for column in CDR_COLUMNS if column not in columns],
                "cdr_status_columns_present": [
                    column for column in CDR_STATUS_COLUMNS if column in columns
                ],
                "all_columns": columns,
            }
        )
    return results


def write_report(split_checks: list[dict]) -> None:
    """Write CDR readiness report for the user-facing experiment decision."""

    total_rows = sum(check["rows"] for check in split_checks)
    ready = all(not check["missing_cdr_columns"] for check in split_checks)
    lines = [
        "# Unified No High Risk CDR-Aware Readiness Report",
        "",
        "## Decision",
        "",
        f"- CDR-aware training can start from the current CSVs: `{ready}`.",
        "- Current action: no CDR-aware model was trained and no new CDR config was created.",
        "- Reason: the current `unified_no_high_risk` split CSVs do not contain standard CDR sequence columns.",
        "- Per instruction, no fixed-index CDR slicing was used as a substitute.",
        "",
        "## Dataset Column Check",
        "",
        f"- Dataset dir: `{DATASET_DIR.relative_to(PROJECT_ROOT)}`",
        f"- Total rows across split CSVs: `{total_rows}`",
        f"- Required CDR columns for the requested input mode: `{CDR_COLUMNS}`",
        "",
        "| split | rows | present CDR columns | missing CDR columns | CDR backend/status columns present |",
        "|---|---:|---|---|---|",
    ]
    for check in split_checks:
        lines.append(
            f"| `{check['split']}` | {check['rows']} | `{check['cdr_columns_present']}` | "
            f"`{check['missing_cdr_columns']}` | `{check['cdr_status_columns_present']}` |"
        )
    lines.extend(
        [
            "",
            "## Existing CDR Work In This Repo",
            "",
            f"- Existing TDC v1 CDR audit: `{TDC_CDR_AUDIT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- Existing backend comparison: `{TDC_BACKEND_PATH.relative_to(PROJECT_ROOT)}`",
            "- That earlier TDC v1 artifact is not enough for this baseline:",
            "  it covers the TDC v1 dataset version rather than the 605-row unified_no_high_risk split,",
            "  and its current CDR backend audit marks the available output as `imgt_index_heuristic`.",
            "- The audit already warns that heuristic CDR lengths are nearly fixed and are not trusted for a formal baseline.",
            "",
            "## Why Standard Extraction Is Needed",
            "",
            "- The requested CDR-aware input is `HCDR1 + HCDR2 + HCDR3 + LCDR1 + LCDR2 + LCDR3 + antigen_sequence`.",
            "- Heavy/light whole sequences contain framework and possible constant-region residues. "
            "A numbering tool should identify variable-domain CDR boundaries before training a focused input baseline.",
            "- A hard raw-index slice would silently bake an annotation error into the dataset and make the baseline hard to interpret.",
            "",
            "## Next Reproducible Plan",
            "",
            "1. Create a separate CDR extraction environment with standard antibody numbering support.",
            "   Preferred path: install AbNumber/ANARCI plus HMMER via Bioconda; the earlier pip-only attempt reached AbNumber but standard numbering failed because `hmmscan` was unavailable.",
            "2. Add a new extraction script for the exact `unified_no_high_risk` train/val/test CSVs.",
            "   The output should be a new dataset directory, not overwrite the current ablation dataset.",
            "3. Use a standard scheme first, recommended `IMGT`, and save provenance columns such as",
            "   `heavy_cdr_backend`, `light_cdr_backend`, `cdr_extract_status`, `cdr_extract_error`.",
            "   Chothia can be a later comparison if you want an alternative CDR definition.",
            "4. Save rows with failed heavy/light numbering separately or filter them explicitly before training.",
            "5. After standard CDR coverage is acceptable, create:",
            "   `src/affinity_cdr_dataset.py`, `src/affinity_cdr_model.py`,",
            "   `run_train_affinity_cdr.py`, and",
            "   `config_affinity_unified_no_high_risk_cdr_aware_lr3e-5_e10.yaml`.",
            "   Keep the original whole-sequence ESM2+LoRA pipeline untouched.",
            "6. Evaluate CDR-aware vs whole-sequence using the same metrics already requested:",
            "   MAE, RMSE, Spearman, prediction std, low/mid/high target-bin MAE, and regression-to-mean residual checks.",
            "",
            "## Planned CDR-Aware Input Mode",
            "",
            "- Name: `cdr_aware` or `cdr_antigen`.",
            "- Shared ESM2+LoRA encoder remains the clean first baseline.",
            "- Encode each CDR and antigen sequence without overwriting the whole-sequence model.",
            "- A simple first fusion is concat pooled embeddings for six CDRs plus antigen, then a scalar regression head.",
            "",
            "## Environment Note",
            "",
            "- AbNumber documentation exposes `Chain(..., scheme='imgt')` and CDR sequence accessors.",
            "- ANARCI provides standard antibody numbering schemes including IMGT and Chothia.",
            "- The repo's prior backend report already identifies Bioconda + HMMER as the practical next environment for standard extraction.",
        ]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Inspect CDR columns and save report."""

    split_checks = inspect_splits()
    write_report(split_checks)
    print("CDR-aware readiness check complete.")
    for check in split_checks:
        print(
            f"{check['split']}: rows={check['rows']}, "
            f"missing_cdr_columns={check['missing_cdr_columns']}"
        )
    print("No CDR-aware training/config was created because required standard CDR columns are absent.")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
