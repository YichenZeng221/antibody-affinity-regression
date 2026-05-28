"""Build TDC v1 plus ready SAbDab supplement affinity dataset.

:
 dataset ,

:
1. TDC v1 antigen-group split  train/val/test
2.  audit  supplement_ready_candidates.csv

 supplement append  train/test?
-  TDC v1 split  antigen_sequence group 
- , split, antigen  split,
   test 
-  full data , antigen-group split

:
    data/processed_affinity/tdc_plus_sabdab_supplement_v1/
 tdc_v1 
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TDC_SPLIT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
SUPPLEMENT_READY_PATH = (
    PROJECT_ROOT / "data" / "processed_affinity" / "sabdab_supplement" / "supplement_ready_candidates.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_plus_sabdab_supplement_v1"
SPLIT_OUTPUT_DIR = OUTPUT_ROOT / "antigen_group_split"
JSON_REPORT_PATH = OUTPUT_ROOT / "processing_report.json"
MARKDOWN_REPORT_PATH = OUTPUT_ROOT / "processing_report.md"
CONFLICT_DUPLICATES_PATH = OUTPUT_ROOT / "conflict_duplicates.csv"

SEED = 42
SPLITS = ["train", "val", "test"]
TRIPLET_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
OUTPUT_COLUMNS = [
    "sample_id",
    "source",
    "original_dataset_split",
    "supplement_candidate_id",
    "antibody_id",
    "antigen_id",
    "heavy_sequence",
    "light_sequence",
    "antigen_sequence",
    "affinity",
    "neg_log10_affinity",
]


def numeric_summary(series: pd.Series) -> dict:
    """Return JSON-friendly target stats."""

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


def load_tdc_rows() -> pd.DataFrame:
    """Load all TDC v1 rows and keep their old split only as provenance."""

    frames = []
    for split_name in SPLITS:
        path = TDC_SPLIT_DIR / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}. Build TDC v1 first.")
        dataframe = pd.read_csv(path)
        dataframe["source"] = "TDC_Protein_SAbDab"
        dataframe["original_dataset_split"] = f"tdc_v1_{split_name}"
        dataframe["supplement_candidate_id"] = ""
        frames.append(dataframe)

    tdc = pd.concat(frames, ignore_index=True)
    required_columns = {
        "sample_id",
        "source",
        "antibody_id",
        "antigen_id",
        *TRIPLET_COLUMNS,
        "affinity",
        "neg_log10_affinity",
    }
    missing_columns = required_columns - set(tdc.columns)
    if missing_columns:
        raise ValueError(f"TDC v1 data is missing columns: {sorted(missing_columns)}")
    return tdc[OUTPUT_COLUMNS].copy()


def load_supplement_rows() -> pd.DataFrame:
    """Convert ready SAbDab supplement rows into the new unified schema."""

    if not SUPPLEMENT_READY_PATH.exists():
        raise FileNotFoundError(f"Cannot find {SUPPLEMENT_READY_PATH}. Run supplement audit first.")

    ready = pd.read_csv(SUPPLEMENT_READY_PATH)
    required_columns = {
        "candidate_row_id",
        "pdb",
        "antigen_name",
        *TRIPLET_COLUMNS,
        "affinity_numeric",
        "computed_neg_log10_affinity",
    }
    missing_columns = required_columns - set(ready.columns)
    if missing_columns:
        raise ValueError(f"Supplement ready candidates missing columns: {sorted(missing_columns)}")

    supplement = pd.DataFrame()
    supplement["sample_id"] = [f"SABDAB_SUPP_{index + 1:06d}" for index in range(len(ready))]
    supplement["source"] = "SAbDab_Supplement"
    supplement["original_dataset_split"] = ready.get("sequence_source_split", "").map(
        lambda split_name: f"sabdab_sequence_only_{split_name}" if str(split_name).strip() else ""
    )
    supplement["supplement_candidate_id"] = ready["candidate_row_id"].astype(str)
    # TDC antibody_id is PDB-like. For supplement rows, pdb is the matching structure ID.
    supplement["antibody_id"] = ready["pdb"].astype(str).str.lower()
    supplement["antigen_id"] = ready["antigen_name"].astype(str)
    for column_name in TRIPLET_COLUMNS:
        supplement[column_name] = ready[column_name].astype(str)
    supplement["affinity"] = pd.to_numeric(ready["affinity_numeric"], errors="coerce")
    supplement["neg_log10_affinity"] = pd.to_numeric(
        ready["computed_neg_log10_affinity"],
        errors="coerce",
    )
    return supplement[OUTPUT_COLUMNS].copy()


def conflict_target(group: pd.DataFrame) -> bool:
    """Return True when duplicate triplet rows disagree on target."""

    targets = pd.to_numeric(group["neg_log10_affinity"], errors="coerce")
    if targets.isna().any():
        return True
    return bool((targets.max() - targets.min()) > 1e-8)


def deduplicate_triplets(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Deduplicate exact triplets and separate target conflicts.

    :
     heavy/light/antigen sequences
    ,;
     target,,
     conflict report 
    """

    kept_groups = []
    conflict_groups = []
    duplicate_rows_removed = 0

    for _, group in dataframe.groupby(TRIPLET_COLUMNS, sort=False, dropna=False):
        if len(group) == 1:
            kept_groups.append(group.iloc[[0]])
            continue

        if conflict_target(group):
            conflicting = group.copy()
            conflicting["conflict_reason"] = "duplicate_triplet_conflicting_neg_log10_affinity"
            conflict_groups.append(conflicting)
            continue

        kept_groups.append(group.iloc[[0]])
        duplicate_rows_removed += len(group) - 1

    deduplicated = pd.concat(kept_groups, ignore_index=True) if kept_groups else dataframe.iloc[0:0].copy()
    conflicts = pd.concat(conflict_groups, ignore_index=True) if conflict_groups else dataframe.iloc[0:0].copy()
    if "conflict_reason" not in conflicts.columns:
        conflicts["conflict_reason"] = pd.Series(dtype=str)
    return deduplicated, conflicts, int(duplicate_rows_removed)


def antigen_group_split(dataframe: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Split by antigen_sequence so each antigen group lives in one split only."""

    antigen_sizes = dataframe.groupby("antigen_sequence").size().reset_index(name="count")
    antigen_sizes = antigen_sizes.sample(frac=1, random_state=seed).reset_index(drop=True)

    total_rows = len(dataframe)
    target_sizes = {
        "train": total_rows * 0.8,
        "val": total_rows * 0.1,
        "test": total_rows * 0.1,
    }
    split_to_antigens = {split_name: [] for split_name in SPLITS}
    split_sizes = {split_name: 0 for split_name in SPLITS}

    #  antigen group ,
    # greedy  group  split
    for _, row in antigen_sizes.iterrows():
        deficits = {
            split_name: target_sizes[split_name] - split_sizes[split_name]
            for split_name in SPLITS
        }
        split_name = max(deficits, key=deficits.get)
        split_to_antigens[split_name].append(row["antigen_sequence"])
        split_sizes[split_name] += int(row["count"])

    return {
        split_name: dataframe[dataframe["antigen_sequence"].isin(antigens)].copy()
        for split_name, antigens in split_to_antigens.items()
    }


def key_set(dataframe: pd.DataFrame, columns: list[str]) -> set[str]:
    """Build comparison keys from one or more columns."""

    if len(dataframe) == 0:
        return set()
    return set(dataframe[columns].astype(str).agg("||".join, axis=1))


def pairwise_overlap_report(splits: dict[str, pd.DataFrame], columns: list[str]) -> dict[str, int]:
    """Count overlap among train/val/test for selected key columns."""

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        f"{left}_vs_{right}": int(len(key_set(splits[left], columns) & key_set(splits[right], columns)))
        for left, right in pairs
    }


def target_distribution_by_split(splits: dict[str, pd.DataFrame]) -> dict:
    """Summarize target values after the new split."""

    return {split_name: numeric_summary(frame["neg_log10_affinity"]) for split_name, frame in splits.items()}


def source_counts_by_split(splits: dict[str, pd.DataFrame]) -> dict:
    """Report how many TDC/supplement rows land in each new split."""

    return {
        split_name: {str(source): int(count) for source, count in frame["source"].value_counts().items()}
        for split_name, frame in splits.items()
    }


def supplement_split_assignments(splits: dict[str, pd.DataFrame]) -> list[dict]:
    """Record where supplement rows ended up after re-splitting."""

    assignments = []
    for split_name, frame in splits.items():
        supplement = frame[frame["source"] == "SAbDab_Supplement"]
        for _, row in supplement.iterrows():
            assignments.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "supplement_candidate_id": str(row["supplement_candidate_id"]),
                    "antibody_id": str(row["antibody_id"]),
                    "antigen_id": str(row["antigen_id"]),
                    "split": split_name,
                }
            )
    return assignments


def build_report(
    tdc_rows: pd.DataFrame,
    supplement_rows: pd.DataFrame,
    merged_rows: pd.DataFrame,
    final_rows: pd.DataFrame,
    conflicts: pd.DataFrame,
    duplicate_rows_removed: int,
    splits: dict[str, pd.DataFrame],
) -> dict:
    """Build JSON report for the new dataset version."""

    report = {
        "dataset_version": "tdc_plus_sabdab_supplement_v1",
        "seed": SEED,
        "inputs": {
            "tdc_v1_split_dir": str(TDC_SPLIT_DIR.relative_to(PROJECT_ROOT)),
            "supplement_ready_candidates": str(SUPPLEMENT_READY_PATH.relative_to(PROJECT_ROOT)),
        },
        "tdc_original_rows": int(len(tdc_rows)),
        "supplement_input_rows": int(len(supplement_rows)),
        "merged_rows_before_dedup": int(len(merged_rows)),
        "duplicate_triplet_rows_removed": int(duplicate_rows_removed),
        "conflict_duplicate_rows": int(len(conflicts)),
        "final_rows": int(len(final_rows)),
        "split_sizes": {split_name: int(len(frame)) for split_name, frame in splits.items()},
        "source_counts_by_split": source_counts_by_split(splits),
        "target_distribution_by_split": target_distribution_by_split(splits),
        "antigen_sequence_overlap_check": pairwise_overlap_report(splits, ["antigen_sequence"]),
        "triplet_overlap_check": pairwise_overlap_report(splits, TRIPLET_COLUMNS),
        "supplement_split_assignments": supplement_split_assignments(splits),
        "worth_training_later": bool(len(supplement_rows) > 0 and len(conflicts) == 0),
        "notes": {
            "split_note": (
                "The merged dataset is re-split by antigen_sequence. Same antigen_sequence should not "
                "cross train/val/test."
            ),
            "training_note": (
                "This build step does not train. The dataset is worth a later controlled comparison "
                "against original TDC v1 because it adds audited non-overlapping supplement rows."
            ),
        },
    }
    return report


def write_csv_outputs(splits: dict[str, pd.DataFrame], conflicts: pd.DataFrame) -> None:
    """Write split CSVs and duplicate conflict CSV without touching old TDC v1."""

    SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split_name, dataframe in splits.items():
        dataframe[OUTPUT_COLUMNS].to_csv(SPLIT_OUTPUT_DIR / f"{split_name}.csv", index=False)
    conflicts.to_csv(CONFLICT_DUPLICATES_PATH, index=False)


def write_markdown(report: dict) -> None:
    """Write human-readable report beside JSON report."""

    lines = [
        "# TDC plus SAbDab Supplement v1 Processing Report",
        "",
        "## Inputs",
        "",
        f"- TDC v1 split dir: `{report['inputs']['tdc_v1_split_dir']}`",
        f"- Supplement ready candidates: `{report['inputs']['supplement_ready_candidates']}`",
        "",
        "## Build Summary",
        "",
        f"- TDC original rows: {report['tdc_original_rows']}",
        f"- Supplement input rows: {report['supplement_input_rows']}",
        f"- Merged rows before dedup: {report['merged_rows_before_dedup']}",
        f"- Duplicate triplet rows removed: {report['duplicate_triplet_rows_removed']}",
        f"- Conflict duplicate rows written to report: {report['conflict_duplicate_rows']}",
        f"- Final rows: {report['final_rows']}",
        "",
        "## New Split",
        "",
        f"- Split sizes: `{report['split_sizes']}`",
        f"- Source counts by split: `{report['source_counts_by_split']}`",
        f"- Target distribution by split: `{report['target_distribution_by_split']}`",
        "",
        "## Overlap Checks",
        "",
        f"- antigen_sequence overlap: `{report['antigen_sequence_overlap_check']}`",
        f"- heavy+light+antigen triplet overlap: `{report['triplet_overlap_check']}`",
        "",
        "## Supplement Split Assignments",
        "",
    ]
    if report["supplement_split_assignments"]:
        lines.extend(
            f"- `{row['supplement_candidate_id']}` / `{row['antibody_id']}` -> `{row['split']}`"
            for row in report["supplement_split_assignments"]
        )
    else:
        lines.append("- No supplement rows entered the final dataset.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Worth later training comparison: `{report['worth_training_later']}`",
            "- This should be compared against original TDC v1 under the same model/training settings.",
            "",
        ]
    )
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: dict) -> None:
    """Print short terminal summary for the build."""

    print("TDC plus SAbDab supplement v1 dataset build complete.")
    print(f"TDC original rows: {report['tdc_original_rows']}")
    print(f"Supplement input rows: {report['supplement_input_rows']}")
    print(f"Final rows: {report['final_rows']}")
    print(f"Split sizes: {report['split_sizes']}")
    print(f"Source counts by split: {report['source_counts_by_split']}")
    print(f"antigen_sequence overlap check: {report['antigen_sequence_overlap_check']}")
    print(f"triplet overlap check: {report['triplet_overlap_check']}")
    print(f"Conflict duplicate rows: {report['conflict_duplicate_rows']}")
    print(f"Worth later training comparison: {report['worth_training_later']}")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Conflict duplicates CSV: {CONFLICT_DUPLICATES_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Build the new merged dataset version."""

    tdc_rows = load_tdc_rows()
    supplement_rows = load_supplement_rows()
    merged_rows = pd.concat([tdc_rows, supplement_rows], ignore_index=True)
    final_rows, conflicts, duplicate_rows_removed = deduplicate_triplets(merged_rows)
    splits = antigen_group_split(final_rows)

    write_csv_outputs(splits, conflicts)
    report = build_report(
        tdc_rows,
        supplement_rows,
        merged_rows,
        final_rows,
        conflicts,
        duplicate_rows_removed,
        splits,
    )
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_markdown(report)
    print_summary(report)


if __name__ == "__main__":
    main()
