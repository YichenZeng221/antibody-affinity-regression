"""Plan conservative antibody-only ANDD v2 antigen-group split.

:
 split , train.csv / val.csv / test.csv

 `expanded_affinity_antibody_v2_audited_flags.csv`,
 `keep_safe=True`  rows, antigen_sequence  train/val/test
:

- antigen group 
-  split  rows
- target/source 
-  current unified test antigen overlap

 split , planning report
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDITED_CSV = (
    ROOT
    / "data/processed_affinity/expanded_affinity_dataset_v2_candidates/"
    / "expanded_affinity_antibody_v2_audited_flags.csv"
)
OUTPUT_DIR = ROOT / "outputs/data_expansion/ANDD_antibody_v2_split_plan"
REPORT_PATH = OUTPUT_DIR / "split_plan_report.md"
JSON_PATH = OUTPUT_DIR / "split_plan_summary.json"

TARGET_COLUMN = "neg_log10_affinity_candidate"
SEED = 42
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def bool_series(series: pd.Series) -> pd.Series:
    """ CSV  bool  bool"""

    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def numeric_summary(values: pd.Series) -> dict:
    """"""

    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "std": None}
    return {
        "count": int(len(clean)),
        "min": float(clean.min()),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "max": float(clean.max()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
    }


def summarize_split(split_df: pd.DataFrame) -> dict:
    """ split  rowsantigen groupstarget/source """

    source_counts = split_df["source"].fillna("unknown").value_counts().to_dict()
    return {
        "rows": int(len(split_df)),
        "antigen_groups": int(split_df["antigen_sequence"].nunique()),
        "target": numeric_summary(split_df[TARGET_COLUMN]),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
    }


def greedy_antigen_group_split(df: pd.DataFrame) -> pd.DataFrame:
    """ antigen_sequence , 80/10/10 split

    :
    1.  antigen_sequence  rows  split
    2.  antigen group , row 
    3.  plan, split; target/source balance
    """

    groups = []
    for antigen, group in df.groupby("antigen_sequence"):
        groups.append(
            {
                "antigen_sequence": antigen,
                "rows": len(group),
                "target_mean": pd.to_numeric(group[TARGET_COLUMN], errors="coerce").mean(),
            }
        )

    random.seed(SEED)
    random.shuffle(groups)
    groups.sort(key=lambda item: item["rows"], reverse=True)

    total_rows = len(df)
    target_rows = {split: total_rows * ratio for split, ratio in RATIOS.items()}
    assigned_rows = {split: 0 for split in RATIOS}
    assignments = {}

    for group in groups:
        #  rows  split
        split = min(
            RATIOS.keys(),
            key=lambda name: (assigned_rows[name] + group["rows"] - target_rows[name], assigned_rows[name] / target_rows[name]),
        )

        #  min  split ;
        split = min(RATIOS.keys(), key=lambda name: assigned_rows[name] / target_rows[name])
        assignments[group["antigen_sequence"]] = split
        assigned_rows[split] += group["rows"]

    planned = df.copy()
    planned["planned_split"] = planned["antigen_sequence"].map(assignments)
    return planned


def markdown_table_from_source_counts(summary: dict) -> list[str]:
    """ source count  Markdown table"""

    all_sources = sorted({source for item in summary.values() for source in item["source_counts"]})
    lines = ["| Split | Rows | Antigen groups | " + " | ".join(f"`{source}`" for source in all_sources) + " |"]
    lines.append("|---|---:|---:|" + "|".join(["---:"] * len(all_sources)) + "|")
    for split in ["train", "val", "test"]:
        item = summary[split]
        counts = [str(item["source_counts"].get(source, 0)) for source in all_sources]
        lines.append(f"| {split} | {item['rows']} | {item['antigen_groups']} | " + " | ".join(counts) + " |")
    return lines


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(AUDITED_CSV)
    df["keep_safe_bool"] = bool_series(df["keep_safe"])
    keep = df[df["keep_safe_bool"]].copy()

    planned = greedy_antigen_group_split(keep)

    split_summary = {}
    for split in ["train", "val", "test"]:
        split_summary[split] = summarize_split(planned[planned["planned_split"] == split])

    all_summary = summarize_split(planned)

    # overlap sanity checkskeep_safe  current unified antigen overlap,
    unified_test_overlap_rows = int(bool_series(planned.get("overlap_current_test_antigen", pd.Series(False, index=planned.index))).sum())
    unified_antigen_overlap_rows = int(bool_series(planned.get("flag_antigen_overlap", pd.Series(False, index=planned.index))).sum())

    split_antigen_sets = {
        split: set(planned.loc[planned["planned_split"] == split, "antigen_sequence"])
        for split in ["train", "val", "test"]
    }
    antigen_overlaps = {
        "train_val": len(split_antigen_sets["train"] & split_antigen_sets["val"]),
        "train_test": len(split_antigen_sets["train"] & split_antigen_sets["test"]),
        "val_test": len(split_antigen_sets["val"] & split_antigen_sets["test"]),
    }

    target_lines = [
        "| Split | Rows | Target min | Target median | Target mean | Target max | Target std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["all", "train", "val", "test"]:
        item = all_summary if split == "all" else split_summary[split]
        target = item["target"]
        target_lines.append(
            f"| {split} | {item['rows']} | {target['min']:.3f} | {target['median']:.3f} | "
            f"{target['mean']:.3f} | {target['max']:.3f} | {target['std']:.3f} |"
        )

    group_sizes = keep.groupby("antigen_sequence").size().sort_values(ascending=False)
    top_group_lines = ["| Antigen group rank | Rows |", "|---:|---:|"]
    for rank, rows in enumerate(group_sizes.head(10), start=1):
        top_group_lines.append(f"| {rank} | {int(rows)} |")

    summary_json = {
        "input_rows": int(len(df)),
        "keep_safe_rows": int(len(keep)),
        "recommended_ratio": RATIOS,
        "seed": SEED,
        "antigen_group_count": int(keep["antigen_sequence"].nunique()),
        "split_summary": split_summary,
        "all_summary": all_summary,
        "antigen_group_overlap_check": antigen_overlaps,
        "current_unified_test_antigen_overlap_rows_in_keep_safe": unified_test_overlap_rows,
        "current_unified_any_antigen_overlap_rows_in_keep_safe": unified_antigen_overlap_rows,
        "top_antigen_group_sizes": [int(value) for value in group_sizes.head(10).tolist()],
    }
    JSON_PATH.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    report = [
        "# Conservative Antibody-only expanded_affinity_dataset_v2 Split Plan",
        "",
        "## Scope",
        "",
        "This is only a split planning report.",
        "",
        "- No model was trained.",
        "- No final `train.csv` / `val.csv` / `test.csv` files were created.",
        "- Existing `unified_no_high_risk` was not modified.",
        "- Only `keep_safe=True` ANDD antibody candidate rows were used.",
        "",
        "## 1. Recommended Split Strategy",
        "",
        "- Recommended ratio: `train/val/test = 80/10/10`.",
        "- Split unit: `antigen_sequence` group.",
        "- Reason: the model should not see the same antigen sequence in both training and testing, otherwise test performance can be inflated by antigen leakage.",
        "- Seed for planning simulation: `42`.",
        "",
        "## 2. Candidate Scope",
        "",
        f"- Audited antibody candidate rows: `{len(df)}`",
        f"- `keep_safe` rows used for planning: `{len(keep)}`",
        f"- Antigen group count among keep_safe rows: `{keep['antigen_sequence'].nunique()}`",
        "",
        "## 3. Planned Split Size Estimate",
        "",
        "| Split | Rows | Antigen groups | Target row ratio |",
        "|---|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        rows = split_summary[split]["rows"]
        report.append(
            f"| {split} | {rows} | {split_summary[split]['antigen_groups']} | {rows / len(keep):.3f} |"
        )

    report.extend(
        [
            "",
            "## 4. Antigen Group Balance",
            "",
            f"- Largest antigen group size: `{int(group_sizes.iloc[0]) if len(group_sizes) else 0}` rows",
            f"- Median antigen group size: `{float(group_sizes.median()) if len(group_sizes) else 0:.1f}` rows",
            "",
            "Top antigen group sizes:",
            "",
        ]
    )
    report.extend(top_group_lines)

    report.extend(
        [
            "",
            "Antigen overlap check under the simulated plan:",
            "",
            f"- train vs val antigen overlap: `{antigen_overlaps['train_val']}`",
            f"- train vs test antigen overlap: `{antigen_overlaps['train_test']}`",
            f"- val vs test antigen overlap: `{antigen_overlaps['val_test']}`",
            "",
            "## 5. Target Distribution Balance",
            "",
        ]
    )
    report.extend(target_lines)

    report.extend(
        [
            "",
            "Interpretation: this split is acceptable as a first plan if train/val/test target means and ranges are not dramatically separated. Because antigen groups can be uneven, exact 80/10/10 row counts are less important than zero antigen leakage.",
            "",
            "## 6. Source Distribution Balance",
            "",
        ]
    )
    report.extend(markdown_table_from_source_counts(split_summary))

    report.extend(
        [
            "",
            "## 7. Current Unified Test Antigen Overlap",
            "",
            f"- `keep_safe` rows overlapping current unified test antigen: `{unified_test_overlap_rows}`",
            f"- `keep_safe` rows overlapping any current unified antigen: `{unified_antigen_overlap_rows}`",
            "",
            "Recommendation: for a clean ANDD-only benchmark, exclude current unified test antigen overlap. In this conservative plan, `keep_safe` already excludes antigen overlap flags, so the overlap count should be zero.",
            "",
            "## 8. ANDD-only vs Merge With unified_no_high_risk",
            "",
            "Recommended first experiment: build an `ANDD antibody-only v2` benchmark separately.",
            "",
            "Reason: ANDD is much larger than the current unified dataset and has different provenance/label distribution. If we immediately merge them, performance changes will be hard to interpret. A separate ANDD-only benchmark tells us whether the model learns better with more antibody-only data.",
            "",
            "After that, a second experiment can merge current `unified_no_high_risk` + conservative ANDD antibody rows, but it should be labeled as a different dataset-version comparison.",
            "",
            "## 9. Next Step Before Creating Final Split",
            "",
            "Before writing final `train.csv` / `val.csv` / `test.csv`, do these checks:",
            "",
            "1. Manually review extreme target edges even inside `keep_safe`.",
            "2. Decide whether to cap antigen length or keep long antigens.",
            "3. Re-run standard AbNumber + IMGT CDR extraction for antibody rows.",
            "4. Create the final antigen-group split only after approving this plan.",
            "",
            "## 10. Output",
            "",
            f"- Planning report: `{REPORT_PATH}`",
            f"- Machine-readable summary: `{JSON_PATH}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"keep_safe rows: {len(keep)}")
    print(f"antigen groups: {keep['antigen_sequence'].nunique()}")
    for split in ["train", "val", "test"]:
        print(f"{split}: rows={split_summary[split]['rows']}, antigen_groups={split_summary[split]['antigen_groups']}")
    print(f"antigen overlap check: {antigen_overlaps}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
