"""Create formal ANDD-only antibody v2 train/val/test split.

:
 split,

:
    expanded_affinity_antibody_v2_audited_flags.csv

:
    keep_safe == True

split :
     antigen_sequence , antigen_sequence 
    train / val / test  split, antigen leakage
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    ROOT
    / "data/processed_affinity/expanded_affinity_dataset_v2_candidates/"
    / "expanded_affinity_antibody_v2_audited_flags.csv"
)
OUTPUT_DIR = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2"
SPLIT_SUMMARY_MD = OUTPUT_DIR / "split_summary.md"
SPLIT_SUMMARY_JSON = OUTPUT_DIR / "split_summary.json"

SEED = 42
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
TARGET_COLUMN = "neg_log10_affinity_candidate"


def bool_series(series: pd.Series) -> pd.Series:
    """ CSV  True/False  bool"""

    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def numeric_summary(values: pd.Series) -> dict:
    """, target """

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


def group_split(df: pd.DataFrame) -> pd.DataFrame:
    """Greedy antigen_sequence group split.

    :
    -  antigen_sequence 
    -  group 
    -  row  split
    """

    groups = []
    for antigen_sequence, group in df.groupby("antigen_sequence"):
        groups.append({"antigen_sequence": antigen_sequence, "rows": len(group)})

    random.seed(SEED)
    random.shuffle(groups)
    groups.sort(key=lambda item: item["rows"], reverse=True)

    total_rows = len(df)
    target_rows = {name: total_rows * ratio for name, ratio in RATIOS.items()}
    assigned_rows = {name: 0 for name in RATIOS}
    assignments = {}

    for group in groups:
        split = min(RATIOS.keys(), key=lambda name: assigned_rows[name] / target_rows[name])
        assignments[group["antigen_sequence"]] = split
        assigned_rows[split] += group["rows"]

    output = df.copy()
    output["split"] = output["antigen_sequence"].map(assignments)
    return output


def summarize_split(df: pd.DataFrame) -> dict:
    """ split rowsantigen group counttarget/source """

    return {
        "rows": int(len(df)),
        "antigen_groups": int(df["antigen_sequence"].nunique()),
        "target": numeric_summary(df[TARGET_COLUMN]),
        "source_counts": {str(k): int(v) for k, v in df["source"].fillna("unknown").value_counts().to_dict().items()},
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(INPUT_CSV)
    keep = raw[bool_series(raw["keep_safe"])].copy().reset_index(drop=True)
    if keep.empty:
        raise ValueError("No keep_safe rows found. Run the antibody v2 audit first.")

    #  sample_idANDD candidate  candidate_id,
    keep["sample_id"] = keep["candidate_id"].astype(str)
    keep["affinity"] = pd.to_numeric(keep["affinity_kd_m"], errors="raise")

    planned = group_split(keep)

    split_frames = {
        split: planned[planned["split"] == split].copy().reset_index(drop=True)
        for split in ["train", "val", "test"]
    }

    antigen_sets = {
        split: set(frame["antigen_sequence"].astype(str))
        for split, frame in split_frames.items()
    }
    antigen_overlap_check = {
        "train_val": len(antigen_sets["train"] & antigen_sets["val"]),
        "train_test": len(antigen_sets["train"] & antigen_sets["test"]),
        "val_test": len(antigen_sets["val"] & antigen_sets["test"]),
    }
    if any(antigen_overlap_check.values()):
        raise ValueError(f"Antigen leakage detected: {antigen_overlap_check}")

    for split, frame in split_frames.items():
        frame.to_csv(OUTPUT_DIR / f"{split}.csv", index=False)

    summary = {
        "input_rows": int(len(raw)),
        "keep_safe_rows": int(len(keep)),
        "ratios": RATIOS,
        "seed": SEED,
        "antigen_group_count": int(keep["antigen_sequence"].nunique()),
        "split_summary": {split: summarize_split(frame) for split, frame in split_frames.items()},
        "antigen_overlap_check": antigen_overlap_check,
    }
    SPLIT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# ANDD-only Antibody v2 Split Summary",
        "",
        "- Dataset: `expanded_affinity_antibody_v2`",
        "- Source rows: `keep_safe=True` from ANDD antibody candidates.",
        "- Split unit: `antigen_sequence`.",
        "- No model was trained by this script.",
        "",
        "## Split Sizes",
        "",
        "| Split | Rows | Antigen groups | Target mean | Target median | Target min | Target max | Source counts |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in ["train", "val", "test"]:
        item = summary["split_summary"][split]
        target = item["target"]
        lines.append(
            f"| {split} | {item['rows']} | {item['antigen_groups']} | "
            f"{target['mean']:.4f} | {target['median']:.4f} | {target['min']:.4f} | {target['max']:.4f} | "
            f"`{item['source_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Leakage Check",
            "",
            f"- train vs val antigen overlap: `{antigen_overlap_check['train_val']}`",
            f"- train vs test antigen overlap: `{antigen_overlap_check['train_test']}`",
            f"- val vs test antigen overlap: `{antigen_overlap_check['val_test']}`",
            "",
            "## Files",
            "",
            f"- `{OUTPUT_DIR / 'train.csv'}`",
            f"- `{OUTPUT_DIR / 'val.csv'}`",
            f"- `{OUTPUT_DIR / 'test.csv'}`",
        ]
    )
    SPLIT_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved split to {OUTPUT_DIR}")
    for split, frame in split_frames.items():
        print(f"{split}: rows={len(frame)}, antigen_groups={frame['antigen_sequence'].nunique()}")
    print(f"antigen overlap check: {antigen_overlap_check}")


if __name__ == "__main__":
    main()
