"""Build a stratified antigen-level split for ANDD antibody v2.

:
 split  antigen , val/test  target 
 `antigen_sequence` , antigen group
 target  low/mid/high , val/test  P5  P95
 split coverage  regression-to-the-mean

:
-  split ,
-  `expanded_affinity_antibody_v2/` 
- all-CDR baseline  IMGT CDR , split 
  `expanded_affinity_antibody_v2_cdr_annotated/`  candidate_id 
  AbNumber + IMGT , source-provided CDR
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2"
ANNOTATED_DIR = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2_cdr_annotated"
OUTPUT_DIR = ROOT / "data/processed_affinity/expanded_affinity_antibody_v2_stratified"
FIGURE_DIR = ROOT / "outputs/andd_antibody_v2_stratified"
TARGET_COLUMN = "neg_log10_affinity_candidate"
GROUP_COLUMN = "antigen_sequence"
ID_COLUMN = "candidate_id"
SPLITS = ["train", "val", "test"]
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
STRATA = ["low", "mid", "high"]
SEED = 42
CDR_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]
CDR_AUDIT_COLUMNS = [
    "heavy_cdr_backend",
    "light_cdr_backend",
    "heavy_cdr_status",
    "light_cdr_status",
    "heavy_cdr_error",
    "light_cdr_error",
]

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_full_rows() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """ split, CDR annotation  CDR """

    old_splits: dict[str, pd.DataFrame] = {}
    raw_frames: list[pd.DataFrame] = []
    annotated_frames: list[pd.DataFrame] = []
    for split in SPLITS:
        raw = pd.read_csv(INPUT_DIR / f"{split}.csv")
        raw["previous_split"] = split
        old_splits[split] = raw.copy()
        raw_frames.append(raw)

        annotated = pd.read_csv(ANNOTATED_DIR / f"{split}.csv")
        annotated_frames.append(annotated[[ID_COLUMN] + CDR_COLUMNS + CDR_AUDIT_COLUMNS])

    full = pd.concat(raw_frames, ignore_index=True)
    annotations = pd.concat(annotated_frames, ignore_index=True)
    if full[ID_COLUMN].duplicated().any() or annotations[ID_COLUMN].duplicated().any():
        raise ValueError("candidate_id must be unique before CDR annotation join.")

    #  ANDD CSV  CDR ; baseline  IMGT 
    full = full.drop(columns=[column for column in CDR_COLUMNS + CDR_AUDIT_COLUMNS if column in full.columns])
    full = full.merge(annotations, how="left", on=ID_COLUMN, validate="one_to_one")
    required_annotation_columns = CDR_COLUMNS + [
        "heavy_cdr_backend",
        "light_cdr_backend",
        "heavy_cdr_status",
        "light_cdr_status",
    ]
    if full[required_annotation_columns].isna().any().any():
        raise ValueError("Some rows did not match existing standard CDR annotation.")
    #  error , annotation
    full[["heavy_cdr_error", "light_cdr_error"]] = full[
        ["heavy_cdr_error", "light_cdr_error"]
    ].fillna("")

    full[TARGET_COLUMN] = pd.to_numeric(full[TARGET_COLUMN], errors="raise")
    return full, old_splits


def build_group_table(full: pd.DataFrame, p05: float, p95: float) -> pd.DataFrame:
    """ antigen , antigen target  low/mid/high """

    groups = (
        full.groupby(GROUP_COLUMN, as_index=False)
        .agg(
            rows=(ID_COLUMN, "size"),
            target_mean=(TARGET_COLUMN, "mean"),
            target_min=(TARGET_COLUMN, "min"),
            target_max=(TARGET_COLUMN, "max"),
        )
        .copy()
    )
    groups["has_p05_tail"] = groups["target_min"] <= p05
    groups["has_p95_tail"] = groups["target_max"] >= p95

    # rank  qcut , group 
    groups["target_stratum"] = pd.qcut(
        groups["target_mean"].rank(method="first"),
        q=3,
        labels=STRATA,
    ).astype(str)
    return groups


def choose_tail_anchor(pool: pd.DataFrame, split: str, direction: str, used: set[str]) -> str:
    """ tail antigen  val/test, P5/P95"""

    available = pool[~pool[GROUP_COLUMN].isin(used)].copy()
    if available.empty:
        raise ValueError(f"Not enough antigen groups to anchor {direction} tail for {split}.")
    #  group, anchor  8:1:1 ; tail
    ascending = direction == "low"
    available = available.sort_values(
        ["rows", "target_mean", GROUP_COLUMN],
        ascending=[True, ascending, True],
    )
    selected = str(available.iloc[0][GROUP_COLUMN])
    used.add(selected)
    return selected


def assign_stratified_groups(groups: pd.DataFrame) -> dict[str, str]:
    """ mean-target stratum  group  8:1:1 """

    assignments: dict[str, str] = {}
    used: set[str] = set()

    low_tail = groups[groups["has_p05_tail"]]
    high_tail = groups[groups["has_p95_tail"]]
    for split in ["val", "test"]:
        assignments[choose_tail_anchor(low_tail, split, "low", used)] = split
        assignments[choose_tail_anchor(high_tail, split, "high", used)] = split

    rng = random.Random(SEED)
    for stratum in STRATA:
        stratum_groups = groups[groups["target_stratum"] == stratum].copy()
        desired = {
            split: float(stratum_groups["rows"].sum()) * ratio
            for split, ratio in RATIOS.items()
        }
        assigned_rows = {split: 0 for split in SPLITS}

        #  tail groups  stratum  row budget
        for _, row in stratum_groups.iterrows():
            group_key = str(row[GROUP_COLUMN])
            if group_key in assignments:
                assigned_rows[assignments[group_key]] += int(row["rows"])

        remaining = stratum_groups[~stratum_groups[GROUP_COLUMN].isin(assignments)].to_dict("records")
        rng.shuffle(remaining)
        remaining.sort(key=lambda row: int(row["rows"]), reverse=True)
        for row in remaining:
            #  row  split  antigen group
            split = max(
                SPLITS,
                key=lambda name: (desired[name] - assigned_rows[name], -assigned_rows[name]),
            )
            group_key = str(row[GROUP_COLUMN])
            assignments[group_key] = split
            assigned_rows[split] += int(row["rows"])

    if len(assignments) != len(groups):
        raise ValueError("Not every antigen group received a split assignment.")
    return assignments


def numeric_summary(values: pd.Series) -> dict:
    """ report  target distribution , P5/P95"""

    target = pd.to_numeric(values, errors="raise")
    return {
        "count": int(len(target)),
        "min": float(target.min()),
        "p05": float(target.quantile(0.05)),
        "mean": float(target.mean()),
        "std": float(target.std()),
        "p95": float(target.quantile(0.95)),
        "max": float(target.max()),
    }


def split_summary(split_frames: dict[str, pd.DataFrame], global_p05: float, global_p95: float) -> dict:
    """ split targetstratum  tail coverage"""

    result: dict[str, dict] = {}
    for split, frame in split_frames.items():
        stratum_counts = (
            {
                str(key): int(value)
                for key, value in frame.drop_duplicates(GROUP_COLUMN)["target_stratum"].value_counts().to_dict().items()
            }
            if "target_stratum" in frame.columns
            else {}
        )
        result[split] = {
            "rows": int(len(frame)),
            "antigen_groups": int(frame[GROUP_COLUMN].nunique()),
            "target": numeric_summary(frame[TARGET_COLUMN]),
            "group_target_stratum_counts": stratum_counts,
            "covers_global_p05": bool((frame[TARGET_COLUMN] <= global_p05).any()),
            "covers_global_p95": bool((frame[TARGET_COLUMN] >= global_p95).any()),
            "global_p05_tail_rows": int((frame[TARGET_COLUMN] <= global_p05).sum()),
            "global_p95_tail_rows": int((frame[TARGET_COLUMN] >= global_p95).sum()),
        }
    return result


def overlaps(split_frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """ antigen_sequence  split """

    keys = {
        split: set(frame[GROUP_COLUMN].astype(str))
        for split, frame in split_frames.items()
    }
    return {
        "train_val": len(keys["train"] & keys["val"]),
        "train_test": len(keys["train"] & keys["test"]),
        "val_test": len(keys["val"] & keys["test"]),
    }


def plot_old_vs_new(
    old_splits: dict[str, pd.DataFrame],
    new_splits: dict[str, pd.DataFrame],
    global_p05: float,
    global_p95: float,
) -> None:
    """ histogram bins / val-test tail coverage"""

    values = pd.concat(
        [frame[TARGET_COLUMN] for frame in list(old_splits.values()) + list(new_splits.values())],
        ignore_index=True,
    )
    bins = np.linspace(float(values.min()), float(values.max()), 21)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    colors = {"val": "#ff7f0e", "test": "#2ca02c"}
    for axis, title, frames in [
        (axes[0], "Original antigen-group split", old_splits),
        (axes[1], "Stratified antigen-group split", new_splits),
    ]:
        for split in ["val", "test"]:
            axis.hist(
                pd.to_numeric(frames[split][TARGET_COLUMN], errors="raise"),
                bins=bins,
                alpha=0.52,
                label=f"{split} (n={len(frames[split])})",
                color=colors[split],
            )
        axis.axvline(global_p05, color="#444444", linestyle="--", linewidth=1.2, label="global P5")
        axis.axvline(global_p95, color="#111111", linestyle=":", linewidth=1.4, label="global P95")
        axis.set_title(title)
        axis.set_xlabel("neg_log10_affinity_candidate")
        axis.set_ylabel("Sample count")
        axis.legend(fontsize=8)
    fig.suptitle("ANDD antibody v2: old vs stratified val/test target tail coverage")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "target_distribution_histogram.png", dpi=200)
    plt.close(fig)


def write_markdown(summary: dict) -> None:
    """ split """

    new = summary["new_split"]
    old = summary["old_split"]
    lines = [
        "# ANDD Antibody v2 Stratified Antigen-Level Split Summary",
        "",
        "## Design",
        "",
        "- Split unit: `antigen_sequence`; an antigen group is assigned to exactly one split.",
        "- Stratification: low/mid/high strata from antigen-level mean target quantiles.",
        "- Requested ratio: 80/10/10; exact row counts can shift slightly because antigen groups are indivisible.",
        "- Tail constraint: both validation and test must include at least one row at or below global P5 and at or above global P95.",
        "- CDR columns are reused from the existing standard `AbNumber + IMGT` annotated dataset for the same `candidate_id` rows.",
        "- No model was trained and the previous split was not overwritten.",
        "",
        f"- Global target P5: `{summary['global_target_thresholds']['p05']:.4f}`",
        f"- Global target P95: `{summary['global_target_thresholds']['p95']:.4f}`",
        "",
        "## New Split Target Summary",
        "",
        "| split | rows | antigen groups | min | P5 | mean | std | P95 | max | covers global P5 | covers global P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for split in SPLITS:
        item = new[split]
        target = item["target"]
        lines.append(
            f"| {split} | {item['rows']} | {item['antigen_groups']} | "
            f"{target['min']:.4f} | {target['p05']:.4f} | {target['mean']:.4f} | {target['std']:.4f} | "
            f"{target['p95']:.4f} | {target['max']:.4f} | {item['covers_global_p05']} | {item['covers_global_p95']} |"
        )
    lines.extend(
        [
            "",
            "## Original Vs New Val/Test Coverage",
            "",
            "| split version | split | rows | min | max | global P5 tail rows | global P95 tail rows |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for version, values in [("original", old), ("stratified", new)]:
        for split in ["val", "test"]:
            item = values[split]
            target = item["target"]
            lines.append(
                f"| {version} | {split} | {item['rows']} | {target['min']:.4f} | {target['max']:.4f} | "
                f"{item['global_p05_tail_rows']} | {item['global_p95_tail_rows']} |"
            )
    overlap = summary["antigen_overlap_check"]
    lines.extend(
        [
            "",
            "## Leakage Check",
            "",
            f"- train vs val antigen overlap: `{overlap['train_val']}`",
            f"- train vs test antigen overlap: `{overlap['train_test']}`",
            f"- val vs test antigen overlap: `{overlap['val_test']}`",
            "",
            "## Next Experiment",
            "",
            "Use exactly the existing all-CDR pooled architecture and MSE loss with the new split. "
            "Only after training manually should metrics be compared; this isolates split coverage as the changed factor.",
        ]
    )
    (OUTPUT_DIR / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full, old_split_frames = load_full_rows()
    global_p05 = float(full[TARGET_COLUMN].quantile(0.05))
    global_p95 = float(full[TARGET_COLUMN].quantile(0.95))
    groups = build_group_table(full, global_p05, global_p95)
    assignments = assign_stratified_groups(groups)

    enriched = full.merge(
        groups[[GROUP_COLUMN, "target_mean", "target_min", "target_max", "target_stratum"]],
        on=GROUP_COLUMN,
        how="left",
        validate="many_to_one",
    )
    enriched["split"] = enriched[GROUP_COLUMN].map(assignments)
    split_frames = {
        split: enriched[enriched["split"] == split].copy().reset_index(drop=True)
        for split in SPLITS
    }
    overlap_check = overlaps(split_frames)
    if any(overlap_check.values()):
        raise ValueError(f"Antigen leakage detected in stratified split: {overlap_check}")

    new_summary = split_summary(split_frames, global_p05, global_p95)
    if not all(new_summary[split]["covers_global_p05"] and new_summary[split]["covers_global_p95"] for split in ["val", "test"]):
        raise ValueError("Tail coverage constraint failed: val/test must cover both global P5 and P95.")

    old_summary = split_summary(old_split_frames, global_p05, global_p95)
    for split, frame in split_frames.items():
        frame.to_csv(OUTPUT_DIR / f"{split}.csv", index=False)

    summary = {
        "input_rows": int(len(full)),
        "antigen_group_count": int(groups[GROUP_COLUMN].nunique()),
        "seed": SEED,
        "requested_ratios": RATIOS,
        "stratification_definition": "tertiles of antigen-level mean target",
        "target_column": TARGET_COLUMN,
        "global_target_thresholds": {"p05": global_p05, "p95": global_p95},
        "old_split": old_summary,
        "new_split": new_summary,
        "antigen_overlap_check": overlap_check,
        "cdr_annotation_source": str(ANNOTATED_DIR.relative_to(ROOT)),
    }
    (OUTPUT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)
    plot_old_vs_new(old_split_frames, split_frames, global_p05, global_p95)

    print(f"Saved stratified ANDD split to {OUTPUT_DIR}")
    print(f"Global target P5/P95: {global_p05:.4f} / {global_p95:.4f}")
    for split in SPLITS:
        item = new_summary[split]
        print(
            f"{split}: rows={item['rows']}, antigen_groups={item['antigen_groups']}, "
            f"min={item['target']['min']:.4f}, max={item['target']['max']:.4f}, "
            f"P5_tail_rows={item['global_p05_tail_rows']}, P95_tail_rows={item['global_p95_tail_rows']}"
        )
    print(f"Antigen overlap check: {overlap_check}")
    print("No model was trained.")


if __name__ == "__main__":
    main()
