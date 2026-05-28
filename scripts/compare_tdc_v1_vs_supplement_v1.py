"""Compare original TDC v1 against TDC plus SAbDab supplement v1.

中文人话说明：
这个脚本只比较两个 dataset，不训练模型。

我们想先回答：
1. supplement v1 比原 TDC v1 多了多少样本？
2. train/val/test 的 target 分布有没有明显改变？
3. 两个版本的 antigen-group split 是否仍然健康？
4. 因为 supplement v1 重新 split 了 full dataset，原 TDC 样本有多少换了 split？

最后一点很重要：
如果同一个模型在两个 dataset 上比较，split 变化本身也会影响结果。
所以 report 会把 split movement 写出来，避免把“数据增加效果”和“测试集换了效果”混在一起。
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TDC_V1_SPLIT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
SUPPLEMENT_V1_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "tdc_plus_sabdab_supplement_v1"
    / "antigen_group_split"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_plus_sabdab_supplement_v1"
JSON_REPORT_PATH = OUTPUT_DIR / "compare_tdc_v1_vs_supplement_v1.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "compare_tdc_v1_vs_supplement_v1.md"

SPLITS = ["train", "val", "test"]
SEQUENCE_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
TARGET_COLUMN = "neg_log10_affinity"


def load_dataset(name: str, split_dir: Path) -> pd.DataFrame:
    """Read one dataset version and attach its split/version labels."""

    frames = []
    for split_name in SPLITS:
        path = split_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}")
        dataframe = pd.read_csv(path)
        dataframe["split"] = split_name
        dataframe["dataset_version"] = name
        frames.append(dataframe)

    dataset = pd.concat(frames, ignore_index=True)
    required_columns = {"sample_id", "source", *SEQUENCE_COLUMNS, TARGET_COLUMN}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {sorted(missing_columns)}")
    return dataset


def numeric_summary(series: pd.Series) -> dict:
    """Return target min/max/mean/std for a report."""

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


def sequence_length_summary(dataframe: pd.DataFrame, column_name: str) -> dict:
    """Return sequence length summary for one sequence column."""

    lengths = dataframe[column_name].fillna("").astype(str).str.len()
    if len(lengths) == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(len(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
    }


def split_summary(dataframe: pd.DataFrame) -> dict:
    """Summarize rows, sources, target, and sequence lengths per split."""

    summary = {}
    for split_name in SPLITS:
        split_frame = dataframe[dataframe["split"] == split_name]
        summary[split_name] = {
            "rows": int(len(split_frame)),
            "source_counts": {
                str(source): int(count) for source, count in split_frame["source"].value_counts().items()
            },
            "target": numeric_summary(split_frame[TARGET_COLUMN]),
            "unique_antigen_sequences": int(split_frame["antigen_sequence"].nunique()),
            "sequence_lengths": {
                column_name: sequence_length_summary(split_frame, column_name)
                for column_name in SEQUENCE_COLUMNS
            },
        }
    return summary


def key_set(dataframe: pd.DataFrame, columns: list[str]) -> set[str]:
    """Create set keys for overlap checks."""

    if len(dataframe) == 0:
        return set()
    return set(dataframe[columns].astype(str).agg("||".join, axis=1))


def pairwise_overlap(dataframe: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Check train/val/test overlap inside one dataset."""

    split_frames = {split_name: dataframe[dataframe["split"] == split_name] for split_name in SPLITS}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        f"{left}_vs_{right}": int(len(key_set(split_frames[left], columns) & key_set(split_frames[right], columns)))
        for left, right in pairs
    }


def dataset_overlap_health(dataframe: pd.DataFrame) -> dict:
    """Report leakage guard checks for one dataset version."""

    return {
        "antigen_sequence_overlap": pairwise_overlap(dataframe, ["antigen_sequence"]),
        "triplet_overlap": pairwise_overlap(dataframe, SEQUENCE_COLUMNS),
    }


def triplet_keys(dataframe: pd.DataFrame) -> pd.Series:
    """Build exact sequence triplet identity used across dataset versions."""

    return dataframe[SEQUENCE_COLUMNS].astype(str).agg("||".join, axis=1)


def compare_triplet_membership(tdc: pd.DataFrame, supplement: pd.DataFrame) -> dict:
    """Measure how much supplement v1 shares with original TDC v1."""

    tdc_keys = set(triplet_keys(tdc))
    supplement_keys = set(triplet_keys(supplement))
    added = supplement_keys - tdc_keys
    removed = tdc_keys - supplement_keys
    return {
        "tdc_unique_triplets": int(len(tdc_keys)),
        "supplement_v1_unique_triplets": int(len(supplement_keys)),
        "triplets_shared_between_versions": int(len(tdc_keys & supplement_keys)),
        "triplets_added_in_supplement_v1": int(len(added)),
        "triplets_missing_from_supplement_v1": int(len(removed)),
    }


def split_movement_for_shared_tdc_rows(tdc: pd.DataFrame, supplement: pd.DataFrame) -> dict:
    """Track original TDC samples that moved split after merged re-split."""

    old_tdc = tdc[["sample_id", "split"]].drop_duplicates("sample_id").rename(columns={"split": "tdc_v1_split"})
    new_tdc = supplement[supplement["source"] == "TDC_Protein_SAbDab"][["sample_id", "split"]]
    new_tdc = new_tdc.drop_duplicates("sample_id").rename(columns={"split": "supplement_v1_split"})
    shared = old_tdc.merge(new_tdc, on="sample_id", how="inner")
    shared["movement"] = shared["tdc_v1_split"] + "->" + shared["supplement_v1_split"]

    movement_counts = {str(key): int(value) for key, value in shared["movement"].value_counts().items()}
    same_split_rows = int((shared["tdc_v1_split"] == shared["supplement_v1_split"]).sum())
    moved_rows = int(len(shared) - same_split_rows)
    return {
        "shared_tdc_sample_ids": int(len(shared)),
        "same_split_rows": same_split_rows,
        "moved_split_rows": moved_rows,
        "movement_counts": movement_counts,
    }


def supplement_rows_report(dataframe: pd.DataFrame) -> dict:
    """Summarize rows whose source is SAbDab supplement."""

    supplement_rows = dataframe[dataframe["source"] == "SAbDab_Supplement"]
    by_split = {
        split_name: int((supplement_rows["split"] == split_name).sum())
        for split_name in SPLITS
    }
    return {
        "rows": int(len(supplement_rows)),
        "unique_antigen_sequences": int(supplement_rows["antigen_sequence"].nunique()),
        "split_counts": by_split,
        "target": numeric_summary(supplement_rows[TARGET_COLUMN]),
    }


def split_target_delta(tdc_summary: dict, supplement_summary: dict) -> dict:
    """Compare target mean/std shifts between dataset versions."""

    deltas = {}
    for split_name in SPLITS:
        old_target = tdc_summary[split_name]["target"]
        new_target = supplement_summary[split_name]["target"]
        deltas[split_name] = {
            "mean_delta_supplement_minus_tdc": float(new_target["mean"] - old_target["mean"]),
            "std_delta_supplement_minus_tdc": float(new_target["std"] - old_target["std"]),
            "row_delta_supplement_minus_tdc": int(
                supplement_summary[split_name]["rows"] - tdc_summary[split_name]["rows"]
            ),
        }
    return deltas


def worth_training_decision(report: dict) -> dict:
    """Give cautious data-only recommendation before any model run."""

    added_rows = report["dataset_sizes"]["supplement_v1_rows"] - report["dataset_sizes"]["tdc_v1_rows"]
    movement = report["tdc_sample_split_movement"]["moved_split_rows"]
    split_health_ok = all(
        count == 0
        for version_health in report["overlap_health"].values()
        for check in version_health.values()
        for count in check.values()
    )

    if not split_health_ok:
        return {
            "worth_training": False,
            "reason": "A split overlap guard failed. Fix dataset split before training.",
        }
    if added_rows <= 0:
        return {
            "worth_training": False,
            "reason": "Supplement v1 does not add final rows over TDC v1.",
        }

    return {
        "worth_training": True,
        "reason": (
            "Supplement v1 adds audited rows and keeps antigen/triplet overlap checks at zero. "
            f"However, {movement} original TDC sample rows moved split after re-splitting, so "
            "later model comparison should be interpreted as dataset+split-version comparison."
        ),
    }


def build_report(tdc: pd.DataFrame, supplement: pd.DataFrame) -> dict:
    """Build structured comparison report."""

    tdc_split_summary = split_summary(tdc)
    supplement_split_summary = split_summary(supplement)
    report = {
        "datasets": {
            "tdc_v1": str(TDC_V1_SPLIT_DIR.relative_to(PROJECT_ROOT)),
            "tdc_plus_sabdab_supplement_v1": str(SUPPLEMENT_V1_SPLIT_DIR.relative_to(PROJECT_ROOT)),
        },
        "dataset_sizes": {
            "tdc_v1_rows": int(len(tdc)),
            "supplement_v1_rows": int(len(supplement)),
            "row_delta_supplement_minus_tdc": int(len(supplement) - len(tdc)),
        },
        "overall_target": {
            "tdc_v1": numeric_summary(tdc[TARGET_COLUMN]),
            "supplement_v1": numeric_summary(supplement[TARGET_COLUMN]),
        },
        "split_summary": {
            "tdc_v1": tdc_split_summary,
            "supplement_v1": supplement_split_summary,
        },
        "split_target_delta": split_target_delta(tdc_split_summary, supplement_split_summary),
        "overlap_health": {
            "tdc_v1": dataset_overlap_health(tdc),
            "supplement_v1": dataset_overlap_health(supplement),
        },
        "triplet_membership_between_versions": compare_triplet_membership(tdc, supplement),
        "supplement_rows_in_supplement_v1": supplement_rows_report(supplement),
        "tdc_sample_split_movement": split_movement_for_shared_tdc_rows(tdc, supplement),
    }
    report["decision"] = worth_training_decision(report)
    return report


def write_markdown(report: dict) -> None:
    """Write beginner-friendly comparison Markdown."""

    split_table_rows = []
    for split_name in SPLITS:
        old_summary = report["split_summary"]["tdc_v1"][split_name]
        new_summary = report["split_summary"]["supplement_v1"][split_name]
        split_table_rows.append(
            f"| {split_name} | {old_summary['rows']} | {old_summary['target']['mean']:.4f} | "
            f"{old_summary['target']['std']:.4f} | {new_summary['rows']} | "
            f"{new_summary['target']['mean']:.4f} | {new_summary['target']['std']:.4f} |"
        )

    lines = [
        "# TDC v1 vs SAbDab Supplement v1 Dataset Comparison",
        "",
        "## Scope",
        "",
        f"- Original TDC v1: `{report['datasets']['tdc_v1']}`",
        f"- Supplement v1: `{report['datasets']['tdc_plus_sabdab_supplement_v1']}`",
        "- This report compares data only. It does not train a model.",
        "",
        "## Size And Overall Target",
        "",
        f"- TDC v1 rows: {report['dataset_sizes']['tdc_v1_rows']}",
        f"- Supplement v1 rows: {report['dataset_sizes']['supplement_v1_rows']}",
        f"- Row delta: {report['dataset_sizes']['row_delta_supplement_minus_tdc']}",
        f"- TDC v1 overall target: `{report['overall_target']['tdc_v1']}`",
        f"- Supplement v1 overall target: `{report['overall_target']['supplement_v1']}`",
        "",
        "## Split And Target Distribution",
        "",
        "| split | TDC rows | TDC target mean | TDC target std | Supplement rows | Supplement target mean | Supplement target std |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *split_table_rows,
        "",
        f"- Target deltas by split: `{report['split_target_delta']}`",
        f"- Supplement row summary: `{report['supplement_rows_in_supplement_v1']}`",
        "",
        "## Split Health",
        "",
        f"- TDC v1 overlap health: `{report['overlap_health']['tdc_v1']}`",
        f"- Supplement v1 overlap health: `{report['overlap_health']['supplement_v1']}`",
        "",
        "## Version Membership",
        "",
        f"- Triplet membership between versions: `{report['triplet_membership_between_versions']}`",
        "",
        "## Split Movement Caution",
        "",
        "Supplement v1 is re-split after merging supplement rows. That is correct for antigen-group split safety, "
        "but it means later model metrics are not only affected by 7 added rows; some original TDC rows changed split.",
        "",
        f"- TDC sample split movement: `{report['tdc_sample_split_movement']}`",
        "",
        "## Decision",
        "",
        f"- Worth training later: `{report['decision']['worth_training']}`",
        f"- Reason: {report['decision']['reason']}",
        "",
    ]
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_summary(report: dict) -> None:
    """Print high-signal comparison result."""

    print("TDC v1 vs supplement v1 dataset comparison complete.")
    print(f"Rows: TDC v1 {report['dataset_sizes']['tdc_v1_rows']} -> supplement v1 {report['dataset_sizes']['supplement_v1_rows']}")
    print(f"Split target deltas: {report['split_target_delta']}")
    print(f"Supplement rows in supplement v1: {report['supplement_rows_in_supplement_v1']}")
    print(f"TDC v1 overlap health: {report['overlap_health']['tdc_v1']}")
    print(f"Supplement v1 overlap health: {report['overlap_health']['supplement_v1']}")
    print(f"TDC sample split movement: {report['tdc_sample_split_movement']}")
    print(f"Worth training later: {report['decision']['worth_training']}")
    print(f"Reason: {report['decision']['reason']}")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Run dataset comparison and write reports."""

    tdc = load_dataset("tdc_v1", TDC_V1_SPLIT_DIR)
    supplement = load_dataset("tdc_plus_sabdab_supplement_v1", SUPPLEMENT_V1_SPLIT_DIR)
    report = build_report(tdc, supplement)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_markdown(report)
    print_summary(report)


if __name__ == "__main__":
    main()
