"""Build a less-strict SAbDab affinity dataset from extracted sequence rows.

中文人话说明：
这个版本的目标不是“越脏越大”，而是：

1. 只从已经有 target 和 heavy/light/antigen sequence 的 ``sequence_only`` CSV 出发；
2. 在 strict metadata screen 的基础上，明确放宽几条规则；
3. 写出每条放宽规则增加了多少样本，以及它带来的风险。

为什么不直接从 summary.tsv 开始？
summary.tsv 里有 metadata，但不保证每行都能提取三条 sequence。
用户这一步要求所有样本必须有可计算 target 和可提取 sequence，
所以 v1 先复用已经完成 PDB sequence extraction 的 sequence_only 结果。

输出是一个新 dataset version，不覆盖 strict / clean_v2 / TDC 数据：

    data/processed_affinity/less_strict_sabdab_affinity_v1/
"""

from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sequence_only"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "less_strict_sabdab_affinity_v1"
SPLIT_DIR = OUTPUT_DIR / "pdb_level_split"
JSON_REPORT_PATH = OUTPUT_DIR / "count_report.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "count_report.md"
EXCLUDED_DUPLICATES_PATH = OUTPUT_DIR / "excluded_duplicate_triplets.csv"

TARGET_COLUMN = "neg_log10_affinity"
SEQUENCE_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
TRIPLET_COLUMNS = SEQUENCE_COLUMNS
SPLITS = ["train", "val", "test"]
SEED = 42


def is_missing(value: object) -> bool:
    """Treat blank cells and common NA spellings as missing."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def looks_suspicious_affinity_method(value: object) -> bool:
    """Flag method values that look like PMID-style pure numbers."""

    text = str(value).strip()
    return bool(text) and bool(re.fullmatch(r"\d+(?:\.\d+)?", text))


def load_sequence_only() -> pd.DataFrame:
    """Load the previously extracted SAbDab sequence-only rows.

    这些 rows 已经走过 PDB sequence extraction。我们仍然会再次做 target/sequence
    sanity check，避免以后有人替换输入 CSV 后 silent 引入坏行。
    """

    frames = []
    for split_name in SPLITS:
        path = INPUT_DIR / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}. Build sequence_only dataset first.")
        frame = pd.read_csv(path)
        frame["source_sequence_only_split"] = split_name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    required_columns = {
        "sample_id",
        "pdb",
        "Hchain",
        "Lchain",
        "antigen_type",
        "affinity",
        TARGET_COLUMN,
        *SEQUENCE_COLUMNS,
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"sequence_only data missing columns: {sorted(missing_columns)}")
    return data


def prepare_flags(data: pd.DataFrame) -> pd.DataFrame:
    """Add strict/relaxed filtering flags."""

    prepared = data.copy()
    prepared["affinity_numeric"] = pd.to_numeric(prepared["affinity"], errors="coerce")
    prepared["target_numeric"] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce")

    missing_sequence = pd.Series(False, index=prepared.index)
    for column_name in SEQUENCE_COLUMNS:
        missing_sequence |= prepared[column_name].map(is_missing)

    antigen_type = prepared["antigen_type"].astype(str).str.lower()
    prepared["has_computable_target"] = (
        prepared["affinity_numeric"].notna()
        & (prepared["affinity_numeric"] > 0)
        & prepared["target_numeric"].notna()
    )
    prepared["has_three_sequences"] = ~missing_sequence
    prepared["base_extracted_pool"] = prepared["has_computable_target"] & prepared["has_three_sequences"]
    prepared["protein_antigen"] = antigen_type.str.contains("protein", na=False) & ~antigen_type.str.contains(
        "hapten", na=False
    )
    prepared["peptide_antigen"] = antigen_type.str.contains("peptide", na=False) & ~antigen_type.str.contains(
        "hapten", na=False
    )
    prepared["same_heavy_light_chain_id"] = (
        prepared["Hchain"].astype(str).str.strip().str.upper()
        == prepared["Lchain"].astype(str).str.strip().str.upper()
    )
    prepared["suspicious_numeric_affinity_method"] = prepared["affinity_method"].map(
        looks_suspicious_affinity_method
    )
    return prepared


def dataset_masks(prepared: pd.DataFrame) -> dict[str, pd.Series]:
    """Create strict and relaxed candidate masks.

    strict here mirrors the previous metadata policy, but only inside rows
    where target and three sequences already exist.
    """

    base = prepared["base_extracted_pool"]
    protein = prepared["protein_antigen"]
    sequence_antigen = prepared["protein_antigen"] | prepared["peptide_antigen"]
    heavy_light_different = ~prepared["same_heavy_light_chain_id"]
    method_not_suspicious = ~prepared["suspicious_numeric_affinity_method"]

    return {
        "base_target_and_sequences": base,
        "strict_extracted_reference": base & protein & heavy_light_different & method_not_suspicious,
        "relax_allow_peptide_only": base & sequence_antigen & heavy_light_different & method_not_suspicious,
        "relax_allow_same_h_l_only": base & protein & method_not_suspicious,
        "relax_allow_suspicious_method_only": base & protein & heavy_light_different,
        "less_strict_all_relaxations": base & sequence_antigen,
    }


def unique_triplet_count(data: pd.DataFrame) -> int:
    """Count distinct model inputs after exact triplet dedup."""

    return int(data.drop_duplicates(TRIPLET_COLUMNS).shape[0])


def mask_summary(name: str, rows: pd.DataFrame, strict_rows: pd.DataFrame) -> dict:
    """Summarize one rule set before final dedup."""

    strict_triplets = set(strict_rows[TRIPLET_COLUMNS].astype(str).agg("||".join, axis=1))
    row_triplets = set(rows[TRIPLET_COLUMNS].astype(str).agg("||".join, axis=1))
    return {
        "rule_set": name,
        "raw_rows": int(len(rows)),
        "unique_pdbs": int(rows["pdb"].nunique()),
        "unique_triplets": unique_triplet_count(rows),
        "raw_row_delta_vs_strict": int(len(rows) - len(strict_rows)),
        "unique_triplet_delta_vs_strict": int(len(row_triplets - strict_triplets)),
        "peptide_rows": int(rows["peptide_antigen"].sum()),
        "same_h_l_chain_id_rows": int(rows["same_heavy_light_chain_id"].sum()),
        "suspicious_numeric_method_rows": int(rows["suspicious_numeric_affinity_method"].sum()),
    }


def risk_notes() -> dict[str, str]:
    """Explain the risk of each relaxation in human language."""

    return {
        "strict_extracted_reference": (
            "Reference policy: protein antigen, Hchain != Lchain, and no numeric/suspicious affinity_method."
        ),
        "relax_allow_peptide_only": (
            "Adds peptide antigen sequences. They are valid sequence inputs, but peptide binding may have "
            "different length/statistics from protein antigens."
        ),
        "relax_allow_same_h_l_only": (
            "Allows Hchain == Lchain metadata. Some are scFv/single-chain or engineered constructs, "
            "so heavy/light semantics can be less standard."
        ),
        "relax_allow_suspicious_method_only": (
            "Allows numeric affinity_method values. These rows have sequence and target, but method metadata "
            "may be shifted or noisy because pure numbers look PMID-like."
        ),
        "less_strict_all_relaxations": (
            "Final v1 relaxed pool: use all sequence-only extracted rows with computable target and "
            "protein/peptide sequence antigens. Exact duplicate triplets are removed before split."
        ),
    }


def deduplicate_triplets(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep first exact triplet and record removed duplicates.

    三条输入 sequence 完全一样时，模型看到的信息也一样。
    去掉 exact duplicate 不会减少输入多样性，还能降低 split 和评估解释风险。
    """

    duplicated = data[data.duplicated(TRIPLET_COLUMNS, keep="first")].copy()
    if len(duplicated):
        duplicated["exclusion_reason"] = "exact_heavy_light_antigen_triplet_duplicate"
    deduplicated = data.drop_duplicates(TRIPLET_COLUMNS, keep="first").copy()
    return deduplicated, duplicated


def pdb_level_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Make reproducible PDB-level train/val/test splits."""

    shuffled_pdbs = (
        pd.Series(data["pdb"].astype(str).unique())
        .sample(frac=1, random_state=SEED)
        .tolist()
    )
    train_end = int(len(shuffled_pdbs) * 0.8)
    val_end = int(len(shuffled_pdbs) * 0.9)
    train_pdbs = set(shuffled_pdbs[:train_end])
    val_pdbs = set(shuffled_pdbs[train_end:val_end])
    test_pdbs = set(shuffled_pdbs[val_end:])
    return {
        "train": data[data["pdb"].astype(str).isin(train_pdbs)].copy(),
        "val": data[data["pdb"].astype(str).isin(val_pdbs)].copy(),
        "test": data[data["pdb"].astype(str).isin(test_pdbs)].copy(),
    }


def overlap_count(first: pd.DataFrame, second: pd.DataFrame, columns: list[str]) -> int:
    """Count exact key overlap across two splits."""

    first_keys = set(first[columns].astype(str).agg("||".join, axis=1))
    second_keys = set(second[columns].astype(str).agg("||".join, axis=1))
    return int(len(first_keys & second_keys))


def split_report(splits: dict[str, pd.DataFrame]) -> dict:
    """Summarize split sizes and leakage checks."""

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    checks = {
        "pdb": ["pdb"],
        "heavy_sequence": ["heavy_sequence"],
        "light_sequence": ["light_sequence"],
        "antigen_sequence": ["antigen_sequence"],
        "heavy_light_antigen_triplet": TRIPLET_COLUMNS,
    }
    overlap = {}
    for first_name, second_name in pairs:
        key = f"{first_name}_vs_{second_name}"
        overlap[key] = {
            check_name: overlap_count(splits[first_name], splits[second_name], columns)
            for check_name, columns in checks.items()
        }

    return {
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "unique_pdb_counts": {name: int(frame["pdb"].nunique()) for name, frame in splits.items()},
        "overlap_check": overlap,
    }


def target_stats(data: pd.DataFrame) -> dict:
    """Return compact target statistics."""

    values = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").dropna()
    return {
        "count": int(len(values)),
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std()) if len(values) else None,
    }


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Write new less-strict train/val/test CSVs only under the new version."""

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, frame in splits.items():
        frame.to_csv(SPLIT_DIR / f"{split_name}.csv", index=False)


def build_report(prepared: pd.DataFrame, masks: dict[str, pd.Series], final_data: pd.DataFrame, duplicates: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> dict:
    """Build count/risk report for the dataset design."""

    strict_rows = prepared[masks["strict_extracted_reference"]].copy()
    rule_summaries = [
        mask_summary(name, prepared[mask].copy(), strict_rows)
        for name, mask in masks.items()
        if name != "base_target_and_sequences"
    ]
    invalid_base_rows = prepared[~masks["base_target_and_sequences"]]
    return {
        "dataset_version": "less_strict_sabdab_affinity_v1",
        "inputs": {
            "sequence_only_dir": str(INPUT_DIR.relative_to(PROJECT_ROOT)),
            "reason": "sequence_only rows already have extracted heavy/light/antigen sequences",
        },
        "design": {
            "target_rule": "affinity > 0 and numeric neg_log10_affinity",
            "sequence_rule": "heavy_sequence, light_sequence, and antigen_sequence must be present",
            "final_relaxations": [
                "allow peptide antigens",
                "allow Hchain == Lchain rows",
                "allow numeric/suspicious affinity_method rows",
            ],
            "dedup_rule": "remove exact heavy_sequence+light_sequence+antigen_sequence duplicates before split",
            "split_rule": "PDB-level train/val/test split with seed 42",
        },
        "risk_notes": risk_notes(),
        "input_rows": int(len(prepared)),
        "rows_failing_target_or_sequence_sanity": int(len(invalid_base_rows)),
        "rule_count_summaries_before_dedup": rule_summaries,
        "final_before_dedup": {
            "rows": int(masks["less_strict_all_relaxations"].sum()),
            "unique_triplets": unique_triplet_count(prepared[masks["less_strict_all_relaxations"]]),
        },
        "exact_duplicate_triplet_rows_removed": int(len(duplicates)),
        "final_after_dedup": {
            "rows": int(len(final_data)),
            "unique_pdbs": int(final_data["pdb"].nunique()),
            "target_stats": target_stats(final_data),
            "antigen_type_counts": final_data["antigen_type"].value_counts(dropna=False).to_dict(),
            "affinity_method_counts": final_data["affinity_method"].value_counts(dropna=False).to_dict(),
        },
        "split_report": split_report(splits),
        "recommendation": (
            "This dataset is worth trying later as a less-strict SAbDab-only version, but metric changes "
            "must be interpreted with peptide/single-chain/method-metadata noise risks in mind."
        ),
    }


def markdown_table_for_rules(rule_rows: list[dict]) -> list[str]:
    """Render the relaxation-count table."""

    lines = [
        "| rule set | raw rows | unique triplets | raw delta vs strict | triplet delta vs strict | peptide rows | same H/L rows | suspicious method rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rule_rows:
        lines.append(
            f"| `{row['rule_set']}` | {row['raw_rows']} | {row['unique_triplets']} | "
            f"{row['raw_row_delta_vs_strict']} | {row['unique_triplet_delta_vs_strict']} | "
            f"{row['peptide_rows']} | {row['same_h_l_chain_id_rows']} | "
            f"{row['suspicious_numeric_method_rows']} |"
        )
    return lines


def write_markdown(report: dict) -> None:
    """Write a beginner-readable design/count report."""

    final = report["final_after_dedup"]
    split = report["split_report"]
    lines = [
        "# Less Strict SAbDab Affinity v1 Count Report",
        "",
        "## Goal",
        "",
        "- Increase future trainable SAbDab affinity rows without touching strict datasets.",
        "- Start only from `sequence_only` rows where sequence extraction already succeeded.",
        "- Keep only rows with a computable `neg_log10_affinity` target and three input sequences.",
        "",
        "## Design",
        "",
        f"- Input rows scanned: {report['input_rows']}",
        f"- Rows failing target/sequence sanity: {report['rows_failing_target_or_sequence_sanity']}",
        f"- Final before exact triplet dedup: `{report['final_before_dedup']}`",
        f"- Exact duplicate triplet rows removed: {report['exact_duplicate_triplet_rows_removed']}",
        f"- Final after dedup: {final['rows']} rows / {final['unique_pdbs']} PDBs",
        f"- Target stats after dedup: `{final['target_stats']}`",
        f"- Output split directory: `{SPLIT_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## Relaxation Counts",
        "",
    ]
    lines.extend(markdown_table_for_rules(report["rule_count_summaries_before_dedup"]))
    lines.extend(["", "## Relaxation Risks", ""])
    for rule_name, note in report["risk_notes"].items():
        lines.append(f"- `{rule_name}`: {note}")

    lines.extend(
        [
            "",
            "## Final Split",
            "",
            f"- Split sizes: `{split['split_sizes']}`",
            f"- Unique PDB counts: `{split['unique_pdb_counts']}`",
            f"- Overlap check: `{split['overlap_check']}`",
            "",
            "## Final Composition",
            "",
            f"- antigen_type counts: `{final['antigen_type_counts']}`",
            f"- affinity_method counts: `{final['affinity_method_counts']}`",
            "",
            "## Interpretation",
            "",
            "- This v1 is deliberately less strict than the strict metadata reference.",
            "- It does not invent rows from summary metadata that lack extracted sequences.",
            "- Dedup makes the final row count smaller than the raw relaxed pool, but keeps distinct model inputs.",
            "- PDB overlap is zero after the split, but individual heavy/light/antigen sequences can still repeat across PDBs; check the overlap table before treating this as a strict generalization split.",
            f"- Recommendation: {report['recommendation']}",
        ]
    )
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: dict) -> None:
    """Print a compact terminal summary."""

    strict = next(
        row for row in report["rule_count_summaries_before_dedup"]
        if row["rule_set"] == "strict_extracted_reference"
    )
    relaxed = next(
        row for row in report["rule_count_summaries_before_dedup"]
        if row["rule_set"] == "less_strict_all_relaxations"
    )
    print("less_strict_sabdab_affinity_v1 build complete.")
    print(f"Strict extracted reference: {strict['raw_rows']} raw rows / {strict['unique_triplets']} unique triplets")
    print(f"Less-strict relaxed pool: {relaxed['raw_rows']} raw rows / {relaxed['unique_triplets']} unique triplets")
    print(
        "Final deduplicated dataset: "
        f"{report['final_after_dedup']['rows']} rows / "
        f"{report['final_after_dedup']['unique_pdbs']} PDBs"
    )
    print(f"Split sizes: {report['split_report']['split_sizes']}")
    print(f"Count report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON report: {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("No model training was run.")


def main() -> None:
    """Build the new less-strict dataset and its count report."""

    prepared = prepare_flags(load_sequence_only())
    masks = dataset_masks(prepared)
    relaxed = prepared[masks["less_strict_all_relaxations"]].copy()
    relaxed = relaxed.drop(
        columns=[
            "affinity_numeric",
            "target_numeric",
            "has_computable_target",
            "has_three_sequences",
            "base_extracted_pool",
            "protein_antigen",
            "peptide_antigen",
            "same_heavy_light_chain_id",
            "suspicious_numeric_affinity_method",
        ],
        errors="ignore",
    )
    final_data, duplicates = deduplicate_triplets(relaxed)
    splits = pdb_level_split(final_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_splits(splits)
    duplicates.to_csv(EXCLUDED_DUPLICATES_PATH, index=False)

    report = build_report(prepared, masks, final_data, duplicates, splits)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print_summary(report)


if __name__ == "__main__":
    main()
