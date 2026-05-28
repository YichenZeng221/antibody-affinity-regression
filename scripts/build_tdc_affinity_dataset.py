"""Build a TDC Protein_SAbDab affinity regression dataset.

:
 TDC ,, clean_v2

 TDC inspection  raw split:
    data/external/tdc_antibody_affinity/raw_train.csv
    data/external/tdc_antibody_affinity/raw_val.csv
    data/external/tdc_antibody_affinity/raw_test.csv

 affinity regression CSV:
    data/processed_affinity/tdc_v1/antigen_group_split/train.csv
    data/processed_affinity/tdc_v1/antigen_group_split/val.csv
    data/processed_affinity/tdc_v1/antigen_group_split/test.csv

 split?
TDC  split  random split, train/test  antigen overlap
 antigen_group_split: antigen_sequence  split 
 antigen
"""

from __future__ import annotations

from pathlib import Path
import ast
import json
import math
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "external" / "tdc_antibody_affinity"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1"
SPLIT_OUTPUT_DIR = OUTPUT_ROOT / "antigen_group_split"
SEED = 42
TARGET_COLUMN = "neg_log10_affinity"

#  20 , X/B/Z/U/O 
AMINO_ACID_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZUO]+$")

OUTPUT_COLUMNS = [
    "sample_id",
    "source",
    "original_split",
    "antibody_id",
    "antigen_id",
    "heavy_sequence",
    "light_sequence",
    "antigen_sequence",
    "affinity",
    "neg_log10_affinity",
    "Antibody_ID",
    "Antigen_ID",
    "Antibody",
    "Antigen",
    "Y",
]


def load_raw_splits() -> pd.DataFrame:
    """Load raw train/val/test CSV files and remember original_split.

    original_split  TDC  random split 
     split, original_split 
    """

    frames = []
    for split_name, filename in [
        ("raw_train", "raw_train.csv"),
        ("raw_val", "raw_val.csv"),
        ("raw_test", "raw_test.csv"),
    ]:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}. Run inspect_tdc_antibody_affinity.py first.")
        dataframe = pd.read_csv(path)
        dataframe["original_split"] = split_name
        frames.append(dataframe)

    return pd.concat(frames, ignore_index=True)


def add_excluded(excluded_records: list[pd.DataFrame], rows: pd.DataFrame, reason: str) -> None:
    """Store excluded rows with a reason so cleaning is auditable."""

    if len(rows) == 0:
        return

    excluded = rows.copy()
    excluded["exclusion_reason"] = reason
    excluded_records.append(excluded)


def clean_sequence(sequence: object) -> str:
    """Normalize sequence text.

    ,
    """

    return re.sub(r"[\s,;|]+", "", str(sequence).strip().upper())


def is_valid_sequence(sequence: object) -> bool:
    """Check whether a sequence looks like amino acid letters."""

    cleaned = clean_sequence(sequence)
    return bool(cleaned) and bool(AMINO_ACID_PATTERN.fullmatch(cleaned))


def parse_antibody_chains(value: object) -> tuple[str | None, str | None]:
    """Parse TDC Antibody column into heavy and light sequences.

    :
    TDC  antibody  heavy chain -> light chain
     Python list :
        "['HEAVYSEQ', 'LIGHTSEQ']"

    , sequence
    , None,
    """

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            heavy = clean_sequence(parsed[0])
            light = clean_sequence(parsed[1])
            return heavy, light
    except (SyntaxError, ValueError):
        pass

    # fallback:/,
    simplified = text.strip("[]()")
    simplified = simplified.replace('"', "").replace("'", "")

    for delimiter in ["|", ";", ","]:
        parts = [clean_sequence(part) for part in simplified.split(delimiter)]
        parts = [part for part in parts if part]
        if len(parts) >= 2:
            return parts[0], parts[1]

    # ,
    parts = [clean_sequence(part) for part in simplified.split()]
    parts = [part for part in parts if part]
    if len(parts) >= 2:
        return parts[0], parts[1]

    return None, None


def filter_target(dataframe: pd.DataFrame, excluded_records: list[pd.DataFrame]) -> pd.DataFrame:
    """Remove rows with missing/non-positive raw affinity Y."""

    numeric_y = pd.to_numeric(dataframe["Y"], errors="coerce")
    bad_mask = numeric_y.isna() | (numeric_y <= 0)
    add_excluded(excluded_records, dataframe[bad_mask].copy(), "missing_non_numeric_or_non_positive_y")

    kept = dataframe[~bad_mask].copy()
    kept["Y"] = pd.to_numeric(kept["Y"], errors="coerce")
    kept["affinity"] = kept["Y"].astype(float)

    # TDC  Y  raw affinity/Kd -log10 
    kept["neg_log10_affinity"] = kept["affinity"].map(lambda value: -math.log10(float(value)))
    return kept


def parse_and_filter_sequences(dataframe: pd.DataFrame, excluded_records: list[pd.DataFrame]) -> pd.DataFrame:
    """Parse heavy/light, clean antigen, and remove invalid sequences."""

    rows = []
    cannot_parse_rows = []
    invalid_sequence_rows = []

    for _, row in dataframe.iterrows():
        heavy_sequence, light_sequence = parse_antibody_chains(row["Antibody"])

        if heavy_sequence is None or light_sequence is None:
            cannot_parse_rows.append(row)
            continue

        antigen_sequence = clean_sequence(row["Antigen"])

        if (
            not is_valid_sequence(heavy_sequence)
            or not is_valid_sequence(light_sequence)
            or not is_valid_sequence(antigen_sequence)
        ):
            invalid_sequence_rows.append(row)
            continue

        new_row = row.copy()
        new_row["heavy_sequence"] = heavy_sequence
        new_row["light_sequence"] = light_sequence
        new_row["antigen_sequence"] = antigen_sequence
        rows.append(new_row)

    if cannot_parse_rows:
        add_excluded(excluded_records, pd.DataFrame(cannot_parse_rows), "cannot_parse_heavy_light")
    if invalid_sequence_rows:
        add_excluded(excluded_records, pd.DataFrame(invalid_sequence_rows), "invalid_amino_acid_sequence")

    if not rows:
        return dataframe.iloc[0:0].copy()

    return pd.DataFrame(rows)


def deduplicate_triplets(dataframe: pd.DataFrame, excluded_records: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Remove exact heavy+light+antigen duplicates, keeping the first row.

     triplet  Y,
    , beginner MVP , report/excluded
    """

    triplet_columns = ["heavy_sequence", "light_sequence", "antigen_sequence"]
    kept_groups = []
    conflicting_duplicate_groups = 0

    for _, group in dataframe.groupby(triplet_columns, sort=False, dropna=False):
        if len(group) == 1:
            kept_groups.append(group.iloc[[0]])
            continue

        target_range = group["affinity"].astype(float).max() - group["affinity"].astype(float).min()
        if target_range > 1e-30:
            conflicting_duplicate_groups += 1
            add_excluded(excluded_records, group.copy(), "conflicting_duplicate_target")
            continue

        kept_groups.append(group.iloc[[0]])
        add_excluded(excluded_records, group.iloc[1:].copy(), "duplicate_triplet_removed")

    if not kept_groups:
        return dataframe.iloc[0:0].copy(), conflicting_duplicate_groups

    return pd.concat(kept_groups, ignore_index=True), conflicting_duplicate_groups


def add_project_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add unified project columns and stable sample_id."""

    output = dataframe.copy().reset_index(drop=True)
    output["sample_id"] = [f"TDC_{index + 1:06d}" for index in range(len(output))]
    output["source"] = "TDC_Protein_SAbDab"
    output["antibody_id"] = output["Antibody_ID"]
    output["antigen_id"] = output["Antigen_ID"]
    return output


def antigen_group_split(dataframe: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Split by antigen_sequence so the same antigen never crosses splits."""

    antigen_sizes = dataframe.groupby("antigen_sequence").size().reset_index(name="count")
    antigen_sizes = antigen_sizes.sample(frac=1, random_state=seed).reset_index(drop=True)

    total_rows = len(dataframe)
    target_train = total_rows * 0.8
    target_val = total_rows * 0.1

    split_to_antigens = {"train": [], "val": [], "test": []}
    split_sizes = {"train": 0, "val": 0, "test": 0}

    #  greedy: antigen group  split
    # group split , antigen group 
    for _, row in antigen_sizes.iterrows():
        deficits = {
            "train": target_train - split_sizes["train"],
            "val": target_val - split_sizes["val"],
            "test": (total_rows - target_train - target_val) - split_sizes["test"],
        }
        split_name = max(deficits, key=deficits.get)
        split_to_antigens[split_name].append(row["antigen_sequence"])
        split_sizes[split_name] += int(row["count"])

    return {
        split_name: dataframe[dataframe["antigen_sequence"].isin(antigens)].copy()
        for split_name, antigens in split_to_antigens.items()
    }


def overlap_count(first: pd.DataFrame, second: pd.DataFrame, columns: list[str]) -> int:
    """Count overlap for one column or combined key."""

    first_keys = set(first[columns].astype(str).agg("||".join, axis=1))
    second_keys = set(second[columns].astype(str).agg("||".join, axis=1))
    return len(first_keys & second_keys)


def split_overlap_report(splits: dict[str, pd.DataFrame]) -> dict:
    """Report train/test overlap requested by the user."""

    train = splits["train"]
    test = splits["test"]
    checks = {
        "antibody_id": ["antibody_id"],
        "antigen_id": ["antigen_id"],
        "heavy_sequence": ["heavy_sequence"],
        "light_sequence": ["light_sequence"],
        "antigen_sequence": ["antigen_sequence"],
        "heavy_light_pair": ["heavy_sequence", "light_sequence"],
        "heavy_light_antigen_triplet": ["heavy_sequence", "light_sequence", "antigen_sequence"],
    }
    return {name: overlap_count(train, test, columns) for name, columns in checks.items()}


def numeric_stats(series: pd.Series) -> dict:
    """Return JSON-friendly numeric stats."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def length_stats(dataframe: pd.DataFrame, column_name: str) -> dict:
    """Return min/max/mean length for a sequence column."""

    lengths = dataframe[column_name].astype(str).str.len()
    if len(lengths) == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
    }


def write_outputs(splits: dict[str, pd.DataFrame], excluded: pd.DataFrame, report: dict) -> None:
    """Write train/val/test CSV, excluded records, and JSON report."""

    SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split_name, dataframe in splits.items():
        dataframe[OUTPUT_COLUMNS].to_csv(SPLIT_OUTPUT_DIR / f"{split_name}.csv", index=False)

    excluded.to_csv(OUTPUT_ROOT / "excluded_records.csv", index=False)

    with open(OUTPUT_ROOT / "processing_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def main() -> None:
    """Build the TDC v1 antigen_group_split dataset."""

    excluded_records: list[pd.DataFrame] = []

    raw = load_raw_splits()
    after_target = filter_target(raw, excluded_records)
    after_parsing = parse_and_filter_sequences(after_target, excluded_records)
    after_dedup, conflicting_duplicate_groups = deduplicate_triplets(after_parsing, excluded_records)
    final_data = add_project_columns(after_dedup)

    splits = antigen_group_split(final_data)

    if excluded_records:
        excluded = pd.concat(excluded_records, ignore_index=True)
    else:
        excluded = raw.iloc[0:0].copy()
        excluded["exclusion_reason"] = []

    report = {
        "raw_total_rows": int(len(raw)),
        "rows_after_target_filtering": int(len(after_target)),
        "rows_after_antibody_parsing": int(len(after_parsing)),
        "rows_after_dedup": int(len(final_data)),
        "conflicting_duplicate_target_groups": int(conflicting_duplicate_groups),
        "excluded_counts_by_reason": excluded["exclusion_reason"].value_counts(dropna=False).to_dict(),
        "target_stats": numeric_stats(final_data[TARGET_COLUMN]),
        "affinity_stats_raw_y": numeric_stats(final_data["affinity"]),
        "length_stats": {
            "heavy_sequence": length_stats(final_data, "heavy_sequence"),
            "light_sequence": length_stats(final_data, "light_sequence"),
            "antigen_sequence": length_stats(final_data, "antigen_sequence"),
        },
        "split_sizes": {split: int(len(df)) for split, df in splits.items()},
        "unique_antigen_count_per_split": {
            split: int(df["antigen_sequence"].nunique()) for split, df in splits.items()
        },
        "train_test_overlap": split_overlap_report(splits),
        "target_distribution_by_split": {
            split: numeric_stats(df[TARGET_COLUMN]) for split, df in splits.items()
        },
        "split_note": (
            "This is an antigen_group_split. The same antigen_sequence is kept within one split. "
            "Ratios may not be exactly 80/10/10 because antigen groups have different sizes."
        ),
    }

    write_outputs(splits, excluded, report)

    print("TDC affinity dataset built successfully.")
    print(f"Output directory: {SPLIT_OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Raw total rows: {report['raw_total_rows']}")
    print(f"Rows after target filtering: {report['rows_after_target_filtering']}")
    print(f"Rows after antibody parsing: {report['rows_after_antibody_parsing']}")
    print(f"Rows after dedup: {report['rows_after_dedup']}")
    print(f"Excluded counts by reason: {report['excluded_counts_by_reason']}")
    print(f"Split sizes: {report['split_sizes']}")
    print(f"Unique antigen count per split: {report['unique_antigen_count_per_split']}")
    print(f"Train/test overlap: {report['train_test_overlap']}")
    print(f"Target stats neg_log10_affinity: {report['target_stats']}")
    print()
    print("Files written:")
    print(f"  {SPLIT_OUTPUT_DIR / 'train.csv'}")
    print(f"  {SPLIT_OUTPUT_DIR / 'val.csv'}")
    print(f"  {SPLIT_OUTPUT_DIR / 'test.csv'}")
    print(f"  {OUTPUT_ROOT / 'processing_report.json'}")
    print(f"  {OUTPUT_ROOT / 'excluded_records.csv'}")


if __name__ == "__main__":
    main()
