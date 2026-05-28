"""Build unified_affinity_dataset_v1 from already audited affinity sources.

:
:

1. /;
2.  sequence-only affinity regression  rows:
    sequence ,target ``neg_log10_affinity`` ;
3.  heavy + light + antigen  sequence  exact triplet ;
4.  target  triplet  conflict file,;
5.  antigen_sequence group  split, train/test antigen leakage

 dataset version:
    data/processed_affinity/unified_affinity_dataset_v1/
 TDC v1supplement v1less-strict v1,
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TDC_V1_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
SUPPLEMENT_V1_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "tdc_plus_sabdab_supplement_v1"
    / "antigen_group_split"
)
LESS_STRICT_ROOT = PROJECT_ROOT / "data" / "processed_affinity" / "less_strict_sabdab_affinity_v1"
PATRICK_CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "sabdab_affinity_search_735_audit"
    / "new_candidate_rows.csv"
)
SEQUENCE_CACHE_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "sequence_only"
RAW_REFERENCE_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "zenodo_antibody_affinity_protein_sabdab.csv",
    PROJECT_ROOT / "data" / "raw" / "sabdab_affinity_search_735_summary.tsv",
    PROJECT_ROOT / "data" / "raw" / "sabdab_summary.tsv",
]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "unified_affinity_dataset_v1"
TRAIN_PATH = OUTPUT_DIR / "train.csv"
VAL_PATH = OUTPUT_DIR / "val.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"
RAW_MERGED_PATH = OUTPUT_DIR / "all_unified_raw_merged.csv"
DEDUP_PATH = OUTPUT_DIR / "all_unified_dedup.csv"
EXCLUDED_DUPLICATES_PATH = OUTPUT_DIR / "excluded_duplicates.csv"
CONFLICTING_TARGETS_PATH = OUTPUT_DIR / "conflicting_targets.csv"
NEEDS_SEQUENCE_PATH = OUTPUT_DIR / "needs_sequence_extraction.csv"
JSON_REPORT_PATH = OUTPUT_DIR / "processing_report.json"
MARKDOWN_REPORT_PATH = OUTPUT_DIR / "processing_report.md"

SPLITS = ["train", "val", "test"]
SEED = 42
TARGET_TOLERANCE = 1e-6
SEQUENCE_COLUMNS = ["heavy_sequence", "light_sequence", "antigen_sequence"]
TRIPLET_COLUMNS = SEQUENCE_COLUMNS
CHAIN_COLUMNS = ["pdb", "Hchain", "Lchain", "antigen_chain"]

#  unified schema source_sample_id / duplicate_sources,
#  dataset 
UNIFIED_COLUMNS = [
    "sample_id",
    "source",
    "original_source",
    "original_split",
    "pdb_or_antibody_id",
    "antigen_id",
    "Hchain",
    "Lchain",
    "antigen_chain",
    "heavy_sequence",
    "light_sequence",
    "antigen_sequence",
    "affinity",
    "neg_log10_affinity",
    "target_source",
    "affinity_method",
    "antigen_type",
    "risk_flags",
    "inclusion_reason",
    "source_sample_id",
    "duplicate_sources",
]

AMINO_ACID_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZUO]+$")


def is_missing(value: object) -> bool:
    """Treat blank cells and common NA spellings as missing."""

    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL", "<NA>"}


def optional_text(row: pd.Series, column_name: str) -> str:
    """Read optional metadata as a clean string."""

    value = row.get(column_name, "")
    return "" if is_missing(value) else str(value).strip()


def normalize_sequence(value: object) -> str:
    """Normalize one amino-acid sequence for exact overlap keys."""

    if is_missing(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip().upper())


def valid_sequence(value: object) -> bool:
    """Require a non-empty amino-acid-like sequence."""

    sequence = normalize_sequence(value)
    return bool(sequence) and bool(AMINO_ACID_PATTERN.fullmatch(sequence))


def numeric(value: object) -> float | None:
    """Convert one cell to float; return None for missing/bad numbers."""

    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(converted) else float(converted)


def safe_neg_log10_affinity(affinity: object) -> float | None:
    """Compute -log10(Kd/affinity) only for positive numeric affinity."""

    value = numeric(affinity)
    if value is None or value <= 0:
        return None
    return -math.log10(value)


def numeric_stats(series: pd.Series) -> dict:
    """Return JSON-friendly numeric summary."""

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


def join_flags(flags: list[str]) -> str:
    """Store risk flags in one readable pipe-separated cell."""

    return "|".join(sorted({flag for flag in flags if str(flag).strip()}))


def suspicious_method(value: object) -> bool:
    """Flag PMID-like method metadata made only of digits."""

    text = str(value).strip()
    return bool(text) and bool(re.fullmatch(r"\d+(?:\.0+)?", text))


def add_standard_row(
    rows: list[dict],
    *,
    source_sample_id: object,
    source: str,
    original_source: str,
    original_split: str,
    pdb_or_antibody_id: object,
    antigen_id: object,
    Hchain: object = "",
    Lchain: object = "",
    antigen_chain: object = "",
    heavy_sequence: object,
    light_sequence: object,
    antigen_sequence: object,
    affinity: object,
    neg_log10_affinity: object,
    target_source: str,
    affinity_method: object = "",
    antigen_type: object = "",
    risk_flags: list[str] | None = None,
    inclusion_reason: str,
) -> None:
    """Append one row in the unified schema before final validity filtering."""

    target = numeric(neg_log10_affinity)
    if target is None:
        target = safe_neg_log10_affinity(affinity)

    rows.append(
        {
            "sample_id": "",  #  ID
            "source": str(source),
            "original_source": str(original_source),
            "original_split": str(original_split),
            "pdb_or_antibody_id": "" if is_missing(pdb_or_antibody_id) else str(pdb_or_antibody_id).strip(),
            "antigen_id": "" if is_missing(antigen_id) else str(antigen_id).strip(),
            "Hchain": "" if is_missing(Hchain) else str(Hchain).strip(),
            "Lchain": "" if is_missing(Lchain) else str(Lchain).strip(),
            "antigen_chain": "" if is_missing(antigen_chain) else str(antigen_chain).strip(),
            "heavy_sequence": normalize_sequence(heavy_sequence),
            "light_sequence": normalize_sequence(light_sequence),
            "antigen_sequence": normalize_sequence(antigen_sequence),
            "affinity": numeric(affinity),
            "neg_log10_affinity": target,
            "target_source": str(target_source),
            "affinity_method": "" if is_missing(affinity_method) else str(affinity_method).strip(),
            "antigen_type": "" if is_missing(antigen_type) else str(antigen_type).strip(),
            "risk_flags": join_flags(risk_flags or []),
            "inclusion_reason": str(inclusion_reason),
            "source_sample_id": "" if is_missing(source_sample_id) else str(source_sample_id).strip(),
            "duplicate_sources": "",
        }
    )


def load_split_dir(split_dir: Path, dataset_name: str) -> list[tuple[str, pd.DataFrame]]:
    """Read train/val/test CSV files from one processed dataset version."""

    frames = []
    for split_name in SPLITS:
        path = split_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {dataset_name} split: {path}")
        frames.append((split_name, pd.read_csv(path)))
    return frames


def resolve_less_strict_split_dir() -> Path:
    """Accept the requested root path and the actual v1 pdb_level_split path."""

    root_has_splits = all((LESS_STRICT_ROOT / f"{split_name}.csv").exists() for split_name in SPLITS)
    nested = LESS_STRICT_ROOT / "pdb_level_split"
    nested_has_splits = all((nested / f"{split_name}.csv").exists() for split_name in SPLITS)
    if root_has_splits:
        return LESS_STRICT_ROOT
    if nested_has_splits:
        return nested
    raise FileNotFoundError(
        f"Cannot find less-strict train/val/test under {LESS_STRICT_ROOT} or {nested}."
    )


def standardize_tdc_v1() -> pd.DataFrame:
    """Convert TDC v1 clean rows to unified schema."""

    rows: list[dict] = []
    for split_name, frame in load_split_dir(TDC_V1_DIR, "tdc_v1"):
        for _, row in frame.iterrows():
            add_standard_row(
                rows,
                source_sample_id=row.get("sample_id", ""),
                source=optional_text(row, "source") or "TDC_Protein_SAbDab",
                original_source="tdc_v1_clean",
                original_split=split_name,
                pdb_or_antibody_id=row.get("antibody_id", row.get("Antibody_ID", "")),
                antigen_id=row.get("antigen_id", row.get("Antigen_ID", "")),
                heavy_sequence=row["heavy_sequence"],
                light_sequence=row["light_sequence"],
                antigen_sequence=row["antigen_sequence"],
                affinity=row.get("affinity", row.get("Y", "")),
                neg_log10_affinity=row.get("neg_log10_affinity", ""),
                target_source="tdc_v1_raw_affinity_Y",
                inclusion_reason="TDC v1 clean row already passed target, sequence, and triplet cleaning.",
            )
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


def standardize_supplement_v1() -> pd.DataFrame:
    """Convert TDC plus SAbDab supplement v1 rows to unified schema."""

    rows: list[dict] = []
    for split_name, frame in load_split_dir(SUPPLEMENT_V1_DIR, "tdc_plus_sabdab_supplement_v1"):
        for _, row in frame.iterrows():
            source = optional_text(row, "source") or "tdc_plus_sabdab_supplement_v1"
            flags = ["audited_sabdab_supplement"] if source == "SAbDab_Supplement" else []
            old_split = optional_text(row, "original_dataset_split")
            add_standard_row(
                rows,
                source_sample_id=row.get("sample_id", ""),
                source=source,
                original_source="tdc_plus_sabdab_supplement_v1",
                original_split=f"{split_name}|{old_split}" if old_split else split_name,
                pdb_or_antibody_id=row.get("antibody_id", ""),
                antigen_id=row.get("antigen_id", ""),
                heavy_sequence=row["heavy_sequence"],
                light_sequence=row["light_sequence"],
                antigen_sequence=row["antigen_sequence"],
                affinity=row.get("affinity", ""),
                neg_log10_affinity=row.get("neg_log10_affinity", ""),
                target_source="tdc_plus_sabdab_supplement_v1_affinity",
                risk_flags=flags,
                inclusion_reason="Row from the audited TDC plus SAbDab supplement dataset version.",
            )
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


def less_strict_risk_flags(row: pd.Series) -> list[str]:
    """Explain why a less-strict SAbDab row is riskier than strict rows."""

    flags = ["less_strict_sabdab"]
    antigen_type = optional_text(row, "antigen_type").lower()
    if "peptide" in antigen_type:
        flags.append("peptide_antigen")
    if optional_text(row, "Hchain").upper() == optional_text(row, "Lchain").upper():
        flags.append("same_Hchain_Lchain_metadata")
    if suspicious_method(row.get("affinity_method", "")):
        flags.append("suspicious_numeric_affinity_method")
    return flags


def standardize_less_strict() -> tuple[pd.DataFrame, Path]:
    """Convert less-strict SAbDab rows to unified schema."""

    split_dir = resolve_less_strict_split_dir()
    rows: list[dict] = []
    for split_name, frame in load_split_dir(split_dir, "less_strict_sabdab_affinity_v1"):
        for _, row in frame.iterrows():
            source_cache_split = optional_text(row, "source_sequence_only_split")
            add_standard_row(
                rows,
                source_sample_id=row.get("sample_id", ""),
                source="SAbDab_Less_Strict",
                original_source="less_strict_sabdab_affinity_v1",
                original_split=f"{split_name}|sequence_only_{source_cache_split}"
                if source_cache_split
                else split_name,
                pdb_or_antibody_id=row.get("pdb", ""),
                antigen_id=row.get("antigen_name", ""),
                Hchain=row.get("Hchain", ""),
                Lchain=row.get("Lchain", ""),
                antigen_chain=row.get("antigen_chain", ""),
                heavy_sequence=row["heavy_sequence"],
                light_sequence=row["light_sequence"],
                antigen_sequence=row["antigen_sequence"],
                affinity=row.get("affinity", ""),
                neg_log10_affinity=row.get("neg_log10_affinity", ""),
                target_source="SAbDab_summary_affinity",
                affinity_method=row.get("affinity_method", ""),
                antigen_type=row.get("antigen_type", ""),
                risk_flags=less_strict_risk_flags(row),
                inclusion_reason="Less-strict SAbDab row already has three extracted sequences and target.",
            )
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS), split_dir


def normalized_chain_key(data: pd.DataFrame) -> pd.Series:
    """Build exact PDB+chain metadata key for Patrick cache lookup."""

    normalized = pd.DataFrame(index=data.index)
    for column_name in CHAIN_COLUMNS:
        value = data[column_name].fillna("").astype(str).str.strip().str.upper()
        normalized[column_name] = value
    return normalized.astype(str).agg("||".join, axis=1)


def load_sequence_cache() -> pd.DataFrame:
    """Load sequence_only rows used as the only Patrick candidate sequence cache."""

    cache_frames = []
    for split_name, frame in load_split_dir(SEQUENCE_CACHE_DIR, "sequence_only"):
        frame = frame.copy()
        frame["sequence_cache_split"] = split_name
        cache_frames.append(frame)
    cache = pd.concat(cache_frames, ignore_index=True)
    required = {*CHAIN_COLUMNS, *SEQUENCE_COLUMNS}
    missing = required - set(cache.columns)
    if missing:
        raise ValueError(f"sequence_only cache missing columns: {sorted(missing)}")
    cache["chain_key"] = normalized_chain_key(cache)
    return cache.drop_duplicates("chain_key", keep="first")


def patrick_needs_row(row: pd.Series, reason: str) -> dict:
    """Store Patrick rows we cannot safely merge yet."""

    result = {str(column): row.get(column, "") for column in row.index}
    result["needs_sequence_reason"] = reason
    return result


def standardize_patrick_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge only Patrick candidate rows with exact local sequence-cache matches."""

    if not PATRICK_CANDIDATE_PATH.exists():
        raise FileNotFoundError(f"Cannot find Patrick candidate CSV: {PATRICK_CANDIDATE_PATH}")
    candidates = pd.read_csv(PATRICK_CANDIDATE_PATH)
    required = {*CHAIN_COLUMNS, "affinity", "affinity_numeric"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Patrick candidate CSV missing columns: {sorted(missing)}")

    cache = load_sequence_cache()
    candidates = candidates.copy()
    candidates["chain_key"] = normalized_chain_key(candidates)
    cache_columns = [
        "chain_key",
        *SEQUENCE_COLUMNS,
        "sequence_cache_split",
        "sample_id",
        "neg_log10_affinity",
    ]
    cache_for_merge = cache[cache_columns].rename(
        columns={
            "heavy_sequence": "cache_heavy_sequence",
            "light_sequence": "cache_light_sequence",
            "antigen_sequence": "cache_antigen_sequence",
            "sample_id": "sequence_cache_sample_id",
            "neg_log10_affinity": "sequence_cache_target",
        }
    )
    attached = candidates.merge(cache_for_merge, on="chain_key", how="left")

    rows: list[dict] = []
    needs_rows: list[dict] = []
    for _, row in attached.iterrows():
        if any(not valid_sequence(row.get(f"cache_{column_name}", "")) for column_name in SEQUENCE_COLUMNS):
            needs_rows.append(patrick_needs_row(row, "no_exact_sequence_only_cache_match"))
            continue

        affinity_value = numeric(row.get("affinity_numeric", row.get("affinity", "")))
        target_value = safe_neg_log10_affinity(affinity_value)
        if target_value is None:
            needs_rows.append(patrick_needs_row(row, "missing_or_non_positive_affinity_target"))
            continue

        flags = ["patrick_735_candidate", "sequence_from_sequence_only_cache"]
        if suspicious_method(row.get("affinity_method", "")):
            flags.append("suspicious_numeric_affinity_method")
        if "peptide" in optional_text(row, "antigen_type").lower():
            flags.append("peptide_antigen")
        add_standard_row(
            rows,
            source_sample_id=row.get("sequence_cache_sample_id", ""),
            source="Patrick_735_Search_Candidate",
            original_source="sabdab_affinity_search_735_audit",
            original_split=f"sequence_only_{optional_text(row, 'sequence_cache_split')}",
            pdb_or_antibody_id=row.get("pdb", ""),
            antigen_id=row.get("antigen_name", ""),
            Hchain=row.get("Hchain", ""),
            Lchain=row.get("Lchain", ""),
            antigen_chain=row.get("antigen_chain", ""),
            heavy_sequence=row["cache_heavy_sequence"],
            light_sequence=row["cache_light_sequence"],
            antigen_sequence=row["cache_antigen_sequence"],
            affinity=affinity_value,
            neg_log10_affinity=target_value,
            target_source="Patrick_735_summary_affinity",
            affinity_method=row.get("affinity_method", ""),
            antigen_type=row.get("antigen_type", ""),
            risk_flags=flags,
            inclusion_reason="Patrick 735 candidate exact-matched to local sequence_only cache.",
        )

    needs = pd.DataFrame(needs_rows)
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS), needs


def filter_trainable_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only rows with valid three sequences and numeric target."""

    filtered = data.copy()
    sequence_ok = pd.Series(True, index=filtered.index)
    for column_name in SEQUENCE_COLUMNS:
        filtered[column_name] = filtered[column_name].map(normalize_sequence)
        sequence_ok &= filtered[column_name].map(valid_sequence)
    target = pd.to_numeric(filtered["neg_log10_affinity"], errors="coerce")
    target_ok = target.notna()
    filtered["neg_log10_affinity"] = target
    report = {
        "input_rows": int(len(filtered)),
        "invalid_sequence_rows": int((~sequence_ok).sum()),
        "missing_or_non_numeric_target_rows": int((~target_ok).sum()),
        "kept_rows": int((sequence_ok & target_ok).sum()),
    }
    return filtered[sequence_ok & target_ok].copy(), report


def add_merged_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Give all merged rows stable temporary IDs before dedup."""

    merged = data.copy().reset_index(drop=True)
    merged["sample_id"] = [f"MERGED_{index + 1:06d}" for index in range(len(merged))]
    return merged


def triplet_key(data: pd.DataFrame) -> pd.Series:
    """Build exact model-input key."""

    return data[TRIPLET_COLUMNS].astype(str).agg("||".join, axis=1)


def representative_priority(row: pd.Series) -> tuple[int, str]:
    """Prefer the audited largest dataset lineage when exact duplicate targets agree."""

    priority = {
        "tdc_plus_sabdab_supplement_v1": 0,
        "tdc_v1_clean": 1,
        "less_strict_sabdab_affinity_v1": 2,
        "sabdab_affinity_search_735_audit": 3,
    }
    return priority.get(str(row["original_source"]), 99), str(row["sample_id"])


def deduplicate_triplets(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep target-consistent unique triplets and isolate conflicts."""

    kept: list[pd.DataFrame] = []
    duplicate_rows: list[pd.DataFrame] = []
    conflict_rows: list[pd.DataFrame] = []

    data = data.copy()
    data["triplet_key"] = triplet_key(data)
    for _, group in data.groupby("triplet_key", sort=False):
        group = group.copy()
        targets = pd.to_numeric(group["neg_log10_affinity"], errors="coerce")
        target_min = float(targets.min())
        target_max = float(targets.max())
        if len(group) > 1 and (target_max - target_min) > TARGET_TOLERANCE:
            group["conflict_reason"] = "same_triplet_conflicting_neg_log10_affinity"
            group["triplet_target_min"] = target_min
            group["triplet_target_max"] = target_max
            group["triplet_target_range"] = target_max - target_min
            conflict_rows.append(group)
            continue

        ordered = group.assign(
            _priority=group.apply(lambda row: representative_priority(row), axis=1)
        ).sort_values("_priority")
        representative = ordered.iloc[[0]].drop(columns=["_priority"]).copy()
        duplicate_sources = sorted(
            {
                f"{row.original_source}:{row.source}"
                for row in group[["original_source", "source"]].itertuples(index=False)
            }
        )
        representative["duplicate_sources"] = "|".join(duplicate_sources)
        kept.append(representative)

        if len(ordered) > 1:
            removed = ordered.iloc[1:].drop(columns=["_priority"]).copy()
            removed["duplicate_reason"] = "same_triplet_target_within_tolerance"
            removed["kept_merged_sample_id"] = str(representative.iloc[0]["sample_id"])
            removed["triplet_target_min"] = target_min
            removed["triplet_target_max"] = target_max
            duplicate_rows.append(removed)

    dedup = pd.concat(kept, ignore_index=True) if kept else data.iloc[0:0].copy()
    duplicates = pd.concat(duplicate_rows, ignore_index=True) if duplicate_rows else data.iloc[0:0].copy()
    conflicts = pd.concat(conflict_rows, ignore_index=True) if conflict_rows else data.iloc[0:0].copy()
    dedup = dedup.drop(columns=["triplet_key"], errors="ignore").reset_index(drop=True)
    dedup["sample_id"] = [f"UNIFIED_{index + 1:06d}" for index in range(len(dedup))]
    return dedup, duplicates, conflicts


def antigen_group_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Greedy 80/10/10 split where one antigen sequence stays in one split."""

    if len(data) == 0:
        raise ValueError("Cannot split an empty unified dataset.")
    antigen_sizes = data.groupby("antigen_sequence").size().reset_index(name="count")
    antigen_sizes = antigen_sizes.sample(frac=1, random_state=SEED).reset_index(drop=True)
    target_sizes = {"train": len(data) * 0.8, "val": len(data) * 0.1, "test": len(data) * 0.1}
    split_antigens = {split_name: [] for split_name in SPLITS}
    split_sizes = {split_name: 0 for split_name in SPLITS}

    for _, row in antigen_sizes.iterrows():
        deficits = {
            split_name: target_sizes[split_name] - split_sizes[split_name]
            for split_name in SPLITS
        }
        split_name = max(deficits, key=deficits.get)
        split_antigens[split_name].append(row["antigen_sequence"])
        split_sizes[split_name] += int(row["count"])

    return {
        split_name: data[data["antigen_sequence"].isin(antigens)].copy()
        for split_name, antigens in split_antigens.items()
    }


def key_set(data: pd.DataFrame, columns: list[str]) -> set[str]:
    """Build exact keys for overlap checks."""

    if len(data) == 0:
        return set()
    return set(data[columns].fillna("").astype(str).agg("||".join, axis=1))


def overlap_report(splits: dict[str, pd.DataFrame], columns: list[str]) -> dict[str, int]:
    """Report pairwise leakage overlap among train/val/test."""

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        f"{left}_vs_{right}": int(len(key_set(splits[left], columns) & key_set(splits[right], columns)))
        for left, right in pairs
    }


def source_counts(data: pd.DataFrame, column_name: str) -> dict[str, int]:
    """Count one provenance field."""

    return {str(key): int(value) for key, value in data[column_name].value_counts(dropna=False).items()}


def risk_flag_distribution(data: pd.DataFrame) -> dict[str, int]:
    """Count each risk flag separately."""

    counter: Counter[str] = Counter()
    for value in data["risk_flags"].fillna("").astype(str):
        flags = [flag for flag in value.split("|") if flag]
        if not flags:
            counter["no_risk_flag"] += 1
        counter.update(flags)
    return {key: int(value) for key, value in sorted(counter.items())}


def source_counts_by_split(splits: dict[str, pd.DataFrame]) -> dict:
    """Show which dataset lineages ended up in each final split."""

    return {
        split_name: {
            "source": source_counts(frame, "source"),
            "original_source": source_counts(frame, "original_source"),
        }
        for split_name, frame in splits.items()
    }


def target_distribution_by_split(splits: dict[str, pd.DataFrame]) -> dict:
    """Summarize target range after final split."""

    return {split_name: numeric_stats(frame["neg_log10_affinity"]) for split_name, frame in splits.items()}


def added_lineage_counts(dedup: pd.DataFrame) -> dict:
    """Count final rows represented by less-strict or Patrick lineages."""

    return {
        "less_strict_representative_rows": int(
            (dedup["original_source"] == "less_strict_sabdab_affinity_v1").sum()
        ),
        "patrick_representative_rows": int(
            (dedup["original_source"] == "sabdab_affinity_search_735_audit").sum()
        ),
        "sabdab_supplement_representative_rows": int((dedup["source"] == "SAbDab_Supplement").sum()),
    }


def bool_all_zero(report: dict[str, int]) -> bool:
    """Return True when every overlap value is zero."""

    return all(value == 0 for value in report.values())


def build_report(
    source_frames: dict[str, pd.DataFrame],
    filter_reports: dict[str, dict[str, int]],
    merged: pd.DataFrame,
    dedup: pd.DataFrame,
    duplicates: pd.DataFrame,
    conflicts: pd.DataFrame,
    needs_sequence: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    less_strict_split_dir: Path,
) -> dict:
    """Build machine-readable build report."""

    antigen_overlap = overlap_report(splits, ["antigen_sequence"])
    triplet_overlap = overlap_report(splits, TRIPLET_COLUMNS)
    source_input_rows = {name: int(len(frame)) for name, frame in source_frames.items()}
    can_train_later = bool(len(dedup) and bool_all_zero(antigen_overlap) and bool_all_zero(triplet_overlap))
    return {
        "dataset_version": "unified_affinity_dataset_v1",
        "seed": SEED,
        "target_conflict_tolerance_log10_units": TARGET_TOLERANCE,
        "inputs": {
            "tdc_v1_clean_dir": str(TDC_V1_DIR.relative_to(PROJECT_ROOT)),
            "tdc_plus_sabdab_supplement_v1_dir": str(SUPPLEMENT_V1_DIR.relative_to(PROJECT_ROOT)),
            "less_strict_dir_used": str(less_strict_split_dir.relative_to(PROJECT_ROOT)),
            "patrick_new_candidate_rows": str(PATRICK_CANDIDATE_PATH.relative_to(PROJECT_ROOT)),
            "sequence_cache_dir": str(SEQUENCE_CACHE_DIR.relative_to(PROJECT_ROOT)),
            "raw_reference_files": {
                str(path.relative_to(PROJECT_ROOT)): bool(path.exists()) for path in RAW_REFERENCE_PATHS
            },
        },
        "source_input_rows_before_trainable_filter": source_input_rows,
        "trainable_filter_by_source": filter_reports,
        "merged_rows_before_triplet_dedup": int(len(merged)),
        "unique_triplets_before_dedup": int(triplet_key(merged).nunique()),
        "dedup_rows_after_conflict_removal": int(len(dedup)),
        "duplicate_rows_excluded": int(len(duplicates)),
        "conflicting_target_rows_excluded": int(len(conflicts)),
        "conflicting_target_triplets": int(triplet_key(conflicts).nunique()) if len(conflicts) else 0,
        "patrick_rows_needing_sequence_extraction": int(len(needs_sequence)),
        "final_increase_vs_tdc_plus_sabdab_supplement_v1_473_rows": int(len(dedup) - 473),
        "added_representative_rows_from_newer_sources": added_lineage_counts(dedup),
        "risk_flags_distribution": risk_flag_distribution(dedup),
        "final_source_counts": {
            "source": source_counts(dedup, "source"),
            "original_source": source_counts(dedup, "original_source"),
        },
        "split_sizes": {split_name: int(len(frame)) for split_name, frame in splits.items()},
        "target_distribution_by_split": target_distribution_by_split(splits),
        "source_counts_by_split": source_counts_by_split(splits),
        "split_overlap_checks": {
            "antigen_sequence": antigen_overlap,
            "heavy_light_antigen_triplet": triplet_overlap,
        },
        "output_files": {
            "train": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
            "val": str(VAL_PATH.relative_to(PROJECT_ROOT)),
            "test": str(TEST_PATH.relative_to(PROJECT_ROOT)),
            "all_unified_raw_merged": str(RAW_MERGED_PATH.relative_to(PROJECT_ROOT)),
            "all_unified_dedup": str(DEDUP_PATH.relative_to(PROJECT_ROOT)),
            "excluded_duplicates": str(EXCLUDED_DUPLICATES_PATH.relative_to(PROJECT_ROOT)),
            "conflicting_targets": str(CONFLICTING_TARGETS_PATH.relative_to(PROJECT_ROOT)),
            "needs_sequence_extraction": str(NEEDS_SEQUENCE_PATH.relative_to(PROJECT_ROOT)),
        },
        "decision": {
            "can_use_for_next_training_step": can_train_later,
            "reason": (
                "Final rows have three sequences and numeric neg_log10_affinity; final antigen_sequence "
                "and triplet overlap checks are zero."
                if can_train_later
                else "Dataset cannot be recommended until empty/overlap problems are resolved."
            ),
            "note": (
                "This is a new dataset-version comparison. It combines lineages and re-splits by "
                "antigen_sequence, so metrics should not be interpreted as only one source's row gain."
            ),
        },
    }


def markdown_table(counter: dict[str, int], first_header: str) -> list[str]:
    """Render a small two-column Markdown count table."""

    lines = [f"| {first_header} | rows |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value} |" for key, value in counter.items())
    return lines


def write_markdown(report: dict) -> None:
    """Write a readable processing report."""

    lines = [
        "# Unified Affinity Dataset v1 Processing Report",
        "",
        "## What This Build Does",
        "",
        "- Combines already cleaned/audited affinity sources into a new dataset version.",
        "- Requires heavy, light, antigen sequences and numeric `neg_log10_affinity`.",
        "- Deduplicates exact heavy+light+antigen triplets.",
        "- Removes triplets whose targets disagree beyond the configured tolerance.",
        "- Re-splits by `antigen_sequence` group to reduce leakage.",
        "- No model training is performed.",
        "",
        "## Inputs",
        "",
        f"- TDC v1: `{report['inputs']['tdc_v1_clean_dir']}`",
        f"- TDC + SAbDab supplement v1: `{report['inputs']['tdc_plus_sabdab_supplement_v1_dir']}`",
        f"- Less-strict SAbDab v1 path used: `{report['inputs']['less_strict_dir_used']}`",
        f"- Patrick candidate audit CSV: `{report['inputs']['patrick_new_candidate_rows']}`",
        f"- Patrick sequence cache only: `{report['inputs']['sequence_cache_dir']}`",
        "",
        "## Source Rows",
        "",
        *markdown_table(report["source_input_rows_before_trainable_filter"], "source frame"),
        "",
        f"- Trainable merged rows before triplet dedup: `{report['merged_rows_before_triplet_dedup']}`",
        f"- Unique triplets before dedup/conflict removal: `{report['unique_triplets_before_dedup']}`",
        f"- Final deduplicated trainable rows: `{report['dedup_rows_after_conflict_removal']}`",
        f"- Increase vs current largest 473-row dataset: `{report['final_increase_vs_tdc_plus_sabdab_supplement_v1_473_rows']}`",
        "",
        "## Dedup And Conflicts",
        "",
        f"- Duplicate rows written to excluded_duplicates.csv: `{report['duplicate_rows_excluded']}`",
        f"- Conflicting target rows written to conflicting_targets.csv: `{report['conflicting_target_rows_excluded']}`",
        f"- Conflicting target triplets: `{report['conflicting_target_triplets']}`",
        f"- Patrick candidate rows needing sequence extraction: `{report['patrick_rows_needing_sequence_extraction']}`",
        "",
        "## Final Lineage",
        "",
        *markdown_table(report["final_source_counts"]["original_source"], "representative original_source"),
        "",
        f"- Newer-source representative rows: `{report['added_representative_rows_from_newer_sources']}`",
        "",
        "## Risk Flags",
        "",
        *markdown_table(report["risk_flags_distribution"], "risk flag"),
        "",
        "## Final Split",
        "",
        f"- Split sizes: `{report['split_sizes']}`",
        f"- Target distribution by split: `{report['target_distribution_by_split']}`",
        f"- Source counts by split: `{report['source_counts_by_split']}`",
        f"- antigen_sequence overlap check: `{report['split_overlap_checks']['antigen_sequence']}`",
        f"- heavy+light+antigen triplet overlap check: `{report['split_overlap_checks']['heavy_light_antigen_triplet']}`",
        "",
        "## Decision",
        "",
        f"- Can this unified dataset be used for the next training step? `{report['decision']['can_use_for_next_training_step']}`",
        f"- Why: {report['decision']['reason']}",
        f"- Interpretation note: {report['decision']['note']}",
        "",
        "## Output Files",
        "",
    ]
    lines.extend(f"- `{name}`: `{path}`" for name, path in report["output_files"].items())
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    merged: pd.DataFrame,
    dedup: pd.DataFrame,
    duplicates: pd.DataFrame,
    conflicts: pd.DataFrame,
    needs_sequence: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> None:
    """Write only the new unified dataset directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(RAW_MERGED_PATH, index=False)
    dedup.to_csv(DEDUP_PATH, index=False)
    duplicates.to_csv(EXCLUDED_DUPLICATES_PATH, index=False)
    conflicts.to_csv(CONFLICTING_TARGETS_PATH, index=False)
    needs_sequence.to_csv(NEEDS_SEQUENCE_PATH, index=False)
    splits["train"].to_csv(TRAIN_PATH, index=False)
    splits["val"].to_csv(VAL_PATH, index=False)
    splits["test"].to_csv(TEST_PATH, index=False)


def print_summary(report: dict) -> None:
    """Print the high-signal terminal result."""

    print("Unified affinity dataset v1 build complete.")
    print(f"Source input rows: {report['source_input_rows_before_trainable_filter']}")
    print(f"Merged trainable rows before dedup: {report['merged_rows_before_triplet_dedup']}")
    print(f"Final unique trainable triplets: {report['dedup_rows_after_conflict_removal']}")
    print(f"Duplicate rows excluded: {report['duplicate_rows_excluded']}")
    print(f"Conflicting target rows excluded: {report['conflicting_target_rows_excluded']}")
    print(f"Patrick rows needing sequence extraction: {report['patrick_rows_needing_sequence_extraction']}")
    print(f"Split sizes: {report['split_sizes']}")
    print(f"Antigen overlap: {report['split_overlap_checks']['antigen_sequence']}")
    print(f"Triplet overlap: {report['split_overlap_checks']['heavy_light_antigen_triplet']}")
    print(f"Increase vs 473-row supplement v1: {report['final_increase_vs_tdc_plus_sabdab_supplement_v1_473_rows']}")
    print(f"Can use for next training step: {report['decision']['can_use_for_next_training_step']}")
    print(f"Markdown report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("No model training was run.")


def main() -> None:
    """Build unified_affinity_dataset_v1."""

    tdc_v1 = standardize_tdc_v1()
    supplement_v1 = standardize_supplement_v1()
    less_strict, less_strict_split_dir = standardize_less_strict()
    patrick, needs_sequence = standardize_patrick_candidates()
    source_frames = {
        "tdc_v1_clean": tdc_v1,
        "tdc_plus_sabdab_supplement_v1": supplement_v1,
        "less_strict_sabdab_affinity_v1": less_strict,
        "patrick_735_cached_candidates": patrick,
    }

    kept_frames = []
    filter_reports = {}
    for source_name, frame in source_frames.items():
        kept, filter_report = filter_trainable_rows(frame)
        kept_frames.append(kept)
        filter_reports[source_name] = filter_report

    merged = add_merged_ids(pd.concat(kept_frames, ignore_index=True))
    dedup, duplicates, conflicts = deduplicate_triplets(merged)
    splits = antigen_group_split(dedup)
    save_outputs(merged, dedup, duplicates, conflicts, needs_sequence, splits)
    report = build_report(
        source_frames,
        filter_reports,
        merged,
        dedup,
        duplicates,
        conflicts,
        needs_sequence,
        splits,
        less_strict_split_dir,
    )
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report)
    print_summary(report)


if __name__ == "__main__":
    main()
