"""Extract CDR-aware features for the TDC v1 affinity dataset.

中文人话说明：
这个脚本是 feasibility check，不训练模型。
它回答一个问题：

    当前 TDC v1 的 heavy/light sequences 能不能稳定得到 CDR 特征？

优先后端：
- 如果当前 Python 环境装了 AbNumber，它会使用 AbNumber。
  AbNumber 是基于 ANARCI 的 Python API，能给出标准 CDR1/CDR2/CDR3。

fallback：
- 如果 AbNumber/ANARCI 环境还没准备好，脚本不会崩。
  它会使用一个明确标记为 imgt_index_heuristic 的粗略预检查方法。
  这个 fallback 只用于“先看数据和图能不能跑通”，不是最终标准 CDR numbering。
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "antigen_group_split"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "cdr_features"
SPLITS = ["train", "val", "test"]

CDR_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]
CDR_LENGTH_COLUMNS = [f"{column}_len" for column in CDR_COLUMNS]
BACKEND_STANDARD = "abnumber_anarci_imgt"
BACKEND_HEURISTIC = "imgt_index_heuristic"
BACKEND_FAILED = "failed"

# IMGT-like fallback ranges on raw sequence indices:
# - IMGT CDR1 positions roughly 27-38
# - IMGT CDR2 positions roughly 56-65
# - IMGT CDR3 positions roughly 105-117
# Python slice end is exclusive, so 1-indexed 27-38 becomes [26:38].
HEURISTIC_CDR_SLICES = {
    "CDR1": slice(26, 38),
    "CDR2": slice(55, 65),
    "CDR3": slice(104, 117),
}


def load_abnumber_chain():
    """Return AbNumber Chain class if available, otherwise None."""

    try:
        from abnumber import Chain
    except ImportError:
        return None
    return Chain


def load_tdc_split(split_name: str) -> pd.DataFrame:
    """Read one TDC v1 split and add its split label."""

    path = INPUT_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Build TDC v1 dataset first.")

    dataframe = pd.read_csv(path)
    required_columns = {
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        "neg_log10_affinity",
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    dataframe = dataframe.copy()
    dataframe["split"] = split_name
    return dataframe


def extract_with_abnumber(sequence: str, expected_chain: str, Chain) -> tuple[dict | None, str | None]:
    """Extract three CDRs from one chain using AbNumber/ANARCI.

    expected_chain is "heavy" or "light".
    We validate chain type so a swapped or malformed sequence is reported clearly.
    """

    try:
        chain = Chain(str(sequence), scheme="imgt")
    except Exception as error:
        return None, f"abnumber_numbering_failed: {error}"

    if expected_chain == "heavy" and not chain.is_heavy_chain():
        return None, f"abnumber_chain_type_mismatch: expected heavy, got {chain.chain_type}"
    if expected_chain == "light" and not chain.is_light_chain():
        return None, f"abnumber_chain_type_mismatch: expected light, got {chain.chain_type}"

    return {
        "CDR1": str(chain.cdr1_seq),
        "CDR2": str(chain.cdr2_seq),
        "CDR3": str(chain.cdr3_seq),
    }, None


def extract_with_heuristic(sequence: str) -> tuple[dict | None, str | None]:
    """Extract rough IMGT-like CDR slices when standard numbering is unavailable.

    中文人话说明：
    这个方法假设 variable region 在 sequence 前半段，按常见 IMGT 位置粗切片。
    它不是标准 ANARCI numbering，所以输出会在 report 里明确标为 heuristic。
    """

    sequence = str(sequence).strip().upper()
    minimum_length = HEURISTIC_CDR_SLICES["CDR3"].stop
    if len(sequence) < minimum_length:
        return None, f"heuristic_sequence_too_short_for_cdr3: length={len(sequence)}"

    cdrs = {name: sequence[positions] for name, positions in HEURISTIC_CDR_SLICES.items()}
    if any(not cdr for cdr in cdrs.values()):
        return None, "heuristic_empty_cdr_slice"
    return cdrs, None


def extract_chain_cdrs(sequence: str, expected_chain: str, Chain) -> tuple[dict | None, str | None, str]:
    """Extract CDRs from one chain and return result, error, backend name.

    中文人话说明：
    先尝试标准 AbNumber/ANARCI。
    如果标准后端因 hmmscan、numbering failure 等原因失败，
    再退回 heuristic fallback，并把标准后端错误写进 error 字段。
    """

    if Chain is not None:
        cdrs, standard_error = extract_with_abnumber(sequence, expected_chain, Chain)
        if cdrs is not None:
            return cdrs, None, BACKEND_STANDARD

        fallback_cdrs, fallback_error = extract_with_heuristic(sequence)
        if fallback_cdrs is not None:
            return fallback_cdrs, f"standard_backend_failed_then_fallback:{standard_error}", BACKEND_HEURISTIC
        return (
            None,
            f"standard_backend_failed:{standard_error} | heuristic_failed:{fallback_error}",
            BACKEND_FAILED,
        )

    cdrs, error = extract_with_heuristic(sequence)
    if cdrs is None:
        return None, error, BACKEND_FAILED
    return cdrs, None, BACKEND_HEURISTIC


def build_feature_row(row: dict, Chain) -> dict:
    """Add CDR features and robust extraction status to one dataframe row."""

    heavy_cdrs, heavy_error, heavy_backend = extract_chain_cdrs(
        row["heavy_sequence"],
        "heavy",
        Chain,
    )
    light_cdrs, light_error, light_backend = extract_chain_cdrs(
        row["light_sequence"],
        "light",
        Chain,
    )

    output = dict(row)
    output["heavy_cdr_backend"] = heavy_backend
    output["light_cdr_backend"] = light_backend
    output["cdr_backend"] = heavy_backend if heavy_backend == light_backend else f"{heavy_backend}|{light_backend}"
    output["heavy_cdr_status"] = "success" if heavy_cdrs is not None else "failed"
    output["light_cdr_status"] = "success" if light_cdrs is not None else "failed"

    errors = []
    if heavy_error:
        errors.append(f"heavy:{heavy_error}")
    if light_error:
        errors.append(f"light:{light_error}")

    if heavy_cdrs is not None and light_cdrs is not None:
        output["cdr_extract_status"] = "success"
    elif heavy_cdrs is not None or light_cdrs is not None:
        output["cdr_extract_status"] = "partial_failure"
    else:
        output["cdr_extract_status"] = "failed"
    output["cdr_extract_error"] = " | ".join(errors)

    for prefix, cdrs in [("H", heavy_cdrs), ("L", light_cdrs)]:
        for cdr_name in ["CDR1", "CDR2", "CDR3"]:
            column_name = f"{prefix}{cdr_name}"
            cdr_sequence = "" if cdrs is None else cdrs[cdr_name]
            output[column_name] = cdr_sequence
            output[f"{column_name}_len"] = len(cdr_sequence)

    output["heavy_len"] = len(str(row["heavy_sequence"]))
    output["light_len"] = len(str(row["light_sequence"]))
    output["antigen_len"] = len(str(row["antigen_sequence"]))
    return output


def summarize_lengths(dataframe: pd.DataFrame, columns: list[str]) -> dict:
    """Return JSON-friendly min/max/mean for length columns."""

    summary = {}
    for column_name in columns:
        values = pd.to_numeric(dataframe[column_name], errors="coerce")
        summary[column_name] = {
            "min": int(values.min()) if len(values) else None,
            "max": int(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
        }
    return summary


def percentage(count: int, total: int) -> str:
    """Format count and percentage for markdown."""

    rate = 0.0 if total == 0 else count / total
    return f"{count}/{total} ({rate:.2%})"


def write_backend_comparison(report: dict, previous_report: dict | None) -> None:
    """Write human-readable heuristic vs standard backend summary."""

    comparison_path = OUTPUT_DIR / "cdr_backend_comparison.md"
    current_status = report["cdr_extract_status_counts"]
    current_total = report["total_rows"]
    standard_chain_count = (
        report["heavy_backend_counts"].get(BACKEND_STANDARD, 0)
        + report["light_backend_counts"].get(BACKEND_STANDARD, 0)
    )

    heuristic_report = None
    if BACKEND_HEURISTIC in report["cdr_backend_counts"]:
        heuristic_report = report
    elif previous_report and BACKEND_HEURISTIC in previous_report.get("cdr_backend_counts", {}):
        heuristic_report = previous_report

    lines = [
        "# TDC v1 CDR Backend Comparison",
        "",
        "## Purpose",
        "",
        "This file compares the rough heuristic feasibility check with a standard AbNumber/ANARCI CDR extraction run.",
        "",
        "## Heuristic feasibility check",
        "",
    ]

    if heuristic_report is None:
        lines.extend(
            [
                "No heuristic report snapshot was available in the current or previous `cdr_report.json`.",
                "",
            ]
        )
    else:
        heuristic_status = heuristic_report["cdr_extract_status_counts"]
        heuristic_total = heuristic_report["total_rows"]
        lines.extend(
            [
                f"- Backend: `{BACKEND_HEURISTIC}`",
                f"- Total rows: {heuristic_total}",
                f"- Success: {percentage(heuristic_status.get('success', 0), heuristic_total)}",
                f"- Partial failure: {percentage(heuristic_status.get('partial_failure', 0), heuristic_total)}",
                f"- Failed: {percentage(heuristic_status.get('failed', 0), heuristic_total)}",
                "- Interpretation: useful for pipeline feasibility only; fixed raw-index slices are not standard CDR annotation.",
                "",
            ]
        )

    lines.extend(["## Standard AbNumber/ANARCI result", ""])
    if standard_chain_count:
        lines.extend(
            [
                f"- Backend label: `{BACKEND_STANDARD}`",
                f"- Total rows in current run: {current_total}",
                f"- Full-row success: {percentage(current_status.get('success', 0), current_total)}",
                f"- Heavy chains using standard backend: {report['heavy_backend_counts'].get(BACKEND_STANDARD, 0)}",
                f"- Light chains using standard backend: {report['light_backend_counts'].get(BACKEND_STANDARD, 0)}",
                "- Interpretation: these standard-numbered CDRs can be considered for a formal CDR-aware feature baseline after failed rows are reviewed.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- Standard result is not available yet in the current run.",
                "- The pip-only isolated environment reached AbNumber but failed standard numbering because `hmmscan`/HMMER was unavailable.",
                "- Recommended next environment: Bioconda AbNumber/ANARCI with HMMER, then rerun extraction.",
                "",
            ]
        )

    lines.extend(
        [
            "## Decision",
            "",
            "- Formal CDR-aware feature baseline: use the standard AbNumber/ANARCI output, not heuristic fixed-index slices.",
            "- Can we enter a simple CDR feature baseline now? Only after the standard backend run succeeds with acceptable coverage.",
            "",
            "## Current run",
            "",
            f"- Row backend counts: `{report['cdr_backend_counts']}`",
            f"- Heavy backend counts: `{report['heavy_backend_counts']}`",
            f"- Light backend counts: `{report['light_backend_counts']}`",
            f"- Extraction status counts: `{current_status}`",
            "",
        ]
    )

    comparison_path.write_text("\n".join(lines), encoding="utf-8")


def load_previous_report() -> dict | None:
    """Read old report before overwrite so comparison markdown can keep context."""

    report_path = OUTPUT_DIR / "cdr_report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_outputs(split_frames: dict[str, pd.DataFrame], all_features: pd.DataFrame) -> None:
    """Write split feature CSVs, all feature CSV, failures, and report."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, dataframe in split_frames.items():
        dataframe.to_csv(OUTPUT_DIR / f"{split_name}_cdr.csv", index=False)

    all_features.to_csv(OUTPUT_DIR / "all_cdr.csv", index=False)

    failed = all_features[all_features["cdr_extract_status"] != "success"].copy()
    failed.to_csv(OUTPUT_DIR / "failed_cdr_records.csv", index=False)

    previous_report = load_previous_report()
    report = {
        "cdr_backend_counts": all_features["cdr_backend"].value_counts(dropna=False).to_dict(),
        "heavy_backend_counts": all_features["heavy_cdr_backend"].value_counts(dropna=False).to_dict(),
        "light_backend_counts": all_features["light_cdr_backend"].value_counts(dropna=False).to_dict(),
        "rows_by_split": {split: int(len(df)) for split, df in split_frames.items()},
        "total_rows": int(len(all_features)),
        "cdr_extract_status_counts": all_features["cdr_extract_status"].value_counts(dropna=False).to_dict(),
        "heavy_cdr_status_counts": all_features["heavy_cdr_status"].value_counts(dropna=False).to_dict(),
        "light_cdr_status_counts": all_features["light_cdr_status"].value_counts(dropna=False).to_dict(),
        "failed_rows": int(len(failed)),
        "failed_error_counts": failed["cdr_extract_error"].value_counts(dropna=False).to_dict(),
        "cdr_length_summary": summarize_lengths(all_features, CDR_LENGTH_COLUMNS),
        "sequence_length_summary": summarize_lengths(all_features, ["heavy_len", "light_len", "antigen_len"]),
        "backend_note": (
            "abnumber_anarci_imgt is standard numbering when AbNumber is installed. "
            "imgt_index_heuristic is a rough fallback for feasibility checks only."
        ),
    }

    with open(OUTPUT_DIR / "cdr_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    write_backend_comparison(report, previous_report)

    print("TDC v1 CDR feature extraction complete.")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"CDR backend counts: {report['cdr_backend_counts']}")
    print(f"CDR extraction status counts: {report['cdr_extract_status_counts']}")
    print(f"Failed rows: {report['failed_rows']}")
    print("Backend note:")
    print(f"  {report['backend_note']}")
    print()
    print("Next step suggestion:")
    if BACKEND_HEURISTIC in report["cdr_backend_counts"]:
        print("  Current outputs are feasibility-only heuristic CDR slices.")
        print("  Install/configure AbNumber + ANARCI for standard numbering before modeling CDR features.")
    else:
        print("  Standard ANARCI-backed CDR extraction worked; inspect CDR distributions next.")


def main() -> None:
    """Run CDR feature extraction for all TDC v1 splits."""

    Chain = load_abnumber_chain()
    if Chain is None:
        print("AbNumber is not installed; using imgt_index_heuristic fallback.")
        print("For standard ANARCI-backed CDR extraction, install/configure AbNumber + ANARCI first.")
        print()
    else:
        print("Using AbNumber/ANARCI IMGT backend for CDR extraction.")
        print()

    split_frames = {}
    for split_name in SPLITS:
        input_frame = load_tdc_split(split_name)
        rows = [build_feature_row(row, Chain) for row in input_frame.to_dict("records")]
        split_frames[split_name] = pd.DataFrame(rows)

    all_features = pd.concat(split_frames.values(), ignore_index=True)
    write_outputs(split_frames, all_features)


if __name__ == "__main__":
    main()
