"""Analyze and visualize TDC v1 CDR feature extraction outputs.

:

 extract_cdr_features_tdc_v1.py  all_cdr.csv,
:

1. heavy/light/all-six CDR ?
2. CDR  antigen ?
3. CDR3  affinity target ?
4.  target ?
"""

from __future__ import annotations

import os
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed_affinity" / "tdc_v1" / "cdr_features"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "cdr_tdc_v1"

# matplotlib ; outputs , home 
MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / "matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


CDR_LENGTH_COLUMNS = [
    "HCDR1_len",
    "HCDR2_len",
    "HCDR3_len",
    "LCDR1_len",
    "LCDR2_len",
    "LCDR3_len",
]


def numeric_summary(series: pd.Series) -> dict:
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


def success_rate(dataframe: pd.DataFrame, column_name: str) -> float:
    """Return success percentage for one status column."""

    return float((dataframe[column_name] == "success").mean())


def cdr_length_summary(dataframe: pd.DataFrame) -> dict:
    """Summarize each CDR length column."""

    return {column_name: numeric_summary(dataframe[column_name]) for column_name in CDR_LENGTH_COLUMNS}


def save_histogram(values: pd.Series, title: str, x_label: str, output_path: Path, label: str) -> None:
    """Save one beginner-friendly histogram."""

    plt.figure(figsize=(8, 6))
    plt.hist(pd.to_numeric(values, errors="coerce").dropna(), bins=25, alpha=0.8, color="#4C78A8", label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_all_cdr_boxplot(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save boxplot comparing all CDR length distributions.

    boxplot  6  CDR 
    """

    values = [dataframe[column].astype(float).tolist() for column in CDR_LENGTH_COLUMNS]
    plt.figure(figsize=(9, 6))
    plt.boxplot(values, tick_labels=CDR_LENGTH_COLUMNS)
    plt.title("All CDR Length Boxplot (TDC v1)")
    plt.xlabel("CDR")
    plt.ylabel("Length")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_cdr_length_vs_target(dataframe: pd.DataFrame, length_column: str, output_path: Path) -> None:
    """Save scatter plot for one CDR length versus affinity target."""

    plt.figure(figsize=(8, 6))
    plt.scatter(
        dataframe[length_column].astype(float),
        dataframe["neg_log10_affinity"].astype(float),
        alpha=0.65,
        color="#54A24B",
        label="samples",
    )
    plt.title(f"{length_column} vs neg_log10_affinity")
    plt.xlabel(length_column)
    plt.ylabel("neg_log10_affinity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_success_failed_target_histogram(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Compare target distribution for full CDR success vs failed/partial rows."""

    success_values = dataframe.loc[
        dataframe["cdr_extract_status"] == "success",
        "neg_log10_affinity",
    ].astype(float)
    failed_values = dataframe.loc[
        dataframe["cdr_extract_status"] != "success",
        "neg_log10_affinity",
    ].astype(float)

    all_values = dataframe["neg_log10_affinity"].astype(float)
    bins = np.linspace(all_values.min(), all_values.max(), 20)

    plt.figure(figsize=(8, 6))
    plt.hist(success_values, bins=bins, alpha=0.65, color="#4C78A8", label="all six CDR success")
    if len(failed_values):
        plt.hist(failed_values, bins=bins, alpha=0.65, color="#E45756", label="failed/partial")
    plt.title("neg_log10_affinity Distribution: CDR Success vs Failed")
    plt.xlabel("neg_log10_affinity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_figures(dataframe: pd.DataFrame) -> list[str]:
    """Create all requested PNG figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    figure_specs = [
        ("HCDR3_len", "HCDR3 Length Distribution", "HCDR3 length", "hcdr3_length_distribution.png"),
        ("LCDR3_len", "LCDR3 Length Distribution", "LCDR3 length", "lcdr3_length_distribution.png"),
        ("antigen_len", "Antigen Length Distribution", "Antigen length", "antigen_length_distribution.png"),
    ]
    for column_name, title, x_label, filename in figure_specs:
        path = FIGURE_DIR / filename
        save_histogram(dataframe[column_name], title, x_label, path, "samples")
        paths.append(str(path.relative_to(PROJECT_ROOT)))

    boxplot_path = FIGURE_DIR / "all_cdr_length_boxplot.png"
    save_all_cdr_boxplot(dataframe, boxplot_path)
    paths.append(str(boxplot_path.relative_to(PROJECT_ROOT)))

    for length_column, filename in [
        ("HCDR3_len", "hcdr3_len_vs_neg_log10_affinity.png"),
        ("LCDR3_len", "lcdr3_len_vs_neg_log10_affinity.png"),
    ]:
        path = FIGURE_DIR / filename
        save_cdr_length_vs_target(dataframe, length_column, path)
        paths.append(str(path.relative_to(PROJECT_ROOT)))

    target_path = FIGURE_DIR / "target_distribution_success_vs_failed.png"
    save_success_failed_target_histogram(dataframe, target_path)
    paths.append(str(target_path.relative_to(PROJECT_ROOT)))
    return paths


def main() -> None:
    """Print/save CDR feature report and visualizations."""

    input_path = INPUT_DIR / "all_cdr.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find {input_path}. Run extract_cdr_features_tdc_v1.py first.")

    dataframe = pd.read_csv(input_path).fillna("")
    failed = dataframe[dataframe["cdr_extract_status"] != "success"].copy()

    report = {
        "total_rows": int(len(dataframe)),
        "heavy_cdr_success_rate": success_rate(dataframe, "heavy_cdr_status"),
        "light_cdr_success_rate": success_rate(dataframe, "light_cdr_status"),
        "all_six_cdr_success_rate": success_rate(dataframe, "cdr_extract_status"),
        "failed_count_by_reason": failed["cdr_extract_error"].value_counts(dropna=False).to_dict(),
        "cdr_backend_counts": dataframe["cdr_backend"].value_counts(dropna=False).to_dict(),
        "cdr_length_summary": cdr_length_summary(dataframe),
        "antigen_length_summary": numeric_summary(dataframe["antigen_len"]),
        "figures": save_figures(dataframe),
    }

    report_path = INPUT_DIR / "cdr_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("TDC v1 CDR feature analysis")
    print(f"Total rows: {report['total_rows']}")
    print(f"Heavy CDR success rate: {report['heavy_cdr_success_rate']:.2%}")
    print(f"Light CDR success rate: {report['light_cdr_success_rate']:.2%}")
    print(f"All six CDR success rate: {report['all_six_cdr_success_rate']:.2%}")
    print(f"CDR backend counts: {report['cdr_backend_counts']}")
    print(f"Failed count by reason: {report['failed_count_by_reason']}")
    print(f"CDR length summary: {report['cdr_length_summary']}")
    print(f"Antigen length summary: {report['antigen_length_summary']}")
    print(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")
    print("Generated figures:")
    for path in report["figures"]:
        print(f"  {path}")
    print()
    print("Next step suggestion:")
    if "imgt_index_heuristic" in report["cdr_backend_counts"]:
        print("  Treat these plots as feasibility checks only; install ANARCI/AbNumber for standard CDRs next.")
    elif report["all_six_cdr_success_rate"] > 0.9:
        print("  CDR extraction coverage is high; consider building a CDR-aware model feature experiment.")
    else:
        print("  Inspect failed chains before using CDR-aware model features.")


if __name__ == "__main__":
    main()
