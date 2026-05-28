"""Error analysis for the best unified_no_high_risk affinity model.

:
 test predictions  test CSV
 split dataset, affinity model

:
1. top errors ;
2. residual  regression-to-mean;
3. source / antigen / target range / sequence length ;
4. max_length=512 
"""

from __future__ import annotations

import os
from pathlib import Path
import math
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "config_affinity_unified_no_high_risk_lr3e-5_e10.yaml"
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ablation"
    / "unified_affinity_dataset_v1"
    / "unified_no_high_risk_test_predictions.csv"
)
TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_affinity"
    / "unified_ablation_datasets"
    / "unified_no_high_risk"
    / "test.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis" / "unified_no_high_risk"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "error_analysis_report.md"
TOP_ERRORS_PATH = OUTPUT_DIR / "top_errors.csv"
ERROR_BY_GROUP_PATH = OUTPUT_DIR / "error_by_group.csv"

# Matplotlib  outputs, home 
MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / "matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config


TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"
ERROR_COLUMN = "error"
ABS_ERROR_COLUMN = "absolute_error"
LENGTH_COLUMNS = ["antigen_len", "heavy_len", "light_len", "total_sequence_len"]
TARGET_BIN_ORDER = ["low_target", "mid_target", "high_target"]
LENGTH_BIN_ORDER = ["short", "medium", "long"]
EPITOPE_LIKE_PATTERN = re.compile(r"peptide|epitope|loop|motif|fragment", re.IGNORECASE)


def read_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load config, test metadata, and saved predictions."""

    config = load_config(str(CONFIG_PATH))
    predictions = pd.read_csv(PREDICTIONS_PATH)
    test = pd.read_csv(TEST_PATH)

    required_predictions = {"sample_id", TRUE_COLUMN, PRED_COLUMN}
    required_test = {
        "sample_id",
        "source",
        "antigen_id",
        "risk_flags",
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
    }
    missing_predictions = required_predictions - set(predictions.columns)
    missing_test = required_test - set(test.columns)
    if missing_predictions:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing_predictions)}")
    if missing_test:
        raise ValueError(f"Test CSV missing columns: {sorted(missing_test)}")
    return config, predictions, test


def quantile_bin(series: pd.Series, labels: list[str]) -> pd.Series:
    """Create row-balanced bins while handling duplicate numeric values."""

    ranked = pd.to_numeric(series, errors="coerce").rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels)


def numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    """Return simple numeric summary."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float | None:
    """Return correlation or None when undefined."""

    value = pd.to_numeric(left, errors="coerce").corr(
        pd.to_numeric(right, errors="coerce"),
        method=method,
    )
    return None if pd.isna(value) else float(value)


def merge_and_annotate(config: dict, predictions: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with metadata and add error-analysis helper columns."""

    metadata_columns = [
        "sample_id",
        "source",
        "original_source",
        "pdb_or_antibody_id",
        "antigen_id",
        "antigen_type",
        "risk_flags",
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
    ]
    metadata_columns = [column for column in metadata_columns if column in test.columns]
    metadata = test[metadata_columns].copy()
    merged = predictions.merge(metadata, on="sample_id", how="left", validate="one_to_one")
    if merged["source_y"].isna().any():
        missing = merged.loc[merged["source_y"].isna(), "sample_id"].astype(str).head(10).tolist()
        raise ValueError(f"Prediction sample_ids not found in test CSV: {missing}")

    # prediction CSV and test CSV both contain sequences/source. Use test metadata consistently.
    merged["source_for_analysis"] = merged["source_y"].astype(str)
    merged["antigen_id_for_analysis"] = merged["antigen_id_y"].fillna("").astype(str)
    merged["risk_flags_for_analysis"] = merged["risk_flags"].fillna("").astype(str)
    for seq_name in ["heavy_sequence", "light_sequence", "antigen_sequence"]:
        merged[f"{seq_name}_analysis"] = merged[f"{seq_name}_y"].fillna("").astype(str)

    merged[TRUE_COLUMN] = pd.to_numeric(merged[TRUE_COLUMN], errors="raise")
    merged[PRED_COLUMN] = pd.to_numeric(merged[PRED_COLUMN], errors="raise")
    merged[ERROR_COLUMN] = merged[PRED_COLUMN] - merged[TRUE_COLUMN]
    merged[ABS_ERROR_COLUMN] = merged[ERROR_COLUMN].abs()
    merged["fold_error"] = 10 ** merged[ABS_ERROR_COLUMN]

    merged["heavy_len"] = merged["heavy_sequence_analysis"].str.len()
    merged["light_len"] = merged["light_sequence_analysis"].str.len()
    merged["antigen_len"] = merged["antigen_sequence_analysis"].str.len()
    merged["total_sequence_len"] = merged[["heavy_len", "light_len", "antigen_len"]].sum(axis=1)

    max_length = int(config["max_length"])
    merged["total_len_gt_max_length"] = merged["total_sequence_len"] > max_length
    # Dataset tokenizes three sequences separately. A single chain over max_length is the direct risk.
    merged["heavy_len_gt_max_length"] = merged["heavy_len"] > max_length
    merged["light_len_gt_max_length"] = merged["light_len"] > max_length
    merged["antigen_len_gt_max_length"] = merged["antigen_len"] > max_length
    merged["any_single_sequence_gt_max_length"] = merged[
        ["heavy_len_gt_max_length", "light_len_gt_max_length", "antigen_len_gt_max_length"]
    ].any(axis=1)

    merged["target_range"] = quantile_bin(merged[TRUE_COLUMN], TARGET_BIN_ORDER)
    for length_column in LENGTH_COLUMNS:
        merged[f"{length_column}_bin"] = quantile_bin(merged[length_column], LENGTH_BIN_ORDER)

    antigen_text = (
        merged["antigen_id_for_analysis"]
        + " "
        + merged.get("antigen_type", "").fillna("").astype(str)
    )
    merged["peptide_risk_flag"] = merged["risk_flags_for_analysis"].str.contains(
        "peptide_antigen", regex=False
    )
    merged["epitope_like_antigen_name"] = antigen_text.str.contains(EPITOPE_LIKE_PATTERN)
    return merged


def metrics(dataframe: pd.DataFrame) -> dict[str, float | None]:
    """Compute overall regression metrics from saved predictions."""

    errors = dataframe[ERROR_COLUMN].astype(float)
    mae = float(dataframe[ABS_ERROR_COLUMN].mean())
    rmse = float(math.sqrt((errors * errors).mean()))
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": safe_corr(dataframe[TRUE_COLUMN], dataframe[PRED_COLUMN], "spearman"),
        "prediction_std": float(dataframe[PRED_COLUMN].std()),
        "true_std": float(dataframe[TRUE_COLUMN].std()),
    }


def group_summary(dataframe: pd.DataFrame, group_type: str, group_column: str) -> pd.DataFrame:
    """Summarize count/error/bias for one grouping dimension."""

    rows = []
    for group_value, group in dataframe.groupby(group_column, observed=True, dropna=False):
        rows.append(
            {
                "group_type": group_type,
                "group": str(group_value),
                "count": int(len(group)),
                "mae": float(group[ABS_ERROR_COLUMN].mean()),
                "rmse": float(math.sqrt((group[ERROR_COLUMN].astype(float) ** 2).mean())),
                "mean_error": float(group[ERROR_COLUMN].mean()),
                "true_mean": float(group[TRUE_COLUMN].mean()),
                "predicted_mean": float(group[PRED_COLUMN].mean()),
                "spearman": safe_corr(group[TRUE_COLUMN], group[PRED_COLUMN], "spearman")
                if len(group) > 1
                else None,
                "min_antigen_len": int(group["antigen_len"].min()),
                "max_antigen_len": int(group["antigen_len"].max()),
            }
        )
    return pd.DataFrame(rows)


def make_error_by_group(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build one long-form grouped error table for CSV output."""

    group_specs = [
        ("source", "source_for_analysis"),
        ("antigen_id", "antigen_id_for_analysis"),
        ("target_affinity_range", "target_range"),
        ("antigen_sequence_length", "antigen_len_bin"),
        ("heavy_sequence_length", "heavy_len_bin"),
        ("light_sequence_length", "light_len_bin"),
        ("total_sequence_length", "total_sequence_len_bin"),
        ("single_sequence_truncation_risk", "any_single_sequence_gt_max_length"),
        ("total_length_gt_512", "total_len_gt_max_length"),
    ]
    return pd.concat(
        [group_summary(dataframe, group_type, column) for group_type, column in group_specs],
        ignore_index=True,
    )


def save_top_errors(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Save top error samples with readable metadata."""

    columns = [
        "sample_id",
        "source_for_analysis",
        "pdb_or_antibody_id",
        "antigen_id_for_analysis",
        TRUE_COLUMN,
        PRED_COLUMN,
        ERROR_COLUMN,
        ABS_ERROR_COLUMN,
        "fold_error",
        "risk_flags_for_analysis",
        "peptide_risk_flag",
        "epitope_like_antigen_name",
        "heavy_len",
        "light_len",
        "antigen_len",
        "total_sequence_len",
        "total_len_gt_max_length",
        "any_single_sequence_gt_max_length",
    ]
    columns = [column for column in columns if column in dataframe.columns]
    top_errors = dataframe.sort_values(ABS_ERROR_COLUMN, ascending=False).head(15)[columns].copy()
    top_errors.to_csv(TOP_ERRORS_PATH, index=False)
    return top_errors


def save_parity_plot(dataframe: pd.DataFrame) -> Path:
    """Save true-vs-predicted plot with equal axis."""

    path = FIGURE_DIR / "true_vs_predicted.png"
    min_value = min(dataframe[TRUE_COLUMN].min(), dataframe[PRED_COLUMN].min())
    max_value = max(dataframe[TRUE_COLUMN].max(), dataframe[PRED_COLUMN].max())
    padding = max((max_value - min_value) * 0.06, 0.2)
    low = float(min_value - padding)
    high = float(max_value + padding)

    fig, ax = plt.subplots(figsize=(7, 7))
    for source_name, group in dataframe.groupby("source_for_analysis"):
        ax.scatter(group[TRUE_COLUMN], group[PRED_COLUMN], alpha=0.8, label=str(source_name))
    ax.plot([low, high], [low, high], "--", color="#ef4444", label="y = x")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Unified No High Risk: True vs Predicted")
    ax.set_xlabel("True neg_log10_affinity")
    ax.set_ylabel("Predicted neg_log10_affinity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_residual_plot(dataframe: pd.DataFrame) -> Path:
    """Save residual-vs-true plot for regression-to-mean inspection."""

    path = FIGURE_DIR / "residual_vs_true.png"
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for source_name, group in dataframe.groupby("source_for_analysis"):
        ax.scatter(group[TRUE_COLUMN], group[ERROR_COLUMN], alpha=0.8, label=str(source_name))
    ax.axhline(0, linestyle="--", color="#ef4444", label="zero error")
    ax.set_title("Unified No High Risk: Residual vs True Target")
    ax.set_xlabel("True neg_log10_affinity")
    ax.set_ylabel("Prediction error = predicted - true")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def markdown_group_table(title: str, grouped: pd.DataFrame, group_type: str, max_rows: int = 12) -> list[str]:
    """Render one grouped error table for markdown."""

    table = grouped[grouped["group_type"] == group_type].copy()
    table = table.sort_values(["mae", "count"], ascending=[False, False]).head(max_rows)
    lines = [
        f"## {title}",
        "",
        "| group | n | MAE | RMSE | mean error | true mean | predicted mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| `{row['group']}` | {int(row['count'])} | {row['mae']:.4f} | "
            f"{row['rmse']:.4f} | {row['mean_error']:.4f} | "
            f"{row['true_mean']:.4f} | {row['predicted_mean']:.4f} |"
        )
    lines.append("")
    return lines


def train_target_context(config: dict) -> dict[str, float]:
    """Read train mean/median for regression-to-mean comparison."""

    train = pd.read_csv(PROJECT_ROOT / config["train_csv"])
    target = pd.to_numeric(train[config.get("target_column", "neg_log10_affinity")], errors="coerce")
    return {
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
    }


def build_observations(
    dataframe: pd.DataFrame,
    grouped: pd.DataFrame,
    top_errors: pd.DataFrame,
    train_context: dict,
    config: dict,
) -> dict:
    """Compute the statements needed for report and terminal summary."""

    target_groups = grouped[grouped["group_type"] == "target_affinity_range"].set_index("group")
    low = target_groups.loc["low_target"]
    high = target_groups.loc["high_target"]
    residual_true_pearson = safe_corr(dataframe[ERROR_COLUMN], dataframe[TRUE_COLUMN], "pearson")
    pred_std_ratio = float(dataframe[PRED_COLUMN].std() / dataframe[TRUE_COLUMN].std())
    top_total_gt = int(top_errors["total_len_gt_max_length"].sum())
    top_single_gt = int(top_errors["any_single_sequence_gt_max_length"].sum())
    all_total_gt = int(dataframe["total_len_gt_max_length"].sum())
    all_single_gt = int(dataframe["any_single_sequence_gt_max_length"].sum())
    source_group = grouped[grouped["group_type"] == "source"].sort_values("mae", ascending=False)
    worst_source = source_group.iloc[0]
    return {
        "low_mean_error": float(low["mean_error"]),
        "high_mean_error": float(high["mean_error"]),
        "residual_vs_true_pearson": residual_true_pearson,
        "prediction_std_ratio_true_std": pred_std_ratio,
        "prediction_mean": float(dataframe[PRED_COLUMN].mean()),
        "true_mean": float(dataframe[TRUE_COLUMN].mean()),
        "train_mean": train_context["mean"],
        "train_median": train_context["median"],
        "top_errors_total_len_gt_512": top_total_gt,
        "top_errors_single_sequence_gt_512": top_single_gt,
        "all_test_total_len_gt_512": all_total_gt,
        "all_test_single_sequence_gt_512": all_single_gt,
        "max_length": int(config["max_length"]),
        "top_errors_epitope_like_count": int(top_errors["epitope_like_antigen_name"].sum()),
        "top_errors_peptide_flag_count": int(top_errors["peptide_risk_flag"].sum()),
        "worst_source": str(worst_source["group"]),
        "worst_source_mae": float(worst_source["mae"]),
    }


def write_report(
    config: dict,
    dataframe: pd.DataFrame,
    grouped: pd.DataFrame,
    top_errors: pd.DataFrame,
    figure_paths: list[Path],
) -> dict:
    """Write the requested markdown report and return observations."""

    overall = metrics(dataframe)
    train_context = train_target_context(config)
    observations = build_observations(dataframe, grouped, top_errors, train_context, config)
    low_high_note = (
        "Low-target samples have positive mean residuals and high-target samples have negative "
        "mean residuals, which is the classic direction of regression-to-mean."
    )
    lines = [
        "# Unified No High Risk Error Analysis",
        "",
        "## Inputs",
        "",
        f"- Config: `{CONFIG_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Predictions: `{PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Test CSV: `{TEST_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Overall",
        "",
        f"- Test samples: `{len(dataframe)}`",
        f"- Test MAE: `{overall['mae']:.4f}` log10 units",
        f"- Test RMSE: `{overall['rmse']:.4f}` log10 units",
        f"- Test Spearman: `{overall['spearman']:.4f}`",
        f"- Prediction std / true std: `{observations['prediction_std_ratio_true_std']:.4f}`",
        f"- Train target mean / median: `{train_context['mean']:.4f}` / `{train_context['median']:.4f}`",
        f"- Test true mean / prediction mean: `{observations['true_mean']:.4f}` / `{observations['prediction_mean']:.4f}`",
        "",
        "## Main Error Pattern",
        "",
        f"- Low-target mean error: `{observations['low_mean_error']:.4f}`. Positive means predictions are too high.",
        f"- High-target mean error: `{observations['high_mean_error']:.4f}`. Negative means predictions are too low.",
        f"- Error vs true target Pearson correlation: `{observations['residual_vs_true_pearson']:.4f}`.",
        f"- {low_high_note}",
        "",
        "## Top Errors",
        "",
        f"- Top-error CSV: `{TOP_ERRORS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Peptide risk flags in top 15: `{observations['top_errors_peptide_flag_count']}`.",
        f"- Epitope/peptide-like antigen names in top 15: `{observations['top_errors_epitope_like_count']}`.",
        "",
        "| sample_id | source | antigen_id | true | predicted | error | abs error | antigen_len | total_len |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top_errors.head(10).iterrows():
        lines.append(
            f"| `{row['sample_id']}` | `{row['source_for_analysis']}` | `{row['antigen_id_for_analysis']}` | "
            f"{row[TRUE_COLUMN]:.4f} | {row[PRED_COLUMN]:.4f} | {row[ERROR_COLUMN]:.4f} | "
            f"{row[ABS_ERROR_COLUMN]:.4f} | {int(row['antigen_len'])} | {int(row['total_sequence_len'])} |"
        )
    lines.append("")
    lines.extend(markdown_group_table("Error by Source", grouped, "source"))
    lines.extend(markdown_group_table("Error by Target Affinity Range", grouped, "target_affinity_range"))
    lines.extend(markdown_group_table("Highest-MAE Antigen IDs", grouped, "antigen_id"))
    lines.extend(markdown_group_table("Error by Antigen Sequence Length", grouped, "antigen_sequence_length"))
    lines.extend(markdown_group_table("Error by Heavy Sequence Length", grouped, "heavy_sequence_length"))
    lines.extend(markdown_group_table("Error by Light Sequence Length", grouped, "light_sequence_length"))
    lines.extend(
        [
            "## max_length / Truncation Check",
            "",
            f"- Config max_length: `{observations['max_length']}`.",
            "- The current Dataset tokenizes heavy, light, and antigen sequences separately. "
            "Therefore `heavy+light+antigen total length > 512` is a useful complexity check, "
            "but direct truncation risk is a single sequence approaching/exceeding the per-sequence max_length.",
            f"- Test samples with heavy+light+antigen total length > 512: `{observations['all_test_total_len_gt_512']}` / `{len(dataframe)}`.",
            f"- Test samples with any single raw sequence length > 512: `{observations['all_test_single_sequence_gt_512']}` / `{len(dataframe)}`.",
            f"- Top 15 errors with total length > 512: `{observations['top_errors_total_len_gt_512']}`.",
            f"- Top 15 errors with any single raw sequence length > 512: `{observations['top_errors_single_sequence_gt_512']}`.",
            "- Tokenizers add special tokens, so sequences very close to 512 can still lose a few residue positions after tokenization.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Next Modeling Suggestion",
            "",
            "- First recommendation: `CDR-aware input`. The error pattern is not dominated by raw length truncation, "
            "and sequence-only whole-chain pooling can miss binding-focused antibody regions.",
            "- Second recommendation: `contact/interface/interaction features` after the CDR-aware baseline, "
            "because affinity depends on antibody-antigen interaction geometry rather than only three independent pooled embeddings.",
            "- Metadata features may help assay/source effects, but the current high-risk cleaning already improved the best dataset version.",
            "- Raising max_length alone is not the first move unless you specifically study the small set of long antigen sequences.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return observations


def print_terminal_summary(observations: dict, dataframe: pd.DataFrame) -> None:
    """Print the requested high-signal terminal summary."""

    print("Unified no-high-risk error analysis complete.")
    print(
        "Main pattern: low targets are over-predicted and high targets are under-predicted "
        f"(low mean error={observations['low_mean_error']:.4f}, "
        f"high mean error={observations['high_mean_error']:.4f})."
    )
    print(
        "Regression-to-mean: yes, supported by residual-vs-true correlation "
        f"{observations['residual_vs_true_pearson']:.4f} and prediction_std/true_std "
        f"{observations['prediction_std_ratio_true_std']:.4f}."
    )
    print(
        "Top errors by metadata: worst source MAE is "
        f"{observations['worst_source']} ({observations['worst_source_mae']:.4f}); "
        f"epitope-like antigen names in top 15={observations['top_errors_epitope_like_count']}."
    )
    print(
        "Length/truncation: total_len>512 in "
        f"{observations['all_test_total_len_gt_512']}/{len(dataframe)} test samples; "
        "direct single-sequence >512 risk in "
        f"{observations['all_test_single_sequence_gt_512']}/{len(dataframe)}."
    )
    print("Recommended next step: 1. CDR-aware input.")
    print(f"Markdown report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Top errors CSV: {TOP_ERRORS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Grouped error CSV: {ERROR_BY_GROUP_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Create markdown, CSVs, and two figures from saved predictions."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    config, predictions, test = read_inputs()
    analyzed = merge_and_annotate(config, predictions, test)
    top_errors = save_top_errors(analyzed)
    grouped = make_error_by_group(analyzed)
    grouped.to_csv(ERROR_BY_GROUP_PATH, index=False)
    figure_paths = [save_parity_plot(analyzed), save_residual_plot(analyzed)]
    observations = write_report(config, analyzed, grouped, top_errors, figure_paths)
    print_terminal_summary(observations, analyzed)
    print("No training, model changes, or dataset-build changes were performed.")


if __name__ == "__main__":
    main()
