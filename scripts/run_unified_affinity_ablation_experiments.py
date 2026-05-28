"""Run and summarize unified affinity ablation experiments.

中文人话说明：
full_629 已经有同参数的 lr=3e-5 run，所以这里直接复用。
三个新 ablation 版本会依次调用项目现有脚本：
1. run_train_affinity.py
2. evaluate_affinity_test_set.py
3. evaluate_affinity_baselines.py

所有 checkpoint / predictions / logs 都用 ablation 专属路径，
不会覆盖已有 unified sweep 结果。
"""

from __future__ import annotations

from pathlib import Path
import math
import subprocess
import sys

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation" / "unified_affinity_dataset_v1"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_PATH = OUTPUT_DIR / "ablation_results.csv"
REPORT_PATH = OUTPUT_DIR / "ablation_report.md"

RUNS = [
    {
        "dataset_version": "unified_full_629",
        "config": "config_affinity_unified_affinity_dataset_v1_lr3e-5_e10.yaml",
        "reuse_existing": True,
        "filter_note": "Current full unified dataset; existing lr=3e-5 result reused.",
    },
    {
        "dataset_version": "unified_no_peptide",
        "config": "config_affinity_unified_no_peptide_lr3e-5_e10.yaml",
        "reuse_existing": False,
        "filter_note": "Removed peptide_antigen risk rows.",
    },
    {
        "dataset_version": "unified_no_high_risk",
        "config": "config_affinity_unified_no_high_risk_lr3e-5_e10.yaml",
        "reuse_existing": False,
        "filter_note": "Removed peptide, same_Hchain_Lchain_metadata, and suspicious method rows.",
    },
    {
        "dataset_version": "unified_no_less_strict",
        "config": "config_affinity_unified_no_less_strict_lr3e-5_e10.yaml",
        "reuse_existing": False,
        "filter_note": "Removed less-strict SAbDab source/risk rows.",
    },
]


def load_config(path: Path) -> dict:
    """Load YAML config with the project's helper."""

    sys.path.append(str(PROJECT_ROOT))
    from src.utils import load_config as project_load_config

    return project_load_config(str(path))


def run_logged(command: list[str], log_path: Path) -> None:
    """Run one existing project command and save terminal output."""

    print(f"Running: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    print(f"Log saved: {log_path.relative_to(PROJECT_ROOT)}")


def compute_metrics(true_values: pd.Series, predicted_values: pd.Series) -> dict:
    """Compute metrics from predictions already saved to CSV."""

    true = pd.to_numeric(true_values, errors="coerce")
    pred = pd.to_numeric(predicted_values, errors="coerce")
    valid = true.notna() & pred.notna()
    true = true[valid]
    pred = pred[valid]
    error = pred - true
    mse = float((error * error).mean())
    return {
        "MAE": float(error.abs().mean()),
        "RMSE": math.sqrt(mse),
        "Spearman": float(true.corr(pred, method="spearman")),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
    }


def mean_baseline(config: dict) -> dict:
    """Evaluate the constant train-mean baseline on that version's test split."""

    target_column = config.get("target_column", "neg_log10_affinity")
    train = pd.read_csv(PROJECT_ROOT / config["train_csv"])
    test = pd.read_csv(PROJECT_ROOT / config["test_csv"])
    mean_value = float(pd.to_numeric(train[target_column], errors="coerce").mean())
    predictions = pd.Series([mean_value] * len(test))
    return compute_metrics(test[target_column], predictions)


def checkpoint_val_metrics(config: dict) -> dict:
    """Read final validation metrics from checkpoint."""

    checkpoint_path = PROJECT_ROOT / config["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint.get("final_val_metrics")
    if not metrics:
        raise ValueError(f"Missing final_val_metrics in {checkpoint_path}")
    return metrics


def split_sizes(config: dict) -> dict[str, int]:
    """Count train/val/test rows from the config CSVs."""

    return {
        "train": int(len(pd.read_csv(PROJECT_ROOT / config["train_csv"]))),
        "val": int(len(pd.read_csv(PROJECT_ROOT / config["val_csv"]))),
        "test": int(len(pd.read_csv(PROJECT_ROOT / config["test_csv"]))),
    }


def collect_row(run: dict) -> dict:
    """Collect metrics for one finished run."""

    config = load_config(PROJECT_ROOT / run["config"])
    predictions = pd.read_csv(PROJECT_ROOT / config["predictions_path"])
    test = compute_metrics(
        predictions["true_neg_log10_affinity"],
        predictions["predicted_neg_log10_affinity"],
    )
    baseline = mean_baseline(config)
    val = checkpoint_val_metrics(config)
    sizes = split_sizes(config)
    rows = sizes["train"] + sizes["val"] + sizes["test"]
    return {
        "dataset_version": run["dataset_version"],
        "rows": rows,
        "train_size": sizes["train"],
        "val_size": sizes["val"],
        "test_size": sizes["test"],
        "val_MAE": float(val["mae"]),
        "val_RMSE": float(val["rmse"]),
        "val_Spearman": float(val["spearman"]),
        "test_MAE": test["MAE"],
        "test_RMSE": test["RMSE"],
        "test_Spearman": test["Spearman"],
        "prediction_mean": test["prediction_mean"],
        "prediction_std": test["prediction_std"],
        "mean_baseline_MAE": baseline["MAE"],
        "mean_baseline_RMSE": baseline["RMSE"],
        "beats_mean_baseline_MAE": bool(test["MAE"] < baseline["MAE"]),
        "beats_mean_baseline_RMSE": bool(test["RMSE"] < baseline["RMSE"]),
        "beats_mean_baseline_both": bool(
            test["MAE"] < baseline["MAE"] and test["RMSE"] < baseline["RMSE"]
        ),
        "config": run["config"],
        "checkpoint_path": config["checkpoint_path"],
        "predictions_path": config["predictions_path"],
        "filter_note": run["filter_note"],
    }


def fmt(value: float) -> str:
    """Short metric formatter for Markdown."""

    return f"{value:.4f}"


def best_version(results: pd.DataFrame) -> pd.Series:
    """Choose best by test MAE, then RMSE."""

    return results.sort_values(["test_MAE", "test_RMSE"]).iloc[0]


def interpretation(results: pd.DataFrame) -> list[str]:
    """Turn ablation differences into cautious experiment conclusions."""

    full = results.loc[results["dataset_version"] == "unified_full_629"].iloc[0]
    lines = []
    for version, question in [
        ("unified_no_peptide", "peptide rows"),
        ("unified_no_high_risk", "configured high-risk rows"),
        ("unified_no_less_strict", "less-strict rows"),
    ]:
        row = results.loc[results["dataset_version"] == version].iloc[0]
        mae_delta = row["test_MAE"] - full["test_MAE"]
        spearman_delta = row["test_Spearman"] - full["test_Spearman"]
        lines.append(
            f"- Removing {question}: test MAE delta vs full `{mae_delta:+.4f}`, "
            f"test Spearman delta `{spearman_delta:+.4f}`. "
            "Because each filtered dataset is re-split, treat this as dataset-version evidence, not a pure row-level causal effect."
        )
    return lines


def write_report(results: pd.DataFrame) -> None:
    """Write human-readable ablation comparison."""

    best = best_version(results)
    best_rank = results.sort_values("test_Spearman", ascending=False).iloc[0]
    lines = [
        "# Unified Affinity Dataset v1 Ablation Report",
        "",
        "## Scope",
        "",
        "- Base model: existing ESM2+LoRA affinity regressor, unchanged.",
        "- Hyperparameters: learning_rate `3e-5`, epochs `10`, seed `42`.",
        "- `unified_full_629` reuses the completed sweep result.",
        "- Filtered versions were re-split by `antigen_sequence` group, so split composition changes between versions.",
        "",
        "## Results",
        "",
        "| dataset | rows | train | val | test | test_MAE | test_RMSE | test_Spearman | prediction_std | mean baseline MAE | mean baseline RMSE | beat mean both |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| `{row['dataset_version']}` | {int(row['rows'])} | {int(row['train_size'])} | "
            f"{int(row['val_size'])} | {int(row['test_size'])} | {fmt(row['test_MAE'])} | "
            f"{fmt(row['test_RMSE'])} | {fmt(row['test_Spearman'])} | {fmt(row['prediction_std'])} | "
            f"{fmt(row['mean_baseline_MAE'])} | {fmt(row['mean_baseline_RMSE'])} | "
            f"`{bool(row['beats_mean_baseline_both'])}` |"
        )
    lines.extend(
        [
            "",
            "## Validation Metrics",
            "",
            "| dataset | val_MAE | val_RMSE | val_Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in results.iterrows():
        lines.append(
            f"| `{row['dataset_version']}` | {fmt(row['val_MAE'])} | "
            f"{fmt(row['val_RMSE'])} | {fmt(row['val_Spearman'])} |"
        )
    lines.extend(
        [
            "",
            "## Judgment",
            "",
            f"- Best by test MAE/RMSE ordering: `{best['dataset_version']}` with test MAE `{best['test_MAE']:.4f}`.",
            f"- Best by test Spearman: `{best_rank['dataset_version']}` with `{best_rank['test_Spearman']:.4f}`.",
            "- These ablations answer whether a filtered dataset version trains better under the same model settings.",
            "",
            "## Risk Questions",
            "",
            *interpretation(results),
            "",
            "## Files",
            "",
            f"- CSV summary: `{RESULTS_PATH.relative_to(PROJECT_ROOT)}`",
            f"- Logs: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run three new ablation trainings and summarize all four versions."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    rows = []
    for run in RUNS:
        if not run["reuse_existing"]:
            config_arg = run["config"]
            name = run["dataset_version"]
            run_logged(
                [python, "run_train_affinity.py", "--config", config_arg],
                LOG_DIR / f"{name}_train.log",
            )
            run_logged(
                [python, "scripts/evaluate_affinity_test_set.py", "--config", config_arg],
                LOG_DIR / f"{name}_test_eval.log",
            )
            run_logged(
                [python, "scripts/evaluate_affinity_baselines.py", "--config", config_arg],
                LOG_DIR / f"{name}_baselines.log",
            )
        rows.append(collect_row(run))

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    write_report(results)
    print(results.to_string(index=False))
    print(f"Saved results: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
