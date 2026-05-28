"""Run and summarize a tiny learning-rate sweep for unified affinity dataset v1.

:
, dataset
 config :

1. ``run_train_affinity.py`` ;
2. ``evaluate_affinity_test_set.py``  test predictions;
3. ``evaluate_affinity_baselines.py``  baseline comparison

 stdout/stderr  log, tqdm  terminal 
:
    outputs/hparam_sweep/unified_affinity_dataset_v1/
"""

from __future__ import annotations

from pathlib import Path
import math
import subprocess
import sys

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hparam_sweep" / "unified_affinity_dataset_v1"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_CSV_PATH = OUTPUT_DIR / "sweep_results.csv"
REPORT_PATH = OUTPUT_DIR / "sweep_report.md"

RUNS = [
    {
        "name": "lr1e-5_e10",
        "learning_rate": 1e-5,
        "config": "config_affinity_unified_affinity_dataset_v1_lr1e-5_e10.yaml",
    },
    {
        "name": "lr3e-5_e10",
        "learning_rate": 3e-5,
        "config": "config_affinity_unified_affinity_dataset_v1_lr3e-5_e10.yaml",
    },
    {
        "name": "lr1e-4_e10",
        "learning_rate": 1e-4,
        "config": "config_affinity_unified_affinity_dataset_v1_lr1e-4_e10.yaml",
    },
]


def load_yaml(path: Path) -> dict:
    """Use the project's YAML loader without touching model code."""

    sys.path.append(str(PROJECT_ROOT))
    from src.utils import load_config

    return load_config(path)


def run_logged(command: list[str], log_path: Path) -> None:
    """Run one existing script and save its full output for audit."""

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
    """Compute test metrics from saved predictions.

    evaluation script already computed the same metrics in its log.
    We recompute here from predictions CSV so the summary table is machine-readable.
    """

    true = pd.to_numeric(true_values, errors="coerce")
    pred = pd.to_numeric(predicted_values, errors="coerce")
    valid = true.notna() & pred.notna()
    true = true[valid]
    pred = pred[valid]
    error = pred - true
    mae = float(error.abs().mean())
    mse = float((error * error).mean())
    rmse = math.sqrt(mse)
    prediction_std = float(pred.std())
    spearman = float(true.corr(pred, method="spearman"))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Spearman": spearman,
        "prediction_std": prediction_std,
    }


def mean_baseline_metrics(config: dict) -> dict:
    """Evaluate constant train-mean baseline on the run's shared test split."""

    target_column = config.get("target_column", "neg_log10_affinity")
    train = pd.read_csv(PROJECT_ROOT / config["train_csv"])
    test = pd.read_csv(PROJECT_ROOT / config["test_csv"])
    mean_prediction = float(pd.to_numeric(train[target_column], errors="coerce").mean())
    predictions = pd.Series([mean_prediction] * len(test))
    return compute_metrics(pd.to_numeric(test[target_column], errors="coerce"), predictions)


def final_val_metrics(checkpoint_path: Path) -> dict:
    """Read final validation metrics saved by the existing training loop."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint.get("final_val_metrics")
    if not metrics:
        raise ValueError(f"Checkpoint missing final_val_metrics: {checkpoint_path}")
    return metrics


def collect_row(run: dict) -> dict:
    """Collect one sweep row after training/evaluation finished."""

    config = load_yaml(PROJECT_ROOT / run["config"])
    checkpoint_path = PROJECT_ROOT / config["checkpoint_path"]
    predictions_path = PROJECT_ROOT / config["predictions_path"]
    val = final_val_metrics(checkpoint_path)
    predictions = pd.read_csv(predictions_path)
    test = compute_metrics(
        predictions["true_neg_log10_affinity"],
        predictions["predicted_neg_log10_affinity"],
    )
    baseline = mean_baseline_metrics(config)
    return {
        "run_name": run["name"],
        "config": run["config"],
        "learning_rate": run["learning_rate"],
        "epochs": int(config["epochs"]),
        "seed": int(config["seed"]),
        "val_MAE": float(val["mae"]),
        "val_RMSE": float(val["rmse"]),
        "val_Spearman": float(val["spearman"]),
        "test_MAE": test["MAE"],
        "test_RMSE": test["RMSE"],
        "test_Spearman": test["Spearman"],
        "prediction_std": test["prediction_std"],
        "mean_baseline_MAE": baseline["MAE"],
        "mean_baseline_RMSE": baseline["RMSE"],
        "beats_mean_baseline_MAE": bool(test["MAE"] < baseline["MAE"]),
        "beats_mean_baseline_RMSE": bool(test["RMSE"] < baseline["RMSE"]),
        "checkpoint_path": config["checkpoint_path"],
        "predictions_path": config["predictions_path"],
    }


def format_metric(value: float) -> str:
    """Format report metrics compactly."""

    return f"{value:.4f}"


def write_report(results: pd.DataFrame) -> None:
    """Write beginner-readable sweep report."""

    best_mae = results.sort_values("test_MAE").iloc[0]
    best_spearman = results.sort_values("test_Spearman", ascending=False).iloc[0]
    lines = [
        "# Unified Affinity Dataset v1 Hyperparameter Sweep",
        "",
        "## Scope",
        "",
        "- Dataset: `data/processed_affinity/unified_affinity_dataset_v1/`",
        "- Search dimension: `learning_rate` only: `1e-5`, `3e-5`, `1e-4`.",
        "- Fixed controls: `epochs=10`, `seed=42`, existing ESM2+LoRA model code and same split CSVs.",
        "- Each run has its own checkpoint, predictions CSV, and log files.",
        "",
        "## Results",
        "",
        "| learning_rate | val_MAE | val_RMSE | val_Spearman | test_MAE | test_RMSE | test_Spearman | prediction_std | mean baseline MAE | mean baseline RMSE | beat mean MAE | beat mean RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| `{row['learning_rate']:.0e}` | {format_metric(row['val_MAE'])} | "
            f"{format_metric(row['val_RMSE'])} | {format_metric(row['val_Spearman'])} | "
            f"{format_metric(row['test_MAE'])} | {format_metric(row['test_RMSE'])} | "
            f"{format_metric(row['test_Spearman'])} | {format_metric(row['prediction_std'])} | "
            f"{format_metric(row['mean_baseline_MAE'])} | {format_metric(row['mean_baseline_RMSE'])} | "
            f"`{bool(row['beats_mean_baseline_MAE'])}` | `{bool(row['beats_mean_baseline_RMSE'])}` |"
        )
    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            f"- Lowest test MAE run: `{best_mae['run_name']}` at learning_rate `{best_mae['learning_rate']:.0e}` with test MAE `{best_mae['test_MAE']:.4f}`.",
            f"- Highest test Spearman run: `{best_spearman['run_name']}` at learning_rate `{best_spearman['learning_rate']:.0e}` with Spearman `{best_spearman['test_Spearman']:.4f}`.",
            "- Baseline comparison is computed on the same unified train/test split for every run.",
            "",
            "## Files",
            "",
            f"- CSV summary: `{RESULTS_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- Logs: `{LOG_DIR.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run all sweep configs and write summary artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    rows = []

    for run in RUNS:
        config_arg = run["config"]
        run_logged(
            [python, "run_train_affinity.py", "--config", config_arg],
            LOG_DIR / f"{run['name']}_train.log",
        )
        run_logged(
            [python, "scripts/evaluate_affinity_test_set.py", "--config", config_arg],
            LOG_DIR / f"{run['name']}_test_eval.log",
        )
        run_logged(
            [python, "scripts/evaluate_affinity_baselines.py", "--config", config_arg],
            LOG_DIR / f"{run['name']}_baselines.log",
        )
        rows.append(collect_row(run))

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_CSV_PATH, index=False)
    write_report(results)
    print(results.to_string(index=False))
    print(f"CSV summary saved: {RESULTS_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Markdown report saved: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
