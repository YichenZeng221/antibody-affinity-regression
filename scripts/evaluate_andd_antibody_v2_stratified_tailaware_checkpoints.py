"""Evaluate tail-aware ANDD stratified cross-attention checkpoints after manual training.

中文说明：
这个脚本不会训练模型。它读取 tail-aware training 保存的四种 validation-selected
checkpoint，在同一个 test split 上统一评估，并与已有 pooled / cross-attention
baseline 对比。所有新增 predictions、报告和图片写入新目录。
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/seqproft_xdg_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/seqproft_matplotlib_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.affinity_cross_attention_dataset import (  # noqa: E402
    CROSS_ATTENTION_CDR_FIELDS,
    CrossAttentionAffinityDataset,
)
from src.affinity_cross_attention_evaluate import (  # noqa: E402
    cross_attention_device,
    evaluate_cross_attention_affinity_model,
)
from src.affinity_cross_attention_model import SeqProFTCrossAttentionAffinityRegressor  # noqa: E402
from src.affinity_cross_attention_tailaware_train import tail_thresholds  # noqa: E402
from src.affinity_cross_attention_train import antigen_length_from_config  # noqa: E402
from src.affinity_evaluate import compute_regression_metrics  # noqa: E402
from src.utils import load_config  # noqa: E402


TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"
BASELINE_LABELS = {
    "stratified_all_cdr_pooled": "Baseline: pooled all-CDR",
    "stratified_all_cdr_cross_attention": "Baseline: cross-attention",
}


def parse_args() -> argparse.Namespace:
    """Read the new tail-aware config."""

    parser = argparse.ArgumentParser(description="Evaluate tail-aware cross-attention checkpoints.")
    parser.add_argument(
        "--config",
        default="config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_tailaware_lr3e-5_e30.yaml",
    )
    return parser.parse_args()


def fmt(value: float | None) -> str:
    """Format optional metrics for readable markdown."""

    return "missing" if value is None or pd.isna(value) else f"{value:.4f}"


def policy_labels(config: dict) -> dict[str, str]:
    """用 config 标签区分 w2/w3 predictions，避免报告里名字混淆。"""

    prefix = str(config.get("experiment_label", "Tail-aware"))
    return {
        "best_val_mae": f"{prefix}: best val MAE",
        "best_val_spearman": f"{prefix}: best val Spearman",
        "best_val_spread": f"{prefix}: best val spread",
        "best_val_tail_mae": f"{prefix}: best val tail MAE",
    }


def metric_bundle(predictions: pd.DataFrame, lower_p10: float, upper_p90: float) -> dict:
    """Compute common metrics using the same train-defined tails for every model."""

    true = pd.to_numeric(predictions[TRUE_COLUMN], errors="raise")
    predicted = pd.to_numeric(predictions[PRED_COLUMN], errors="raise")
    error = predicted - true
    absolute_error = error.abs()
    true_std = float(true.std())
    prediction_std = float(predicted.std())
    below_mask = true <= lower_p10
    above_mask = true >= upper_p90
    return {
        **compute_regression_metrics(true.tolist(), predicted.tolist()),
        "rows": int(len(predictions)),
        "prediction_std": prediction_std,
        "true_std": true_std,
        "pred_std_over_true_std": prediction_std / true_std if true_std else None,
        "error_vs_true_pearson": float(error.corr(true, method="pearson")),
        "below_train_p10_rows": int(below_mask.sum()),
        "above_train_p90_rows": int(above_mask.sum()),
        "below_train_p10_mae": float(absolute_error[below_mask].mean()) if bool(below_mask.any()) else None,
        "above_train_p90_mae": float(absolute_error[above_mask].mean()) if bool(above_mask.any()) else None,
    }


def add_target_bin_metrics(metrics: dict, predictions: pd.DataFrame, low_edge: float, high_edge: float) -> None:
    """Append train-tertile low/mid/high MAE to one metric dictionary."""

    frame = predictions.copy()
    frame["absolute_error_for_metrics"] = (
        pd.to_numeric(frame[PRED_COLUMN], errors="raise")
        - pd.to_numeric(frame[TRUE_COLUMN], errors="raise")
    ).abs()
    frame["target_bin"] = pd.cut(
        pd.to_numeric(frame[TRUE_COLUMN], errors="raise"),
        bins=[-math.inf, low_edge, high_edge, math.inf],
        labels=["low_target", "mid_target", "high_target"],
        include_lowest=True,
    )
    metrics["low_target_mae"] = float(
        frame.loc[frame["target_bin"] == "low_target", "absolute_error_for_metrics"].mean()
    )
    metrics["mid_target_mae"] = float(
        frame.loc[frame["target_bin"] == "mid_target", "absolute_error_for_metrics"].mean()
    )
    metrics["high_target_mae"] = float(
        frame.loc[frame["target_bin"] == "high_target", "absolute_error_for_metrics"].mean()
    )
    tail_values = [metrics["below_train_p10_mae"], metrics["above_train_p90_mae"]]
    tail_values = [value for value in tail_values if value is not None and not pd.isna(value)]
    metrics["tail_mae"] = sum(tail_values) / len(tail_values) if tail_values else None


def prediction_rows(dataset, true_values: list[float], predicted_values: list[float]) -> pd.DataFrame:
    """Save test metadata together with predictions for later inspection."""

    rows = []
    for metadata, true_value, prediction in zip(dataset.data.to_dict("records"), true_values, predicted_values):
        error = float(prediction - true_value)
        row = {
            "sample_id": metadata.get("sample_id", metadata.get("candidate_id", "")),
            "candidate_id": metadata.get("candidate_id", ""),
            "source": metadata.get("source", ""),
            "pdb_id": metadata.get("pdb_id", ""),
            "ag_name": metadata.get("ag_name", ""),
            "antigen_sequence": metadata.get("antigen_sequence", ""),
            TRUE_COLUMN: float(true_value),
            PRED_COLUMN: float(prediction),
            "error": error,
            "absolute_error": abs(error),
            "fold_error": 10 ** abs(error),
        }
        for cdr_field in CROSS_ATTENTION_CDR_FIELDS:
            row[cdr_field] = metadata.get(cdr_field, "")
        rows.append(row)
    return pd.DataFrame(rows)


def load_baseline(path: Path, expected_ids: set[str]) -> pd.DataFrame | None:
    """Load baseline output only when it is present and matches this test set."""

    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"sample_id", TRUE_COLUMN, PRED_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    if set(frame["sample_id"].astype(str)) != expected_ids:
        raise ValueError(f"{path} does not contain exactly the same stratified test sample IDs.")
    return frame


def metrics_row(name: str, kind: str, epoch: int | str, metrics: dict) -> dict:
    """Flatten metrics for CSV and plotting."""

    return {
        "model": name,
        "kind": kind,
        "best_epoch": epoch,
        "rows": metrics["rows"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "Spearman": metrics["spearman"],
        "prediction_std": metrics["prediction_std"],
        "true_std": metrics["true_std"],
        "pred_std_true_std": metrics["pred_std_over_true_std"],
        "error_vs_true_Pearson": metrics["error_vs_true_pearson"],
        "low_target_MAE": metrics["low_target_mae"],
        "mid_target_MAE": metrics["mid_target_mae"],
        "high_target_MAE": metrics["high_target_mae"],
        "below_train_p10_MAE": metrics["below_train_p10_mae"],
        "above_train_p90_MAE": metrics["above_train_p90_mae"],
        "tail_MAE": metrics["tail_mae"],
    }


def save_comparison_figure(metrics_frame: pd.DataFrame, path: Path) -> None:
    """Plot checkpoint metrics; lower errors and ratio nearer 1 are preferable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = metrics_frame["model"].str.replace("Baseline: ", "", regex=False)
    colors_by_kind = {"baseline": "#507DBC", "comparison": "#9C8BC1", "tailaware": "#D66C44"}
    colors = [colors_by_kind.get(kind, "#D66C44") for kind in metrics_frame["kind"]]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    panels = [
        ("MAE", "MAE (lower is better)"),
        ("Spearman", "Spearman (higher is better)"),
        ("pred_std_true_std", "Pred std / true std (closer to 1)"),
        ("tail_MAE", "Average P10/P90 tail MAE (lower is better)"),
    ]
    for axis, (column, title) in zip(axes.flat, panels):
        axis.bar(labels, metrics_frame[column], color=colors, alpha=0.88)
        if column == "pred_std_true_std":
            axis.axhline(1.0, color="#505050", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("ANDD stratified: tail-aware checkpoint comparison", fontsize=15, fontweight="bold")
    figure.savefig(path, dpi=300)
    plt.close(figure)


def save_residual_figure(
    prediction_frames: dict[str, pd.DataFrame],
    path: Path,
    current_tail_label: str,
    reference_tail_label: str | None,
) -> None:
    """Show whether the best tail checkpoint weakens the downward residual trend."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    styles = {
        "Baseline: pooled all-CDR": ("#247BA0", "o"),
        "Baseline: cross-attention": ("#6C757D", "s"),
        current_tail_label: ("#D66C44", "^"),
    }
    if reference_tail_label:
        styles[reference_tail_label] = ("#9C8BC1", "D")
    for label, frame in prediction_frames.items():
        if label not in styles:
            continue
        color, marker = styles[label]
        true = pd.to_numeric(frame[TRUE_COLUMN], errors="raise")
        residual = pd.to_numeric(frame[PRED_COLUMN], errors="raise") - true
        axis.scatter(true, residual, s=28, alpha=0.42, marker=marker, color=color, label=label)
        if len(frame) >= 2:
            slope, intercept = pd.Series(residual).cov(pd.Series(true)) / pd.Series(true).var(), None
            intercept = float(residual.mean() - slope * true.mean())
            xs = pd.Series([float(true.min()), float(true.max())])
            axis.plot(xs, slope * xs + intercept, color=color, linewidth=2)
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    axis.set_xlabel("True neg_log10_affinity")
    axis.set_ylabel("Prediction - true")
    axis.set_title("Tail-aware residual comparison on stratified test set")
    axis.text(
        0.02,
        0.04,
        "Downward residual trend = regression to the mean",
        transform=axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def comparison_statement(new: float, baseline: float, favorable: str) -> str:
    """Produce honest comparison language for the generated report."""

    improved = new < baseline if favorable == "lower" else new > baseline
    return "improved" if improved else "did not improve"


def write_report(config: dict, metrics_frame: pd.DataFrame, missing_baselines: list[str]) -> Path:
    """Write a report that answers the requested questions after manual evaluation."""

    report_path = Path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    current_labels = policy_labels(config)
    tail_row = metrics_frame.loc[metrics_frame["model"] == current_labels["best_val_tail_mae"]].iloc[0]
    baseline_rows = metrics_frame.loc[metrics_frame["kind"] == "baseline"]
    cross = baseline_rows.loc[
        baseline_rows["model"] == BASELINE_LABELS["stratified_all_cdr_cross_attention"]
    ]
    lines = [
        f"# {config.get('report_title', 'ANDD Antibody v2 Stratified: Tail-Aware Cross-Attention Experiment')}",
        "",
        "## Experiment",
        "",
        "- Model architecture: unchanged all-CDR learnable cross-attention model.",
        f"- Loss: tail-weighted MSE; train targets at or below P10 and at or above P90 receive weight `{float(config.get('tail_sample_weight', 3.0)):.1f}`, other rows receive weight `{float(config.get('regular_sample_weight', 1.0)):.1f}`.",
        "- Checkpoint selection: independently saved by validation MAE, Spearman, prediction-spread closeness to 1, and validation tail MAE.",
        "- Comparison is on the same stratified antigen-level test split.",
        "- This is a **single-seed experiment**; conclusions are provisional.",
        "",
        "## Test Metrics By Validation-Selected Checkpoint",
        "",
        metrics_frame.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Primary Validation-Tail-Selected Reading",
        "",
        "The checkpoint selected by validation tail MAE is the primary reading; it is not selected using test labels.",
        "",
    ]
    if cross.empty:
        lines.append("- Existing stratified cross-attention baseline predictions were missing; automated comparisons are marked missing.")
    else:
        base = cross.iloc[0]
        lines.extend(
            [
                f"- Prediction spread: `{tail_row['pred_std_true_std']:.4f}` vs baseline `{base['pred_std_true_std']:.4f}`; "
                f"the current setting {'improved' if abs(tail_row['pred_std_true_std'] - 1.0) < abs(base['pred_std_true_std'] - 1.0) else 'did not improve'} spread toward 1.",
                f"- Error-vs-true Pearson: `{tail_row['error_vs_true_Pearson']:.4f}` vs baseline `{base['error_vs_true_Pearson']:.4f}`; "
                f"the current setting {'improved' if abs(tail_row['error_vs_true_Pearson']) < abs(base['error_vs_true_Pearson']) else 'did not improve'} compression trend toward 0.",
                f"- Below-P10 MAE: `{tail_row['below_train_p10_MAE']:.4f}` vs `{base['below_train_p10_MAE']:.4f}`; "
                f"{comparison_statement(tail_row['below_train_p10_MAE'], base['below_train_p10_MAE'], 'lower')}.",
                f"- Above-P90 MAE: `{tail_row['above_train_p90_MAE']:.4f}` vs `{base['above_train_p90_MAE']:.4f}`; "
                f"{comparison_statement(tail_row['above_train_p90_MAE'], base['above_train_p90_MAE'], 'lower')}.",
                f"- Spearman: `{tail_row['Spearman']:.4f}` vs `{base['Spearman']:.4f}`; "
                f"{comparison_statement(tail_row['Spearman'], base['Spearman'], 'higher')}.",
                f"- Overall MAE: `{tail_row['MAE']:.4f}` vs `{base['MAE']:.4f}`; "
                f"{comparison_statement(tail_row['MAE'], base['MAE'], 'lower')}.",
                "",
                "If tail or spread metrics improve while overall MAE gets worse, that is a real tradeoff rather than a universal win.",
            ]
        )
    reference_label = config.get("primary_tailaware_reference_label")
    reference = metrics_frame.loc[metrics_frame["model"] == reference_label] if reference_label else pd.DataFrame()
    if not reference.empty:
        prior = reference.iloc[0]
        w2_more_stable = (
            tail_row["MAE"] <= prior["MAE"]
            and tail_row["Spearman"] >= prior["Spearman"]
        )
        lines.extend(
            [
                "",
                "## Comparison With Prior Tail-Aware Setting",
                "",
                f"The requested prior reference is `{reference_label}`. The primary current reading is `{current_labels['best_val_tail_mae']}`.",
                "",
                f"- Overall MAE: current `{tail_row['MAE']:.4f}` vs prior `{prior['MAE']:.4f}`; "
                f"{comparison_statement(tail_row['MAE'], prior['MAE'], 'lower')}.",
                f"- Spearman: current `{tail_row['Spearman']:.4f}` vs prior `{prior['Spearman']:.4f}`; "
                f"{comparison_statement(tail_row['Spearman'], prior['Spearman'], 'higher')}.",
                f"- Prediction spread ratio: current `{tail_row['pred_std_true_std']:.4f}` vs prior `{prior['pred_std_true_std']:.4f}`.",
                f"- Error-vs-true Pearson: current `{tail_row['error_vs_true_Pearson']:.4f}` vs prior `{prior['error_vs_true_Pearson']:.4f}`.",
                f"- Average P10/P90 tail MAE: current `{tail_row['tail_MAE']:.4f}` vs prior `{prior['tail_MAE']:.4f}`; "
                f"{comparison_statement(tail_row['tail_MAE'], prior['tail_MAE'], 'lower')}.",
                "",
                f"- Stability reading: the milder setting is "
                f"{'more stable by the requested overall-MAE-and-Spearman criterion' if w2_more_stable else 'not clearly more stable by both overall MAE and Spearman together'}.",
                "",
                "If the milder weighting preserves spread/tail gains while improving overall MAE or Spearman, it is the more stable variant. "
                "If tradeoffs remain, tail-aware objective is useful but not a final solution; structure/contact-aware features remain a serious next step.",
            ]
        )
    if missing_baselines:
        lines.extend(["", f"- Missing baseline files: `{', '.join(missing_baselines)}`."])
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Checkpoint comparison CSV: `{config['checkpoint_comparison_path']}`",
            f"- Comparison figure: `{config['checkpoint_comparison_figure_path']}`",
            f"- Residual figure: `{config['residual_figure_path']}`",
            "",
            "## Conclusion Boundary",
            "",
            "This experiment tests one conservative tail-weighted loss at one seed. It can reveal a promising tradeoff, but it cannot establish a stable improvement until repeated across seeds or validation policies.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    """Evaluate all validation-selected checkpoints and generate new comparison outputs."""

    config = load_config(parse_args().config)
    missing_checkpoints = [
        path for path in config["checkpoint_paths"].values() if not Path(path).exists()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Tail-aware checkpoints do not exist yet. Run manual training first. Missing: "
            + ", ".join(missing_checkpoints)
        )
    device = cross_attention_device(config)
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_dataset = CrossAttentionAffinityDataset(
        csv_path=config["train_csv"],
        tokenizer=tokenizer,
        antigen_max_length=antigen_length_from_config(config),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    test_dataset = CrossAttentionAffinityDataset(
        csv_path=config["test_csv"],
        tokenizer=tokenizer,
        antigen_max_length=antigen_length_from_config(config),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    lower_p10, upper_p90 = tail_thresholds(train_dataset.targets)
    train_series = pd.Series(train_dataset.targets, dtype=float)
    low_edge = float(train_series.quantile(1 / 3))
    high_edge = float(train_series.quantile(2 / 3))
    test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    output_rows: list[dict] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    current_labels = policy_labels(config)
    for policy_name, checkpoint_path in config["checkpoint_paths"].items():
        model = SeqProFTCrossAttentionAffinityRegressor(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, true_values, predicted_values = evaluate_cross_attention_affinity_model(model, test_loader, device)
        predictions = prediction_rows(test_dataset, true_values, predicted_values)
        prediction_path = Path(config["prediction_paths"][policy_name])
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_path, index=False)
        metrics = metric_bundle(predictions, lower_p10, upper_p90)
        add_target_bin_metrics(metrics, predictions, low_edge, high_edge)
        label = current_labels[policy_name]
        output_rows.append(
            metrics_row(
                label,
                str(config.get("experiment_kind", "tailaware")),
                checkpoint.get("best_epoch", ""),
                metrics,
            )
        )
        prediction_frames[label] = predictions

    expected_ids = set(test_dataset.data["sample_id"].astype(str))
    missing_baselines = []
    baseline_paths = {
        "stratified_all_cdr_pooled": Path(config["pooled_baseline_predictions_path"]),
        "stratified_all_cdr_cross_attention": Path(config["cross_attention_baseline_predictions_path"]),
    }
    for baseline_name, path in baseline_paths.items():
        baseline = load_baseline(path, expected_ids)
        if baseline is None:
            missing_baselines.append(str(path))
            continue
        metrics = metric_bundle(baseline, lower_p10, upper_p90)
        add_target_bin_metrics(metrics, baseline, low_edge, high_edge)
        label = BASELINE_LABELS[baseline_name]
        output_rows.append(metrics_row(label, "baseline", "", metrics))
        prediction_frames[label] = baseline

    for _, details in config.get("additional_comparison_predictions", {}).items():
        comparison_path = Path(details["path"])
        comparison = load_baseline(comparison_path, expected_ids)
        if comparison is None:
            missing_baselines.append(str(comparison_path))
            continue
        metrics = metric_bundle(comparison, lower_p10, upper_p90)
        add_target_bin_metrics(metrics, comparison, low_edge, high_edge)
        label = str(details["label"])
        output_rows.append(metrics_row(label, "comparison", "", metrics))
        prediction_frames[label] = comparison

    metrics_frame = pd.DataFrame(output_rows)
    comparison_path = Path(config["checkpoint_comparison_path"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(comparison_path, index=False)
    save_comparison_figure(metrics_frame, Path(config["checkpoint_comparison_figure_path"]))
    save_residual_figure(
        prediction_frames,
        Path(config["residual_figure_path"]),
        current_tail_label=current_labels["best_val_tail_mae"],
        reference_tail_label=config.get("primary_tailaware_reference_label"),
    )
    report_path = write_report(config, metrics_frame, missing_baselines)
    print(f"Saved checkpoint comparison to {comparison_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved figures to {config['checkpoint_comparison_figure_path']} and {config['residual_figure_path']}")


if __name__ == "__main__":
    main()
