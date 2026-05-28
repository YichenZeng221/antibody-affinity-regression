"""ANDD stratified pooled / cross-attention  train-vs-eval fit diagnosis

:
 checkpoint, predictions  split  inference
, dataset, test predictions 

:
-  train ?
-  train  test , underfit / representation bottleneck?
-  sequence length features  target / error ?
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

#  PNG,;
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/seqproft_xdg_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/seqproft_matplotlib_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.affinity_cdr_dataset import CDRAwareAffinityDataset, SUCCESS_STATUS_VALUES  # noqa: E402
from src.affinity_cdr_evaluate import evaluate_cdr_affinity_model  # noqa: E402
from src.affinity_cdr_model import SeqProFTCDRAwareAffinityRegressor  # noqa: E402
from src.affinity_cross_attention_dataset import CrossAttentionAffinityDataset  # noqa: E402
from src.affinity_cross_attention_evaluate import (  # noqa: E402
    cross_attention_device,
    evaluate_cross_attention_affinity_model,
)
from src.affinity_cross_attention_model import SeqProFTCrossAttentionAffinityRegressor  # noqa: E402
from src.affinity_cross_attention_train import antigen_length_from_config  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402


POOLED_CONFIG_PATH = ROOT / "config_affinity_andd_antibody_v2_stratified_all_cdr_pooled_lr3e-5_e10.yaml"
CROSS_CONFIG_PATH = (
    ROOT / "config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_lr3e-5_e10.yaml"
)
OUTPUT_DIR = ROOT / "outputs" / "andd_antibody_v2_stratified" / "fit_diagnosis"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
REPORT_PATH = OUTPUT_DIR / "fit_diagnosis_report.md"
METRICS_PATH = OUTPUT_DIR / "fit_metrics_by_split.csv"
CORRELATION_PATH = OUTPUT_DIR / "feature_correlation_summary.csv"
FIGURE_DIR = ROOT / "outputs" / "final_reports" / "figures"
TRUE_PRED_FIGURE_PATH = FIGURE_DIR / "train_eval_true_predicted_scatter.png"
RESIDUAL_FIGURE_PATH = FIGURE_DIR / "train_eval_residual_scatter.png"

TARGET_COLUMN = "neg_log10_affinity_candidate"
TRUE_COLUMN = "true_neg_log10_affinity"
PRED_COLUMN = "predicted_neg_log10_affinity"
SPLITS = ["train", "val", "test"]
MODELS = ["all_cdr_pooled", "all_cdr_cross_attention"]
MODEL_LABELS = {
    "all_cdr_pooled": "All-CDR pooled",
    "all_cdr_cross_attention": "All-CDR cross-attention",
}
COLORS = {
    "all_cdr_pooled": "#247BA0",
    "all_cdr_cross_attention": "#C65A4A",
}
MARKERS = {
    "all_cdr_pooled": "o",
    "all_cdr_cross_attention": "^",
}
SIMPLE_FEATURES = ["HCDR3_len", "LCDR3_len", "total_CDR_len", "antigen_len"]
CDR_FIELDS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]
OPTIONAL_STRUCTURE_FEATURES = ["contact_count", "min_distance", "interface_residue_count"]


def successful_rows(csv_path: str | Path) -> pd.DataFrame:
    """ Dataset  CDR """
    frame = pd.read_csv(csv_path)
    heavy_ok = frame["heavy_cdr_status"].fillna("").astype(str).str.lower().isin(SUCCESS_STATUS_VALUES)
    light_ok = frame["light_cdr_status"].fillna("").astype(str).str.lower().isin(SUCCESS_STATUS_VALUES)
    return add_length_features(frame.loc[heavy_ok & light_ok].reset_index(drop=True).copy())


def sequence_length(value: object) -> float:
    """ sequence , NaN"""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    return float(len(text)) if text else float("nan")


def add_length_features(frame: pd.DataFrame) -> pd.DataFrame:
    """ CDR / antigen """
    for field in ["HCDR3", "LCDR3"]:
        frame[f"{field}_len"] = frame[field].map(sequence_length)
    frame["total_CDR_len"] = frame[CDR_FIELDS].apply(
        lambda row: sum(len(str(value).strip()) for value in row if pd.notna(value)),
        axis=1,
    ).astype(float)
    frame["antigen_len"] = frame["antigen_sequence"].map(sequence_length)
    return frame


def model_output_path(model_name: str, split: str) -> Path:
    """ supporting predictions """
    return PREDICTION_DIR / f"{model_name}_{split}_predictions.csv"


def existing_test_path(model_name: str) -> Path:
    """ test predictions ,"""
    if model_name == "all_cdr_pooled":
        return (
            ROOT
            / "outputs/andd_antibody_v2_stratified/all_cdr_pooled/"
            / "andd_antibody_v2_stratified_all_cdr_pooled_test_predictions.csv"
        )
    return ROOT / "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs/test_predictions.csv"


def prediction_frame(metadata: pd.DataFrame, true_values: list[float], pred_values: list[float]) -> pd.DataFrame:
    """ inference  metadata """
    columns = [
        "sample_id",
        "candidate_id",
        "source",
        "pdb_id",
        "ag_name",
        *CDR_FIELDS,
        "antigen_sequence",
    ]
    output = metadata[[column for column in columns if column in metadata.columns]].copy()
    output[TRUE_COLUMN] = true_values
    output[PRED_COLUMN] = pred_values
    output["error"] = output[PRED_COLUMN] - output[TRUE_COLUMN]
    output["absolute_error"] = output["error"].abs()
    return add_length_features(output)


def run_pooled_inference(config: dict, split: str, tokenizer, device: torch.device, model) -> pd.DataFrame:
    """ pooled checkpoint  split  inference"""
    dataset = CDRAwareAffinityDataset(
        csv_path=config[f"{split}_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        input_cdr_fields=config.get("input_cdr_fields"),
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    _, true_values, pred_values = evaluate_cdr_affinity_model(model, loader, device)
    return prediction_frame(dataset.data, true_values, pred_values)


def run_cross_inference(config: dict, split: str, tokenizer, device: torch.device, model) -> pd.DataFrame:
    """ cross-attention checkpoint  split  inference"""
    dataset = CrossAttentionAffinityDataset(
        csv_path=config[f"{split}_csv"],
        tokenizer=tokenizer,
        antigen_max_length=antigen_length_from_config(config),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    # Cross-attention  token matrices; inference  batch 
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    _, true_values, pred_values = evaluate_cross_attention_affinity_model(model, loader, device)
    return prediction_frame(dataset.data, true_values, pred_values)


def load_existing_test_predictions(model_name: str, split_frame: pd.DataFrame) -> pd.DataFrame:
    """ test output  length features"""
    predictions = pd.read_csv(existing_test_path(model_name))
    feature_sequence_columns = [*CDR_FIELDS, "antigen_sequence"]
    keep_columns = [
        "sample_id",
        "candidate_id",
        "source",
        "pdb_id",
        "ag_name",
        *feature_sequence_columns,
        TRUE_COLUMN,
        PRED_COLUMN,
        "error",
        "absolute_error",
    ]
    existing = predictions[[column for column in keep_columns if column in predictions.columns]].copy()
    missing_metadata = [column for column in feature_sequence_columns if column not in existing.columns]
    if missing_metadata:
        existing = existing.merge(
            split_frame[["sample_id", *missing_metadata]],
            how="left",
            on="sample_id",
        )
    if "error" not in existing.columns:
        existing["error"] = existing[PRED_COLUMN] - existing[TRUE_COLUMN]
    if "absolute_error" not in existing.columns:
        existing["absolute_error"] = existing["error"].abs()
    return add_length_features(existing)


def generate_missing_predictions() -> tuple[dict[tuple[str, str], pd.DataFrame], str]:
    """ checkpoint, train/val  predictions, test CSV"""
    pooled_config = load_config(str(POOLED_CONFIG_PATH))
    cross_config = load_config(str(CROSS_CONFIG_PATH))
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    devices = {
        "all_cdr_pooled": get_device(),
        "all_cdr_cross_attention": cross_attention_device(cross_config),
    }
    print(f"Pooled inference device: {devices['all_cdr_pooled']}")
    print(f"Cross-attention inference device: {devices['all_cdr_cross_attention']}")

    tokenizer = AutoTokenizer.from_pretrained(pooled_config["model_name"])
    pooled_model = SeqProFTCDRAwareAffinityRegressor(pooled_config).to(devices["all_cdr_pooled"])
    pooled_checkpoint = torch.load(pooled_config["checkpoint_path"], map_location=devices["all_cdr_pooled"])
    pooled_model.load_state_dict(pooled_checkpoint["model_state_dict"])
    pooled_model.eval()

    cross_model = SeqProFTCrossAttentionAffinityRegressor(cross_config).to(
        devices["all_cdr_cross_attention"]
    )
    cross_checkpoint = torch.load(cross_config["checkpoint_path"], map_location=devices["all_cdr_cross_attention"])
    cross_model.load_state_dict(cross_checkpoint["model_state_dict"])
    cross_model.eval()

    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    split_data = {
        split: successful_rows(pooled_config[f"{split}_csv"])
        for split in SPLITS
    }

    for split in SPLITS:
        if split == "test" and existing_test_path("all_cdr_pooled").exists():
            pooled_predictions = load_existing_test_predictions("all_cdr_pooled", split_data[split])
        else:
            print(f"Inference: all_cdr_pooled / {split}")
            pooled_predictions = run_pooled_inference(
                pooled_config, split, tokenizer, devices["all_cdr_pooled"], pooled_model
            )
        predictions[("all_cdr_pooled", split)] = pooled_predictions
        pooled_predictions.to_csv(model_output_path("all_cdr_pooled", split), index=False)

        if split == "test" and existing_test_path("all_cdr_cross_attention").exists():
            cross_predictions = load_existing_test_predictions(
                "all_cdr_cross_attention", split_data[split]
            )
        else:
            print(f"Inference: all_cdr_cross_attention / {split}")
            cross_predictions = run_cross_inference(
                cross_config, split, tokenizer, devices["all_cdr_cross_attention"], cross_model
            )
        predictions[("all_cdr_cross_attention", split)] = cross_predictions
        cross_predictions.to_csv(model_output_path("all_cdr_cross_attention", split), index=False)

    device_text = (
        f"pooled={devices['all_cdr_pooled']}, cross_attention={devices['all_cdr_cross_attention']}"
    )
    return predictions, device_text


def correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    """; NaN"""
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def metric_row(model_name: str, split: str, predictions: pd.DataFrame, train_target: pd.Series) -> dict[str, object]:
    """ train-defined bins/tails """
    true = predictions[TRUE_COLUMN].astype(float)
    pred = predictions[PRED_COLUMN].astype(float)
    errors = pred - true
    abs_error = errors.abs()
    low_edge = float(train_target.quantile(1 / 3))
    high_edge = float(train_target.quantile(2 / 3))
    p10 = float(train_target.quantile(0.10))
    p90 = float(train_target.quantile(0.90))
    bins = {
        "low": true <= low_edge,
        "mid": (true > low_edge) & (true <= high_edge),
        "high": true > high_edge,
        "below_p10": true <= p10,
        "above_p90": true >= p90,
    }

    def group_mae(mask: pd.Series) -> float:
        return float(abs_error[mask].mean()) if int(mask.sum()) else float("nan")

    return {
        "model": model_name,
        "split": split,
        "rows": int(len(predictions)),
        "MAE": float(abs_error.mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "Spearman": correlation(true, pred, "spearman"),
        "prediction_std": float(pred.std(ddof=1)),
        "true_std": float(true.std(ddof=1)),
        "pred_std_true_std": float(pred.std(ddof=1) / true.std(ddof=1)),
        "error_vs_true_Pearson": correlation(errors, true, "pearson"),
        "low_rows": int(bins["low"].sum()),
        "low_MAE": group_mae(bins["low"]),
        "mid_rows": int(bins["mid"].sum()),
        "mid_MAE": group_mae(bins["mid"]),
        "high_rows": int(bins["high"].sum()),
        "high_MAE": group_mae(bins["high"]),
        "below_P10_rows": int(bins["below_p10"].sum()),
        "below_P10_MAE": group_mae(bins["below_p10"]),
        "above_P90_rows": int(bins["above_p90"].sum()),
        "above_P90_MAE": group_mae(bins["above_p90"]),
    }


def build_feature_correlations(
    predictions: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """ target/error ; missing"""
    rows: list[dict[str, object]] = []
    missing_features: list[str] = []
    for split in SPLITS:
        reference = predictions[("all_cdr_pooled", split)]
        for feature in SIMPLE_FEATURES + OPTIONAL_STRUCTURE_FEATURES:
            if feature not in reference.columns:
                if feature not in missing_features:
                    missing_features.append(feature)
                rows.append(
                    {
                        "analysis": "target_vs_feature",
                        "model": "not_applicable",
                        "split": split,
                        "feature": feature,
                        "status": "missing_feature",
                        "n": 0,
                        "pearson": np.nan,
                        "spearman": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "analysis": "target_vs_feature",
                    "model": "not_applicable",
                    "split": split,
                    "feature": feature,
                    "status": "available",
                    "n": int(reference[[TRUE_COLUMN, feature]].dropna().shape[0]),
                    "pearson": correlation(reference[TRUE_COLUMN], reference[feature], "pearson"),
                    "spearman": correlation(reference[TRUE_COLUMN], reference[feature], "spearman"),
                }
            )
        for model_name in MODELS:
            model_frame = predictions[(model_name, split)]
            for feature in SIMPLE_FEATURES + OPTIONAL_STRUCTURE_FEATURES:
                if feature not in model_frame.columns:
                    rows.append(
                        {
                            "analysis": "absolute_error_vs_feature",
                            "model": model_name,
                            "split": split,
                            "feature": feature,
                            "status": "missing_feature",
                            "n": 0,
                            "pearson": np.nan,
                            "spearman": np.nan,
                        }
                    )
                    continue
                rows.append(
                    {
                        "analysis": "absolute_error_vs_feature",
                        "model": model_name,
                        "split": split,
                        "feature": feature,
                        "status": "available",
                        "n": int(model_frame[["absolute_error", feature]].dropna().shape[0]),
                        "pearson": correlation(
                            model_frame["absolute_error"], model_frame[feature], "pearson"
                        ),
                        "spearman": correlation(
                            model_frame["absolute_error"], model_frame[feature], "spearman"
                        ),
                    }
                )
    return pd.DataFrame(rows), missing_features


def draw_split_scatter(
    predictions: dict[tuple[str, str], pd.DataFrame],
    y_kind: str,
    output_path: Path,
) -> None:
    """ train/val/test """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    all_true = pd.concat([predictions[(model, split)][TRUE_COLUMN] for model in MODELS for split in SPLITS])
    if y_kind == "prediction":
        all_y = pd.concat([predictions[(model, split)][PRED_COLUMN] for model in MODELS for split in SPLITS])
        combined = pd.concat([all_true, all_y])
        pad = max((combined.max() - combined.min()) * 0.05, 0.12)
        shared_limits = (float(combined.min() - pad), float(combined.max() + pad))
    else:
        all_y = pd.concat([predictions[(model, split)]["error"] for model in MODELS for split in SPLITS])
        pad = max((all_y.max() - all_y.min()) * 0.05, 0.12)
        shared_limits = (float(all_y.min() - pad), float(all_y.max() + pad))

    for axis, split in zip(axes, SPLITS):
        for model_name in MODELS:
            frame = predictions[(model_name, split)]
            y_column = PRED_COLUMN if y_kind == "prediction" else "error"
            axis.scatter(
                frame[TRUE_COLUMN],
                frame[y_column],
                s=22 if split == "train" else 38,
                marker=MARKERS[model_name],
                color=COLORS[model_name],
                alpha=0.34 if split == "train" else 0.60,
                edgecolors="none",
                label=MODEL_LABELS[model_name],
            )
            fit = np.polyfit(frame[TRUE_COLUMN], frame[y_column], 1)
            x_grid = np.linspace(frame[TRUE_COLUMN].min(), frame[TRUE_COLUMN].max(), 100)
            axis.plot(x_grid, np.polyval(fit, x_grid), color=COLORS[model_name], linewidth=1.5)

        if y_kind == "prediction":
            axis.plot(
                shared_limits,
                shared_limits,
                linestyle="--",
                color="#4C5560",
                linewidth=1,
                label="Ideal y = x",
            )
            axis.set_xlim(shared_limits)
            axis.set_ylim(shared_limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_ylabel("Predicted target" if split == "train" else "")
        else:
            axis.axhline(0, linestyle="--", color="#4C5560", linewidth=1, label="Ideal residual = 0")
            axis.set_ylim(shared_limits)
            axis.set_ylabel("Prediction - true" if split == "train" else "")
            if split == "test":
                axis.text(
                    0.04,
                    0.05,
                    "Downward trend =\nregression to the mean",
                    transform=axis.transAxes,
                    fontsize=8.5,
                    bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "boxstyle": "round,pad=0.3"},
                )

        axis.set_title(f"{split.capitalize()} split")
        axis.set_xlabel("True target")
        axis.grid(color="#DDE2E8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if split == "train":
            axis.legend(frameon=False, fontsize=8, loc="best")

    title = (
        "True vs predicted: train / validation / test"
        if y_kind == "prediction"
        else "Residual vs true: train / validation / test"
    )
    fig.suptitle(f"ANDD stratified model fit diagnosis - {title}", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_value(value: object) -> str:
    """Markdown """
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def metrics_markdown(metrics: pd.DataFrame) -> str:
    """ metrics  Markdown ,"""
    columns = [
        "model", "split", "rows", "MAE", "RMSE", "Spearman", "pred_std_true_std",
        "error_vs_true_Pearson", "low_MAE", "mid_MAE", "high_MAE",
        "below_P10_MAE", "above_P90_MAE",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    result = [header, separator]
    for _, row in metrics[columns].iterrows():
        values = []
        for column in columns:
            if column in {"model", "split"}:
                values.append(str(row[column]))
            elif column == "rows":
                values.append(str(int(row[column])))
            else:
                values.append(format_value(row[column]))
        result.append("| " + " | ".join(values) + " |")
    return "\n".join(result)


def strongest_available_relationships(correlations: pd.DataFrame, analysis: str, model: str | None = None) -> pd.DataFrame:
    """ Spearman  simple-feature relationships"""
    frame = correlations[
        (correlations["analysis"] == analysis)
        & (correlations["status"] == "available")
        & (correlations["feature"].isin(SIMPLE_FEATURES))
    ].copy()
    if model is not None:
        frame = frame[frame["model"] == model]
    frame["abs_spearman"] = frame["spearman"].abs()
    return frame.sort_values("abs_spearman", ascending=False).head(5)


def write_report(
    metrics: pd.DataFrame,
    correlations: pd.DataFrame,
    missing_features: list[str],
    device_text: str,
) -> None:
    """ train-vs-test """
    pooled_train = metrics[(metrics["model"] == "all_cdr_pooled") & (metrics["split"] == "train")].iloc[0]
    pooled_test = metrics[(metrics["model"] == "all_cdr_pooled") & (metrics["split"] == "test")].iloc[0]
    cross_train = metrics[(metrics["model"] == "all_cdr_cross_attention") & (metrics["split"] == "train")].iloc[0]
    cross_test = metrics[(metrics["model"] == "all_cdr_cross_attention") & (metrics["split"] == "test")].iloc[0]

    compression_on_train = (
        pooled_train["pred_std_true_std"] < 0.6
        and cross_train["pred_std_true_std"] < 0.6
        and pooled_train["error_vs_true_Pearson"] < -0.7
        and cross_train["error_vs_true_Pearson"] < -0.7
    )
    if compression_on_train:
        fit_conclusion = (
            " train split  prediction compression  residual trend,"
            " underfit / representation  objective bottleneck,"
            "train  validation/test  overfit"
        )
    else:
        fit_conclusion = (
            "train  evaluation split ,;"
            " metrics  overfit"
        )

    target_top = strongest_available_relationships(correlations, "target_vs_feature")
    pooled_error_top = strongest_available_relationships(
        correlations, "absolute_error_vs_feature", "all_cdr_pooled"
    )
    cross_error_top = strongest_available_relationships(
        correlations, "absolute_error_vs_feature", "all_cdr_cross_attention"
    )

    def relationship_lines(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "-  available sequence feature correlation"
        lines = []
        for _, row in frame.iterrows():
            lines.append(
                f"- `{row['split']}` / `{row['feature']}`: "
                f"Pearson={format_value(row['pearson'])}, Spearman={format_value(row['spearman'])}"
            )
        return "\n".join(lines)

    missing_text = ", ".join(f"`{feature}`" for feature in missing_features) or "none"
    report = f"""# ANDD Stratified Train vs Eval/Test Model Fit Diagnosis

## 

 checkpoint  inference,, dataset 

- Dataset: `data/processed_affinity/expanded_affinity_antibody_v2_stratified/`
- Models: `all_cdr_pooled`  `all_cdr_cross_attention`
- Inference device in this run: `{device_text}`
- Existing test predictions were read from original output paths; missing train/val predictions were saved only under `fit_diagnosis/predictions/`.
- Low/mid/high bins  P10/P90 tails  **train target distribution** 

## Metrics By Model And Split

{metrics_markdown(metrics)}

## 1. Train  Regression-To-The-Mean?

- Pooled train: `pred_std / true_std={pooled_train['pred_std_true_std']:.4f}`, `error_vs_true_Pearson={pooled_train['error_vs_true_Pearson']:.4f}`
- Cross-attention train: `pred_std / true_std={cross_train['pred_std_true_std']:.4f}`, `error_vs_true_Pearson={cross_train['error_vs_true_Pearson']:.4f}`
- Pooled test: `pred_std / true_std={pooled_test['pred_std_true_std']:.4f}`, `error_vs_true_Pearson={pooled_test['error_vs_true_Pearson']:.4f}`
- Cross-attention test: `pred_std / true_std={cross_test['pred_std_true_std']:.4f}`, `error_vs_true_Pearson={cross_test['error_vs_true_Pearson']:.4f}`

{fit_conclusion}

## 2. UnderfitOverfit  Representation Bottleneck?

-  train  MAE  prediction spread , val/test , overfit
-  train  `pred_std / true_std`  `error_vs_true_Pearson`, target extremes
- :`regression-to-the-mean / representation-or-objective bottleneck`; overfit
- Cross-attention  pooled  learnable interaction , calibration/tail error

## 3. Simple Sequence Feature Relationships

### Target vs sequence features: Spearman  available relationships

{relationship_lines(target_top)}

### Pooled absolute error vs sequence features

{relationship_lines(pooled_error_top)}

### Cross-attention absolute error vs sequence features

{relationship_lines(cross_error_top)}

,; antigen groups  target 

## 4. Contact / Structure Feature Availability

 stratified dataset  structure/contact feature columns, correlation analysis:

- {missing_text}

, dataset 

## 5. 

1. **Multi-seed / checkpoint policy**: single-seed baseline, compression  cross-attention 
2. **Tail-aware training  calibration**: affinity extremes, tail-aware weightingranking/calibration objective  validation tail  checkpoint
3. **Structure/contact-aware features**: train , pooled sequence ;CDR-antigen interface/contact information 

## 

- Metrics: `outputs/andd_antibody_v2_stratified/fit_diagnosis/fit_metrics_by_split.csv`
- Feature correlations: `outputs/andd_antibody_v2_stratified/fit_diagnosis/feature_correlation_summary.csv`
- True-vs-predicted figure: `outputs/final_reports/figures/train_eval_true_predicted_scatter.png`
- Residual figure: `outputs/final_reports/figures/train_eval_residual_scatter.png`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    predictions, device_text = generate_missing_predictions()

    pooled_config = load_config(str(POOLED_CONFIG_PATH))
    train_target = successful_rows(pooled_config["train_csv"])[TARGET_COLUMN].astype(float)
    metric_rows = [
        metric_row(model_name, split, predictions[(model_name, split)], train_target)
        for model_name in MODELS
        for split in SPLITS
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(METRICS_PATH, index=False)

    correlations, missing_features = build_feature_correlations(predictions)
    correlations.to_csv(CORRELATION_PATH, index=False)
    draw_split_scatter(predictions, "prediction", TRUE_PRED_FIGURE_PATH)
    draw_split_scatter(predictions, "residual", RESIDUAL_FIGURE_PATH)
    write_report(metrics, correlations, missing_features, device_text)

    print(f"Saved report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(ROOT)}")
    print(f"Saved correlations: {CORRELATION_PATH.relative_to(ROOT)}")
    print(f"Saved figures: {TRUE_PRED_FIGURE_PATH.relative_to(ROOT)}, {RESIDUAL_FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
