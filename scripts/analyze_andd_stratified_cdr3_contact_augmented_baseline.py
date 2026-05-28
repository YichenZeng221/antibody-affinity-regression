"""Post-hoc CDR3 contact feature augmentation on ANDD stratified subsets.

:
 sequence prediction, contact-covered
train subset  Ridge :

    residual = target - sequence_prediction

 residual  test sequence prediction, CDR3 
 tail compression

:
-  dataset, predictions/checkpoints/reports
-  mapping validation  contact features
- HCDR3+LCDR3 subset  CDR3 loops ;
  `cdr_min_distance`  validation  all-six-CDR , all-CDR subset
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/seqproft_xdg_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/seqproft_matplotlib_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
#  Hugging Face cache;,
#  post-hoc audit 
os.environ.setdefault("HF_HOME", "/Users/yichenzeng/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.affinity_cross_attention_dataset import CrossAttentionAffinityDataset  # noqa: E402
from src.affinity_cross_attention_evaluate import (  # noqa: E402
    cross_attention_device,
    evaluate_cross_attention_affinity_model,
)
from src.affinity_cross_attention_model import SeqProFTCrossAttentionAffinityRegressor  # noqa: E402
from src.affinity_cross_attention_train import antigen_length_from_config  # noqa: E402
from src.utils import load_config  # noqa: E402


CONTACT_PATH = Path(
    "outputs/andd_antibody_v2_stratified/contact_feature_audit/preliminary_cdr_contact_features.csv"
)
DATA_DIR = Path("data/processed_affinity/expanded_affinity_antibody_v2_stratified")
OUTPUT_DIR = Path("outputs/andd_antibody_v2_stratified/contact_feature_audit")
METRICS_PATH = OUTPUT_DIR / "cdr3_contact_augmented_metrics.csv"
REPORT_PATH = OUTPUT_DIR / "cdr3_contact_augmented_baseline_report.md"
FIGURE_PATH = Path("outputs/final_reports/figures/cdr3_contact_augmented_baseline.png")

UNWEIGHTED_PREDICTION_DIR = Path(
    "outputs/andd_antibody_v2_stratified/fit_diagnosis/predictions"
)
UNWEIGHTED_PREDICTIONS = {
    split: UNWEIGHTED_PREDICTION_DIR / f"all_cdr_cross_attention_{split}_predictions.csv"
    for split in ("train", "test")
}
W2_CONFIG_PATH = Path(
    "config_affinity_andd_antibody_v2_stratified_cross_attention_all_cdrs_tailaware_w2_lr3e-5_e20.yaml"
)
W2_TEST_PREDICTION_PATH = Path(
    "outputs/andd_antibody_v2_stratified/cross_attention_all_cdrs_tailaware_w2/"
    "tailaware_w2_test_predictions_best_val_tail_mae.csv"
)

TARGET = "neg_log10_affinity_candidate"
TRUE = "true_neg_log10_affinity"
PRED = "predicted_neg_log10_affinity"
RIDGE_ALPHA = 1.0
SUBSETS = {
    "hcdr3_lcdr3_contact_safe": {
        "eligibility": "hcdr3_lcdr3_contact_feature_eligible",
        "features": [
            "hcdr3_contact_count_5A",
            "lcdr3_contact_count_5A",
            "hcdr3_contact_fraction_5A",
            "lcdr3_contact_fraction_5A",
        ],
        "description": "HCDR3+LCDR3 contact-safe subset",
    },
    "all_cdr_contact_safe": {
        "eligibility": "cdr_contact_feature_eligible",
        "features": [
            "hcdr3_contact_count_5A",
            "lcdr3_contact_count_5A",
            "hcdr3_contact_fraction_5A",
            "lcdr3_contact_fraction_5A",
            "cdr_min_distance",
            "all_cdr_contact_count_5A",
        ],
        "description": "All-CDR contact-safe subset",
    },
}


def load_contact_features() -> pd.DataFrame:
    frame = pd.read_csv(CONTACT_PATH)
    required = {
        "sample_id",
        "split",
        "hcdr3_lcdr3_contact_feature_eligible",
        "cdr_contact_feature_eligible",
        *SUBSETS["all_cdr_contact_safe"]["features"],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Preliminary contact table missing columns: {sorted(missing)}")
    return frame


def prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", TRUE, PRED}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    result = frame[["sample_id", TRUE, PRED]].copy()
    result["error"] = result[PRED] - result[TRUE]
    result["absolute_error"] = result["error"].abs()
    return result


def load_unweighted_predictions() -> dict[str, pd.DataFrame]:
    outputs = {}
    for split, path in UNWEIGHTED_PREDICTIONS.items():
        if not path.exists():
            raise FileNotFoundError(f"Existing unweighted prediction file missing: {path}")
        outputs[split] = prediction_frame(pd.read_csv(path))
    return outputs


def w2_inference_for_train() -> pd.DataFrame:
    """Use the existing trained w2 checkpoint for inference only; no fitting happens here."""
    config = load_config(str(W2_CONFIG_PATH))
    checkpoint_path = Path(config["checkpoint_paths"]["best_val_tail_mae"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Tail-aware w2 checkpoint missing: {checkpoint_path}")
    device = cross_attention_device(config)
    print(f"Tail-aware w2 train inference device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = CrossAttentionAffinityDataset(
        csv_path=config["train_csv"],
        tokenizer=tokenizer,
        antigen_max_length=antigen_length_from_config(config),
        cdr_max_length=int(config.get("cdr_max_length", 64)),
        target_column=config["target_column"],
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = SeqProFTCrossAttentionAffinityRegressor(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, true_values, predicted_values = evaluate_cross_attention_affinity_model(model, loader, device)
    result = pd.DataFrame(
        {
            "sample_id": dataset.data["sample_id"].astype(str),
            TRUE: true_values,
            PRED: predicted_values,
        }
    )
    return prediction_frame(result)


def load_w2_predictions() -> dict[str, pd.DataFrame]:
    if not W2_TEST_PREDICTION_PATH.exists():
        raise FileNotFoundError(f"Existing tail-aware w2 test predictions missing: {W2_TEST_PREDICTION_PATH}")
    return {
        "train": w2_inference_for_train(),
        "test": prediction_frame(pd.read_csv(W2_TEST_PREDICTION_PATH)),
    }


def train_thresholds() -> tuple[float, float]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    targets = pd.to_numeric(train[TARGET], errors="raise")
    return float(targets.quantile(0.10)), float(targets.quantile(0.90))


def regression_metrics(true: pd.Series, predicted: pd.Series, lower: float, upper: float) -> dict[str, float]:
    true = pd.to_numeric(true, errors="raise").astype(float)
    predicted = pd.to_numeric(predicted, errors="raise").astype(float)
    error = predicted - true
    absolute = error.abs()
    true_std = float(true.std())
    pred_std = float(predicted.std())
    below = true <= lower
    above = true >= upper
    below_mae = float(absolute[below].mean()) if below.any() else float("nan")
    above_mae = float(absolute[above].mean()) if above.any() else float("nan")
    return {
        "MAE": float(absolute.mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "Spearman": float(true.corr(predicted, method="spearman")),
        "prediction_std": pred_std,
        "true_std": true_std,
        "pred_std_true_std": pred_std / true_std if true_std else float("nan"),
        "error_vs_true_Pearson": float(error.corr(true, method="pearson")),
        "below_P10_MAE": below_mae,
        "above_P90_MAE": above_mae,
        "tail_MAE": float(np.nanmean([below_mae, above_mae])),
        "below_P10_rows": int(below.sum()),
        "above_P90_rows": int(above.sum()),
    }


def fit_residual_correction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, Pipeline]:
    """Fit one deliberately small linear correction using train contacts only."""
    residual = train[TRUE] - train[PRED]
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )
    model.fit(train[features], residual)
    predicted_residual = model.predict(test[features])
    corrected = test[PRED].to_numpy(dtype=float) + predicted_residual
    return corrected, model


def subset_with_predictions(
    contacts: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    eligibility: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = contacts[contacts[eligibility].fillna(False).astype(bool)].copy()
    result = {}
    for split in ("train", "test"):
        feature_split = filtered[filtered["split"] == split].copy()
        result[split] = feature_split.merge(
            predictions[split], on="sample_id", how="inner", validate="one_to_one"
        )
    return result["train"], result["test"]


def analyze_one_baseline(
    contacts: pd.DataFrame,
    baseline_label: str,
    predictions: dict[str, pd.DataFrame],
    lower: float,
    upper: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    for subset_name, definition in SUBSETS.items():
        features = definition["features"]
        train, test = subset_with_predictions(contacts, predictions, definition["eligibility"])
        if train[features].isna().any().any() or test[features].isna().any().any():
            raise ValueError(f"{subset_name} contains missing validated contact features.")
        raw = regression_metrics(test[TRUE], test[PRED], lower, upper)
        raw.update(
            {
                "subset": subset_name,
                "subset_description": definition["description"],
                "sequence_baseline": baseline_label,
                "method": "sequence_only_prediction",
                "train_rows": len(train),
                "test_rows": len(test),
                "features": "",
            }
        )
        metric_rows.append(raw)
        corrected_prediction, ridge = fit_residual_correction(train, test, features)
        corrected = regression_metrics(test[TRUE], pd.Series(corrected_prediction), lower, upper)
        corrected.update(
            {
                "subset": subset_name,
                "subset_description": definition["description"],
                "sequence_baseline": baseline_label,
                "method": "sequence_plus_contact_ridge_residual",
                "train_rows": len(train),
                "test_rows": len(test),
                "features": ";".join(features),
            }
        )
        metric_rows.append(corrected)
        raw_row = raw
        for metric in (
            "MAE",
            "RMSE",
            "Spearman",
            "pred_std_true_std",
            "error_vs_true_Pearson",
            "below_P10_MAE",
            "above_P90_MAE",
            "tail_MAE",
        ):
            corrected[f"delta_vs_sequence_{metric}"] = corrected[metric] - raw_row[metric]
        ridge_model = ridge.named_steps["ridge"]
        for feature, coefficient in zip(features, ridge_model.coef_):
            coefficients.append(
                {
                    "subset": subset_name,
                    "sequence_baseline": baseline_label,
                    "feature": feature,
                    "standardized_ridge_coefficient": float(coefficient),
                }
            )
    return metric_rows, coefficients


def make_figure(metrics: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("MAE", "MAE (lower is better)"),
        ("Spearman", "Spearman (higher is better)"),
        ("pred_std_true_std", "Prediction std / true std (closer to 1)"),
        ("tail_MAE", "P10/P90 tail MAE (lower is better)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    subset_order = list(SUBSETS)
    baseline_order = ["unweighted_cross_attention", "tailaware_w2_best_val_tail_mae"]
    positions = np.arange(len(subset_order) * len(baseline_order))
    labels = [
        f"{subset.replace('_contact_safe', '')}\n{baseline.replace('_best_val_tail_mae', '')}"
        for subset in subset_order
        for baseline in baseline_order
    ]
    for axis, (metric, title) in zip(axes.flat, panels):
        raw_values = []
        corrected_values = []
        for subset in subset_order:
            for baseline in baseline_order:
                pair = metrics[
                    (metrics["subset"] == subset) & (metrics["sequence_baseline"] == baseline)
                ]
                raw_values.append(
                    float(pair[pair["method"] == "sequence_only_prediction"][metric].iloc[0])
                )
                corrected_values.append(
                    float(pair[pair["method"] == "sequence_plus_contact_ridge_residual"][metric].iloc[0])
                )
        width = 0.36
        axis.bar(positions - width / 2, raw_values, width, label="Sequence only", color="#517DB6")
        axis.bar(
            positions + width / 2,
            corrected_values,
            width,
            label="+ CDR3 contact Ridge correction",
            color="#D16B42",
        )
        if metric == "pred_std_true_std":
            axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle(
        "Post-hoc CDR3 contact feature augmentation on contact-covered subsets",
        fontsize=15,
    )
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    output = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        output.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return output


def write_report(metrics: pd.DataFrame, coefficients: pd.DataFrame, lower: float, upper: float) -> None:
    view = metrics[
        [
            "subset_description",
            "sequence_baseline",
            "method",
            "train_rows",
            "test_rows",
            "MAE",
            "RMSE",
            "Spearman",
            "pred_std_true_std",
            "error_vs_true_Pearson",
            "below_P10_MAE",
            "above_P90_MAE",
            "tail_MAE",
        ]
    ].copy()
    lines = [
        "# ANDD Stratified CDR3 Contact Feature Augmented Baseline",
        "",
        "## Question",
        "",
        " CDR3 interface geometry features ,"
        " cross-attention predictions , tail compression?",
        "",
        "## Experimental Boundary",
        "",
        "-  **contact-covered subset analysis**, full 1,168-row benchmark",
        "-  chain/CDR-to-structure mapping validation ; ambiguous chain mapping",
        "- :sequence predictions  cross-attention models",
        "- Augmentation : train subset  `Ridge(alpha=1.0)`  "
        "`target - sequence_prediction`, subset  test rows  residual correction",
        "-  test labels  feature ",
        f"- Tail thresholds  stratified train split:P10 = `{lower:.4f}`,P90 = `{upper:.4f}`",
        "",
        "## Subsets and Features",
        "",
        "- `HCDR3+LCDR3 contact-safe`: 467 total validated rows; "
        "`hcdr3_contact_count_5A`, `lcdr3_contact_count_5A`, "
        "`hcdr3_contact_fraction_5A`, `lcdr3_contact_fraction_5A`",
        "- `All-CDR contact-safe`: 422 total validated rows; "
        "`cdr_min_distance`  `all_cdr_contact_count_5A`",
        "- `cdr_min_distance`  all-six-CDR  antigen ,"
        " CDR3-only subset,",
        "",
        "## Test Metrics Within The Same Subset",
        "",
    ]
    lines.extend(markdown_table(view, list(view.columns)))
    lines.extend(["", "## Delta After Contact Residual Correction", ""])
    corrected = metrics[metrics["method"] == "sequence_plus_contact_ridge_residual"].copy()
    delta_columns = [
        "subset_description",
        "sequence_baseline",
        "delta_vs_sequence_MAE",
        "delta_vs_sequence_RMSE",
        "delta_vs_sequence_Spearman",
        "delta_vs_sequence_pred_std_true_std",
        "delta_vs_sequence_error_vs_true_Pearson",
        "delta_vs_sequence_tail_MAE",
    ]
    lines.extend(markdown_table(corrected, delta_columns))
    lines.extend(
        [
            "",
            ":MAE/RMSE/tail MAE  delta  0 ;Spearman  delta  0 ;"
            "`pred_std/true_std`  1,`error_vs_true_Pearson`  0",
            "",
            "## Standardized Ridge Coefficients",
            "",
            " correction ,",
        ]
    )
    coef_view = coefficients.copy()
    lines.extend(markdown_table(coef_view, list(coef_view.columns)))

    interpretation = []
    for _, row in corrected.iterrows():
        raw = metrics[
            (metrics["subset"] == row["subset"])
            & (metrics["sequence_baseline"] == row["sequence_baseline"])
            & (metrics["method"] == "sequence_only_prediction")
        ].iloc[0]
        wins = []
        if row["MAE"] < raw["MAE"]:
            wins.append("MAE")
        if row["Spearman"] > raw["Spearman"]:
            wins.append("Spearman")
        if abs(row["pred_std_true_std"] - 1) < abs(raw["pred_std_true_std"] - 1):
            wins.append("prediction spread")
        if abs(row["error_vs_true_Pearson"]) < abs(raw["error_vs_true_Pearson"]):
            wins.append("residual compression")
        if row["tail_MAE"] < raw["tail_MAE"]:
            wins.append("tail MAE")
        interpretation.append(
            f"- {row['subset_description']} / `{row['sequence_baseline']}`: "
            + (f"improved {', '.join(wins)}." if wins else "no monitored metric improved.")
        )
    w2_reading = corrected[corrected["sequence_baseline"] == "tailaware_w2_best_val_tail_mae"]
    w2_lines = []
    for _, row in w2_reading.iterrows():
        raw = metrics[
            (metrics["subset"] == row["subset"])
            & (metrics["sequence_baseline"] == row["sequence_baseline"])
            & (metrics["method"] == "sequence_only_prediction")
        ].iloc[0]
        w2_lines.append(
            f"- `{row['subset_description']}`: MAE `{raw['MAE']:.4f} -> {row['MAE']:.4f}`, "
            f"Spearman `{raw['Spearman']:.4f} -> {row['Spearman']:.4f}`, "
            f"tail MAE `{raw['tail_MAE']:.4f} -> {row['tail_MAE']:.4f}`, "
            f"pred_std/true_std `{raw['pred_std_true_std']:.4f} -> {row['pred_std_true_std']:.4f}`, "
            f"error-vs-true Pearson `{raw['error_vs_true_Pearson']:.4f} -> "
            f"{row['error_vs_true_Pearson']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Honest Interpretation",
            "",
            *interpretation,
            "",
            "### Primary Reading For Tail-Aware w2",
            "",
            *w2_lines,
            "",
            "-  tail-aware w2,CDR3 contact correction  subset  "
            "MAE/RMSE/Spearman/tail-MAE , interface geometry ",
            "-  prediction spread  1 ,error-vs-true Pearson  0 ;"
            " correction ** regression-to-the-mean **",
            "-  unweighted baseline : subset , error "
            " ranking/tail contact features ,"
            "",
            "",
            "-  correction  subset , full 1,168-row model:"
            " mapping ",
            "-  baseline  subset  tail/spread , CDR3 geometry "
            ";, contact counts ,",
            "-  post-hoc linear correction, structure model,",
            "",
            "## Outputs",
            "",
            f"- Metrics: `{METRICS_PATH}`",
            f"- Figure: `{FIGURE_PATH}`",
            f"- Report: `{REPORT_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contacts = load_contact_features()
    lower, upper = train_thresholds()
    prediction_sets = {
        "unweighted_cross_attention": load_unweighted_predictions(),
        "tailaware_w2_best_val_tail_mae": load_w2_predictions(),
    }
    rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for label, predictions in prediction_sets.items():
        metric_rows, coefs = analyze_one_baseline(contacts, label, predictions, lower, upper)
        rows.extend(metric_rows)
        coefficient_rows.extend(coefs)
    metrics = pd.DataFrame(rows)
    coefficients = pd.DataFrame(coefficient_rows)
    metrics.to_csv(METRICS_PATH, index=False)
    make_figure(metrics)
    write_report(metrics, coefficients, lower, upper)
    print("CDR3 contact augmented residual-correction baseline complete.")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
