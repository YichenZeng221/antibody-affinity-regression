"""Evaluate the trained affinity regression model on the test set.
:
test set ,
 checkpoint, test.csv ,
, CSV
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.affinity_dataset import AffinityRegressionDataset
from src.affinity_evaluate import compute_regression_metrics, evaluate_affinity_model
from src.affinity_model import SeqProFTAffinityRegressor
from src.utils import get_device, load_config


OPTIONAL_METADATA_COLUMNS = ["pdb", "antibody_id", "antigen_id", "source"]


def parse_args() -> argparse.Namespace:
    """Read command line arguments.

     config_affinity.yaml
     --config  clean_v2 all_methods / spr_only  checkpoint
    """

    parser = argparse.ArgumentParser(description="Evaluate affinity regression test set.")
    parser.add_argument("--config", default="config_affinity.yaml")
    return parser.parse_args()


def build_prediction_row(row: dict, true_value: float, predicted_value: float) -> dict:
    """Build one predictions.csv row without assuming every dataset has pdb.

    :
    sequence_only / clean_v2  pdb 
    TDC v1  pdb, antibody_id / antigen_id / source
     row.get(...)  metadata, KeyError
    """

    error = predicted_value - true_value
    absolute_error = abs(error)

    #  target  -log10(affinity),log10 scale 
    # approximate fold error absolute_error=1  10 
    fold_error = 10 ** absolute_error

    prediction_row = {
        "sample_id": row.get("sample_id", ""),
        "true_neg_log10_affinity": true_value,
        "predicted_neg_log10_affinity": predicted_value,
        "error": error,
        "absolute_error": absolute_error,
        "fold_error": fold_error,
        "heavy_sequence": row.get("heavy_sequence", ""),
        "light_sequence": row.get("light_sequence", ""),
        "antigen_sequence": row.get("antigen_sequence", ""),
    }

    for column_name in OPTIONAL_METADATA_COLUMNS:
        if column_name in row:
            prediction_row[column_name] = row.get(column_name, "")

    return prediction_row


def top_error_columns(predictions: pd.DataFrame) -> list[str]:
    """Choose columns for top-error printing based on what exists."""

    columns = ["sample_id"]
    for column_name in ["pdb", "antibody_id", "antigen_id", "source"]:
        if column_name in predictions.columns:
            columns.append(column_name)

    columns.extend(
        [
            "true_neg_log10_affinity",
            "predicted_neg_log10_affinity",
            "error",
            "absolute_error",
            "fold_error",
        ]
    )
    return columns


def main() -> None:
    """Load checkpoint, evaluate test set, and save per-sample predictions."""

    args = parse_args()

    # ,max_length
    config = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    # tokenizer  model_name 
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    test_dataset = AffinityRegressionDataset(
        csv_path=config["test_csv"],
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        target_column=config["target_column"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    # , checkpoint 
    model = SeqProFTAffinityRegressor(config).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # evaluate_affinity_model  model.eval()  torch.no_grad(),
    # 
    metrics, true_values, predicted_values = evaluate_affinity_model(
        model, test_dataloader, device
    )

    print()
    print("Affinity test set evaluation")
    print(f"Test MAE: {metrics['mae']:.4f} log10 units")
    print(f"Approx average fold error from MAE: {metrics['approx_mae_fold_error']:.1f}x")
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f} log10 units")
    print(f"Approx fold error from RMSE: {metrics['approx_rmse_fold_error']:.1f}x")
    print(f"Test Spearman: {metrics['spearman']:.4f}")

    # raw_test  sample_id/pdb/sequence ,
    #  predictions CSV ,
    raw_test = pd.read_csv(config["test_csv"])
    prediction_rows = []

    for row, true_value, predicted_value in zip(
        raw_test.to_dict("records"),
        true_values,
        predicted_values,
    ):
        prediction_rows.append(build_prediction_row(row, true_value, predicted_value))

    predictions = pd.DataFrame(prediction_rows)
    predictions_path = Path(config["predictions_path"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    print()
    print("Top 10 largest prediction errors")
    if predictions.empty:
        print("No predictions to show.")
    else:
        columns = top_error_columns(predictions)
        print(
            predictions.sort_values("absolute_error", ascending=False)
            .head(10)[columns]
            .to_string(index=False)
        )

    print()
    print(f"Saved affinity test predictions to {predictions_path}")


if __name__ == "__main__":
    main()
