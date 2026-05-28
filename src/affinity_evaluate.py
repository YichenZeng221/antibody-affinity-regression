"""Evaluation metrics for affinity regression.

:
 loss ;


 regression metrics:
- MAE:,,
- MSE:,
- RMSE:MSE , target 
- Spearman:,/ binding 
"""

import math

import pandas as pd
import torch


def compute_regression_metrics(true_values: list[float], predicted_values: list[float]) -> dict:
    """Compute MAE, MSE, RMSE, Spearman, and approximate fold error.

    :
     target  -log10(affinity)
     1.0  log10 , affinity  10 
    """

    if len(true_values) == 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "spearman": 0.0,
            "approx_mae_fold_error": 1.0,
            "approx_rmse_fold_error": 1.0,
        }

    # error = prediction - true
    # ,
    errors = [pred - true for true, pred in zip(true_values, predicted_values)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]

    # MAE  abs(error), log10 unit
    mae = sum(absolute_errors) / len(absolute_errors)

    # MSE/RMSE 
    mse = sum(squared_errors) / len(squared_errors)
    rmse = math.sqrt(mse)

    # Spearman ,
    # ,Spearman  NaN
    spearman = pd.Series(true_values).corr(pd.Series(predicted_values), method="spearman")
    if pd.isna(spearman):
        spearman = 0.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "spearman": float(spearman),
        "approx_mae_fold_error": 10 ** mae,
        "approx_rmse_fold_error": 10 ** rmse,
    }


def evaluate_affinity_model(model, dataloader, device) -> tuple[dict, list[float], list[float]]:
    """Run inference on a dataloader and return metrics plus raw values.

    :
    evaluate ,:
    - model.eval(): dropout 
    - torch.no_grad():,,
    """

    # eval mode  train mode :eval mode  dropout 
    model.eval()
    true_values = []
    predicted_values = []

    # no_grad ,
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)

            outputs = model(
                heavy_input_ids=batch["heavy_input_ids"].to(device),
                heavy_attention_mask=batch["heavy_attention_mask"].to(device),
                light_input_ids=batch["light_input_ids"].to(device),
                light_attention_mask=batch["light_attention_mask"].to(device),
                antigen_input_ids=batch["antigen_input_ids"].to(device),
                antigen_attention_mask=batch["antigen_attention_mask"].to(device),
            )

            #  CPU list, pandas/math  CSV
            true_values.extend(labels.cpu().tolist())
            predicted_values.extend(outputs["predictions"].cpu().tolist())

    metrics = compute_regression_metrics(true_values, predicted_values)
    return metrics, true_values, predicted_values
