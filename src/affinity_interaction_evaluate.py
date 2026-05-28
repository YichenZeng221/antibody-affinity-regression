"""Evaluation helpers for interaction-aware affinity regression."""

from __future__ import annotations

import torch

from src.affinity_evaluate import compute_regression_metrics


INTERACTION_BATCH_KEYS = [
    "hcdr3_input_ids",
    "hcdr3_attention_mask",
    "lcdr3_input_ids",
    "lcdr3_attention_mask",
    "antigen_input_ids",
    "antigen_attention_mask",
]


def interaction_model_inputs(batch: dict, device) -> dict:
    """Move HCDR3/LCDR3/antigen tensors onto the requested device."""

    return {key: batch[key].to(device) for key in INTERACTION_BATCH_KEYS}


def evaluate_interaction_affinity_model(model, dataloader, device) -> tuple[dict, list, list]:
    """Run eval-mode predictions and return regression metrics plus raw values."""

    model.eval()
    true_values = []
    predicted_values = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            outputs = model(**interaction_model_inputs(batch, device))
            true_values.extend(labels.cpu().tolist())
            predicted_values.extend(outputs["predictions"].cpu().tolist())
    metrics = compute_regression_metrics(true_values, predicted_values)
    return metrics, true_values, predicted_values
