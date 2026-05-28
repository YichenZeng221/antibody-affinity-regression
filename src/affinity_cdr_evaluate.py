"""Evaluation helpers for the CDR-aware affinity baseline."""

from __future__ import annotations

import torch

from src.affinity_cdr_dataset import CDR_MODEL_SEQUENCE_KEYS, normalize_input_cdr_fields
from src.affinity_evaluate import compute_regression_metrics


def cdr_model_inputs(batch: dict, device, input_cdr_fields=None) -> dict:
    """Move selected CDR inputs plus antigen input onto the selected device."""

    model_inputs = {}
    for cdr_field in normalize_input_cdr_fields(input_cdr_fields):
        key = CDR_MODEL_SEQUENCE_KEYS[cdr_field]
        model_inputs[f"{key}_input_ids"] = batch[f"{key}_input_ids"].to(device)
        model_inputs[f"{key}_attention_mask"] = batch[f"{key}_attention_mask"].to(device)
    model_inputs["antigen_input_ids"] = batch["antigen_input_ids"].to(device)
    model_inputs["antigen_attention_mask"] = batch["antigen_attention_mask"].to(device)
    return model_inputs


def evaluate_cdr_affinity_model(model, dataloader, device) -> tuple[dict, list[float], list[float]]:
    """Run eval mode inference for CDR-aware batches."""

    model.eval()
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            outputs = model(**cdr_model_inputs(batch, device, model.input_cdr_fields))
            true_values.extend(labels.cpu().tolist())
            predicted_values.extend(outputs["predictions"].cpu().tolist())

    metrics = compute_regression_metrics(true_values, predicted_values)
    return metrics, true_values, predicted_values
