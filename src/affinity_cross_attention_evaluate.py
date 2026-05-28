"""Evaluation helpers shared by cross-attention train/evaluate/dry-run code."""

from __future__ import annotations

import torch

from src.affinity_cross_attention_dataset import CDR_MODEL_KEYS, CROSS_ATTENTION_CDR_FIELDS
from src.affinity_evaluate import compute_regression_metrics
from src.utils import get_device


def cross_attention_device(config: dict) -> torch.device:
    """Honor `mps_if_available` for this branch while keeping a CPU fallback."""

    device_request = str(config.get("device", "auto")).lower()
    if device_request in {"mps", "mps_if_available"}:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_request == "cpu":
        return torch.device("cpu")
    if device_request == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return get_device()


def cross_attention_model_inputs(batch: dict, device) -> dict:
    """Move six CDR streams plus antigen stream onto device."""

    model_inputs = {}
    for cdr_field in CROSS_ATTENTION_CDR_FIELDS:
        model_key = CDR_MODEL_KEYS[cdr_field]
        model_inputs[f"{model_key}_input_ids"] = batch[f"{model_key}_input_ids"].to(device)
        model_inputs[f"{model_key}_attention_mask"] = batch[
            f"{model_key}_attention_mask"
        ].to(device)
    model_inputs["antigen_input_ids"] = batch["antigen_input_ids"].to(device)
    model_inputs["antigen_attention_mask"] = batch["antigen_attention_mask"].to(device)
    return model_inputs


def evaluate_cross_attention_affinity_model(model, dataloader, device) -> tuple[dict, list, list]:
    """Run eval-mode inference and compute regression metrics."""

    model.eval()
    true_values = []
    predicted_values = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            outputs = model(**cross_attention_model_inputs(batch, device))
            true_values.extend(labels.cpu().tolist())
            predicted_values.extend(outputs["predictions"].cpu().tolist())
    return compute_regression_metrics(true_values, predicted_values), true_values, predicted_values
