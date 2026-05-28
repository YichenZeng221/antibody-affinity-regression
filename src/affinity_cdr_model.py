"""ESM-2 + LoRA model for the first CDR-aware affinity baseline.

:
 whole-sequence baseline  shared ESM-2 + LoRA backbone
:
    six CDR sequences + antigen sequence

mean pool, 7  pooled embeddings concat,
 linear regression head  scalar
"""

from __future__ import annotations

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from src.affinity_cdr_dataset import CDR_MODEL_SEQUENCE_KEYS, normalize_input_cdr_fields


class SeqProFTCDRAwareAffinityRegressor(nn.Module):
    """Shared ESM-2 + LoRA regressor for six CDRs and one antigen."""

    def __init__(self, config: dict):
        super().__init__()
        self.input_cdr_fields = normalize_input_cdr_fields(config.get("input_cdr_fields"))
        self.esm = AutoModel.from_pretrained(
            config["model_name"],
            add_pooling_layer=False,
        )

        #  whole-sequence baseline  LoRA target modules ,
        # 
        lora_config = LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=["query", "value"],
            bias="none",
        )
        self.esm = get_peft_model(self.esm, lora_config)

        hidden_size = self.esm.config.hidden_size
        self.regression_head = nn.Linear(hidden_size * (len(self.input_cdr_fields) + 1), 1)
        self.loss_fn = nn.MSELoss()

    def mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average real token embeddings while ignoring padding tokens."""

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed_embeddings = (token_embeddings * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed_embeddings / token_counts

    def encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode one CDR or antigen sequence with the shared protein encoder."""

        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(outputs.last_hidden_state, attention_mask)

    def forward(self, antigen_input_ids, antigen_attention_mask, labels=None, **sequence_inputs) -> dict:
        """Run one regression forward pass and return predictions with shape [B]."""

        embeddings = []
        for cdr_field in self.input_cdr_fields:
            model_key = CDR_MODEL_SEQUENCE_KEYS[cdr_field]
            embeddings.append(
                self.encode_sequence(
                    sequence_inputs[f"{model_key}_input_ids"],
                    sequence_inputs[f"{model_key}_attention_mask"],
                )
            )
        embeddings.append(self.encode_sequence(antigen_input_ids, antigen_attention_mask))
        combined_embedding = torch.cat(embeddings, dim=-1)
        predictions = self.regression_head(combined_embedding).squeeze(-1)

        output = {"predictions": predictions}
        if labels is not None:
            output["loss"] = self.loss_fn(predictions, labels)
        return output
