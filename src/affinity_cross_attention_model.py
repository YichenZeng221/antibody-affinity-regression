"""All-CDR CDR-to-antigen learnable cross-attention affinity model.

中文人话说明：
这版不再用 dot-product interaction matrix 的固定 summary statistics。
它让 MultiheadAttention 学习：

    query = all CDR residue tokens
    key/value = antigen residue tokens

输出的 attended CDR tokens 是“看过 antigen 后”的 CDR 表示。
然后模型把：
1. attention-pooled attended CDR 表示
2. mean-pooled 原始 CDR 表示
3. mean-pooled antigen 表示
拼接起来，交给一个小 MLP 做 affinity regression。
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel

from src.affinity_cross_attention_dataset import CDR_MODEL_KEYS, CROSS_ATTENTION_CDR_FIELDS


class MaskedAttentionPooling(nn.Module):
    """Learn one scalar importance weight for each real token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return weighted token sum while excluding padding positions."""

        scores = self.score(token_embeddings).squeeze(-1)
        scores = scores.masked_fill(~attention_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), token_embeddings).squeeze(1)


class SeqProFTCrossAttentionAffinityRegressor(nn.Module):
    """Shared ESM-2 + LoRA with all-CDR queries attending to antigen tokens."""

    def __init__(self, config: dict):
        super().__init__()
        self.esm = AutoModel.from_pretrained(
            config["model_name"],
            add_pooling_layer=False,
        )
        lora_config = LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=["query", "value"],
            bias="none",
        )
        self.esm = get_peft_model(self.esm, lora_config)

        hidden_size = self.esm.config.hidden_size
        num_heads = int(config["num_attention_heads"])
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_heads}."
            )
        dropout = float(config.get("dropout", config.get("lora_dropout", 0.1)))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attended_cdr_pooling = MaskedAttentionPooling(hidden_size)
        mlp_hidden_size = int(config.get("regression_hidden_size", hidden_size))
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size * 3, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 1),
        )
        self.loss_fn = nn.MSELoss()

    def mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool real token embeddings and ignore padding."""

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_token_matrix(self, input_ids, attention_mask) -> torch.Tensor:
        """Encode one tokenized protein sequence with shared ESM-2 + LoRA."""

        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def forward(
        self,
        antigen_input_ids,
        antigen_attention_mask,
        labels=None,
        return_debug_shapes: bool = False,
        **cdr_inputs,
    ) -> dict:
        """Run all-CDR cross-attention and return `[batch_size]` predictions."""

        cdr_token_matrices = []
        cdr_masks = []
        per_cdr_shapes = {}
        for cdr_field in CROSS_ATTENTION_CDR_FIELDS:
            model_key = CDR_MODEL_KEYS[cdr_field]
            cdr_tokens = self.encode_token_matrix(
                cdr_inputs[f"{model_key}_input_ids"],
                cdr_inputs[f"{model_key}_attention_mask"],
            )
            cdr_token_matrices.append(cdr_tokens)
            cdr_masks.append(cdr_inputs[f"{model_key}_attention_mask"])
            per_cdr_shapes[f"{model_key}_token_matrix"] = tuple(cdr_tokens.shape)

        all_cdr_tokens = torch.cat(cdr_token_matrices, dim=1)
        all_cdr_mask = torch.cat(cdr_masks, dim=1)
        antigen_tokens = self.encode_token_matrix(antigen_input_ids, antigen_attention_mask)

        # MultiheadAttention 的 key_padding_mask 中 True 表示“忽略该 antigen 位置”。
        attended_cdr_tokens, _ = self.cross_attention(
            query=all_cdr_tokens,
            key=antigen_tokens,
            value=antigen_tokens,
            key_padding_mask=~antigen_attention_mask.bool(),
            need_weights=False,
        )
        attended_cdr_pooled = self.attended_cdr_pooling(attended_cdr_tokens, all_cdr_mask)
        original_cdr_pooled = self.mean_pool(all_cdr_tokens, all_cdr_mask)
        antigen_pooled = self.mean_pool(antigen_tokens, antigen_attention_mask)

        pooled_features = torch.cat(
            [attended_cdr_pooled, original_cdr_pooled, antigen_pooled],
            dim=-1,
        )
        predictions = self.regression_head(pooled_features).squeeze(-1)
        output = {"predictions": predictions}
        if labels is not None:
            output["loss"] = self.loss_fn(predictions, labels)
        if return_debug_shapes:
            output["debug_shapes"] = {
                **per_cdr_shapes,
                "all_cdr_tokens": tuple(all_cdr_tokens.shape),
                "antigen_tokens": tuple(antigen_tokens.shape),
                "attended_cdr_tokens": tuple(attended_cdr_tokens.shape),
                "attended_cdr_pooled": tuple(attended_cdr_pooled.shape),
                "original_cdr_pooled": tuple(original_cdr_pooled.shape),
                "antigen_pooled": tuple(antigen_pooled.shape),
                "pooled_features": tuple(pooled_features.shape),
                "predictions": tuple(predictions.shape),
            }
        return output
