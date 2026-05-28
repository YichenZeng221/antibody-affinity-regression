"""Residue-level CDR-antigen interaction baseline for affinity regression.

中文人话说明：
pooled CDR baseline 会先把每条 CDR 序列平均成一个向量。
这很清楚，也很轻量，但 residue-to-residue 的配对信息会被压掉。

这个最小 interaction baseline 仍然不用 3D structure，也不做复杂 attention head。
它只做一件更显式的事：
1. 用 shared ESM-2 + LoRA 得到 HCDR3、LCDR3、antigen 的 token embeddings。
2. 把两个 CDR3 token matrix 拼成一个 CDR token matrix。
3. 计算 CDR token 和 antigen token 的 dot-product interaction matrix。
4. 从矩阵取几个简单 summary features，和 pooled embeddings 一起做 regression。
"""

from __future__ import annotations

import math

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel


class SeqProFTInteractionAffinityRegressor(nn.Module):
    """Shared ESM-2 + LoRA regressor with a simple residue interaction matrix."""

    interaction_feature_count = 5

    def __init__(self, config: dict):
        super().__init__()
        self.interaction_top_k = int(config.get("interaction_top_k", 5))
        if self.interaction_top_k < 1:
            raise ValueError("interaction_top_k must be >= 1.")

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
        # Features = pooled CDR embedding + pooled antigen embedding + 5 matrix summaries.
        self.regression_head = nn.Linear(
            hidden_size * 2 + self.interaction_feature_count,
            1,
        )
        self.loss_fn = nn.MSELoss()

    def mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average real token embeddings while ignoring padded positions."""

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed_embeddings = (token_embeddings * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed_embeddings / token_counts

    def encode_token_matrix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one protein sequence into token matrix plus pooled embedding."""

        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        pooled_embedding = self.mean_pool(token_embeddings, attention_mask)
        return token_embeddings, pooled_embedding

    def summarize_interaction_matrix(
        self,
        cdr_tokens: torch.Tensor,
        cdr_mask: torch.Tensor,
        antigen_tokens: torch.Tensor,
        antigen_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build masked interaction matrix and extract simple numeric summaries.

        interaction shape:
            [batch_size, cdr_token_length, antigen_token_length]

        padding positions exist only so batches have fixed tensor shapes.
        `pair_mask` makes sure matrix summaries ignore those fake positions.
        """

        interaction = torch.bmm(cdr_tokens, antigen_tokens.transpose(1, 2))
        pair_mask = cdr_mask.bool().unsqueeze(-1) & antigen_mask.bool().unsqueeze(1)
        pair_mask_float = pair_mask.to(interaction.dtype)

        pair_counts = pair_mask_float.sum(dim=(1, 2)).clamp(min=1.0)
        masked_sum = (interaction * pair_mask_float).sum(dim=(1, 2))
        interaction_mean = masked_sum / pair_counts

        very_low = torch.finfo(interaction.dtype).min
        masked_interaction = interaction.masked_fill(~pair_mask, very_low)
        interaction_max = masked_interaction.amax(dim=(1, 2))

        top_k_means = []
        row_max_means = []
        column_max_means = []
        # 不同 sample 的真实 token 数不同，top-k 在这里逐个 sample 计算最直观。
        for sample_index in range(interaction.shape[0]):
            valid_scores = interaction[sample_index][pair_mask[sample_index]]
            sample_k = min(self.interaction_top_k, int(valid_scores.numel()))
            top_k_means.append(valid_scores.topk(sample_k).values.mean())

            valid_cdr_rows = cdr_mask[sample_index].bool()
            valid_antigen_columns = antigen_mask[sample_index].bool()
            row_maxes = masked_interaction[sample_index].amax(dim=1)[valid_cdr_rows]
            column_maxes = masked_interaction[sample_index].amax(dim=0)[valid_antigen_columns]
            row_max_means.append(row_maxes.mean())
            column_max_means.append(column_maxes.mean())

        features = torch.stack(
            [
                interaction_mean,
                interaction_max,
                torch.stack(top_k_means),
                torch.stack(row_max_means),
                torch.stack(column_max_means),
            ],
            dim=-1,
        )
        return interaction, features

    def forward(
        self,
        hcdr3_input_ids,
        hcdr3_attention_mask,
        lcdr3_input_ids,
        lcdr3_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        labels=None,
        return_debug_shapes: bool = False,
    ) -> dict:
        """Run one HCDR3/LCDR3-antigen interaction regression forward pass."""

        hcdr3_tokens, _ = self.encode_token_matrix(hcdr3_input_ids, hcdr3_attention_mask)
        lcdr3_tokens, _ = self.encode_token_matrix(lcdr3_input_ids, lcdr3_attention_mask)
        antigen_tokens, antigen_pooled = self.encode_token_matrix(
            antigen_input_ids,
            antigen_attention_mask,
        )

        # 这里拼的是 token matrix，不是 raw sequence string。
        # shape 从两个 [B, cdr_max_length, H] 变成 [B, 2*cdr_max_length, H]。
        cdr_tokens = torch.cat([hcdr3_tokens, lcdr3_tokens], dim=1)
        cdr_mask = torch.cat([hcdr3_attention_mask, lcdr3_attention_mask], dim=1)
        cdr_pooled = self.mean_pool(cdr_tokens, cdr_mask)

        interaction, interaction_features = self.summarize_interaction_matrix(
            cdr_tokens,
            cdr_mask,
            antigen_tokens,
            antigen_attention_mask,
        )
        combined_features = torch.cat(
            [cdr_pooled, antigen_pooled, interaction_features],
            dim=-1,
        )
        predictions = self.regression_head(combined_features).squeeze(-1)

        output = {
            "predictions": predictions,
            "interaction_features": interaction_features,
        }
        if labels is not None:
            output["loss"] = self.loss_fn(predictions, labels)
        if return_debug_shapes:
            output["debug_shapes"] = {
                "hcdr3_token_matrix": tuple(hcdr3_tokens.shape),
                "lcdr3_token_matrix": tuple(lcdr3_tokens.shape),
                "cdr_token_matrix": tuple(cdr_tokens.shape),
                "antigen_token_matrix": tuple(antigen_tokens.shape),
                "interaction_matrix": tuple(interaction.shape),
                "interaction_features": tuple(interaction_features.shape),
                "combined_features": tuple(combined_features.shape),
                "predictions": tuple(predictions.shape),
            }
        return output
