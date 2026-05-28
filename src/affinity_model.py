"""ESM-2 + LoRA model for affinity regression.

中文人话说明：
这个模型使用一个 shared ESM-2 + LoRA encoder。
也就是说 heavy、light、antigen 三条序列共用同一个 ESM-2 模型。

流程：
    heavy sequence  -> shared ESM-2 -> mean pooling -> heavy embedding
    light sequence  -> shared ESM-2 -> mean pooling -> light embedding
    antigen sequence -> shared ESM-2 -> mean pooling -> antigen embedding

然后把三个 embedding 拼接起来，经过 regression head，输出一个 scalar。
"""

import torch
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class SeqProFTAffinityRegressor(nn.Module):
    """Shared ESM-2 + LoRA affinity regression model.

    中文人话说明：
    shared backbone 的意思是：
    heavy / light / antigen 三条序列都用同一个 ESM-2 encoder。

    为什么共用？
    因为它们本质上都是蛋白质序列，ESM-2 学到的是通用蛋白语言表示。
    如果创建三个独立 ESM-2，参数量会变成 3 倍，Stage 1 没必要。
    """

    def __init__(self, config: dict):
        super().__init__()

        # AutoModel 会从 Hugging Face 加载 ESM-2 backbone。
        # add_pooling_layer=False 表示我们自己做 mean pooling，不用模型默认 pooling。
        self.esm = AutoModel.from_pretrained(
            config["model_name"],
            add_pooling_layer=False,
        )

        # LoRA 可以理解成“给大模型旁边加一小组可训练 adapter”。
        # 原始 ESM-2 大部分参数保持不动，只训练 LoRA 参数和 regression head。
        # 这样显存/内存更省，也更适合小数据。
        lora_config = LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=["query", "value"],
            bias="none",
        )
        self.esm = get_peft_model(self.esm, lora_config)

        hidden_size = self.esm.config.hidden_size

        # 三个 pooled embedding 拼接，所以输入维度是 hidden_size * 3。
        self.regression_head = nn.Linear(hidden_size * 3, 1)

        # MSELoss = mean squared error。
        # 它会惩罚 prediction 和 label 的平方差，是回归任务最常见的 loss 之一。
        self.loss_fn = nn.MSELoss()

    def mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average token embeddings while ignoring padding tokens.

        中文人话说明：
        ESM-2 输出的是每个 token 的 embedding，形状大概是：
            [batch_size, sequence_length, hidden_size]

        但 regression head 想要每条蛋白一个固定长度向量。
        mean pooling 就是把所有 token embedding 求平均，得到：
            [batch_size, hidden_size]

        注意：padding token 是为了补齐长度的“假 token”，不能算进平均值。
        所以这里用 attention_mask 把 padding 的位置乘成 0。
        """

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed_embeddings = (token_embeddings * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed_embeddings / token_counts

    def encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode one sequence type with the shared ESM-2 model.

        这个函数不关心传进来的是 heavy、light 还是 antigen。
        它只负责：token ids -> ESM token embeddings -> pooled sequence embedding。
        """

        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(outputs.last_hidden_state, attention_mask)

    def forward(
        self,
        heavy_input_ids,
        heavy_attention_mask,
        light_input_ids,
        light_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
        labels=None,
    ) -> dict:
        """Run one forward pass.

        Prediction shape is [batch_size], matching labels shape [batch_size].

        中文人话说明：
        forward 就是模型“从输入算到输出”的过程。
        训练时 labels 不为空，会额外计算 loss。
        推理/评估时 labels 可以为空，只返回 predictions。
        """

        # 三条序列共用同一个 ESM-2 encoder，但各自得到自己的 embedding。
        heavy_embedding = self.encode_sequence(heavy_input_ids, heavy_attention_mask)
        light_embedding = self.encode_sequence(light_input_ids, light_attention_mask)
        antigen_embedding = self.encode_sequence(antigen_input_ids, antigen_attention_mask)

        # 拼接后，模型才能同时看到 antibody heavy、light 和 antigen 的信息。
        combined_embedding = torch.cat(
            [heavy_embedding, light_embedding, antigen_embedding],
            dim=-1,
        )

        # Linear(..., 1) 输出形状是 [batch_size, 1]。
        # squeeze(-1) 把最后一维去掉，变成 [batch_size]，和 labels 形状一致。
        predictions = self.regression_head(combined_embedding).squeeze(-1)
        output = {"predictions": predictions}

        if labels is not None:
            output["loss"] = self.loss_fn(predictions, labels)

        return output
