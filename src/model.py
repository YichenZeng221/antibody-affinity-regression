"""模型代码：ESM-2 + LoRA + mean pooling + classification head。

人话解释：
这个文件是项目最核心的地方。
它定义了一个模型类：SeqProFTMiniClassifier。

模型流程是：

protein sequence tokens
-> ESM-2
-> LoRA adapter
-> mean pooling
-> linear classifier
-> logits

注意：
这里是最小可运行版本，不包含 contact map、CM-MAH、多任务 benchmark。
"""

import torch
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class SeqProFTMiniClassifier(nn.Module):
    """一个很小的 ESM-2 + LoRA 分类模型。

    人话解释：
    ESM-2 会给序列里的每个 token 输出一个向量。
    但分类任务通常需要“整条蛋白”的一个向量。

    所以我们做 mean pooling：
    把所有真实 token 的向量平均起来，得到一条蛋白的整体表示。
    """

    def __init__(self, config: dict):
        super().__init__()

        # 二分类时 num_labels = 2。
        # 模型最后会输出两个分数：class 0 的分数、class 1 的分数。
        self.num_labels = int(config["num_labels"])

        # AutoModel 会加载 ESM-2 主干模型 backbone。
        # 它只负责把 protein sequence 变成 hidden states，不自带我们的分类头。
        #
        # add_pooling_layer=False：
        # 我们自己写 mean pooling，所以不需要 Hugging Face 默认 pooler。
        self.esm = AutoModel.from_pretrained(
            config["model_name"],
            add_pooling_layer=False,
        )

        # LoRA 的作用：
        # 不直接训练 ESM-2 里所有大参数，而是在注意力层旁边加一些小 adapter。
        # 这样训练的参数少很多，适合小机器和教学项目。
        lora_config = LoraConfig(
            # r 是 LoRA 的秩，可以理解成 adapter 的“容量”。
            # r 越大，能学的东西越多，但训练参数也越多。
            r=int(config["lora_r"]),

            # lora_alpha 是缩放系数，控制 LoRA 更新的强度。
            lora_alpha=int(config["lora_alpha"]),

            # dropout 可以减少过拟合。
            lora_dropout=float(config["lora_dropout"]),

            # ESM-2 的 attention 里有 query/key/value。
            # 这里我们只给 query 和 value 加 LoRA，是常见的轻量做法。
            target_modules=["query", "value"],

            # 不额外训练 bias，保持 MVP 简单。
            bias="none",
        )
        self.esm = get_peft_model(self.esm, lora_config)

        # ESM-2 每个 token 输出向量的维度。
        # facebook/esm2_t6_8M_UR50D 的 hidden_size 是 320。
        hidden_size = self.esm.config.hidden_size

        # 分类头：
        # 输入：一条蛋白的 pooled vector，形状 [batch_size, hidden_size]
        # 输出：类别分数 logits，形状 [batch_size, num_labels]
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # CrossEntropyLoss 是多分类/二分类常用 loss。
        # 它内部会处理 softmax，所以训练时不用自己先 softmax。
        self.loss_fn = nn.CrossEntropyLoss()

    def mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """对 token embeddings 做平均池化，同时忽略 padding。

        人话解释：
        ESM-2 输出的是每个 token 一个向量：

            token_embeddings shape = [batch_size, sequence_length, hidden_size]

        但分类头需要每条蛋白一个向量：

            pooled_features shape = [batch_size, hidden_size]

        attention_mask 里：
        - 1 表示真实 token
        - 0 表示 padding

        所以平均时要忽略 padding，不然短序列会被很多 padding 影响。
        """

        # attention_mask 原本形状是 [batch_size, sequence_length]。
        # unsqueeze(-1) 后变成 [batch_size, sequence_length, 1]，
        # 这样才能和 token_embeddings 相乘。
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)

        # padding 位置的 mask 是 0，相乘后 padding 向量就变成 0。
        summed_embeddings = (token_embeddings * mask).sum(dim=1)

        # 每条序列真实 token 的数量。
        # clamp 是为了避免极端情况下除以 0。
        token_counts = mask.sum(dim=1).clamp(min=1e-9)

        # 平均：真实 token 向量总和 / 真实 token 数量。
        return summed_embeddings / token_counts

    def forward(self, input_ids, attention_mask, labels=None) -> dict:
        """模型前向传播：输入 token，输出 logits，可选输出 loss。

        人话解释：
        forward 就是“模型跑一遍”的过程。

        训练时：
        - 传入 labels
        - 返回 logits 和 loss

        推理时：
        - 不传 labels
        - 只返回 logits
        """

        # 1. 把 token ids 输入 ESM-2，得到每个 token 的 hidden vector。
        esm_outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = esm_outputs.last_hidden_state

        # 2. 把 token-level 表示变成 protein-level 表示。
        pooled_features = self.mean_pool(token_embeddings, attention_mask)

        # 3. 分类头输出每个类别的原始分数 logits。
        logits = self.classifier(pooled_features)

        output = {"logits": logits}

        # 4. 如果训练时传入了 labels，就顺便计算 loss。
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)

        return output
