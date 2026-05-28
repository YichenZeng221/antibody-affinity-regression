"""PyTorch Dataset for antibody-antigen affinity regression.

中文人话说明：
classification 任务里，每个样本只有一条 sequence。
regression 任务里，每个样本有三条 sequence：

1. heavy_sequence
2. light_sequence
3. antigen_sequence

模型会分别 tokenize 这三条序列，再预测一个连续数值 target。
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class AffinityRegressionDataset(Dataset):
    """Load affinity regression CSV files.

    中文人话说明：
    PyTorch 的 Dataset 像一个“样本仓库”。
    DataLoader 每次会问 Dataset：“请给我第 index 个样本。”

    对这个任务来说，一个样本包含：
    - heavy_sequence
    - light_sequence
    - antigen_sequence
    - 一个 float target：neg_log10_affinity

    注意：regression 的 label 是连续数值，所以是 float；
    classification 的 label 才通常是 0/1/2 这种 class id。
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int, target_column: str):
        # CSV 是已经处理好的表格数据，每一行是一个 antibody-antigen complex。
        self.data = pd.read_csv(csv_path)

        # tokenizer 负责把氨基酸字母序列变成模型能读的数字 token。
        # 例如 "EVQL..." 会变成 input_ids。
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_column = target_column

        # 这里提前检查列名，能让错误更早、更清楚地暴露出来。
        required_columns = {
            "heavy_sequence",
            "light_sequence",
            "antigen_sequence",
            target_column,
        }
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

        # 转成 list 后，__getitem__ 按 index 取样本会更直接。
        self.heavy_sequences = self.data["heavy_sequence"].astype(str).tolist()
        self.light_sequences = self.data["light_sequence"].astype(str).tolist()
        self.antigen_sequences = self.data["antigen_sequence"].astype(str).tolist()

        # 回归任务的 target 是浮点数，比如 8.34，不是 class id。
        self.targets = self.data[target_column].astype(float).tolist()

    def __len__(self) -> int:
        """Return number of samples."""

        return len(self.data)

    def tokenize_sequence(self, sequence: str) -> dict:
        """Convert one amino acid sequence into ESM-2 input tensors.

        中文人话说明：
        - input_ids：每个氨基酸 token 对应的整数编号。
        - attention_mask：哪些位置是真 token，哪些位置是 padding。
          1 通常表示真实 token，0 表示 padding。
        - padding="max_length"：所有序列补齐到同样长度，方便组成 batch。
        - truncation=True：太长的序列会被截断到 max_length。

        padding 很重要，因为神经网络一个 batch 里的张量形状必须一致。
        """

        encoded = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            # tokenizer 返回的形状是 [1, max_length]。
            # squeeze(0) 去掉最前面的 1，变成 [max_length]，
            # 后面 DataLoader 会再把多个样本堆回 [batch_size, max_length]。
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, index: int) -> dict:
        """Return one model-ready sample.

        DataLoader 会把这里返回的 dict 自动拼成 batch。
        所以 key 名要和 model.forward(...) 的参数名对得上。
        """

        heavy = self.tokenize_sequence(self.heavy_sequences[index])
        light = self.tokenize_sequence(self.light_sequences[index])
        antigen = self.tokenize_sequence(self.antigen_sequences[index])

        return {
            "heavy_input_ids": heavy["input_ids"],
            "heavy_attention_mask": heavy["attention_mask"],
            "light_input_ids": light["input_ids"],
            "light_attention_mask": light["attention_mask"],
            "antigen_input_ids": antigen["input_ids"],
            "antigen_attention_mask": antigen["attention_mask"],
            # dtype=torch.float32 是 MSELoss 常用的回归 label 类型。
            "labels": torch.tensor(self.targets[index], dtype=torch.float32),
        }
