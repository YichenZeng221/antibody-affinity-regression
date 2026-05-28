"""数据读取代码：把 CSV 里的蛋白序列变成模型能吃的 tensor。

人话解释：
PyTorch 训练模型时，通常不会直接把 CSV 文件丢给模型。
我们会写一个 Dataset 类，它负责：

1. 读取 CSV
2. 取出 protein sequence 和 label
3. 用 ESM-2 tokenizer 把氨基酸字母变成 token id
4. 返回 PyTorch tensor

模型只能处理数字 tensor，不能直接处理 "MKTAY..." 这种字符串。
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class ProteinSequenceDataset(Dataset):
    """从 CSV 文件读取蛋白序列和标签。

    CSV 格式应该长这样：

    sequence,label
    MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP,0
    KKKKKKKKRRRRRRRRHHHHHHHHKKRRHKRH,1

    人话解释：
    - sequence：蛋白质氨基酸序列
    - label：分类标签，这里是 0 或 1

    现在的数据是 toy dataset，只是为了学习流程。
    它还不是一个真正有生物学意义的 benchmark。
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int):
        # 读取 CSV 文件。pandas 会把它变成一个表格对象。
        self.data = pd.read_csv(csv_path)

        # tokenizer 来自 Hugging Face，会把氨基酸字母转成模型需要的 token id。
        self.tokenizer = tokenizer

        # max_length 控制最长序列长度。太长的序列会被截断。
        self.max_length = max_length

        # 确认 CSV 里必须有 sequence 和 label 两列。
        # 如果列名写错，早点报错会更容易 debug。
        required_columns = {"sequence", "label"}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

        # 把表格里的两列保存成 Python list，后面 __getitem__ 会按 index 取。
        self.sequences = self.data["sequence"].astype(str).tolist()
        self.labels = self.data["label"].astype(int).tolist()

    def __len__(self) -> int:
        """返回数据集中一共有多少条样本。

        PyTorch DataLoader 会调用这个函数来知道 dataset 的大小。
        """

        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        """把一条 protein sequence 转成模型需要的输入。

        人话解释：
        DataLoader 每次会问 Dataset：“请给我第 index 条数据。”
        这里我们返回一个字典，里面有：

        - input_ids：序列 token 编号
        - attention_mask：哪些位置是真 token，哪些位置是 padding
        - labels：正确答案，0 或 1
        """

        sequence = self.sequences[index]
        label = self.labels[index]

        # tokenizer 做三件事：
        # 1. 把蛋白序列字符串转成 token ids
        # 2. padding="max_length"：补齐到固定长度，方便组成 batch
        # 3. truncation=True：如果序列太长，就截断到 max_length
        encoded = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # tokenizer 返回的 tensor 形状通常是 [1, max_length]。
        # squeeze(0) 去掉最前面的 1，变成 [max_length]。
        # DataLoader 之后会把多条样本自动堆叠成 [batch_size, max_length]。
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
