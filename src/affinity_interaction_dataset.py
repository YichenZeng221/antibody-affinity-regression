"""Dataset for the first residue-level CDR-antigen interaction baseline.

中文人话说明：
这个分支不再把每条序列一开始就压成一个 pooled embedding。
它先保留 HCDR3、LCDR3、antigen 的 token-level 表示，后面的模型才能构建：

    CDR residue tokens x antigen residue tokens

第一版 interaction baseline 只使用：
    HCDR3 + LCDR3 + antigen_sequence

CDR 已经由 AbNumber + IMGT 写进 annotated CSV。这里不重新切 CDR，
只负责：
1. 过滤标准 CDR extraction 失败的 row。
2. 把序列交给 ESM tokenizer。
3. 返回训练模型需要的 tensors。
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


SUCCESS_STATUS_VALUES = {"ok", "success"}
INTERACTION_CDR_FIELDS = ["HCDR3", "LCDR3"]


class InteractionAffinityDataset(Dataset):
    """Read annotated CSVs for the HCDR3/LCDR3-antigen interaction model."""

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int,
        target_column: str,
        cdr_max_length: int = 64,
    ):
        raw_data = pd.read_csv(csv_path)
        self.raw_row_count = int(len(raw_data))
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.cdr_max_length = int(cdr_max_length)
        self.target_column = target_column

        required_columns = {
            "HCDR3",
            "LCDR3",
            "antigen_sequence",
            "heavy_cdr_status",
            "light_cdr_status",
            target_column,
        }
        missing_columns = required_columns - set(raw_data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing interaction columns: {sorted(missing_columns)}")

        # 只有标准 CDR extraction 成功的 row 才适合这个 baseline。
        # 原始 annotated CSV 保留失败 row；过滤只发生在这个 loader 内。
        heavy_ok = raw_data["heavy_cdr_status"].fillna("").astype(str).str.lower().isin(
            SUCCESS_STATUS_VALUES
        )
        light_ok = raw_data["light_cdr_status"].fillna("").astype(str).str.lower().isin(
            SUCCESS_STATUS_VALUES
        )
        cdr_values_exist = raw_data[INTERACTION_CDR_FIELDS].notna().all(axis=1)
        antigen_exists = raw_data["antigen_sequence"].notna()

        self.data = raw_data[heavy_ok & light_ok & cdr_values_exist & antigen_exists].reset_index(
            drop=True
        )
        self.filtered_out_count = self.raw_row_count - int(len(self.data))
        if self.data.empty:
            raise ValueError(f"{csv_path} has no rows ready for interaction-aware training.")

        # Regression label 是连续数值，不是 classification class id，所以用 float tensor。
        self.targets = pd.to_numeric(self.data[target_column], errors="raise").astype(float).tolist()

    def __len__(self) -> int:
        """Return row count after in-loader CDR extraction filtering."""

        return len(self.data)

    def tokenize_sequence(self, sequence: str, max_length: int) -> dict:
        """Tokenize one CDR or antigen sequence with fixed padded length.

        padding 让一个 batch 内 tensor 形状一致；
        attention_mask 会告诉 interaction model 哪些 token 是真实 token，
        哪些 token 只是 padding。
        """

        encoded = self.tokenizer(
            str(sequence),
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, index: int) -> dict:
        """Return HCDR3, LCDR3, antigen inputs and one float target."""

        row = self.data.iloc[index]
        hcdr3 = self.tokenize_sequence(row["HCDR3"], self.cdr_max_length)
        lcdr3 = self.tokenize_sequence(row["LCDR3"], self.cdr_max_length)
        antigen = self.tokenize_sequence(row["antigen_sequence"], self.max_length)

        return {
            "hcdr3_input_ids": hcdr3["input_ids"],
            "hcdr3_attention_mask": hcdr3["attention_mask"],
            "lcdr3_input_ids": lcdr3["input_ids"],
            "lcdr3_attention_mask": lcdr3["attention_mask"],
            "antigen_input_ids": antigen["input_ids"],
            "antigen_attention_mask": antigen["attention_mask"],
            "labels": torch.tensor(self.targets[index], dtype=torch.float32),
        }
