"""Dataset for the all-CDR CDR-to-antigen cross-attention affinity baseline.

中文人话说明：
这个分支使用 standard AbNumber + IMGT 已经提取好的六个 CDR：

    HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3

Dataset 不重新做 CDR extraction，也不改变 annotated CSV。它只负责：
1. 过滤 heavy/light CDR extraction 失败的 row。
2. 把六个 CDR 和 antigen sequence tokenize 成 ESM-2 tensors。
3. 返回 attention_mask，让模型可以忽略 padding token。
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


SUCCESS_STATUS_VALUES = {"ok", "success"}
CROSS_ATTENTION_CDR_FIELDS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]
CDR_MODEL_KEYS = {
    "HCDR1": "hcdr1",
    "HCDR2": "hcdr2",
    "HCDR3": "hcdr3",
    "LCDR1": "lcdr1",
    "LCDR2": "lcdr2",
    "LCDR3": "lcdr3",
}


class CrossAttentionAffinityDataset(Dataset):
    """Return six CDR token streams, antigen token stream, and float target."""

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        antigen_max_length: int,
        target_column: str,
        cdr_max_length: int = 64,
    ):
        raw_data = pd.read_csv(csv_path)
        self.raw_row_count = int(len(raw_data))
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.antigen_max_length = int(antigen_max_length)
        self.cdr_max_length = int(cdr_max_length)
        self.target_column = target_column

        required_columns = set(CROSS_ATTENTION_CDR_FIELDS) | {
            "antigen_sequence",
            "heavy_cdr_status",
            "light_cdr_status",
            target_column,
        }
        missing_columns = required_columns - set(raw_data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing cross-attention columns: {sorted(missing_columns)}")

        heavy_ok = raw_data["heavy_cdr_status"].fillna("").astype(str).str.lower().isin(
            SUCCESS_STATUS_VALUES
        )
        light_ok = raw_data["light_cdr_status"].fillna("").astype(str).str.lower().isin(
            SUCCESS_STATUS_VALUES
        )
        cdr_values_exist = raw_data[CROSS_ATTENTION_CDR_FIELDS].notna().all(axis=1)
        antigen_exists = raw_data["antigen_sequence"].notna()
        self.data = raw_data[heavy_ok & light_ok & cdr_values_exist & antigen_exists].reset_index(
            drop=True
        )
        self.filtered_out_count = self.raw_row_count - int(len(self.data))
        if self.data.empty:
            raise ValueError(f"{csv_path} has no rows ready for all-CDR cross-attention.")

        # Affinity regression 的 label 是连续 target，所以这里使用 float。
        self.targets = pd.to_numeric(self.data[target_column], errors="raise").astype(float).tolist()

    def __len__(self) -> int:
        """Return sample count after CDR-success filtering."""

        return len(self.data)

    def tokenize_sequence(self, sequence: str, max_length: int) -> dict:
        """Tokenize one CDR or antigen sequence with padding and truncation."""

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
        """Tokenize all six CDRs and antigen for one sample."""

        row = self.data.iloc[index]
        item = {"labels": torch.tensor(self.targets[index], dtype=torch.float32)}
        for cdr_field in CROSS_ATTENTION_CDR_FIELDS:
            encoded = self.tokenize_sequence(row[cdr_field], self.cdr_max_length)
            model_key = CDR_MODEL_KEYS[cdr_field]
            item[f"{model_key}_input_ids"] = encoded["input_ids"]
            item[f"{model_key}_attention_mask"] = encoded["attention_mask"]

        antigen = self.tokenize_sequence(row["antigen_sequence"], self.antigen_max_length)
        item["antigen_input_ids"] = antigen["input_ids"]
        item["antigen_attention_mask"] = antigen["attention_mask"]
        return item
