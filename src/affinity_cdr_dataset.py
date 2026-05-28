"""Dataset for the first CDR-aware affinity regression baseline.

中文人话说明：
whole-sequence baseline 读三条序列：
    heavy_sequence + light_sequence + antigen_sequence

这个 CDR-aware baseline 改成更聚焦的七条序列：
    HCDR1 + HCDR2 + HCDR3
    LCDR1 + LCDR2 + LCDR3
    antigen_sequence

CDR 已经由 AbNumber + IMGT 提取好。这里不重新做 CDR extraction，
只负责过滤提取失败的 row、tokenize 序列、返回 PyTorch tensors。
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


CDR_SEQUENCE_COLUMNS = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]
DEFAULT_INPUT_CDR_FIELDS = CDR_SEQUENCE_COLUMNS.copy()
CDR_MODEL_SEQUENCE_KEYS = {
    "HCDR1": "hcdr1",
    "HCDR2": "hcdr2",
    "HCDR3": "hcdr3",
    "LCDR1": "lcdr1",
    "LCDR2": "lcdr2",
    "LCDR3": "lcdr3",
}
SUCCESS_STATUS_VALUES = {"ok", "success"}


def normalize_input_cdr_fields(input_cdr_fields=None) -> list[str]:
    """Validate the config-selected CDR subset for ablation experiments."""

    fields = DEFAULT_INPUT_CDR_FIELDS if input_cdr_fields is None else list(input_cdr_fields)
    if not fields:
        raise ValueError("input_cdr_fields must contain at least one CDR column.")
    unknown_fields = [field for field in fields if field not in CDR_SEQUENCE_COLUMNS]
    if unknown_fields:
        raise ValueError(f"Unknown CDR fields: {unknown_fields}. Allowed: {CDR_SEQUENCE_COLUMNS}")
    if len(fields) != len(set(fields)):
        raise ValueError(f"input_cdr_fields contains duplicate fields: {fields}")
    return fields


class CDRAwareAffinityDataset(Dataset):
    """Read annotated CDR CSVs and return model-ready CDR/antigen tensors."""

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int,
        target_column: str,
        cdr_max_length: int = 64,
        input_cdr_fields=None,
    ):
        raw_data = pd.read_csv(csv_path)
        self.raw_row_count = int(len(raw_data))
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cdr_max_length = cdr_max_length
        self.input_cdr_fields = normalize_input_cdr_fields(input_cdr_fields)
        self.target_column = target_column

        required_columns = set(self.input_cdr_fields) | {
            "antigen_sequence",
            "heavy_cdr_status",
            "light_cdr_status",
            target_column,
        }
        missing_columns = required_columns - set(raw_data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")

        # 注释数据目前写的是 `success`；用户的过滤规则写的是 `ok`。
        # 这里两个都接受，表达的是同一件事：heavy/light CDR 都提取成功。
        heavy_ok = raw_data["heavy_cdr_status"].fillna("").astype(str).str.lower().isin(
            SUCCESS_STATUS_VALUES
        )
        light_ok = raw_data["light_cdr_status"].fillna("").astype(str).str.lower().isin(
            SUCCESS_STATUS_VALUES
        )
        self.data = raw_data[heavy_ok & light_ok].reset_index(drop=True).copy()
        self.filtered_out_count = self.raw_row_count - int(len(self.data))

        if self.data.empty:
            raise ValueError(
                f"{csv_path} has no rows with successful heavy and light CDR extraction."
            )

        # CDR-aware regression 仍然预测连续 target，所以 label 是 float。
        self.targets = pd.to_numeric(self.data[target_column], errors="raise").astype(float).tolist()

    def __len__(self) -> int:
        """Return filtered training/evaluation sample count."""

        return len(self.data)

    def tokenize_sequence(self, sequence: str, max_length: int) -> dict:
        """Tokenize one CDR or antigen sequence for ESM-2.

        CDR 通常很短，antigen 可能较长；两者仍用同一 tokenizer。
        padding 让 batch tensor 形状一致，attention_mask 告诉 pooling 哪些 token 真实存在。
        """

        encoded = self.tokenizer(
            str(sequence),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, index: int) -> dict:
        """Return six CDR inputs, antigen input, and float target."""

        row = self.data.iloc[index]
        item = {"labels": torch.tensor(self.targets[index], dtype=torch.float32)}

        for column_name in self.input_cdr_fields:
            encoded = self.tokenize_sequence(row[column_name], self.cdr_max_length)
            model_key = CDR_MODEL_SEQUENCE_KEYS[column_name]
            item[f"{model_key}_input_ids"] = encoded["input_ids"]
            item[f"{model_key}_attention_mask"] = encoded["attention_mask"]

        antigen = self.tokenize_sequence(row["antigen_sequence"], self.max_length)
        item["antigen_input_ids"] = antigen["input_ids"]
        item["antigen_attention_mask"] = antigen["attention_mask"]
        return item
