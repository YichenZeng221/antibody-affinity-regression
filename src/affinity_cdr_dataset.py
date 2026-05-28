"""Dataset for the first CDR-aware affinity regression baseline.

:
whole-sequence baseline :
    heavy_sequence + light_sequence + antigen_sequence

 CDR-aware baseline :
    HCDR1 + HCDR2 + HCDR3
    LCDR1 + LCDR2 + LCDR3
    antigen_sequence

CDR  AbNumber + IMGT  CDR extraction,
 rowtokenize  PyTorch tensors
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

        #  `success`; `ok`
        # ,:heavy/light CDR 
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

        # CDR-aware regression  target, label  float
        self.targets = pd.to_numeric(self.data[target_column], errors="raise").astype(float).tolist()

    def __len__(self) -> int:
        """Return filtered training/evaluation sample count."""

        return len(self.data)

    def tokenize_sequence(self, sequence: str, max_length: int) -> dict:
        """Tokenize one CDR or antigen sequence for ESM-2.

        CDR ,antigen ; tokenizer
        padding  batch tensor ,attention_mask  pooling  token 
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
