"""PyTorch Dataset for antibody-antigen affinity regression.

:
classification , sequence
regression , sequence:

1. heavy_sequence
2. light_sequence
3. antigen_sequence

 tokenize , target
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class AffinityRegressionDataset(Dataset):
    """Load affinity regression CSV files.

    :
    PyTorch  Dataset 
    DataLoader  Dataset: index 

    ,:
    - heavy_sequence
    - light_sequence
    - antigen_sequence
    -  float target:neg_log10_affinity

    :regression  label , float;
    classification  label  0/1/2  class id
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int, target_column: str):
        # CSV , antibody-antigen complex
        self.data = pd.read_csv(csv_path)

        # tokenizer  token
        #  "EVQL..."  input_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_column = target_column

        # ,
        required_columns = {
            "heavy_sequence",
            "light_sequence",
            "antigen_sequence",
            target_column,
        }
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

        #  list ,__getitem__  index 
        self.heavy_sequences = self.data["heavy_sequence"].astype(str).tolist()
        self.light_sequences = self.data["light_sequence"].astype(str).tolist()
        self.antigen_sequences = self.data["antigen_sequence"].astype(str).tolist()

        #  target , 8.34, class id
        self.targets = self.data[target_column].astype(float).tolist()

    def __len__(self) -> int:
        """Return number of samples."""

        return len(self.data)

    def tokenize_sequence(self, sequence: str) -> dict:
        """Convert one amino acid sequence into ESM-2 input tensors.

        :
        - input_ids: token 
        - attention_mask: token, padding
          1  token,0  padding
        - padding="max_length":, batch
        - truncation=True: max_length

        padding , batch 
        """

        encoded = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            # tokenizer  [1, max_length]
            # squeeze(0)  1, [max_length],
            #  DataLoader  [batch_size, max_length]
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, index: int) -> dict:
        """Return one model-ready sample.

        DataLoader  dict  batch
         key  model.forward(...) 
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
            # dtype=torch.float32  MSELoss  label 
            "labels": torch.tensor(self.targets[index], dtype=torch.float32),
        }
