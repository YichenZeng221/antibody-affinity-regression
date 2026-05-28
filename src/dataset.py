""": CSV  tensor

:
PyTorch , CSV 
 Dataset ,:

1.  CSV
2.  protein sequence  label
3.  ESM-2 tokenizer  token id
4.  PyTorch tensor

 tensor, "MKTAY..." 
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class ProteinSequenceDataset(Dataset):
    """ CSV 

    CSV :

    sequence,label
    MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP,0
    KKKKKKKKRRRRRRRRHHHHHHHHKKRRHKRH,1

    :
    - sequence:
    - label:, 0  1

     toy dataset,
     benchmark
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int):
        #  CSV pandas 
        self.data = pd.read_csv(csv_path)

        # tokenizer  Hugging Face, token id
        self.tokenizer = tokenizer

        # max_length 
        self.max_length = max_length

        #  CSV  sequence  label 
        # , debug
        required_columns = {"sequence", "label"}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing columns: {missing_columns}")

        #  Python list, __getitem__  index 
        self.sequences = self.data["sequence"].astype(str).tolist()
        self.labels = self.data["label"].astype(int).tolist()

    def __len__(self) -> int:
        """

        PyTorch DataLoader  dataset 
        """

        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        """ protein sequence 

        :
        DataLoader  Dataset: index 
        ,:

        - input_ids: token 
        - attention_mask: token, padding
        - labels:,0  1
        """

        sequence = self.sequences[index]
        label = self.labels[index]

        # tokenizer :
        # 1.  token ids
        # 2. padding="max_length":, batch
        # 3. truncation=True:, max_length
        encoded = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # tokenizer  tensor  [1, max_length]
        # squeeze(0)  1, [max_length]
        # DataLoader  [batch_size, max_length]
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
