import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict

class MultilingualTextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 512):
        self.texts = dataframe['DATA'].tolist()
        self.labels = torch.tensor((dataframe['LABEL'] == 'AI').astype(int).values)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': self.labels[idx]
        }