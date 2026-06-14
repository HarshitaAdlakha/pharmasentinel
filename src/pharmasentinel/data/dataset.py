"""
PyTorch Dataset classes for PharmaSentinel.

Supports single-task and multi-task learning modes.
The multi-task variant exposes four simultaneous targets:
  - rating_norm   (float) – continuous drug rating [0, 1]
  - sentiment     (int)   – negative / neutral / positive
  - condition     (int)   – medical condition label
  - helpful_score (float) – log-normalised review helpfulness
"""

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class DrugReviewDataset(Dataset):
    """
    Tokenises drug reviews and packages all four multi-task targets.

    Args:
        texts          – list of cleaned review strings
        rating_norms   – continuous [0, 1] ratings
        sentiments     – integer sentiment labels (0/1/2)
        conditions     – integer condition labels
        helpful_scores – continuous [0, 1] helpfulness scores
        tokenizer      – any HuggingFace tokenizer
        max_length     – max token length (default 256)
    """

    def __init__(
        self,
        texts: list,
        rating_norms: list,
        sentiments: list,
        conditions: list,
        helpful_scores: list,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        self.texts = texts
        self.rating_norms = rating_norms
        self.sentiments = sentiments
        self.conditions = conditions
        self.helpful_scores = helpful_scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "rating_norm":    torch.tensor(self.rating_norms[idx],   dtype=torch.float),
            "sentiment":      torch.tensor(self.sentiments[idx],     dtype=torch.long),
            "condition":      torch.tensor(self.conditions[idx],     dtype=torch.long),
            "helpful_score":  torch.tensor(self.helpful_scores[idx], dtype=torch.float),
        }


class InferenceDataset(Dataset):
    """
    Lightweight dataset for inference — no labels required.
    Used by the Streamlit app and FastAPI service.
    """

    def __init__(
        self,
        texts: list,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
