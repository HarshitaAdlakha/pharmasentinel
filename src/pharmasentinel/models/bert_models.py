"""
Single-task BERT-based classifiers for PharmaSentinel.

Wraps HuggingFace DistilBERT / BERT / Bio_ClinicalBERT with a
custom classification head. These are the single-task upper-bound
models used in the ablation study (Table 3 of the paper).

All models expose .predict() and .predict_proba() for easy integration
with SHAP and the Streamlit front-end.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel


# Default checkpoints used in the paper benchmark
CHECKPOINT_REGISTRY = {
    "distilbert":      "distilbert-base-uncased",
    "bert":            "bert-base-uncased",
    "bio-clinical":    "emilyalsentzer/Bio_ClinicalBERT",
    "biobert":         "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert":      "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}


class DrugReviewClassifier(nn.Module):
    """
    BERT encoder → dropout → linear head for single-task classification.

    Supports Monte Carlo Dropout (MC-Dropout) for uncertainty estimation:
    call model.train() during inference and forward() multiple times.
    """

    def __init__(
        self,
        checkpoint: str,
        num_labels: int,
        dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))

    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
        self.eval()
        logits = self(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def mc_dropout_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_samples: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout uncertainty estimation.

        Returns:
            mean_proba – mean predicted probabilities over n_samples
            epistemic_uncertainty – std of probabilities (a proxy for model uncertainty)
        """
        self.train()  # enable dropout during inference
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self(input_ids, attention_mask)
                samples.append(torch.softmax(logits, dim=-1).cpu().numpy())
        self.eval()

        samples = np.stack(samples, axis=0)          # (n_samples, batch, num_labels)
        mean_proba = samples.mean(axis=0)
        epistemic_unc = samples.std(axis=0).mean(axis=-1)  # scalar per sample
        return mean_proba, epistemic_unc


class DrugRatingRegressor(nn.Module):
    """
    BERT encoder → dropout → linear head for continuous rating regression.
    Output is sigmoid-activated, representing a rating in [0, 1].
    """

    def __init__(self, checkpoint: str, dropout_prob: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(hidden, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return torch.sigmoid(self.regressor(self.dropout(pooled))).squeeze(-1)

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Return per-layer attention matrices for visualisation."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        return outputs.attentions
