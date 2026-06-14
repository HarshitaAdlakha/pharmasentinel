"""
PharmaSentinel Multi-Task Learning Model.

The core research contribution of this work: a single BERT encoder
with four task-specific heads trained jointly.

Architecture:
  ┌──────────────────────────────────────────┐
  │  DistilBERT / ClinicalBERT  Encoder      │
  │  (shared representation, 768-dim CLS)    │
  └──────────┬───────────────────────────────┘
             │
    ┌────────┼────────────────┬───────────────┐
    ▼        ▼                ▼               ▼
  Task 1   Task 2           Task 3          Task 4
  Rating   Sentiment        Condition       Helpfulness
 (regression) (3-class clf) (N-class clf)  (regression)

Loss:
  L_total = w1·MSE(rating) + w2·CE(sentiment) + w3·CE(condition) + w4·MSE(helpful)

The task weights w1–w4 are hyper-parameters (default: 0.3, 1.0, 1.0, 0.2).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class SharedEncoder(nn.Module):
    """Frozen-or-finetuneable transformer encoder with a projection head."""

    def __init__(self, checkpoint: str, projection_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(checkpoint)
        hidden = self.transformer.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            projected  – (batch, projection_dim) shared representation
            cls_hidden – (batch, hidden_size) raw CLS for attention export
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_hidden), cls_hidden


class TaskHead(nn.Module):
    """Configurable task-specific head (classification or regression)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        task_type: str = "classification",   # 'classification' | 'regression'
    ):
        super().__init__()
        self.task_type = task_type
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        if task_type == "regression":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.task_type == "regression":
            return self.activation(out).squeeze(-1)
        return out


class PharmaSentinelMTL(nn.Module):
    """
    Multi-Task Learning model: four heads on a shared BERT encoder.

    Args:
        checkpoint      – HuggingFace model checkpoint
        num_conditions  – number of medical condition classes
        num_sentiments  – number of sentiment classes (default 3)
        projection_dim  – shared representation size
        task_weights    – (w_rating, w_sentiment, w_condition, w_helpful)
    """

    def __init__(
        self,
        checkpoint: str = "distilbert-base-uncased",
        num_conditions: int = 51,
        num_sentiments: int = 3,
        projection_dim: int = 256,
        task_weights: Tuple[float, float, float, float] = (0.3, 1.0, 1.0, 0.2),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.task_weights = task_weights

        self.encoder = SharedEncoder(checkpoint, projection_dim, dropout)

        self.head_rating = TaskHead(
            projection_dim, 1, task_type="regression"
        )
        self.head_sentiment = TaskHead(
            projection_dim, num_sentiments, task_type="classification"
        )
        self.head_condition = TaskHead(
            projection_dim, num_conditions, task_type="classification"
        )
        self.head_helpful = TaskHead(
            projection_dim, 1, task_type="regression"
        )

        self._loss_rating = nn.MSELoss()
        self._loss_sentiment = nn.CrossEntropyLoss()
        self._loss_condition = nn.CrossEntropyLoss()
        self._loss_helpful = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rating_norm: Optional[torch.Tensor] = None,
        sentiment: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        helpful_score: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        During training, pass all label tensors; the method returns a
        'loss' key containing the weighted multi-task loss.

        During inference, omit labels; the method returns raw logits/scores.
        """
        shared_repr, _ = self.encoder(input_ids, attention_mask)

        out_rating    = self.head_rating(shared_repr)
        out_sentiment = self.head_sentiment(shared_repr)
        out_condition = self.head_condition(shared_repr)
        out_helpful   = self.head_helpful(shared_repr)

        result = {
            "rating":    out_rating,
            "sentiment": out_sentiment,
            "condition": out_condition,
            "helpful":   out_helpful,
        }

        if all(t is not None for t in [rating_norm, sentiment, condition, helpful_score]):
            w1, w2, w3, w4 = self.task_weights
            loss = (
                w1 * self._loss_rating(out_rating, rating_norm)
                + w2 * self._loss_sentiment(out_sentiment, sentiment)
                + w3 * self._loss_condition(out_condition, condition)
                + w4 * self._loss_helpful(out_helpful, helpful_score)
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Convenience wrapper for inference-only forward pass."""
        self.eval()
        return self.forward(input_ids, attention_mask)

    def mc_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_samples: int = 30,
    ) -> Dict[str, dict]:
        """
        MC-Dropout uncertainty for all four tasks.

        Returns per-task dicts with 'mean' and 'std' tensors.
        """
        self.train()
        samples = {k: [] for k in ["rating", "sentiment", "condition", "helpful"]}

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(input_ids, attention_mask)
                for k in samples:
                    samples[k].append(out[k].cpu())

        self.eval()

        import torch as _t
        return {
            k: {
                "mean": _t.stack(v).mean(0),
                "std":  _t.stack(v).std(0),
            }
            for k, v in samples.items()
        }
