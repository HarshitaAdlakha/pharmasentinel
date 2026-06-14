"""
Unit tests for model forward passes (CPU-only, no GPU required).
Uses a tiny mock encoder to keep tests fast (<10s).
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pharmasentinel.models.multitask import TaskHead, SharedEncoder, PharmaSentinelMTL
from pharmasentinel.models.baseline import (
    logistic_regression_baseline,
    evaluate_classifier,
)


# ── TaskHead ──────────────────────────────────────────────────────────────────
class TestTaskHead:
    def test_classification_output_shape(self):
        head = TaskHead(in_dim=64, out_dim=3, task_type="classification")
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 3)

    def test_regression_output_range(self):
        head = TaskHead(in_dim=64, out_dim=1, task_type="regression")
        x = torch.randn(32, 64)
        out = head(x)
        assert out.shape == (32,)
        assert (out >= 0).all() and (out <= 1).all()


# ── Baseline models ───────────────────────────────────────────────────────────
class TestBaselineModels:
    def test_lr_pipeline_fit_predict(self):
        X = ["I love this drug", "It made me sick", "Average results", "Great medication", "Horrible side effects"]
        y = [2, 0, 1, 2, 0]
        model = logistic_regression_baseline()
        model.fit(X * 20, y * 20)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_evaluate_classifier_keys(self):
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 2]
        import numpy as np
        result = evaluate_classifier(
            logistic_regression_baseline().fit(
                ["a", "b", "c", "d", "e"] * 20,
                [0, 1, 2, 0, 1] * 20,
            ),
            ["a", "b", "c", "d", "e"],
            np.array(y_true),
        )
        assert "accuracy" in result
        assert "f1_weighted" in result


# ── MTL model (mocked encoder) ────────────────────────────────────────────────
class _TinyEncoder(nn.Module):
    """Replaces the HuggingFace transformer for fast unit testing."""
    class _Config:
        hidden_size = 64

    config = _Config()

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 64)

    def forward(self, input_ids, attention_mask, **kwargs):
        class _Out:
            pass
        o = _Out()
        o.last_hidden_state = self.emb(input_ids)
        return o


def _patched_mtl():
    """Build an MTL model with a tiny encoder that avoids downloading weights."""
    model = PharmaSentinelMTL.__new__(PharmaSentinelMTL)
    nn.Module.__init__(model)
    model.task_weights = (0.3, 1.0, 1.0, 0.2)

    from pharmasentinel.models.multitask import SharedEncoder, TaskHead
    encoder_stub = SharedEncoder.__new__(SharedEncoder)
    nn.Module.__init__(encoder_stub)
    encoder_stub.transformer = _TinyEncoder()
    encoder_stub.projection = nn.Sequential(
        nn.Linear(64, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.0)
    )
    model.encoder = encoder_stub

    model.head_rating    = TaskHead(256, 1, task_type="regression")
    model.head_sentiment = TaskHead(256, 3, task_type="classification")
    model.head_condition = TaskHead(256, 5, task_type="classification")
    model.head_helpful   = TaskHead(256, 1, task_type="regression")
    model._loss_rating    = nn.MSELoss()
    model._loss_sentiment = nn.CrossEntropyLoss()
    model._loss_condition = nn.CrossEntropyLoss()
    model._loss_helpful   = nn.MSELoss()
    return model


class TestMTLModel:
    def test_forward_inference_keys(self):
        model = _patched_mtl()
        ids   = torch.randint(0, 100, (2, 16))
        mask  = torch.ones(2, 16, dtype=torch.long)
        out   = model.predict(ids, mask)
        assert "rating" in out
        assert "sentiment" in out
        assert "condition" in out
        assert "helpful" in out

    def test_forward_training_returns_loss(self):
        model = _patched_mtl()
        ids   = torch.randint(0, 100, (4, 16))
        mask  = torch.ones(4, 16, dtype=torch.long)
        out   = model(
            ids, mask,
            rating_norm   = torch.rand(4),
            sentiment     = torch.randint(0, 3, (4,)),
            condition     = torch.randint(0, 5, (4,)),
            helpful_score = torch.rand(4),
        )
        assert "loss" in out
        assert out["loss"].item() > 0
