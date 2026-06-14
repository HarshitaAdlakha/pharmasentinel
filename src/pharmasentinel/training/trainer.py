"""
Training loop for PharmaSentinel MTL model.

Implements:
  - Gradient accumulation for large effective batch sizes
  - Cosine annealing with warm restarts (SGDR)
  - Early stopping with configurable patience
  - Per-task loss logging for tensorboard / W&B
  - Checkpoint saving (best val loss)
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class PharmaSentinelTrainer:
    """
    Stateful trainer for the MTL model.

    Typical usage:
        trainer = PharmaSentinelTrainer(model, config)
        trainer.train(train_loader, val_loader)
        trainer.load_best()
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "checkpoints/",
        lr: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        grad_accumulation: int = 4,
        patience: int = 5,
        device: Optional[str] = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.grad_accumulation = grad_accumulation
        self.early_stopping = EarlyStopping(patience=patience)

        os.makedirs(output_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

        logger.info("Trainer initialised on device: %s", self.device)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        batch = self._move_batch(batch)
        outputs = self.model(
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            rating_norm    = batch.get("rating_norm"),
            sentiment      = batch.get("sentiment"),
            condition      = batch.get("condition"),
            helpful_score  = batch.get("helpful_score"),
        )
        return outputs["loss"], outputs

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
            loss, _ = self._forward(batch)
            (loss / self.grad_accumulation).backward()

            if (step + 1) % self.grad_accumulation == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Val", leave=False):
            loss, _ = self._forward(batch)
            total_loss += loss.item()

        return total_loss / len(loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Epoch %d/%d — train_loss: %.4f | val_loss: %.4f",
                epoch, epochs, train_loss, val_loss,
            )

            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint("best_model.pt")
                logger.info("  ✓ New best checkpoint saved (val_loss=%.4f)", best_val)

            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        return history

    def save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_best(self, filename: str = "best_model.pt") -> None:
        path = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded best checkpoint from %s", path)
