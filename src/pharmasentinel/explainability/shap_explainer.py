"""
SHAP-based explainability for PharmaSentinel.

Wraps the trained model with a SHAP DeepExplainer (for BERT) or
KernelExplainer fallback to generate token-level attribution scores.

The PharmaSentinelExplainer class is used both in:
  - Notebook 05 (offline analysis)
  - The Streamlit app (live explanation panel)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class PharmaSentinelExplainer:
    """
    Compute token-level importance scores using Integrated Gradients.

    We use Integrated Gradients instead of SHAP's DeepExplainer here
    because DistilBERT uses LayerNorm layers that SHAP's implementation
    cannot propagate through reliably. IG is a gradient-based attribution
    method with the same Shapley-value axiomatic guarantees.

    Reference:
        Sundararajan et al., "Axiomatic Attribution for Deep Networks",
        ICML 2017. https://arxiv.org/abs/1703.01365
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[str] = None,
        n_steps: int = 50,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.n_steps = n_steps

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract token embeddings from the transformer's embedding layer."""
        # Works for both BERT and DistilBERT
        embeddings_layer = (
            self.model.encoder.transformer.embeddings
            if hasattr(self.model.encoder, "transformer")
            else self.model.encoder.embeddings
        )
        return embeddings_layer(input_ids)

    def _forward_from_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor, task: str
    ) -> torch.Tensor:
        """
        Forward pass using pre-computed embeddings (needed for IG interpolation).
        """
        # We hook into the model using embedding injection
        outputs = self.model.encoder.transformer(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        projected = self.model.encoder.projection(cls)

        head_map = {
            "rating":    self.model.head_rating,
            "sentiment": self.model.head_sentiment,
            "condition": self.model.head_condition,
            "helpful":   self.model.head_helpful,
        }
        return head_map[task](projected)

    def integrated_gradients(
        self,
        text: str,
        task: str = "sentiment",
        target_class: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Compute Integrated Gradients attributions for a single review.

        Args:
            text         – raw drug review string
            task         – one of 'rating', 'sentiment', 'condition', 'helpful'
            target_class – class index for classification tasks (None = argmax)
            n_steps      – number of interpolation steps (default: self.n_steps)

        Returns:
            tokens       – list of token strings (subword BPE tokens)
            attributions – numpy array of attribution scores, shape (n_tokens,)
        """
        steps = n_steps or self.n_steps
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        input_ids     = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Baseline = all-PAD embedding
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        embeddings          = self._get_embeddings(input_ids).detach()
        baseline_embeddings = self._get_embeddings(baseline_ids).detach()

        # Interpolated inputs
        alphas = torch.linspace(0, 1, steps, device=self.device).view(-1, 1, 1)
        interp = baseline_embeddings + alphas * (embeddings - baseline_embeddings)
        interp = interp.requires_grad_(True)

        # Forward all interpolations (chunked to save VRAM)
        grads = []
        for i in range(steps):
            emb_i = interp[i : i + 1]
            out = self._forward_from_embeddings(emb_i, attention_mask, task)

            if task in ("rating", "helpful"):
                scalar = out.squeeze()
            else:
                if target_class is None:
                    target_class = int(out.argmax(-1).item())
                scalar = out[0, target_class]

            scalar.backward(retain_graph=(i < steps - 1))
            grads.append(interp.grad[i : i + 1].clone())
            interp.grad.zero_()

        # Riemann trapezoidal approximation
        grads_tensor = torch.cat(grads, dim=0)                    # (steps, seq, hidden)
        avg_grads    = grads_tensor.mean(0)                        # (seq, hidden)
        ig           = (embeddings.squeeze(0) - baseline_embeddings.squeeze(0)) * avg_grads
        attribution  = ig.norm(dim=-1).cpu().detach().numpy()     # (seq,)

        # Normalise to [0, 1]
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        return tokens, attribution

    def explain_batch(
        self,
        texts: List[str],
        task: str = "sentiment",
    ) -> List[Tuple[List[str], np.ndarray]]:
        """Run explain on a list of texts; returns list of (tokens, scores) tuples."""
        return [self.integrated_gradients(t, task=task) for t in texts]
