"""
Attention weight extraction and visualisation helpers.

Generates per-layer, per-head attention matrices from BERT models,
and returns the data in formats suitable for:
  - Matplotlib heatmaps (notebooks)
  - Streamlit / Plotly interactive charts (app)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def extract_attention(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_length: int = 128,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Extract per-layer attention matrices for a given text.

    Returns:
        tokens  – list of token strings
        layers  – list of (n_heads, seq_len, seq_len) numpy arrays, one per layer
    """
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        # Access the underlying transformer; handle MTL wrapper or single-task model
        encoder = getattr(model, "encoder", model)
        transformer = getattr(encoder, "transformer", encoder)
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    layers = [attn[0].cpu().numpy() for attn in outputs.attentions]
    return tokens, layers


def mean_attention_across_heads(layers: List[np.ndarray]) -> np.ndarray:
    """
    Average attention across all layers and all heads.
    Returns a (seq_len, seq_len) summary attention matrix.
    """
    all_heads = np.concatenate([l for l in layers], axis=0)  # (total_heads, seq, seq)
    return all_heads.mean(axis=0)


def attention_rollout(layers: List[np.ndarray]) -> np.ndarray:
    """
    Compute Attention Rollout (Abnar & Zuidema, 2020).

    Multiplies residual-connected attention maps across layers to
    trace information flow from input tokens to the CLS token.

    Reference: https://arxiv.org/abs/2005.00928
    """
    n_layers = len(layers)
    rollout = np.eye(layers[0].shape[-1])  # identity baseline

    for layer in layers:
        avg_head = layer.mean(axis=0)                    # (seq, seq)
        # Add residual connection
        avg_head_res = 0.5 * avg_head + 0.5 * np.eye(avg_head.shape[0])
        avg_head_res /= avg_head_res.sum(axis=-1, keepdims=True)
        rollout = rollout @ avg_head_res

    # CLS token's attention to all other tokens
    return rollout[0]  # shape: (seq_len,)


def top_k_tokens(
    tokens: List[str],
    scores: np.ndarray,
    k: int = 10,
    skip_special: bool = True,
) -> List[Tuple[str, float]]:
    """
    Return the k tokens with the highest attribution / attention scores.

    Filters out [CLS], [SEP], [PAD] tokens by default.
    """
    special = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    pairs = [
        (tok, float(scores[i]))
        for i, tok in enumerate(tokens)
        if not (skip_special and tok in special)
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def format_for_plotly(
    tokens: List[str],
    scores: np.ndarray,
    max_tokens: int = 30,
) -> Dict[str, list]:
    """
    Return a dict ready for a Plotly bar chart in the Streamlit app.
    """
    special = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    pairs = [
        (tok, float(scores[i]))
        for i, tok in enumerate(tokens[:max_tokens])
        if tok not in special
    ]
    return {
        "tokens": [p[0] for p in pairs],
        "scores": [p[1] for p in pairs],
    }
