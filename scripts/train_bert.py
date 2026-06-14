"""
Script: Fine-tune a BERT-family model on the PharmaSentinel MTL objective.

Usage:
    python scripts/train_bert.py \
        --train  data/drugsComTrain_raw.tsv   \
        --test   data/drugsComTest_raw.tsv    \
        --checkpoint distilbert-base-uncased  \
        --output checkpoints/distilbert_mtl/  \
        --epochs 5 \
        --batch-size 32

Supported checkpoints (--checkpoint):
    distilbert-base-uncased
    bert-base-uncased
    emilyalsentzer/Bio_ClinicalBERT
    dmis-lab/biobert-base-cased-v1.2
    microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pharmasentinel.data import load_raw_data, preprocess, build_splits, DrugReviewDataset
from pharmasentinel.models import PharmaSentinelMTL
from pharmasentinel.training import (
    PharmaSentinelTrainer,
    sentiment_metrics,
    condition_metrics,
    rating_metrics,
    build_benchmark_table,
)
from pharmasentinel.utils import set_seed, setup_logging, save_json, count_parameters, format_number

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",      required=True)
    p.add_argument("--test",       default=None)
    p.add_argument("--checkpoint", default="distilbert-base-uncased")
    p.add_argument("--output",     default="checkpoints/")
    p.add_argument("--epochs",     type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len",    type=int, default=256)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--top-k-conditions", type=int, default=50)
    return p.parse_args()


def make_loader(df, tokenizer, batch_size, max_len, shuffle=False):
    ds = DrugReviewDataset(
        texts          = df["review_clean"].tolist(),
        rating_norms   = df["rating_norm"].tolist(),
        sentiments     = df["sentiment"].tolist(),
        conditions     = df["condition_label"].tolist(),
        helpful_scores = df["helpful_score"].tolist(),
        tokenizer      = tokenizer,
        max_length     = max_len,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_logging()

    # ── Data ─────────────────────────────────────────────────────────────────
    raw = load_raw_data(args.train, args.test)
    df, cond_enc, _ = preprocess(raw, top_k_conditions=args.top_k_conditions)
    train_df, val_df, test_df = build_splits(df)
    num_conditions = df["condition"].nunique()

    logger.info("Conditions: %d | Train: %d | Val: %d | Test: %d",
                num_conditions, len(train_df), len(val_df), len(test_df))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    train_loader = make_loader(train_df, tokenizer, args.batch_size, args.max_len, shuffle=True)
    val_loader   = make_loader(val_df,   tokenizer, args.batch_size, args.max_len)
    test_loader  = make_loader(test_df,  tokenizer, args.batch_size, args.max_len)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PharmaSentinelMTL(
        checkpoint     = args.checkpoint,
        num_conditions = num_conditions,
    )
    params = count_parameters(model)
    logger.info("Model params — total: %s | trainable: %s",
                format_number(params["total"]), format_number(params["trainable"]))

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = PharmaSentinelTrainer(
        model      = model,
        output_dir = args.output,
        lr         = args.lr,
    )
    history = trainer.train(train_loader, val_loader, epochs=args.epochs)
    trainer.load_best()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    all_sentiment, all_condition, all_rating = [], [], []
    true_sentiment, true_condition, true_rating = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            out = model.predict(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_sentiment.extend(out["sentiment"].argmax(-1).cpu().numpy())
            all_condition.extend(out["condition"].argmax(-1).cpu().numpy())
            all_rating.extend(out["rating"].cpu().numpy())
            true_sentiment.extend(batch["sentiment"].numpy())
            true_condition.extend(batch["condition"].numpy())
            true_rating.extend(batch["rating_norm"].numpy())

    import numpy as np
    results = {
        "sentiment": sentiment_metrics(np.array(true_sentiment), np.array(all_sentiment)),
        "condition": condition_metrics(np.array(true_condition), np.array(all_condition)),
        "rating":    rating_metrics(np.array(true_rating),    np.array(all_rating)),
    }

    for task, metrics in results.items():
        print(f"\n## {task.capitalize()} (MTL DistilBERT)\n")
        print(build_benchmark_table({args.checkpoint: metrics}))

    save_json(results, os.path.join(args.output, "eval_results.json"))
    save_json(history, os.path.join(args.output, "training_history.json"))
    logger.info("Done. Results saved to %s", args.output)


if __name__ == "__main__":
    main()
