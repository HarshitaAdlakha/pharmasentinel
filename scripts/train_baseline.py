"""
Script: Train and evaluate all classical baseline models.

Usage:
    python scripts/train_baseline.py \
        --train data/drugsComTrain_raw.tsv \
        --test  data/drugsComTest_raw.tsv  \
        --output results/baselines/

Results are saved as JSON and printed as a Markdown table.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pharmasentinel.data import load_raw_data, preprocess, build_splits
from pharmasentinel.models import run_baselines, BASELINE_REGISTRY
from pharmasentinel.training import (
    sentiment_metrics,
    condition_metrics,
    build_benchmark_table,
)
from pharmasentinel.utils import set_seed, setup_logging, save_json

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",  required=True, help="Path to train TSV")
    p.add_argument("--test",   default=None,  help="Path to test TSV (optional)")
    p.add_argument("--output", default="results/baselines/")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--top-k-conditions", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_logging()
    os.makedirs(args.output, exist_ok=True)

    # ── Load & preprocess ─────────────────────────────────────────────────────
    logger.info("Loading data from %s", args.train)
    raw = load_raw_data(args.train, args.test)
    df, cond_enc, drug_enc = preprocess(raw, top_k_conditions=args.top_k_conditions)
    train_df, val_df, test_df = build_splits(df)

    X_train = train_df["review_clean"].tolist()
    X_test  = test_df["review_clean"].tolist()

    all_results = {}

    # ── Sentiment task ────────────────────────────────────────────────────────
    logger.info("Running sentiment baselines …")
    sent_results = run_baselines(
        X_train, train_df["sentiment"].values,
        X_test,  test_df["sentiment"].values,
        task="sentiment",
    )
    all_results["sentiment"] = sent_results
    print("\n## Sentiment Classification Baselines\n")
    print(build_benchmark_table(sent_results))

    # ── Condition task ────────────────────────────────────────────────────────
    logger.info("Running condition baselines …")
    cond_results = run_baselines(
        X_train, train_df["condition_label"].values,
        X_test,  test_df["condition_label"].values,
        task="condition",
    )
    all_results["condition"] = cond_results
    print("\n## Condition Classification Baselines\n")
    print(build_benchmark_table(cond_results))

    # ── Rating regression task ────────────────────────────────────────────────
    logger.info("Running rating regression baselines …")
    rating_results = run_baselines(
        X_train, train_df["rating_norm"].values,
        X_test,  test_df["rating_norm"].values,
        task="rating",
    )
    all_results["rating"] = rating_results
    print("\n## Drug Rating Regression Baselines\n")
    print(build_benchmark_table(rating_results))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output, "baseline_results.json")
    save_json(all_results, out_path)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
