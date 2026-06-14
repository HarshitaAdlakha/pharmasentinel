"""
Data preprocessing pipeline for the UCI Drug Review Dataset.

Handles HTML cleaning, text normalization, label encoding,
condition filtering, and train/val/test splitting.
"""

import re
import html
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# Conditions with fewer than this many reviews are collapsed into "Other"
MIN_CONDITION_COUNT = 100


def clean_review_text(text: str) -> str:
    """
    Normalise a raw drug review string.

    Steps:
    1. Decode HTML entities (&#039; → ')
    2. Strip residual HTML tags
    3. Lower-case and collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def discretise_rating(rating: float) -> int:
    """
    Map a 1–10 star rating to three sentiment classes:
      0 = Negative  (1–4)
      1 = Neutral   (5–6)
      2 = Positive  (7–10)
    """
    if rating <= 4:
        return 0
    if rating <= 6:
        return 1
    return 2


def load_raw_data(train_path: str, test_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and concatenate the UCI Drug Review CSV files.

    The UCI dataset ships as separate train/test TSV files from Kaggle:
      drugsComTrain_raw.tsv  (161,297 rows)
      drugsComTest_raw.tsv   (53,766 rows)

    Returns a single concatenated DataFrame with an 'split' column.
    """
    df_train = pd.read_csv(train_path, sep="\t", index_col=0)
    df_train["split"] = "train"

    if test_path:
        df_test = pd.read_csv(test_path, sep="\t", index_col=0)
        df_test["split"] = "test"
        return pd.concat([df_train, df_test], ignore_index=True)

    return df_train


def preprocess(
    df: pd.DataFrame,
    min_review_len: int = 10,
    top_k_conditions: Optional[int] = 50,
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Full preprocessing pipeline.

    Returns:
        df_clean      – cleaned DataFrame with engineered columns
        cond_encoder  – fitted LabelEncoder for medical conditions
        drug_encoder  – fitted LabelEncoder for drug names
    """
    df = df.copy()

    # ── Text cleaning ────────────────────────────────────────────────────────
    df["review_clean"] = df["review"].apply(clean_review_text)
    df = df[df["review_clean"].str.len() >= min_review_len].copy()

    # ── Rating targets ────────────────────────────────────────────────────────
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df["rating_norm"] = (df["rating"] - 1) / 9.0          # [0, 1] for regression
    df["sentiment"] = df["rating"].apply(discretise_rating)

    # ── Condition label ───────────────────────────────────────────────────────
    df["condition"] = df["condition"].fillna("Unknown").str.strip()
    # Remove entries where condition leaked drug info (a known dataset artifact)
    df = df[~df["condition"].str.contains(r"\d+ users found this comment helpful", na=False)].copy()

    if top_k_conditions:
        top_conditions = (
            df["condition"].value_counts().head(top_k_conditions).index.tolist()
        )
        df["condition"] = df["condition"].apply(
            lambda c: c if c in top_conditions else "Other"
        )

    # ── Drug name label ───────────────────────────────────────────────────────
    df["drugName"] = df["drugName"].str.strip().str.lower()

    # ── Helpfulness (log-scaled usefulCount) ─────────────────────────────────
    df["useful_count"] = pd.to_numeric(df["usefulCount"], errors="coerce").fillna(0)
    df["helpful_score"] = np.log1p(df["useful_count"]) / np.log1p(df["useful_count"].max())

    # ── Review length features ────────────────────────────────────────────────
    df["review_len"] = df["review_clean"].str.split().str.len()

    # ── Encode labels ─────────────────────────────────────────────────────────
    cond_encoder = LabelEncoder()
    df["condition_label"] = cond_encoder.fit_transform(df["condition"])

    drug_encoder = LabelEncoder()
    df["drug_label"] = drug_encoder.fit_transform(df["drugName"])

    logger.info(
        "Preprocessing complete: %d rows | %d conditions | %d drugs",
        len(df),
        df["condition"].nunique(),
        df["drugName"].nunique(),
    )
    return df, cond_encoder, drug_encoder


def build_splits(
    df: pd.DataFrame,
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train / validation / test splits.

    Stratification is done on 'sentiment' to keep class distribution
    consistent across all three partitions — important for fair evaluation.
    """
    # Honour existing 'split' column when present (UCI ships pre-split)
    if "split" in df.columns and df["split"].nunique() > 1:
        train_df = df[df["split"] == "train"].copy()
        test_df  = df[df["split"] == "test"].copy()
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size / (1 - test_size),
            stratify=train_df["sentiment"],
            random_state=random_state,
        )
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=val_size + test_size,
            stratify=df["sentiment"],
            random_state=random_state,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (val_size + test_size),
            stratify=temp_df["sentiment"],
            random_state=random_state,
        )

    logger.info(
        "Splits — train: %d | val: %d | test: %d", len(train_df), len(val_df), len(test_df)
    )
    return train_df, val_df, test_df
