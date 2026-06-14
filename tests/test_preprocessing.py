"""
Unit tests for the data preprocessing pipeline.
"""

import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pharmasentinel.data.preprocessing import (
    clean_review_text,
    discretise_rating,
    preprocess,
    build_splits,
)


# ── clean_review_text ─────────────────────────────────────────────────────────
class TestCleanReviewText:
    def test_html_entities_decoded(self):
        assert "'" in clean_review_text("it&#039;s great")

    def test_html_tags_removed(self):
        assert "<b>" not in clean_review_text("<b>Bold</b> text")

    def test_lower_cased(self):
        assert clean_review_text("UPPERCASE") == "uppercase"

    def test_whitespace_collapsed(self):
        assert "  " not in clean_review_text("too   many   spaces")

    def test_non_string_returns_empty(self):
        assert clean_review_text(None) == ""
        assert clean_review_text(42) == ""


# ── discretise_rating ─────────────────────────────────────────────────────────
class TestDiscretiseRating:
    @pytest.mark.parametrize("rating,expected", [
        (1, 0), (2, 0), (4, 0),
        (5, 1), (6, 1),
        (7, 2), (9, 2), (10, 2),
    ])
    def test_boundaries(self, rating, expected):
        assert discretise_rating(rating) == expected


# ── preprocess ────────────────────────────────────────────────────────────────
def make_sample_df(n: int = 200) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "review":     [f"This drug {'helped' if rng.random() > 0.5 else 'hurt'} me a lot." for _ in range(n)],
        "rating":     rng.integers(1, 11, size=n).astype(float),
        "condition":  rng.choice(["Depression", "Anxiety", "Pain", "Diabetes"], size=n),
        "drugName":   rng.choice(["Sertraline", "Metformin", "Ibuprofen"], size=n),
        "usefulCount":rng.integers(0, 100, size=n).astype(float),
    })


class TestPreprocess:
    def test_returns_three_values(self):
        df = make_sample_df()
        result = preprocess(df)
        assert len(result) == 3

    def test_review_clean_column_exists(self):
        df, _, _ = preprocess(make_sample_df())
        assert "review_clean" in df.columns

    def test_sentiment_column_in_range(self):
        df, _, _ = preprocess(make_sample_df())
        assert df["sentiment"].isin([0, 1, 2]).all()

    def test_rating_norm_in_range(self):
        df, _, _ = preprocess(make_sample_df())
        assert df["rating_norm"].between(0, 1).all()

    def test_no_nan_in_key_columns(self):
        df, _, _ = preprocess(make_sample_df())
        for col in ["review_clean", "rating_norm", "sentiment", "condition_label"]:
            assert df[col].isna().sum() == 0, f"NaN found in {col}"


# ── build_splits ──────────────────────────────────────────────────────────────
class TestBuildSplits:
    def test_no_overlap(self):
        df, _, _ = preprocess(make_sample_df(300))
        train, val, test = build_splits(df)
        train_idx = set(train.index)
        val_idx   = set(val.index)
        test_idx  = set(test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_total_rows_preserved(self):
        df, _, _ = preprocess(make_sample_df(300))
        train, val, test = build_splits(df)
        assert len(train) + len(val) + len(test) == len(df)
