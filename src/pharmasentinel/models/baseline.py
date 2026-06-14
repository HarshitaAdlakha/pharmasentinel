"""
Classical baseline models for PharmaSentinel.

Provides TF-IDF + scikit-learn pipelines that serve as the
reproducible lower-bound in the benchmark table (Table 2 in the paper).

Models:
  - TF-IDF + Logistic Regression (fast, strong baseline)
  - TF-IDF + Random Forest       (ensemble baseline)
  - TF-IDF + LinearSVC           (margin-based baseline)
  - TF-IDF + XGBoost             (gradient-boost baseline)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf_pipeline(
    classifier: Any,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 50_000,
    sublinear_tf: bool = True,
) -> Pipeline:
    """Return a (TF-IDF → classifier) sklearn Pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    sublinear_tf=sublinear_tf,
                    strip_accents="unicode",
                    analyzer="word",
                    min_df=2,
                ),
            ),
            ("clf", classifier),
        ]
    )


# ── Named baseline constructors ───────────────────────────────────────────────

def logistic_regression_baseline(C: float = 1.0, **kwargs) -> Pipeline:
    return build_tfidf_pipeline(
        LogisticRegression(C=C, max_iter=1000, class_weight="balanced", **kwargs)
    )


def svm_baseline(C: float = 1.0, **kwargs) -> Pipeline:
    return build_tfidf_pipeline(
        LinearSVC(C=C, max_iter=2000, class_weight="balanced", **kwargs)
    )


def random_forest_baseline(n_estimators: int = 200, **kwargs) -> Pipeline:
    return build_tfidf_pipeline(
        RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced", n_jobs=-1, **kwargs
        )
    )


def ridge_regression_baseline(alpha: float = 1.0, **kwargs) -> Pipeline:
    """For continuous drug-rating regression."""
    return build_tfidf_pipeline(Ridge(alpha=alpha, **kwargs))


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_classifier(
    model: Pipeline,
    X_test: list,
    y_test: np.ndarray,
    average: str = "weighted",
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average=average, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def evaluate_regressor(
    model: Pipeline,
    X_test: list,
    y_test: np.ndarray,
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "LR":  logistic_regression_baseline,
    "SVM": svm_baseline,
    "RF":  random_forest_baseline,
}


def run_baselines(
    X_train: list,
    y_train: np.ndarray,
    X_test: list,
    y_test: np.ndarray,
    task: str = "sentiment",
) -> Dict[str, Dict[str, float]]:
    """
    Train all baseline models and return a metrics dictionary.

    Args:
        task – one of 'sentiment', 'condition', 'rating'
    """
    results: Dict[str, Dict[str, float]] = {}

    for name, builder in BASELINE_REGISTRY.items():
        model = builder()
        model.fit(X_train, y_train)
        if task == "rating":
            results[name] = evaluate_regressor(model, X_test, y_test)
        else:
            results[name] = evaluate_classifier(model, X_test, y_test)

    return results
