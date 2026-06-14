from .baseline import (
    logistic_regression_baseline,
    svm_baseline,
    random_forest_baseline,
    ridge_regression_baseline,
    run_baselines,
    BASELINE_REGISTRY,
)

__all__ = [
    "logistic_regression_baseline",
    "svm_baseline",
    "random_forest_baseline",
    "ridge_regression_baseline",
    "run_baselines",
    "BASELINE_REGISTRY",
]

# Torch-dependent models — imported lazily
def __getattr__(name):
    if name in ("DrugReviewClassifier", "DrugRatingRegressor", "CHECKPOINT_REGISTRY"):
        from .bert_models import DrugReviewClassifier, DrugRatingRegressor, CHECKPOINT_REGISTRY
        return {"DrugReviewClassifier": DrugReviewClassifier,
                "DrugRatingRegressor": DrugRatingRegressor,
                "CHECKPOINT_REGISTRY": CHECKPOINT_REGISTRY}[name]
    if name == "PharmaSentinelMTL":
        from .multitask import PharmaSentinelMTL
        return PharmaSentinelMTL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
