from .preprocessing import preprocess, load_raw_data, build_splits, clean_review_text

__all__ = [
    "preprocess",
    "load_raw_data",
    "build_splits",
    "clean_review_text",
]

# DrugReviewDataset and InferenceDataset require torch — import lazily
def __getattr__(name):
    if name in ("DrugReviewDataset", "InferenceDataset"):
        from .dataset import DrugReviewDataset, InferenceDataset
        return {"DrugReviewDataset": DrugReviewDataset, "InferenceDataset": InferenceDataset}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
