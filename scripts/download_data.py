"""
Script: Download the UCI Drug Review Dataset from Kaggle.

Prerequisites:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
    (or place kaggle.json in ~/.kaggle/)

Usage:
    python scripts/download_data.py --output data/

The dataset will be extracted to:
    data/drugsComTrain_raw.tsv
    data/drugsComTest_raw.tsv
"""

import argparse
import os
import zipfile
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")


def download_from_kaggle(output_dir: str) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Then set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )

    os.makedirs(output_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    dataset = "jessicali9530/kuc-hackathon-winter-2018"
    logger.info("Downloading dataset: %s", dataset)
    api.dataset_download_files(dataset, path=output_dir, unzip=True)
    logger.info("Dataset downloaded to %s", output_dir)


def verify_files(output_dir: str) -> bool:
    expected = ["drugsComTrain_raw.tsv", "drugsComTest_raw.tsv"]
    found = all(os.path.isfile(os.path.join(output_dir, f)) for f in expected)
    if found:
        logger.info("All expected data files present.")
    else:
        logger.warning("Some files missing. Check %s", output_dir)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/", help="Output directory")
    args = parser.parse_args()

    download_from_kaggle(args.output)
    verify_files(args.output)


if __name__ == "__main__":
    main()
