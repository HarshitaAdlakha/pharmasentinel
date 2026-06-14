# Data Directory

This directory is intentionally empty in the repository.

## Download Instructions

The UCI Drug Review Dataset must be downloaded separately due to its size (~70MB).

### Option 1: Kaggle CLI (recommended)

```bash
# Install Kaggle
pip install kaggle

# Set credentials (get from https://www.kaggle.com/settings/account → API)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download
python scripts/download_data.py --output data/
```

### Option 2: Manual Download

1. Go to: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
2. Download `archive.zip`
3. Extract both TSV files here:
   - `data/drugsComTrain_raw.tsv`
   - `data/drugsComTest_raw.tsv`

## Expected Files After Download

```
data/
├── drugsComTrain_raw.tsv   (161,297 rows, ~50MB)
├── drugsComTest_raw.tsv    ( 53,766 rows, ~18MB)
└── README.md               (this file)
```

## Data Schema

| Column | Type | Description |
|---|---|---|
| uniqueID | int | Row identifier |
| drugName | str | Drug name (e.g. "Sertraline") |
| condition | str | Medical condition (e.g. "Depression") |
| review | str | Patient review text (HTML-encoded) |
| rating | float | 1–10 satisfaction rating |
| date | str | Review date |
| usefulCount | int | Number of users who found review helpful |

## Citation

Graber, F. & Graber, M. (2018). *Drug Review Dataset (Drugs.com)*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5SK5S
