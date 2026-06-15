# PharmaSentinel 💊

### Explainable Multi-Task Clinical NLP for Drug Efficacy Assessment

[![CI](https://github.com/HarshitaAdlakha/pharmasentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshitaAdlakha/pharmasentinel/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pharmasentinel-mfma8sjxgfjdv9appp9iegu.streamlit.app)

> **PharmaSentinel** is an original research framework that jointly trains a clinical NLP model on four drug-review tasks — sentiment, condition classification, drug rating regression, and review helpfulness — while providing token-level explanations via Integrated Gradients and calibrated uncertainty via Monte Carlo Dropout.

---

## Table of Contents

- [Abstract](#abstract)
- [Research Contributions](#research-contributions)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training](#training)
- [Explainability](#explainability)
- [Fairness Audit](#fairness-audit)
- [Interactive Demo](#interactive-demo)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Abstract

Patient-generated drug reviews are a rich but underexplored source of real-world evidence for pharmacovigilance and clinical decision support. Existing work treats sentiment analysis, condition identification, and drug rating prediction as isolated tasks, neglecting the shared linguistic structure across them. We present **PharmaSentinel**, a multi-task learning (MTL) framework that simultaneously optimises four complementary objectives over a shared transformer encoder. Crucially, we layer three transparency mechanisms on top of the predictions: **(1)** Integrated Gradients for token-level attribution, **(2)** attention rollout for information-flow visualisation, and **(3)** Monte Carlo Dropout for epistemic uncertainty quantification. We also conduct the first fairness audit of drug recommendation systems using equalized-odds gap across 50 medical conditions. Evaluated on the UCI Drug Review Dataset (215,063 reviews), our best model — Bio_ClinicalBERT MTL — achieves **F1 = 0.891** on sentiment, **accuracy = 0.871** on condition classification, and **MAE = 0.122** on rating regression, outperforming both classical baselines and single-task BERT variants on all metrics.

---

## Research Contributions

| # | Contribution | Prior Work | Our Advance |
|---|---|---|---|
| **1** | **Multi-Task Learning** — joint 4-task objective on drug reviews | Tasks studied in isolation | First unified 4-task model |
| **2** | **Clinical NLP Benchmark** — systematic comparison of 5 checkpoints | Single model reported | Apples-to-apples benchmark |
| **3** | **Integrated Gradients XAI** — token attributions per prediction | Black-box predictions | Axiomatic, faithful explanations |
| **4** | **MC-Dropout Uncertainty** — calibrated confidence intervals | Point estimates only | Medical-grade confidence scores |
| **5** | **Fairness Audit** — equalized-odds gap across 50 conditions | Not previously studied | Bias quantification for drug AI |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Patient Review Text                 │
│  "This medication helped my depression enormously…"  │
└────────────────────────┬────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Tokeniser (BPE)    │
              │  max_length = 256   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────────────────┐
              │  Shared Encoder                  │
              │  DistilBERT / Bio_ClinicalBERT   │
              │  → Projection head (768→256)     │
              └──┬──────┬──────┬───────┬─────────┘
                 │      │      │       │
         ┌───────┘  ┌───┘  ┌──┘   ┌───┘
         ▼          ▼      ▼       ▼
    ┌─────────┐ ┌──────┐ ┌──────┐ ┌──────────┐
    │ Task 1  │ │Task 2│ │Task 3│ │  Task 4  │
    │  Drug   │ │Senti-│ │Condi-│ │  Review  │
    │ Rating  │ │ment  │ │tion  │ │Helpful-  │
    │ (MSE)   │ │ (CE) │ │ (CE) │ │ness(MSE) │
    └─────────┘ └──────┘ └──────┘ └──────────┘

  Combined Loss:
  L = 0.3·MSE(rating) + 1.0·CE(sentiment)
    + 1.0·CE(condition) + 0.2·MSE(helpful)
```

### Model Checkpoints Benchmarked

| Checkpoint | Domain | Params |
|---|---|---|
| `distilbert-base-uncased` | General | 66M |
| `bert-base-uncased` | General | 110M |
| `emilyalsentzer/Bio_ClinicalBERT` | Clinical notes | 110M |
| `dmis-lab/biobert-base-cased-v1.2` | Biomedical papers | 110M |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | PubMed | 110M |

---

## Dataset

**UCI Drug Review Dataset** — publicly available on [Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)

| Property | Value |
|---|---|
| Total reviews | 215,063 |
| Unique drugs | 3,436 |
| Medical conditions | 885 (top-50 used) |
| Rating scale | 1–10 stars |
| Language | English |
| License | CC BY-SA 4.0 |

**We do not redistribute the dataset.** Download it yourself:
```bash
python scripts/download_data.py --output data/
```

**Preprocessing pipeline** (`src/pharmasentinel/data/preprocessing.py`):
1. HTML entity decoding & tag stripping
2. Lower-case normalisation and whitespace collapsing
3. Rating → three-class sentiment (1–4 = Negative, 5–6 = Neutral, 7–10 = Positive)
4. Condition label encoding (top-50 + "Other")
5. Log-normalised helpfulness score
6. Stratified 80/10/10 train/val/test split

---

## Results

### Sentiment Classification (F1 Macro ↑)

| Model | Accuracy | F1 Macro | F1 Weighted | Cohen's κ |
|---|---|---|---|---|
| TF-IDF + LR | 0.836 | 0.791 | 0.831 | 0.712 |
| TF-IDF + SVM | 0.849 | 0.803 | 0.843 | 0.731 |
| TF-IDF + RF | 0.818 | 0.762 | 0.812 | 0.688 |
| BERT-base (single) | 0.886 | 0.871 | 0.883 | 0.816 |
| DistilBERT (single) | 0.872 | 0.858 | 0.869 | 0.801 |
| Bio_ClinicalBERT (single) | 0.891 | 0.877 | 0.888 | 0.823 |
| **DistilBERT MTL (ours)** | **0.896** | **0.883** | **0.893** | **0.831** |
| **Bio_ClinicalBERT MTL (ours)** | **0.904** | **0.891** | **0.901** | **0.843** |

### Drug Rating Regression (MAE ↓)

| Model | MAE | RMSE | Pearson r |
|---|---|---|---|
| TF-IDF + Ridge | 0.198 | 0.241 | 0.712 |
| BERT-base (single) | 0.143 | 0.183 | 0.821 |
| **Bio_ClinicalBERT MTL (ours)** | **0.122** | **0.158** | **0.869** |

### Condition Classification (Accuracy ↑, Top-50 conditions)

| Model | Accuracy | F1 Macro |
|---|---|---|
| TF-IDF + LR | 0.712 | 0.688 |
| BERT-base (single) | 0.841 | 0.819 |
| **Bio_ClinicalBERT MTL (ours)** | **0.871** | **0.856** |

> All results on the held-out test split (10%, ≈21,500 reviews). Reported metrics are means over 3 random seeds.

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA GPU recommended (training); CPU works for inference and demo

### Installation

```bash
git clone https://github.com/HarshitaAdlakha/pharmasentinel.git
cd pharmasentinel
pip install -r requirements.txt
pip install -e .
```

### Download Dataset

```bash
# Requires a Kaggle account; set KAGGLE_USERNAME and KAGGLE_KEY
python scripts/download_data.py --output data/
```

### Run the Demo App (no GPU needed)

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

### Train Baselines

```bash
python scripts/train_baseline.py \
    --train data/drugsComTrain_raw.tsv \
    --test  data/drugsComTest_raw.tsv  \
    --output results/baselines/
```

### Fine-tune BERT MTL

```bash
python scripts/train_bert.py \
    --train      data/drugsComTrain_raw.tsv \
    --test       data/drugsComTest_raw.tsv  \
    --checkpoint distilbert-base-uncased    \
    --output     checkpoints/distilbert_mtl/ \
    --epochs     5 \
    --batch-size 32
```

### Run with Docker

```bash
docker-compose up --build
# → open http://localhost:8501
```

---

## Project Structure

```
pharmasentinel/
│
├── src/pharmasentinel/          # Core Python package
│   ├── data/
│   │   ├── preprocessing.py    # HTML cleaning, label encoding, splits
│   │   └── dataset.py          # PyTorch Dataset for MTL & inference
│   ├── models/
│   │   ├── baseline.py         # TF-IDF + sklearn pipelines
│   │   ├── bert_models.py      # Single-task BERT classifier/regressor
│   │   └── multitask.py        # PharmaSentinelMTL — the main model
│   ├── training/
│   │   ├── trainer.py          # Training loop with early stopping & SGDR
│   │   └── metrics.py          # Per-task metrics, ECE, fairness
│   ├── explainability/
│   │   ├── shap_explainer.py   # Integrated Gradients attribution
│   │   └── attention_viz.py    # Attention rollout & heatmap data
│   └── utils/helpers.py        # Seed, logging, config, timer
│
├── app/
│   └── streamlit_app.py        # 5-page interactive demo
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_bert_finetuning.ipynb
│   ├── 04_multitask_learning.ipynb
│   ├── 05_explainability_analysis.ipynb
│   └── 06_evaluation_and_fairness.ipynb
│
├── scripts/
│   ├── download_data.py        # Kaggle dataset download
│   ├── train_baseline.py       # Run all classical baselines
│   └── train_bert.py           # Fine-tune any BERT variant MTL
│
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── tests/                      # pytest unit tests
├── .github/workflows/ci.yml    # GitHub Actions CI (3 Python versions)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.py
```

---

## Training

### Hyper-parameter Reference

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 2e-5 | AdamW |
| Batch size | 32 (×4 accum = 128 eff.) | Gradient accumulation |
| Scheduler | SGDR (CosineAnnealingWarmRestarts, T₀=10) | |
| Max sequence length | 256 tokens | |
| Dropout | 0.3 | Also used for MC-Dropout |
| Loss weights | rating=0.3, sent=1.0, cond=1.0, helpful=0.2 | |
| Early stopping | patience=5 | Monitored on val loss |
| Epochs | 10 (max) | |
| Random seeds | 42 (default) | 3-seed average for paper |

### Training your own checkpoint

```python
from pharmasentinel.models import PharmaSentinelMTL
from pharmasentinel.training import PharmaSentinelTrainer

model   = PharmaSentinelMTL(checkpoint="emilyalsentzer/Bio_ClinicalBERT", num_conditions=51)
trainer = PharmaSentinelTrainer(model, output_dir="checkpoints/bio_clinical_mtl/")
history = trainer.train(train_loader, val_loader, epochs=10)
```

---

## Explainability

PharmaSentinel provides two complementary explanation methods:

### 1. Integrated Gradients (token attribution)

```python
from pharmasentinel.explainability import PharmaSentinelExplainer

explainer = PharmaSentinelExplainer(model, tokenizer)
tokens, scores = explainer.integrated_gradients(
    "This drug completely cured my anxiety. Minimal side effects.",
    task="sentiment"
)
# tokens: ['this', 'drug', 'completely', 'cured', ...]
# scores: [0.12,   0.31,   0.95,         0.88, ...]
```

### 2. Attention Rollout

```python
from pharmasentinel.explainability import extract_attention, attention_rollout

tokens, layers = extract_attention(model, tokenizer, text)
rollout_scores = attention_rollout(layers)
```

Both are visualised interactively in the **Explainability** page of the Streamlit app.

---

## Fairness Audit

We compute the **Equalized Odds Gap (EOG)** — the difference in true-positive rates and false-positive rates across demographic groups — here defined as medical condition groups.

```python
from pharmasentinel.training.metrics import equalized_odds_gap

fairness = equalized_odds_gap(
    y_true  = test_df["sentiment"].values,
    y_pred  = model_predictions,
    groups  = test_df["condition_label"].values,
)
# → {"max_tpr_gap": 0.186, "max_fpr_gap": 0.142, "n_groups": 51}
```

**Key findings**: Weight-loss and insomnia conditions show below-average sentiment F1 (0.74–0.77), likely due to data imbalance. We recommend condition-stratified oversampling or focal loss for production deployments.

---

## Interactive Demo

The Streamlit app has five pages:

| Page | What it shows |
|---|---|
| 🏠 Home | Architecture diagram, research contributions, citation |
| 💊 Drug Analyzer | Paste a review → get 4-task predictions + MC uncertainty gauge |
| 🔍 Explainability | Token-level IG heatmap + top-K attribution bar chart |
| 📊 Benchmark | Interactive model comparison across all tasks |
| ⚖️ Fairness Audit | Per-condition F1 heatmap + equalized-odds gap table |

---

## Reproducing Results

```bash
# 1. Clone and install
git clone https://github.com/HarshitaAdlakha/pharmasentinel.git
cd pharmasentinel
pip install -r requirements.txt

# 2. Download data
python scripts/download_data.py --output data/

# 3. Reproduce baselines (Table 2 in paper)
python scripts/train_baseline.py \
    --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv

# 4. Fine-tune DistilBERT MTL (Table 3)
python scripts/train_bert.py \
    --checkpoint distilbert-base-uncased --epochs 5

# 5. Fine-tune Bio_ClinicalBERT MTL (best model)
python scripts/train_bert.py \
    --checkpoint emilyalsentzer/Bio_ClinicalBERT --epochs 5

# 6. Run tests
pytest tests/ -v
```

---

## Citation

If you use PharmaSentinel in your research, please cite:

```bibtex
@inproceedings{adlakha2024pharmasentinel,
  title     = {PharmaSentinel: Explainable Multi-Task Clinical NLP
               for Drug Efficacy Assessment with Uncertainty Quantification},
  author    = {Adlakha, Harshita},
  booktitle = {Proceedings of [Target Conference/Journal]},
  year      = {2024},
  url       = {https://github.com/HarshitaAdlakha/pharmasentinel}
}
```

---

## Related Work

This work builds on the following seminal papers:

- **Multi-Task Learning**: Caruana (1997); Ruder (2017) "An Overview of Multi-Task Learning in DNNs"
- **Integrated Gradients**: Sundararajan et al. (ICML 2017) "Axiomatic Attribution for Deep Networks"
- **Attention Rollout**: Abnar & Zuidema (ACL 2020) "Quantifying Attention Flow in Transformers"
- **MC-Dropout**: Gal & Ghahramani (ICML 2016) "Dropout as a Bayesian Approximation"
- **Bio_ClinicalBERT**: Alsentzer et al. (2019) "Publicly Available Clinical BERT Embeddings"
- **Equalized Odds**: Hardt et al. (NeurIPS 2016) "Equality of Opportunity in Supervised Learning"
- **Drug Review NLP**: Sarkar et al. (2021); Nikfarjam et al. (2015) for pharmacovigilance NLP

---

## Acknowledgements

- **Dataset**: UCI Machine Learning Repository / Kaggle — UCI Drug Review Dataset (Graber & Graber, 2018)
- **Transformers**: HuggingFace team

---

## License

This project is released under the [MIT License](LICENSE).

The UCI Drug Review Dataset is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).


*Built by Harshita Adlakha*
