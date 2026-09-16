# Kaggle Experimentation Guide

This directory contains experimental pipelines designed for training and embedding precomputation on Kaggle GPU instances.

## Shopee Dataset on Kaggle
The dataset is available directly on Kaggle:
- [Shopee - Price Match Guarantee Dataset](https://www.kaggle.com/competitions/shopee-product-matching/data)
- Input path on Kaggle: `/kaggle/input/shopee-product-matching/`

## Workflow

### 1. Data Split
We split `train.csv` strictly by `label_group` into:
- `train_split.csv` (~70%): for model fitting and fine-tuning
- `val_split.csv` (~15%): for threshold tuning and hyperparameter selection
- `test_demo_split.csv` (~15%): held-out unseen product groups for unbiased reporting and interactive Streamlit demo

### 2. Experiments
- **Experiment 0 — TF-IDF Baseline**:
  - Unigram + bigram TF-IDF with sublinear term frequency
  - Direct sparse cosine similarity
  - Threshold tuned on validation split, evaluated on test split
- **Experiment 1 — Transformer Text Embeddings**:
  - Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional)
  - Precomputes L2-normalized text vectors
- **Experiment 2 — Improved / Multilingual Text Embeddings**:
  - Model: `paraphrase-multilingual-MiniLM-L12-v2` or contrastive fine-tuning
- **Experiment 3 — Visual Image Embeddings**:
  - Model: `timm` (e.g. `efficientnet_b0`) feature extraction
  - Precomputes visual representations for all images
- **Experiment 4 — Multimodal Fusion**:
  - Early concatenation fusion: `[sqrt(alpha) * text || sqrt(1-alpha) * image]`
  - Late score fusion: `alpha * sim_text + (1-alpha) * sim_image`
  - Tunes alpha on validation split
- **Experiment 5 — Final Optimized Model**:
  - Optimal threshold + alpha evaluated on held-out test split

### 3. Downloading Artifacts for Local Streamlit Demo
After running the notebook on Kaggle, download:
1. `test_text_embeddings.npy` (embeddings for the test/demo split)
2. `test_image_embeddings.npy` (embeddings for the test/demo split)
3. `results.json` (experiment benchmark results)

Place the embeddings in `data/embeddings/` and `results.json` in `experiments/results.json`. The Streamlit app will immediately pick them up!
