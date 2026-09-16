# Multimodal Product Matching System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![NUS ISS](https://img.shields.io/badge/NUS%20ISS-Pattern%20Recognition%20Systems-purple.svg)](https://www.iss.nus.edu.sg/)

An academic pattern recognition and multimodal deep learning system that identifies and groups equivalent e-commerce product listings across merchants despite variations in titles, languages, and photography.

Based on the [Kaggle Shopee Product Matching dataset](https://www.kaggle.com/competitions/shopee-product-matching/data).

---

## 📌 Key Highlights

- **Zero Data Leakage Split**: Dataset partitioned strictly by `label_group`:
  - **Train split** (~70%): Model fitting, fine-tuning, embedding computation.
  - **Validation split** (~15%): Hyperparameter tuning (similarity threshold $\tau$, multimodal fusion weight $\alpha$).
  - **Test/Demo split** (~15%, ~5,100 listings): Held-out unseen product groups for unbiased reporting and interactive Streamlit demo.
- **Experimental Progression**:
  - **Exp 0 — Classical Baseline**: TF-IDF unigram/bigram features with sparse cosine similarity.
  - **Exp 1 — Text Deep Learning**: Transformer embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384-d).
  - **Exp 3 — Vision Deep Learning**: Visual CNN feature extraction (`timm` EfficientNet-B0, 1280-d).
  - **Exp 4 — Multimodal Fusion**: Joint early vector concatenation & late weighted score fusion ($\alpha$).
  - **Exp 5 — Final Optimization**: Validation-calibrated thresholds evaluated on unseen test groups.
- **Dynamic Streamlit Demo**:
  - **3-Section Precision/Recall Inspector**:
    - 🟢 **MATCHED PRODUCTS** (Score $\ge \tau$): True Matches & False Positives.
    - ⚪ **SIMILAR BUT REJECTED** (Score $< \tau$): Highest-scoring hard negatives safely filtered out.
    - 🟠 **MISSED TRUE MATCHES** (Score $< \tau$): True matches missed.
  - Interactive threshold slider demonstrating precision/recall trade-offs live!
  - Side-by-side model comparison mode.
  - Academic Experiment Results Dashboard with interactive Plotly charts.

---

## 🛠️ Project Structure

```
productmatching/
├── .gitattributes                       # Git LFS tracking for demo_thumbnails.zip
├── data/
│   ├── demo_thumbnails.zip              # Packaged test/demo thumbnails (Git LFS)
│   ├── shopee-product-matching.zip      # Raw dataset archive (optional, Kaggle)
│   ├── raw/                             # Extracted train.csv / test.csv
│   ├── processed/                       # train_split.csv, val_split.csv, test_demo_split.csv
│   ├── thumbnails/                      # 224x224 thumbnails (extracted on-demand)
│   └── embeddings/                      # Precomputed .npy embeddings
├── app/                                 # Streamlit Multi-Page Demo
│   ├── __init__.py                      # Package marker for cloud/local execution
│   ├── app.py                           # Landing & System Architecture
│   ├── pages/
│   │   ├── 1_Product_Matcher.py         # Mode 1: Core Dataset Product Matcher
│   │   ├── 2_Experiment_Results.py      # Quantitative Benchmarks & Error Analysis
│   │   └── 3_Live_Matcher.py            # Mode 2: Consumer-Facing Web Search Stub
│   └── components/
│       ├── __init__.py                  # Component package marker
│       ├── product_card.py              # Status badges & on-demand image loader
│       └── sidebar_controls.py          # Interactive sliders & dropdowns
├── src/                                 # Core Python Package
│   ├── config.py                        # Constants & directory configurations
│   ├── data/                            # Loader, cleaner, and zero-leakage splitter
│   ├── models/                          # TF-IDF, SentenceTransformers, timm, Fusion
│   ├── retrieval/                       # FAISS index & precision-recall matching
│   └── evaluation/                      # Shopee metrics & ExperimentTracker
├── notebooks/
│   ├── README.md                        # Kaggle GPU workflow guide
│   └── kaggle_pipeline.py               # Self-contained end-to-end Kaggle pipeline
├── scripts/
│   ├── extract_data.py                  # Extracts CSVs and creates test thumbnails
│   └── run_experiments.py              # Local experiment evaluation runner
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/limcrong/productmatching.git
cd productmatching

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets / API tokens
copy .env.example .env        # On Linux/macOS: cp .env.example .env
```

### 2. Data Setup & Git LFS

| Data Path | Status in Git | Description / Instructions |
| :--- | :--- | :--- |
| **`data/processed/*.csv`** | ✅ **Tracked in Git** | Split CSVs available immediately upon `git clone`. Zero configuration required. |
| **`experiments/results.json`** | ✅ **Tracked in Git** | Academic benchmark records across experimental stages for instant dashboard rendering. |
| **`data/demo_thumbnails.zip`** | 📦 **Tracked via Git LFS** | All 4,880 test/demo product images (55 MB). Extracted automatically and on-demand by the app. |
| **`data/thumbnails/`** | ⚡ **On-Demand Cache** | Extracted dynamically in milliseconds as products are rendered in the UI. |
| **`data/embeddings/`** | ❌ Ignored (`.npy`) | Precomputed vectors. Generated via `kaggle_pipeline.py` or computed locally. |
| **`data/shopee-product-matching.zip`** | ❌ Ignored (~1.8 GB) | Full raw dataset (optional, only if retraining vision CNNs from raw full dataset). |

> [!TIP]
> **Git LFS**: When cloning to a new machine, ensure Git LFS is installed to fetch the demo thumbnails:
> ```bash
> git lfs install
> git lfs pull
> ```

### 3. Run Baseline Experiment (Exp 0: TF-IDF)
Evaluate TF-IDF classical baseline (works immediately with tracked processed CSVs):
```powershell
python scripts/run_experiments.py --experiment 0
```

### 4. Launch Streamlit Demo (Local)
```powershell
streamlit run app/app.py
```

### 5. Streamlit Community Cloud Deployment
The repository is fully configured for continuous deployment on [Streamlit Community Cloud](https://share.streamlit.io/):
- **Repository**: `limcrong/productmatching`
- **Branch**: `main`
- **Main file path**: `app/app.py`
- Automatic `sys.path` package resolution is configured in `app/app.py` and page scripts.
- Git LFS pulls `data/demo_thumbnails.zip` automatically, and the app extracts thumbnails on-demand without startup latency.


---

## 🎓 NUS ISS Requirement Alignment

| Assessment Aspect | System Feature |
|---|---|
| **Supervised / Unsupervised Learning** | Unsupervised cosine similarity retrieval; supervised threshold tuning & metric evaluation via `label_group`. |
| **Machine Learning / Deep Learning** | Scikit-learn TF-IDF bigrams (Classical ML); PyTorch SentenceTransformers & timm EfficientNet (Deep Learning). |
| **Hybrid / Ensemble Approach** | Early vector concatenation and late convex combination ($\alpha \cdot \text{Sim}_{text} + (1-\alpha) \cdot \text{Sim}_{img}$). |
| **Intelligent Sensing / Sense Making** | Visual image understanding resolving text ambiguity and false positives in e-commerce listings. |
