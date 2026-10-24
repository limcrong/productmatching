# Multimodal Product Matching System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Demo-brightgreen.svg?logo=streamlit)](https://pmatching-cm.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![NUS ISS](https://img.shields.io/badge/NUS%20ISS-Pattern%20Recognition%20Systems-purple.svg)](https://www.iss.nus.edu.sg/)

An academic pattern recognition and multimodal deep learning system that identifies and groups equivalent e-commerce product listings across merchants despite variations in titles, languages, and photography.

Based on the [Kaggle Shopee Product Matching dataset](https://www.kaggle.com/competitions/shopee-product-matching/data).

---

## 📌 Key Highlights

- **Zero Data Leakage Split**: Dataset partitioned strictly by `label_group`:
  - **Train split** (~70%): Model fitting, metric learning fine-tuning, embedding computation.
  - **Validation split** (~15%): Hyperparameter tuning (similarity threshold $\tau$, multimodal fusion weight $\alpha$, model selection).
  - **Test/Demo split** (~15%, 5,138 listings): Held-out unseen product groups for unbiased reporting and interactive Streamlit demo.
- **Experimental Progression (Exps 0 to 5)**:
  - **Exp 0 — Classical Baseline**: TF-IDF word unigrams/bigrams with sublinear TF and sparse cosine similarity.
  - **Exp 1 — Text Deep Learning**: Comparison of pretrained dense transformers (`all-MiniLM-L6-v2` vs `paraphrase-multilingual-MiniLM-L12-v2`).
  - **Exp 2 — Domain Adaptation**: Supervised metric learning via `MultipleNegativesRankingLoss` with group-aware batching.
  - **Exp 3 — Vision Deep Learning**: Pretrained CNN (`timm` EfficientNet-B0) vs Vision Transformer (`openai/clip-vit-base-patch32`).
  - **Exp 4 — Multimodal Fusion**: Validation grid sweep (Exp 4a) vs Supervised Logistic Regression learned fusion (Exp 4b).
  - **Exp 5 — Final Optimization**: FAISS index latency benchmarks and post-processing hypothesis testing evaluated on unseen test groups.
- **Two Complete Demonstration Modes**:
  - **Mode 1 — Dataset Product Matcher**: 3-section inspector (🟢 Matched Products $\ge \tau$, ⚪ Similar but Rejected $<\tau$, 🟠 Missed Matches) with live threshold slider, side-by-side model comparison, and profile-calibrated hyperparameters on held-out Kaggle Shopee data.
  - **Mode 2 — Image-Assisted Cross-Merchant Product Matching and Price Comparison**: Consumer-facing POC demonstrating how Google Shopping candidate retrieval combined with our trained multimodal model filters accessories and mismatched variants to surface the lowest price among retrieved matches, with dual-model (Text-Only vs Multimodal) explainability.

---

## 🛠️ Project Structure

```
productmatching/
├── .gitattributes                       # Git LFS tracking for demo_thumbnails.zip, embeddings, and models
├── data/
│   ├── demo_thumbnails.zip              # Packaged test/demo thumbnails (Git LFS)
│   ├── processed/                       # train_split.csv, val_split.csv, test_demo_split.csv (Tracked)
│   ├── thumbnails/                      # 224x224 thumbnails (extracted on-demand)
│   ├── embeddings/                      # Dense .npy embeddings (Git LFS)
│   └── live_demo/                       # Curated offline test cases for Image-Assisted Matcher POC
├── app/                                 # Streamlit Multi-Page Web Application
│   ├── app.py                           # Landing & System Architecture
│   ├── pages/
│   │   ├── 1_Product_Matcher.py         # Mode 1: Core Dataset Product Matcher
│   │   ├── 2_Experiment_Results.py      # Quantitative Benchmarks & Error Analysis Dashboard
│   │   └── 3_Live_Matcher.py            # Mode 2: Image-Assisted Product Matching & Price Comparison POC
│   └── components/
│       ├── product_card.py              # Visual status badges & thumbnail loader
│       └── sidebar_controls.py          # Dynamic sliders with calibrated defaults
├── src/                                 # Core Python Package
│   ├── config.py                        # Constants & directory configurations
│   ├── app_config.py                    # Benchmark & hyperparameter profile loader
│   ├── data/                            # Loader, cleaner, and zero-leakage splitter
│   ├── models/                          # TF-IDF, SentenceTransformers, timm, Fusion
│   ├── retrieval/                       # FAISS index, matching logic, and SerpAPI client
│   ├── pipeline/                        # Live candidate matching & re-ranking
│   └── evaluation/                      # Top-K protocol, metrics, and ExperimentTracker
├── notebooks/
│   ├── README.md                        # Kaggle GPU execution guide
│   ├── kaggle_pipeline.py               # Self-contained end-to-end GPU pipeline (Exps 0 to 5)
│   └── kernel-metadata.json             # Kaggle kernel configuration
├── scripts/
│   ├── extract_data.py                  # Extracts CSVs and creates test thumbnails
│   ├── run_experiments.py              # Local baseline & FAISS latency runner
│   └── kaggle_run.py                    # Automated Kaggle kernel push, monitor & artifact sync
├── requirements.txt
└── README.md
```

---

## 🌐 System Access & Evaluation Guide

For evaluators and professors, the application can be evaluated either through the **live hosted cloud deployment** (zero configuration) or by **running it locally**.

---

### Option A: Live Hosted Web App (Recommended — Zero Setup)

The application is deployed on Streamlit Community Cloud with all models and demo data pre-loaded:

👉 **Hosted URL:** [https://pmatching-cm.streamlit.app/](https://pmatching-cm.streamlit.app/)

#### Instructions for Hosted Evaluation:
1. **Browse to the URL**: Open [https://pmatching-cm.streamlit.app/](https://pmatching-cm.streamlit.app/) in any modern web browser.
2. **Waking the App**: Streamlit Community Cloud automatically puts inactive applications to sleep to save cloud resources. If you encounter the sleeping page:
   - Click the blue button: **"Yes, get this app back up"**.
3. **Wait for Startup**: Please wait approximately **1 to 2 minutes** while the cloud container boots up, prepares the runtime environment, and extracts the test thumbnails.
4. **Interactive Demonstration**: Once loaded, use the sidebar navigation to evaluate all three modules:
   - **🔍 Mode 1: Dataset Product Matcher** — Interactive threshold tuning, ground-truth match inspection, and side-by-side comparison across all trained models (TF-IDF, dense text transformers, fine-tuned metric learning, vision CNNs/ViTs, and multimodal fusion).
   - **📈 Experiment Results Dashboard** — Benchmark tables, validation sweeps, and academic performance charts across Experiments 0 to 5.
   - **🛒 Mode 2: Live Matcher POC** — Image-assisted cross-merchant price comparison using curated offline scenarios (or live web search).

---

### Option B: Local Setup Instructions (For Professor / Local Verification)

Follow the steps below to set up and run the project locally on Windows, macOS, or Linux.

#### 1. System Requirements & Prerequisites
- **Python**: `3.10`, `3.11`, or `3.12` installed ([python.org](https://www.python.org/downloads/)).
- **Git** and **Git LFS**: Required to download the repository and binary data artifacts (thumbnails and precomputed dense embeddings).
  - Check if Git LFS is installed:
    ```bash
    git lfs --version
    ```
  - If not installed: install via `git-lfs` (e.g., `brew install git-lfs` on macOS, or download from [git-lfs.com](https://git-lfs.com/)).

#### 2. Clone Repository & Pull Git LFS Data
```bash
# Clone the repository
git clone https://github.com/limcrong/productmatching.git
cd productmatching

# Initialize Git LFS and pull tracked binary artifacts
git lfs install
git lfs pull
```
> [!NOTE]
> `git lfs pull` fetches `data/demo_thumbnails.zip` (55 MB) and precomputed evaluation vectors in `data/embeddings/` (~48 MB). These allow all models to run immediately without needing GPU training.

#### 3. Create & Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, or activate via Command Prompt using `.\.venv\Scripts\activate.bat`)*

- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Environment Configuration (Optional)
Create an environment file from the provided template:
- **Windows:** `copy .env.example .env`
- **macOS / Linux:** `cp .env.example .env`

> [!TIP]
> **API Keys are NOT required for evaluation!**
> - **Mode 1 (Dataset Product Matcher)** and **Experiment Dashboard** run 100% offline using the tracked split CSVs, cached thumbnails, and local embeddings.
> - **Mode 2 (Live Matcher POC)** contains pre-cached **⚡ Curated Demo Scenarios** that run offline with zero external API calls.
> - *(Optional)* A `serpapi_key` in `.env` is only necessary if you choose to perform arbitrary real-time Google Shopping web searches.

#### 6. Launch the Streamlit Application
```bash
streamlit run app/app.py
```
The application will launch and automatically open in your default browser at:
`http://localhost:8501`

#### 7. Verify Baseline Experiment from CLI (Optional)
To verify the classical baseline (Exp 0: TF-IDF with zero data leakage) directly via terminal:
```bash
python scripts/run_experiments.py --experiment 0
```

---

## 📦 Data Artifacts & Git Tracking Status

| Data Path | Status in Git | Description / Instructions |
| :--- | :--- | :--- |
| **`data/processed/*.csv`** | ✅ **Tracked in Git** | Zero-leakage split CSVs (`train_split.csv`, `val_split.csv`, `test_demo_split.csv`). Available immediately upon `git clone`. |
| **`experiments/results.json`** | ✅ **Tracked in Git** | Complete academic benchmark records (Exp 0 to Exp 5) for instant dashboard rendering. |
| **`data/demo_thumbnails.zip`** | 📦 **Tracked via Git LFS** | All 4,880 test/demo product images (55 MB). Extracted automatically and on-demand by the app. |
| **`data/embeddings/*.npy`** | 📦 **Tracked via Git LFS** | Precomputed dense embeddings for test/validation splits (`test_text_*.npy`, `test_img_*.npy`). Downloaded via `git lfs pull`. |
| **`data/thumbnails/`** | ⚡ **On-Demand Cache** | Extracted dynamically in milliseconds as products are rendered in the UI. |
| **`data/shopee-product-matching.zip`** | ❌ Optional / Ignored (~1.8 GB) | Full raw Shopee dataset (only required if retraining vision models or reproducing full dataset extraction from scratch). |


---

## 🎓 NUS ISS Requirement Alignment

| Assessment Aspect | System Feature |
|---|---|
| **Supervised / Unsupervised Learning** | Unsupervised cosine similarity retrieval; supervised threshold tuning & metric evaluation via `label_group`. |
| **Machine Learning / Deep Learning** | Scikit-learn TF-IDF bigrams (Classical ML); PyTorch SentenceTransformers & timm EfficientNet (Deep Learning). |
| **Hybrid / Ensemble Approach** | Early vector concatenation and late convex combination ($\alpha \cdot \text{Sim}_{text} + (1-\alpha) \cdot \text{Sim}_{img}$). |
| **Intelligent Sensing / Sense Making** | Visual image understanding resolving text ambiguity and false positives in e-commerce listings. |
