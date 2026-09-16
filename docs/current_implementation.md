# Multimodal Product Matching — Implementation Walkthrough

## Summary of Accomplishments

We implemented the complete Python-based **Multimodal Product Matching System** with an interactive **Streamlit Demo** adhering strictly to NUS ISS requirements and your review specifications:

1. **Environment & Architecture**:
   - Built a dedicated, isolated virtual environment (`.venv`) with pinned dependencies.
   - Core local stack operates cleanly with `streamlit`, `faiss-cpu`, `scikit-learn`, `pandas`, `numpy`, and `Pillow`.

2. **Zero-Leakage Data Partitioning**:
   - Partitioned the Shopee dataset strictly by `label_group` to guarantee zero data leakage:
     - **Train Split** (`train_split.csv`): 23,971 listings (~70%) for model fitting only.
     - **Validation Split** (`val_split.csv`): 5,142 listings (~15%) for hyperparameter & threshold tuning.
     - **Test/Demo Split** (`test_demo_split.csv`): 5,137 listings (~15%) with unseen product groups for unbiased reporting and interactive demo.
   - Extracted compact 224×224 thumbnails **only for the test/demo listings** (4,880 images, ~45MB) to keep local storage minimal while guaranteeing 100% demo image availability.

3. **Models & Experimental Pipeline**:
   - **Experiment 0 (TF-IDF Baseline)**: Unigram/bigram `TfidfVectorizer` with sublinear TF and sparse cosine similarity.
   - **Experiment 1 (Text DL)**: `SentenceTransformer` (`all-MiniLM-L6-v2`, 384-d).
   - **Experiment 3 (Vision DL)**: `timm` (EfficientNet-B0, 1280-d).
   - **Experiment 4 (Multimodal Fusion)**: Early vector concatenation and late convex combination ($\alpha \cdot \text{Sim}_{text} + (1-\alpha) \cdot \text{Sim}_{img}$).
   - **Experiment 5 (Optimized)**: Validation-calibrated thresholding evaluated on unseen test groups.

4. **Streamlit Multi-Page Demo (`app/`)**:
   - **Landing Page (`app.py`)**: System overview, dataset metrics, and architecture diagram.
   - **Dataset Product Matcher (`pages/1_Product_Matcher.py`)**:
     - **Dynamic Precision/Recall Inspector** with the 3 sections requested:
       - 🟢 **MATCHED PRODUCTS** (Score $\ge \tau$): True Matches & False Positives.
       - ⚪ **SIMILAR BUT REJECTED** (Score $< \tau$): Highest-scoring hard negatives safely filtered out.
       - 🟠 **MISSED TRUE MATCHES** (Score $< \tau$): True matches missed.
     - Real-time precision, recall, and F1 updates as the threshold slider is adjusted.
     - Side-by-side model comparison mode.
   - **Experiment Dashboard (`pages/2_Experiment_Results.py`)**:
     - Benchmark table, Plotly F1 progression chart, Precision/Recall breakdown, and qualitative error analysis (phone vs. case, color variants, multilingual titles, cross-generation confusion).
   - **Live Web Matcher Stub (`pages/3_Live_Matcher.py`)**:
     - Consumer-facing cross-merchant comparison concept stub.

5. **Kaggle GPU Execution (`notebooks/`)**:
   - Created `notebooks/kaggle_pipeline.py` and `notebooks/README.md` for running full deep learning embedding extraction on Kaggle GPUs and saving `test_text_embeddings.npy`, `test_image_embeddings.npy`, and `results.json` for download.

---

## Verification Results

### 1. Data Splitting & Thumbnail Generation
```
Total rows: 34250
Train rows: 23971 (7744 groups)
Validation rows: 5142 (1660 groups)
Test/Demo rows: 5137 (1659 groups)
Successfully generated 4880 thumbnails in data/thumbnails
```

### 2. Experiment 0 (TF-IDF Baseline Evaluation)
```
Fitting TfidfEncoder on 23,971 training product titles...
Tuning threshold on validation set...
Optimal threshold on Validation Split: 0.45 (Val F1: 0.7406)
Evaluating on held-out Test/Demo Split (unseen product groups)...

Mean Precision: 0.8424
Mean Recall:    0.7434
Mean F1 Score:  0.7409
Saved benchmark to experiments/results.json
```

### 3. Streamlit Service Health Check
- Port `8501`: `LISTENING`
- HTTP Status: `200 OK` (Serving at `http://localhost:8501/`)

---

## How to Run Locally

### Activate Environment & Start Demo
```powershell
.\.venv\Scripts\activate
streamlit run app/app.py
```
Open browser at: `http://localhost:8501`

### Run Local Experiments
```powershell
.\.venv\Scripts\python scripts/run_experiments.py --experiment 0
```
