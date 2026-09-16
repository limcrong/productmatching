# Multimodal Product Matching — Phase 2 Implementation Walkthrough

## Summary of Completed Work

In this phase, we implemented the complete experimental and evaluation framework for **Experiments 1 through 5**, fully incorporating all academic feedback and review decisions:

1. **Pretrained Text Selection & Domain Adaptation (Exp 1 & 2)**:
   - **Experiment 1**: Pretrained transformer text model selection comparing `all-MiniLM-L6-v2` against `paraphrase-multilingual-MiniLM-L12-v2` on the validation split.
   - **Experiment 2**: Domain adaptation using `MultipleNegativesRankingLoss` on `train_split`. Positive pairs are constructed with group-aware batching (ensuring listings from the same `label_group` are never treated as in-batch negatives), establishing a clean academic progression from pretrained representation $\rightarrow$ task-specific learning.

2. **Vision Feature Extraction & Model Selection (Exp 3)**:
   - Evaluates `timm / efficientnet_b0` (1280-d features) as the primary visual baseline.
   - Selected based on validation performance prior to multimodal fusion.

3. **Two-Stage Weighted Score Multimodal Fusion (Exp 4)**:
   - Formulates score fusion: $\text{Sim}_{\text{fused}} = \alpha \cdot \text{Sim}_{\text{text}} + (1 - \alpha) \cdot \text{Sim}_{\text{image}}$.
   - **Coarse sweep**: $\alpha \in \{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$ including unimodal anchors.
   - **Fine search**: Explores around the optimal region ($\pm 0.15$ in $0.05$ increments) jointly with threshold $\tau$.
   - Locks optimal $(\alpha^*, \tau^*)$ strictly on validation data before final test evaluation.

4. **Hypothesis-Driven Post-Processing & Retrieval Benchmarking (Exp 5)**:
   - **Hypothesis 5A (Reciprocal NN)** & **Hypothesis 5B (Transitive Connected Components)** implemented as testable options in [src/retrieval/matching.py](file:///c:/Users/Rong/dev/productmatching/src/retrieval/matching.py). Connected components evaluated against validation data with strict awareness of false-positive cluster chaining risks.
   - **Hypothesis 5C (FAISS Efficiency Benchmark)** in [src/retrieval/faiss_index.py](file:///c:/Users/Rong/dev/productmatching/src/retrieval/faiss_index.py): Separates computational efficiency (latency in ms, QPS) from matching quality (F1), benchmarking exact brute force against `faiss.IndexFlatIP`.

5. **Self-Contained Kaggle GPU Pipeline (`notebooks/kaggle_pipeline.py`)**:
   - Complete end-to-end script ready to run on Kaggle GPU (T4 / P100).
   - Generates all 6 experimental stages, performs validation-only tuning, freezes configurations, evaluates the held-out test split, and exports:
     - `results.json`
     - `test_text_embeddings.npy`
     - `test_image_embeddings.npy`
     - `val_text_embeddings.npy`
     - `val_image_embeddings.npy`
     - `test_demo_split.csv`

6. **Enhanced Streamlit Demo (`app/`)**:
   - **`1_Product_Matcher.py`**: Added data split selector (Validation Split for tuning/development vs. Test Split for frozen presentation), plus experimental hypothesis toggles.
   - **`2_Experiment_Results.py`**: Updated with full 6-stage comparison table, $\Delta$ vs. baseline tracking, validation vs. test F1 progression chart, retrieval latency table, and qualitative error analysis.

---

## Verification Results

### 1. FAISS Latency & Computational Efficiency Benchmark
Executed via: `.\.venv\Scripts\python scripts/run_experiments.py --benchmark-faiss`
```
Database Size:                   5,138 vectors
Embedding Dimension:             384
Brute Force Latency per query:   0.046 ms (21,666.1 QPS)
FAISS IndexFlatIP Latency:       0.224 ms (4,469.0 QPS)
Recall@50:                       100.0% (Exact cosine search)
```

### 2. Streamlit Web App Health Check
- Port `8501`: `LISTENING`
- HTTP Status: `200 OK`
- Active URL: `http://localhost:8501/`

---

## Next Steps for the User

1. **Run Kaggle GPU Pipeline**:
   - Create a Kaggle notebook with GPU accelerator attached to the `shopee-product-matching` dataset.
   - Copy-paste and run [notebooks/kaggle_pipeline.py](file:///c:/Users/Rong/dev/productmatching/notebooks/kaggle_pipeline.py).
   - Download the generated `.npy` embeddings into `data/embeddings/` and `results.json` into `experiments/results.json`.
2. **Inspect Locally**:
   - The Streamlit demo at `http://localhost:8501/` will immediately display the complete progression across all 6 experimental stages!
