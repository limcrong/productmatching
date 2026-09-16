# Multimodal Product Matching — Phase 2: Experiments 1 to 5 Implementation Plan (Revised)

## Goal
Execute, evaluate, and benchmark **Experiments 1 through 5** following a rigorous academic progression:
1. Pretrained text representation & model selection (Exp 1)
2. Task-specific fine-tuning / domain adaptation (Exp 2)
3. Vision feature extraction & encoder selection (Exp 3)
4. Score-level multimodal fusion (Exp 4)
5. Retrieval & post-processing hypothesis testing (Exp 5)

Every experiment introduces **one clearly identifiable change**, measures absolute performance and $\Delta$ against the preceding baseline on validation data, and evaluates on the held-out test split (unseen product groups) only after configuration is fixed.

---

## Experimental Progression & Research Story

```text
Experiment 0: TF-IDF Classical Baseline (DONE — Val F1: 0.7406, Test F1: 0.7409)
      │
      ▼
Experiment 1: Pretrained Transformer Text Embeddings & Model Selection
              (Compare all-MiniLM-L6-v2 vs paraphrase-multilingual on Val; select best)
      │
      ▼
Experiment 2: Domain-Adapted / Fine-Tuned Text Encoder
              (Fine-tune selected text encoder with ArcFace / metric learning on train label_group)
      │
      ▼
Experiment 3: Vision Feature Extraction & Encoder Selection
              (EfficientNet-B0 primary; benchmark CLIP-ViT; select best on Val)
      │
      ▼
Experiment 4: Weighted Score-Level Multimodal Fusion
              (Sim_fused = α·Sim_text + (1-α)·Sim_image; evaluate α ∈ [0, 1] including boundaries)
      │
      ▼
Experiment 5: Retrieval & Post-Processing Hypothesis Testing
              (Hypothesis 1: Reciprocal NN; Hypothesis 2: Graph Transitive Clustering; FAISS Latency Benchmark)
```

---

## Detailed Experimental Design

### Experiment 1: Pretrained Transformer Text Embeddings & Model Selection
- **Core Change**: Transition from sparse bigrams to dense semantic embeddings.
- **Model Selection on Validation Split**:
  - Model A: `sentence-transformers/all-MiniLM-L6-v2` (monolingual/English-optimized, 384-d).
  - Model B: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual Indonesian/English, 384-d).
- **Validation Protocol**:
  - Tune similarity threshold $\tau$ on validation split for both models.
  - Select winner based on validation F1.
- **Reporting**: Report absolute metrics and $\Delta$ over Experiment 0.

### Experiment 2: Domain Adaptation via Metric Learning / Fine-Tuning
- **Core Change**: Pretrained generic embeddings $\rightarrow$ task-specific representations learned from Shopee `label_group`.
- **Objective & Loss**:
  - Primary metric-learning objective: `MultipleNegativesRankingLoss` with `SentenceTransformer`.
  - ArcFace maintained as an optional secondary comparison.
- **Sampling Protocol & Constraints**:
  - Positive pairs constructed from listings sharing the same `label_group`.
  - **Group-Aware Batching**: When using in-batch negatives, ensure listings belonging to the same `label_group` are never treated as negative examples.
  - **Hard Negatives**: Where practical, mine hard negatives from different `label_group`s with high lexical/TF-IDF similarity to teach the model to discriminate between variants, accessories, and actual products.
  - Hyperparameters recorded for reproducibility: loss name, sampling strategy, epochs (e.g. 3–5), batch size (e.g. 32–64), learning rate (e.g. $2 \times 10^{-5}$ with warmup).
- **Validation Protocol**:
  - Re-tune threshold $\tau$ on validation split.
  - Measure $\Delta \text{F1}$ over pretrained Experiment 1 to quantify the contribution of supervised domain adaptation.

### Experiment 3: Pretrained Vision Feature Extraction & Model Selection
- **Core Change**: Add unimodal visual signal independent of text titles.
- **Model Selection on Validation Split**:
  - Model A (Primary Baseline): `timm / efficientnet_b0` (ImageNet pretrained, 1280-d pooled features).
  - Model B (Zero-Shot Vision): `openai/clip-vit-base-patch32` visual trunk (512-d).
- **Validation Protocol**:
  - Tune threshold $\tau$ on validation split.
  - Select best vision encoder for multimodal fusion in Experiment 4.
  - Measure unimodal vision performance vs. unimodal text performance.

### Experiment 4: Weighted Score-Level Multimodal Fusion
- **Core Change**: Combine text and visual similarity scores to resolve single-modality ambiguities (e.g. phone vs. phone case, color variants).
- **Formulation**:
  $$\text{Sim}_{\text{fused}} = \alpha \cdot \text{Sim}_{\text{text}} + (1 - \alpha) \cdot \text{Sim}_{\text{image}}$$
  *(Note: Weighted score fusion is mathematically equivalent to L2-normalized weighted vector concatenation, so we focus on score-level fusion for clarity and direct interpretability).*
- **Two-Stage Validation Tuning Protocol**:
  1. **Coarse Sweep**: Sweep $\alpha \in \{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}$ on the validation split. Boundary references $\alpha = 1.0$ (pure text) and $\alpha = 0.0$ (pure image) verify unimodal anchor points.
  2. **Fine Search**: Perform a finer search around the best $\alpha$ region (e.g. $\pm 0.2$ in $0.05$ increments) jointly with threshold tuning $\tau \in [0.40, 0.85]$.
  3. **Freeze**: Lock the optimal $(\alpha^*, \tau^*)$ pair based strictly on validation data before any test evaluation.
  4. Quantify whether multimodal synergy provides a positive $\Delta \text{F1}$ over both the best unimodal text and best unimodal image models.

### Experiment 5: Retrieval & Post-Processing Hypothesis Testing
Rather than assuming post-processing steps are optimizations, treat them as **hypotheses to rigorously test** against Experiment 4:

1. **Hypothesis 5A — Reciprocal Nearest Neighbors**:
   - *Hypothesis*: Requiring mutual nearest neighbor agreement ($A \in \text{TopK}(B)$ and $B \in \text{TopK}(A)$) reduces false positives.
   - *Test*: Evaluate on validation split. If precision increases without disproportionate recall loss, retain; otherwise discard.
2. **Hypothesis 5B — Graph-Based Transitive Grouping (Connected Components)**:
   - *Hypothesis*: Transitive grouping ($A \sim B$ and $B \sim C \implies A \sim C$) improves recall for dispersed product clusters.
   - *Risk*: Risk of chaining false positive merges across distinct products.
   - *Test*: Evaluate on validation split. Retain only if validation F1 increases.
3. **Retrieval Latency & Efficiency Benchmark (FAISS)**:
   - *Metric separation*: Report computational efficiency (query latency in ms, QPS) separately from matching quality (F1).
   - Compare brute-force cosine search vs. `faiss.IndexFlatIP` (exact search, identical F1, lower latency).
   - Benchmark approximate `faiss.IndexIVFFlat` measuring Recall@K vs. search latency.

---

## Academic Validation & Leakage Prevention Protocol

| Phase | Dataset Split Used | Permitted Actions |
|---|---|---|
| **Training / Fine-Tuning** | `train_split.csv` (~70%, 23,971 rows) | Model fitting, TF-IDF vocabulary, `MultipleNegativesRankingLoss` fine-tuning. |
| **Model Selection & Tuning** | `val_split.csv` (~15%, 5,142 rows) | Threshold $\tau$ tuning, coarse/fine $\alpha$ selection, Top-K selection, post-processing hypothesis decisions. |
| **Final Benchmark** | `test_demo_split.csv` (~15%, 5,137 rows) | **Strictly zero tuning**. Evaluated once across all frozen experiment configurations after Experiments 1–5 are completed. Unseen product groups, zero leakage. |
| **Interactive Streamlit UI** | Validation split (during development); frozen Test split (final presentation) | Interactive inspection without using test data to guide model iterations. |

---

## Experiment Tracking Matrix (Saved in `experiments/results.json`)

| Exp ID | Approach / Identifiable Change | Model Details | Optimal Params (Val) | Val F1 | Test F1 | $\Delta$ vs. Baseline | Decision / Outcome |
|:---|:---|:---|:---|:---|:---|:---|:---|
| `exp_00` | Classical Bigram Baseline | TF-IDF (sublinear, 25k) | $\tau = 0.45$ | 0.7406 | **0.7409** | Baseline | Reference |
| `exp_01` | Pretrained Text Embeddings | MiniLM-L6 vs Multilingual | $\tau = \text{TBD}$ | TBD | TBD | TBD | Select best text encoder |
| `exp_02` | Domain Adaptation | Fine-tuned selected text encoder | $\tau = \text{TBD}$ | TBD | TBD | TBD | Pretrained vs Adapted |
| `exp_03` | Vision Feature Extraction | EfficientNet-B0 vs CLIP | $\tau = \text{TBD}$ | TBD | TBD | TBD | Select best vision encoder |
| `exp_04` | Weighted Score Multimodal Fusion | Best Text + Best Vision | $\alpha^*, \tau^*$ | TBD | TBD | TBD | Multimodal synergy check |
| `exp_05` | Post-Processing & Grouping | Reciprocal NN / Connected Comp | Params TBD | TBD | TBD | TBD | Keep only if $\Delta \text{Val} > 0$ |

---

## Execution Workflow (Kaggle-First with Local Artifact Ingestion)

1. **Kaggle GPU Script (`notebooks/kaggle_pipeline.py`)**:
   - Primary computational engine for GPU training and batch feature extraction.
   - Outputs:
     - `test_text_embeddings.npy` (winning text encoder)
     - `test_image_embeddings.npy` (winning vision encoder)
     - `val_text_embeddings.npy`, `val_image_embeddings.npy`
     - `results.json` (complete structured experiment records)
2. **Local Repository Ingestion**:
   - Download artifacts into `data/embeddings/` and `experiments/results.json`.
   - Local CLI `scripts/run_experiments.py` loads cached embeddings to re-run evaluations, generate report tables, or test FAISS latency locally.
3. **Streamlit UI Presentation**:
   - `pages/1_Product_Matcher.py`: Dynamically inspect queries using cached embeddings and the 3-section precision/recall inspector.
   - `pages/2_Experiment_Results.py`: Render the complete progression table, F1 progression chart, and hypothesis validation notes.
