# Multimodal Product Matching — Implementation Plan

## Goal

Build a Python-based multimodal product matching system on the Shopee dataset that progresses from a TF-IDF baseline through deep-learning text, image, and multimodal embeddings, with a Streamlit demo showcasing interactive product matching and experiment results.

## Key Constraints & Decisions

| Concern | Decision |
|---|---|
| Training environment | Kaggle notebooks (GPU); download resulting artifacts (models, embeddings, metrics) |
| Local environment | CPU-only; runs Streamlit demo + inference with pre-computed embeddings |
| Dataset scope | Full ~34 k listings; a held-out demo subset (~5 k) for the Streamlit demo |
| Text encoder baseline | `sentence-transformers/all-MiniLM-L6-v2` |
| Image encoder | `timm` (e.g. EfficientNet-B0 or ViT-small); exact model chosen in Experiment 3 |
| Retrieval | FAISS (flat index for accuracy, IVF for scale) |
| Demo modes | Mode 1 (Dataset Matcher) as core; Mode 2 (Live Web) as a stub |
| Team size | Solo developer |

---

## Project Structure

```
productmatching/
├── data/
│   ├── shopee-product-matching.zip      # raw download (already present)
│   ├── raw/                             # extracted CSVs + images
│   ├── processed/                       # cleaned CSV, train/val/demo splits
│   └── embeddings/                      # pre-computed .npy / .pkl files
├── docs/                                # existing docs
├── notebooks/                           # Kaggle-oriented notebooks
│   ├── 01_eda.ipynb
│   ├── 02_tfidf_baseline.ipynb
│   ├── 03_text_embeddings.ipynb
│   ├── 04_image_embeddings.ipynb
│   ├── 05_multimodal_fusion.ipynb
│   └── 06_optimization.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                        # paths, constants, model names
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                    # load CSVs, resolve image paths
│   │   ├── preprocessing.py             # text cleaning, image transforms
│   │   └── splitter.py                  # train / val / demo split logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_encoder.py              # TF-IDF + sentence-transformer encoders
│   │   ├── image_encoder.py             # timm-based image encoder
│   │   └── multimodal.py                # fusion strategies (concat, weighted)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── faiss_index.py               # build / query FAISS indices
│   │   └── matching.py                  # threshold, grouping, deduplication
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                   # precision, recall, F1 (per-query & global)
│   │   └── experiment_tracker.py        # save/load experiment result JSONs
│   └── utils.py                         # helpers (timing, logging, display)
├── app/
│   ├── app.py                           # Streamlit entry point
│   ├── pages/
│   │   ├── 1_Product_Matcher.py         # Mode 1 — Dataset Product Matcher
│   │   ├── 2_Experiment_Results.py      # experiment comparison dashboard
│   │   └── 3_Live_Matcher.py            # Mode 2 — stub / placeholder
│   └── components/
│       ├── product_card.py              # reusable product card (image + title + score)
│       └── sidebar_controls.py          # model selector, threshold slider, etc.
├── experiments/                         # saved experiment results (JSON/CSV)
│   └── results.json
├── scripts/
│   ├── extract_data.py                  # unzip & organize raw data
│   ├── precompute_embeddings.py         # batch encode all listings (for local use)
│   └── build_faiss_index.py             # build FAISS indices from embeddings
├── requirements.txt
└── README.md
```

---

## NUS ISS Requirements Coverage

The project requirements demand at least **3 of 4** aspects. Here is how they are satisfied:

| Requirement | How addressed |
|---|---|
| ✅ Supervised / unsupervised learning | TF-IDF + cosine similarity (unsupervised); label-group supervision for evaluation & threshold tuning |
| ✅ Machine learning / deep learning | TF-IDF (ML); sentence-transformers & timm CNNs/ViTs (DL) |
| ✅ Hybrid / ensemble approach | Multimodal fusion of text + image embeddings; weighted scoring |
| ✅ Intelligent sensing / sense making | Image-based product recognition; visual similarity as a signal |

All four are covered.

---

## Data Pipeline

### Phase 0 — Data Extraction & Splits

#### [NEW] [extract_data.py](file:///c:/Users/Rong/dev/productmatching/scripts/extract_data.py)
- Unzips `shopee-product-matching.zip` into `data/raw/`
- Loads `train.csv`, verifies image paths exist

#### [NEW] [splitter.py](file:///c:/Users/Rong/dev/productmatching/src/data/splitter.py)
- Splits by `label_group` to avoid data leakage (all listings from the same group stay in the same split)
- **Train split** (~70 %): for model fitting, fine-tuning, and embedding computation only
- **Validation split** (~15 %): for hyperparameter tuning (thresholds, fusion weights $\alpha$, top-k)
- **Test / Demo split** (~15 %, ≥ 5 k listings): held-out unseen `label_group`s; used for final unbiased evaluation reporting AND powers the interactive Streamlit demo
- Saves split CSVs to `data/processed/`

> [!NOTE]
> Hyperparameters are optimized using the validation set. Final reported benchmark performance is measured on the held-out test/demo set containing previously unseen product groups, guaranteeing zero data leakage and unbiased evaluation.

> [!IMPORTANT]
> The Shopee dataset only provides `train.csv` (the Kaggle test set has no labels). All splits come from `train.csv`, grouped by `label_group` to prevent leakage.

#### [NEW] [preprocessing.py](file:///c:/Users/Rong/dev/productmatching/src/data/preprocessing.py)
- Text: lowercase, strip HTML/special chars, optional transliteration for mixed-language titles
- Images: resize, normalize (per-model requirements)

---

## Experiments

### Experiment 0 — TF-IDF Baseline

**Notebook**: `02_tfidf_baseline.ipynb` (runs on Kaggle or locally — no GPU needed)

| Step | Detail |
|---|---|
| Encoder | `TfidfVectorizer` (scikit-learn), bigrams, sublinear TF |
| Retrieval | Sparse cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) — keep it simple, no FAISS at this stage |
| Threshold | Grid-search on validation split |
| Metrics | Precision, Recall, F1 per-query and macro |

**Artifacts to download**: TF-IDF vocabulary/matrix (optional, small); experiment results JSON.

#### [NEW] [text_encoder.py](file:///c:/Users/Rong/dev/productmatching/src/models/text_encoder.py) — `TfidfEncoder` class
- `fit(titles)` → fits the vectorizer
- `encode(titles)` → returns dense or sparse matrix
- `save()` / `load()` for persistence

---

### Experiment 1 — Transformer Text Embeddings

**Notebook**: `03_text_embeddings.ipynb` (Kaggle GPU)

| Step | Detail |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Encoding | Batch encode all titles → `text_embeddings.npy` |
| Retrieval | FAISS `IndexFlatIP` on L2-normalized embeddings |
| Threshold | Tune on validation split |
| Metrics | Compare against Experiment 0 |

**Artifacts to download**: `text_embeddings.npy` (~34 k × 384 × 4 bytes ≈ 50 MB), results JSON.

#### [NEW] [text_encoder.py](file:///c:/Users/Rong/dev/productmatching/src/models/text_encoder.py) — `SentenceTransformerEncoder` class
- `encode(titles, batch_size)` → np.ndarray
- Model name configurable

---

### Experiment 2 — Improved Text Embeddings (Optional fine-tuning)

**Notebook**: `03_text_embeddings.ipynb` (continued) or a separate section

Options (choose during experimentation):
1. Try a larger / multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`)
2. Fine-tune with contrastive loss using `label_group` pairs
3. Both

If fine-tuning:
- Create positive pairs from same `label_group`, hard negatives from similar but different groups
- Fine-tune with `SentenceTransformer.fit()` + `MultipleNegativesRankingLoss`
- Save fine-tuned model to `models/` directory

**Artifacts**: fine-tuned model weights (if applicable), updated embeddings, results JSON.

---

### Experiment 3 — Image Embeddings

**Notebook**: `04_image_embeddings.ipynb` (Kaggle GPU)

| Step | Detail |
|---|---|
| Model | `timm` — EfficientNet-B0 or ViT-small (feature extraction, no classifier head) |
| Encoding | Batch encode all images → `image_embeddings.npy` |
| Retrieval | FAISS on image embeddings |
| Threshold | Tune separately |
| Metrics | Compare image-only against text-only approaches |

**Artifacts to download**: `image_embeddings.npy` (~34 k × 512–768 × 4 bytes ≈ 70–100 MB), results JSON.

#### [NEW] [image_encoder.py](file:///c:/Users/Rong/dev/productmatching/src/models/image_encoder.py) — `ImageEncoder` class
- `encode(image_paths, batch_size)` → np.ndarray
- Uses `timm.create_model(model_name, pretrained=True, num_classes=0)` for feature extraction

---

### Experiment 4 — Multimodal Fusion

**Notebook**: `05_multimodal_fusion.ipynb` (Kaggle or local)

| Strategy | Description |
|---|---|
| Early fusion (concat) | `[text_emb ‖ image_emb]` → normalize → FAISS |
| Late fusion (weighted avg) | `α · sim_text + (1-α) · sim_image`, tune α |
| Score-level fusion | Retrieve top-K from each modality, merge & re-rank |

- Compare all fusion strategies against unimodal results
- Tune α / weights on validation split

#### [NEW] [multimodal.py](file:///c:/Users/Rong/dev/productmatching/src/models/multimodal.py)
- `ConcatFusion`, `WeightedScoreFusion`, `ReRankFusion` classes
- Common interface: `fuse(text_emb, image_emb)` → combined embedding or score

---

### Experiment 5 — Retrieval & Threshold Optimization

**Notebook**: `06_optimization.ipynb`

- Optimal threshold per modality/fusion strategy (validation-set grid search)
- Two-stage retrieval: coarse FAISS → fine re-ranking
- Connected-components grouping vs. greedy matching
- Final evaluation on the full validation set

---

## Evaluation Module

#### [NEW] [metrics.py](file:///c:/Users/Rong/dev/productmatching/src/evaluation/metrics.py)
- `per_query_precision_recall_f1(predicted_matches, true_matches)` → per-query scores
- `macro_metrics(all_queries)` → macro-averaged P/R/F1
- Compatible with Shopee evaluation: for each listing, predict a set of matching posting_ids; compare against `label_group`

#### [NEW] [experiment_tracker.py](file:///c:/Users/Rong/dev/productmatching/src/evaluation/experiment_tracker.py)
- Saves experiment configs + metrics to `experiments/results.json`
- Schema:
```json
{
  "experiment_id": "exp_04_multimodal_concat",
  "model_config": { "text_model": "...", "image_model": "...", "fusion": "concat" },
  "hyperparameters": { "threshold": 0.65, "top_k": 50, "alpha": 0.6 },
  "metrics": { "precision": 0.82, "recall": 0.78, "f1": 0.80 },
  "timestamp": "2026-09-20T14:30:00"
}
```

---

## FAISS Retrieval Module

#### [NEW] [faiss_index.py](file:///c:/Users/Rong/dev/productmatching/src/retrieval/faiss_index.py)
- `build_index(embeddings, index_type="flat")` → FAISS index
- `query_index(index, query_emb, top_k)` → distances, indices
- `save_index()` / `load_index()`

#### [NEW] [matching.py](file:///c:/Users/Rong/dev/productmatching/src/retrieval/matching.py)
- `apply_threshold(distances, indices, threshold)` → filtered matches
- `group_matches(matches)` → predicted product groups (connected components)

---

## Streamlit Application

### App Structure

Uses Streamlit's built-in **multi-page app** pattern (files in `app/pages/`).

#### [NEW] [app.py](file:///c:/Users/Rong/dev/productmatching/app/app.py) — Home / landing page
- Project overview, dataset summary stats
- Navigation to sub-pages

---

### Page 1 — Dataset Product Matcher (Mode 1 — Core)

#### [NEW] [1_Product_Matcher.py](file:///c:/Users/Rong/dev/productmatching/app/pages/1_Product_Matcher.py)

**Sidebar Controls** (via [sidebar_controls.py](file:///c:/Users/Rong/dev/productmatching/app/components/sidebar_controls.py)):
- Model selector: TF-IDF | Text Embedding | Image Embedding | Multimodal
- Similarity threshold slider (0.0 – 1.0)
- Text/Image fusion weight α (when multimodal)
- Top-K results slider (5 – 50)

**Main Area**:
1. **Search bar** — search/filter product titles from the demo split
2. **Reference product card** — shows selected product's image + title
3. **Results display (Dynamic Precision / Recall Trade-off Inspector)**:
   Retrieved candidates dynamically split into three clear visual sections based on the active threshold slider:
   - **MATCHED PRODUCTS** (Score $\ge$ threshold):
     - ✅ `Correct Match` — predicted match & same `label_group`
     - ❌ `False Match` — predicted match but different `label_group`
   - **SIMILAR BUT REJECTED** (Highest-scoring candidates immediately below threshold):
     - ✅ `Correct Rejection` — hard negative / similar candidate safely rejected because score < threshold
   - **MISSED TRUE MATCHES** (Ground truth matches that fell below threshold):
     - ⚠️ `Missed Match` — ground truth match missed because score < threshold
   *Moving the threshold slider visually demonstrates false positives appearing vs true matches getting missed in real-time.*
4. **Per-query stats** — Precision, Recall, F1 for this specific query at current threshold
5. **Comparison mode** — toggle to run the same query with two different model configs side-by-side

---

### Page 2 — Experiment Results Dashboard

#### [NEW] [2_Experiment_Results.py](file:///c:/Users/Rong/dev/productmatching/app/pages/2_Experiment_Results.py)

- Loads `experiments/results.json`
- **Summary table** — all experiments with P/R/F1
- **Bar/radar chart** — visual metric comparison across experiments
- **Progression chart** — F1 improvement from baseline → final model
- **Per-experiment detail** — expandable sections with config, hyperparams, and notes
- **Error analysis** section — curated difficult examples (accessories vs. products, variants, etc.)

---

### Page 3 — Live Web Matcher (Mode 2 — Stub)

#### [NEW] [3_Live_Matcher.py](file:///c:/Users/Rong/dev/productmatching/app/pages/3_Live_Matcher.py)

- Placeholder UI with a search box and a "Coming Soon" message
- Documents what this mode would do when implemented
- No backend logic until Mode 1 and all experiments are complete

---

### Reusable Components

#### [NEW] [product_card.py](file:///c:/Users/Rong/dev/productmatching/app/components/product_card.py)
- `render_product_card(image_path, title, score, match_status)` — renders a styled product card with `st.image` + `st.markdown`

#### [NEW] [sidebar_controls.py](file:///c:/Users/Rong/dev/productmatching/app/components/sidebar_controls.py)
- Returns a dict of selected settings (model, threshold, α, top_k)

---

## Implementation Phases

### Phase 1 — Foundation (Days 1–2)
- [ ] Extract & explore dataset (EDA notebook)
- [ ] Implement data loading, cleaning, splitting (`src/data/`)
- [ ] Build evaluation module (`src/evaluation/`)
- [ ] Set up project structure, `requirements.txt`, `config.py`

### Phase 2 — Baseline & Text (Days 3–4)
- [ ] Experiment 0: TF-IDF baseline (notebook + `src/models/text_encoder.py`)
- [ ] Experiment 1: Sentence-transformer embeddings
- [ ] Build FAISS retrieval module (`src/retrieval/`)
- [ ] Pre-compute and save text embeddings on Kaggle
- [ ] Download embedding artifacts locally

### Phase 3 — Image & Multimodal (Days 5–6)
- [ ] Experiment 3: Image embeddings (notebook + `src/models/image_encoder.py`)
- [ ] Experiment 4: Multimodal fusion (`src/models/multimodal.py`)
- [ ] Pre-compute and save image embeddings on Kaggle
- [ ] Download embedding artifacts locally

### Phase 4 — Optimization (Day 7)
- [ ] Experiment 2: Improved/fine-tuned text embeddings (if time allows)
- [ ] Experiment 5: Retrieval & threshold optimization
- [ ] Final evaluation on full validation set

### Phase 5 — Streamlit App (Days 8–9)
- [ ] Build app skeleton + sidebar controls
- [ ] Page 1: Dataset Product Matcher
- [ ] Page 2: Experiment Results Dashboard
- [ ] Page 3: Live Matcher stub
- [ ] Polish UI, add comparison mode
- [ ] End-to-end testing

### Phase 6 — Report & Presentation (Day 10)
- [ ] Final report
- [ ] Video presentation
- [ ] Code cleanup & README

---

## Verification Plan

### Automated Tests
```bash
# Run evaluation metrics on validation split for each experiment
python -m src.evaluation.metrics --split val --experiment all

# Smoke-test the Streamlit app
streamlit run app/app.py
```

### Manual Verification
- Verify each experiment's metrics match the saved results JSON
- Visually inspect the Streamlit demo for correct match badges
- Test edge cases: accessories, product variants, different generations
- Confirm all pre-computed embeddings load correctly on the local machine

### Metrics Targets (approximate, for reference)
| Experiment | Expected F1 range |
|---|---|
| TF-IDF Baseline | 0.40 – 0.55 |
| Text Embeddings | 0.55 – 0.70 |
| Image Embeddings | 0.45 – 0.60 |
| Multimodal Fusion | 0.65 – 0.80 |
| Optimized Final | 0.75 – 0.85 |

---

## Open Questions

> [!TIP]
> **Demo images**: Extract and generate lightweight thumbnails *only* for the test/demo set (~5k images). This minimizes local storage (<100MB) while ensuring lightning-fast rendering in the Streamlit app. Full images stay zipped or on Kaggle.

> [!NOTE]
> **Experiment 2 scope**: Fine-tuning the sentence-transformer with contrastive loss would produce the best results but takes more effort. If time is tight, we can skip fine-tuning and use a larger pre-trained multilingual model instead. I'll keep the code modular so either approach can be swapped in.

## Dependencies

```txt
# requirements.txt
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
sentence-transformers>=2.2
torch>=2.0
torchvision>=0.15
timm>=0.9
faiss-cpu>=1.7
Pillow>=10.0
matplotlib>=3.7
plotly>=5.15
```
