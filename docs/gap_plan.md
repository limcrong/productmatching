# Gap Remediation Plan — Proposal → Final System

> **Audience:** the next implementing agent/developer.
> **Source of truth for scope:** `docs/PRS_Group43_Proposal_final_ppt_final.pptx` (proposal slides) + this document.
> **Created:** 2026-10-10. Work through phases **in order**; each phase ends with explicit acceptance criteria.

---

## 0. Context & Decisions (read first)

### 0.1 Project in one paragraph
Multimodal product matching on the Kaggle Shopee dataset. A listing (title + image) is encoded by a text encoder and an image encoder; candidates are retrieved by nearest-neighbour search; text and image similarities are fused with a weighted score and thresholded to decide "same product". Evaluated with per-query Precision / Recall / F1 against `label_group` on **unseen** product groups. Delivered as a Streamlit app with (A) a Dataset Matcher and (B) a Visual Shopping Assistant that matches a user's product image against live web listings and returns the cheapest exact match.

### 0.2 Decisions already made by the user (do NOT re-ask)
| # | Decision |
|---|---|
| D1 | **Shopping Assistant retrieval = SerpAPI.** User will put the key in `.env` as `serpapi_key=...`. |
| D2 | **Kaggle GPU is available.** Credentials are in `.env` as `kaggle_token=...`. Use Kaggle for all GPU work. |
| D3 | **No ArcFace / image metric learning.** Only follow what the proposal lists. |
| D4 | **Fusion = weighted score fusion, with weights optimised by training** (learned, not only grid-searched). Ignore the "voting" wording on slide 11. |

### 0.3 Proposal promises that must be visibly satisfied
| Slide | Promise | Phase that delivers it |
|---|---|---|
| S3 Obj 1 | End-to-end working prototype | A–D |
| S3 Obj 2 | Compare classical, text, image, multimodal on equal footing | A, C |
| S3 Obj 3 | Quantify every optimisation with P / R / F1 | A, C |
| S3 Obj 4 | Interactive demo against held-out ground truth | B |
| S3 Obj 5 | Apply model on live web results | D |
| S6 | Text enc + image enc → embeddings → candidate retrieval (FAISS) → fusion + threshold | A, B |
| S9 | Supervised learning on `label_group`; TF-IDF, transformer text embeddings, **CNN/CLIP** visual, metric learning (text); hybrid fusion | A |
| S10 | Eval: Top-K retrieve → threshold → check label_group → TP/FP/FN | A |
| S11 | Validation-selected threshold; learns similarity not classes | A |
| S12 | Works on unseen products (no class lookup) | A (test split = unseen groups), E (report) |
| S13A | Dataset Matcher: choose TF-IDF/text/image/multimodal, scores, true/false/missed, per-query P/R/F1 | B |
| S13B/S14 | Upload screenshot/image + optional text hint → external search → re-rank/filter → merchant + price → cheapest | D |

### 0.4 Current state snapshot (verified 2026-10-10)
- ✅ Splits exist and are committed: `data/processed/{train,val,test_demo}_split.csv` (split by `label_group`, seed 42, 70/15/15). Test = 5,138 rows.
- ✅ Exp 0 TF-IDF ran locally → `experiments/results.json` (test F1 0.741, τ=0.45).
- ❌ Exp 1–5 never ran. `data/embeddings/` is empty.
- ❌ `notebooks/kaggle_pipeline.py` will crash (`val_text_val_f1` undefined at L247 & L439; imports `src.retrieval.matching` at L475 which doesn't exist on Kaggle).
- ❌ Demo silently falls back to TF-IDF when embeddings are missing.
- ❌ `app/pages/3_Live_Matcher.py` is a hardcoded mockup.
- ❌ `app/pages/2_Experiment_Results.py` writes fabricated default numbers to `results.json` when empty.

---

## 1. Target Experiment Design (final)

All tuning on **val**, final numbers on **test** with frozen settings. Every experiment logs to `experiments/results.json` via the schema in §1.3.

| ID | Name | What | Selected by |
|---|---|---|---|
| `exp_00` | TF-IDF baseline | Already done. Keep local config (`max_features=25000`, `ngram_range=(1,2)`, `sublinear_tf=True`, **no stop words**). Re-run with Top-K protocol (§1.2). | val F1 → τ |
| `exp_01` | Pretrained text | `all-MiniLM-L6-v2` vs `paraphrase-multilingual-MiniLM-L12-v2` (Shopee titles are Indonesian/English mix). | val F1 |
| `exp_02` | Fine-tuned text (metric learning) | Winner of exp_01 fine-tuned with `MultipleNegativesRankingLoss` on **train_fit** groups (§1.1) using a **group-aware batch sampler** (no two pairs from the same `label_group` in one batch). 2–3 epochs, batch 64. **Save the model.** | val F1; adopt only if > exp_01 |
| `exp_03` | Image encoder | `timm efficientnet_b0` (1280-d, CNN) vs `openai/clip-vit-base-patch32` image tower (512-d). Pretrained only (D3). | val F1 |
| `exp_04a` | Weighted fusion — grid | `s = α·s_text + (1-α)·s_img`; coarse α ∈ {0,.1,…,1} then fine ±0.1 step .02; τ grid per α. Uses best text (exp_01/02) + best image (exp_03). | val F1 |
| `exp_04b` | **Weighted fusion — learned (D4)** | Logistic regression on candidate pairs from **train_fusion** groups: features `[s_text, s_img]`, label = same `label_group`. Fused score = `σ(w_t·s_text + w_i·s_img + b)`. Report effective `α = w_t/(w_t+w_i)`. τ on val. | val F1 vs 04a; winner becomes "Multimodal" |
| `exp_05` | Retrieval & post-processing | FAISS `IndexFlatIP` Top-K (K=50) vs brute force: latency/QPS + Recall@K. Hypothesis: transitive connected components on top of best fusion — keep only if val F1 improves. **Compute test metrics for the final config (no copying).** | val F1 |

### 1.1 Sub-splitting train (needed for honest learned fusion)
The fine-tuned text encoder has seen train pairs, so train similarities are over-optimistic. Split **train groups** deterministically (seed 42, by `label_group`):
- `train_fit` (~85% of train groups) → text fine-tuning (exp_02).
- `train_fusion` (~15% of train groups) → fit logistic-regression fusion weights (exp_04b).
Val and test are untouched. Persist the group lists to `data/processed/train_fusion_groups.csv` so it is reproducible.

### 1.2 Evaluation protocol (matches slide 10)
For each query in the split:
1. Retrieve Top-K (K=50) candidates **within the same split** by the model's score (FAISS for dense; sparse dot for TF-IDF). For fusion: union of text Top-K and image Top-K, then score each with the fused function.
2. Keep candidates with score ≥ τ. Always include the query itself.
3. P/R/F1 per query vs the set of postings with the same `label_group` in that split. Mean over queries.
Implement once in `src/evaluation/protocol.py` and reuse in local scripts **and** a copy in the Kaggle script (Kaggle has no `src`, see Phase A).

### 1.3 `results.json` record schema (extend the existing one)
```json
{
  "experiment_id": "exp_04b",
  "name": "Experiment 4b — Learned Weighted Fusion",
  "approach_type": "Multimodal Fusion",
  "model_name": "MiniLM-L12 (ft) + CLIP-B/32",
  "hyperparameters": {"threshold": 0.71, "alpha": 0.63, "w_text": 7.1, "w_image": 4.2, "bias": -6.3, "top_k": 50},
  "validation_metrics": {"mean_precision": 0, "mean_recall": 0, "mean_f1": 0, "num_queries": 0},
  "test_metrics":       {"mean_precision": 0, "mean_recall": 0, "mean_f1": 0, "num_queries": 0},
  "delta_val_f1": 0.0,
  "delta_reference": "exp_04a",
  "selected": true,
  "artifacts": {"text_emb": "test_text_embeddings.npy", "image_emb": "test_image_embeddings.npy"},
  "notes": "...",
  "timestamp": "ISO-8601"
}
```
Additionally write `experiments/sweeps.json` (α×τ grid F1 for heatmap, threshold curves per model), `experiments/latency.json` (exp_05), and `experiments/error_cases.json` (Phase C).

---

## Phase A — Fix & Run the Experiment Pipeline (Kaggle GPU) · highest priority

### A1. Make `notebooks/kaggle_pipeline.py` correct and truly self-contained
File: [kaggle_pipeline.py](file:///c:/Users/Rong/dev/productmatching/notebooks/kaggle_pipeline.py)
- [ ] Fix undefined `val_text_val_f1` (L247, L439) → use `best_text_val_f1`.
- [ ] Remove `from src.retrieval.matching import apply_connected_components`; inline a union-find implementation.
- [ ] **Do not re-split.** Load the committed CSVs from a Kaggle dataset input (see A2). Assert row counts: train/val/test must equal local CSV lengths, and `posting_id` order identical.
- [ ] Exp 0: same config as local (no `stop_words`).
- [ ] Implement the evaluation protocol in §1.2 (Top-K=50, FAISS on GPU/CPU).
- [ ] Implement `train_fit` / `train_fusion` sub-split (§1.1).
- [ ] Exp 2: group-aware sampler. Either `sentence_transformers.datasets.NoDuplicatesDataLoader` is **not** sufficient (it dedups texts, not groups) → write a custom batch builder that takes at most one pair per `label_group` per batch. Save model to `/kaggle/working/models/text_ft/`.
- [ ] Exp 3: add CLIP ViT-B/32 (`open_clip` or `transformers.CLIPModel.get_image_features`, L2-normalised). Keep EfficientNet. Select on val.
- [ ] Exp 4a + 4b as in §1. For 4b: build pairs from `train_fusion` with the same Top-K union candidate generation; use `sklearn.linear_model.LogisticRegression(class_weight="balanced")`; store `w_text, w_image, bias`.
- [ ] Exp 5: FAISS latency/QPS + Recall@50; connected-components hypothesis; **real test evaluation**.
- [ ] Wrap each experiment in try/except + checkpoint writes so a late failure doesn't lose earlier results. Set seeds (torch, numpy, random).
- [ ] Export to `/kaggle/working/`:
  - `results.json`, `sweeps.json`, `latency.json`, `error_cases.json` (A1 can compute error cases; see C3 for format)
  - Test embeddings for **every** candidate model, aligned to `test_demo_split.csv` row order, float16 to save space:
    `test_text_minilm_l6.npy`, `test_text_minilm_l12.npy`, `test_text_ft.npy`, `test_img_effnet.npy`, `test_img_clip.npy`
  - Same for val (`val_*.npy`) for the dashboard's threshold curves.
  - **Train-split embeddings** of the selected text and image models (`train_text_selected.npy`, `train_img_selected.npy`) — used as the offline candidate index fallback / sanity checks.
  - `models/text_ft/` (fine-tuned SentenceTransformer) if exp_02 is adopted.
  - `final_config.json`: `{text_model, image_model, fusion: {type, alpha, w_text, w_image, bias}, threshold, top_k}` — **single source of truth for the app**.

### A2. Kaggle automation (use `kaggle_token` from `.env`)
- [ ] Add `scripts/kaggle_run.py` that:
  1. Loads `.env` (use `python-dotenv`; add to requirements) and exports Kaggle credentials. Verify the token format: new-style tokens use env `KAGGLE_API_TOKEN` (kaggle ≥ 1.7); legacy use `KAGGLE_USERNAME` + `KAGGLE_KEY`. Handle both. **Never print the token.**
  2. Creates/versions a private Kaggle dataset `productmatching-splits` containing `data/processed/*.csv` (`kaggle datasets create/version`).
  3. Pushes `notebooks/kaggle_pipeline.py` as a script kernel with `kernel-metadata.json`: `enable_gpu: true`, `enable_internet: true` (to download HF models), `competition_sources: ["shopee-product-matching"]`, `dataset_sources: ["<user>/productmatching-splits"]`.
  4. Polls `kaggle kernels status`, then `kaggle kernels output` into `experiments/kaggle_output/`.
  5. Copies artifacts: `*.json` → `experiments/`, `*.npy` → `data/embeddings/`, `models/text_ft` → `models/text_ft`, `final_config.json` → `experiments/final_config.json`.
- Note: the user must have accepted the Shopee competition rules on Kaggle; surface a clear error if the competition source is forbidden.

### A3. Repo storage
- [ ] Track `data/embeddings/test_*.npy`, `data/embeddings/val_*.npy` and `models/text_ft/**` with **Git LFS** (update `.gitattributes`; adjust `.gitignore` so these are no longer ignored). Train embeddings may stay ignored if > 50 MB.
- [ ] Update local `scripts/run_experiments.py` exp 0 to use the shared protocol (`src/evaluation/protocol.py`) so local and Kaggle numbers match. Add `--benchmark-faiss` (the docs already claim it exists).

### Acceptance criteria — Phase A
- `experiments/results.json` contains real records for exp_00 … exp_05 (incl. 04a and 04b) with both val and test metrics; exactly one multimodal record has `"selected": true`.
- `experiments/final_config.json` exists and references embedding files that exist in `data/embeddings/`.
- Embedding row count == `len(test_demo_split.csv)` for every test file.
- Local exp_00 F1 equals Kaggle exp_00 F1 (±0.001).

---

## Phase B — Make the Dataset Matcher Honest (slide 13A)

Files: [1_Product_Matcher.py](file:///c:/Users/Rong/dev/productmatching/app/pages/1_Product_Matcher.py), [sidebar_controls.py](file:///c:/Users/Rong/dev/productmatching/app/components/sidebar_controls.py), [matching.py](file:///c:/Users/Rong/dev/productmatching/src/retrieval/matching.py), [multimodal.py](file:///c:/Users/Rong/dev/productmatching/src/models/multimodal.py)

- [ ] Add `src/app_config.py` (or a loader in `src/config.py`) that reads `experiments/final_config.json` + `results.json` and exposes per-model `{embedding_file, threshold, alpha/weights}`.
- [ ] **Remove the silent TF-IDF fallback** in `retrieve_candidates`. Models whose embeddings are missing are disabled in the sidebar with a caption ("run Phase A pipeline").
- [ ] TF-IDF: fit on **train split** with exp_00 config, transform test (cache with `st.cache_resource`).
- [ ] Model options: TF-IDF, Text (pretrained), Text (fine-tuned), Image (EfficientNet), Image (CLIP), Multimodal (selected fusion). Default = Multimodal.
- [ ] Slider defaults: τ (and α for grid fusion) loaded from the tuned values per model; show "tuned τ = x" marker. Comparison model uses **its own** tuned settings (not hardcoded α=0.5).
- [ ] Multimodal scoring: use the same function as exp_04 winner — `score_fusion` for 04a, or `sigmoid(w·s+b)` for 04b. Add `learned_fusion()` to `multimodal.py`. Candidate set = union of text Top-K and image Top-K (same as protocol).
- [ ] `categorize_and_filter_candidates`: add true matches that were **not retrieved** in Top-K to "Missed" (score shown as computed full-similarity or "not retrieved").
- [ ] Add a "Pick a multi-match example" helper (only show queries whose group size ≥ 3) — makes the demo/video clearer.
- [ ] Remove/adjust the claimed "data split selector" from docs, or implement val/test toggle (embeddings for val exist after Phase A — implement it).

### Acceptance criteria — Phase B
- Selecting each model actually uses that model's embeddings (verify by differing scores).
- With default (tuned) settings, mean F1 over 200 random test queries computed via the page's code path ≈ test F1 in `results.json` (±0.02).

---

## Phase C — Data-Driven Experiment Dashboard

File: [2_Experiment_Results.py](file:///c:/Users/Rong/dev/productmatching/app/pages/2_Experiment_Results.py)

- [ ] **Delete the fabricated `default_records` block** (L27–83). Empty state = `st.warning` with instructions.
- [ ] C1 Ablation table: each experiment, val F1, test P/R/F1, Δ vs reference (`delta_reference`), selected flag. Highlight best per modality.
- [ ] C2 Charts: test F1 progression; val vs test F1 (generalisation gap); precision–recall vs τ curves per model (from `sweeps.json`); α×τ F1 heatmap (04a); bar of learned weights (04b) with effective α.
- [ ] C3 **Real error analysis** from `error_cases.json`. Format per case: `{category, query_posting_id, candidate_posting_id, s_text, s_img, s_fused, same_group}`. Categories:
  - `fusion_fixed_text_fp` (text ≥ τ_text, fused < τ, different group) — e.g. phone vs case
  - `fusion_fixed_image_fp`
  - `fusion_recovered_fn` (missed by best unimodal, found by fusion)
  - `fusion_still_wrong` (top residual FPs/FNs)
  Render with thumbnails via existing `render_product_card`. Keep the existing narrative cards only as captions that cite real examples.
- [ ] C4 Latency table from `latency.json` (brute force vs FAISS, ms/query, QPS, Recall@50).
- [ ] C5 "Generalisation" callout: test groups are disjoint from train (show counts and assert 0 overlap).

### Acceptance criteria — Phase C
- No hardcoded metric numbers anywhere in `app/`.
- Every chart renders from files produced by Phase A.

---

## Phase D — Visual Shopping Assistant with SerpAPI (slides 13B, 14; Obj 5)

File: replace [3_Live_Matcher.py](file:///c:/Users/Rong/dev/productmatching/app/pages/3_Live_Matcher.py) (rename to `3_Shopping_Assistant.py`; update README/app nav).

### D1. Config & secrets
- [ ] `.env.example`: add `serpapi_key=your_serpapi_key_here`. Load via `python-dotenv` locally and `st.secrets["serpapi_key"]` on Streamlit Cloud (support both, env first). Add `google-search-results` (or plain `requests`) and `python-dotenv` to `requirements.txt`.
- [ ] If key missing → page shows setup instructions and runs only cached demo queries (D5).

### D2. Inputs
- Image **upload** (png/jpg/webp, e.g. a social-media screenshot) **or** image URL, plus **optional text hint** (e.g. "Sony WH-1000XM5"), country (`gl`, default `sg`), currency display.

### D3. Candidate retrieval (`src/retrieval/web_search.py`)
⚠️ SerpAPI **Google Lens** needs a **public image URL** (`engine=google_lens&url=...`); it does not accept file uploads. Implement:
1. **Image URL provided** → `google_lens` → use `visual_matches` (fields: `title`, `link`, `source`, `thumbnail`, `price.extracted_value`, `price.currency`). Optionally also call with `type=products` if available.
2. **Uploaded image + text hint** → `google_shopping` (`q=hint`, `gl`, `hl=en`) → `shopping_results` (fields: `title`, `source`, `extracted_price`, `thumbnail`, `product_link`/`link`).
3. **Uploaded image without hint** → ask the user for a hint (simplest, honest). Optional enhancement: upload to a temporary public host to enable Lens — only if trivial; document the privacy implication.
4. Union + dedupe candidates by link; cap at 40. Download candidate thumbnails concurrently (timeout 5 s, skip failures).
- Return a list of `WebCandidate(title, merchant, price, currency, link, thumbnail_url, image: PIL|None)`.

### D4. Matching (`src/pipeline/live_matcher.py`)
- Load encoders per `final_config.json` (CPU): fine-tuned or pretrained SentenceTransformer from `models/text_ft` / HF id; EfficientNet or CLIP image tower. Cache with `st.cache_resource`.
- Query text = user hint (if absent, use the top Lens result title as a pseudo-hint and say so in the UI).
- Compute `s_text(query_text, cand.title)`, `s_img(query_image, cand.image)`; fuse with the **selected fusion** (04a α or 04b weights) and threshold with the tuned τ. If a candidate has no image, fall back to text-only score with the text model's own τ, and flag it.
- Output three sections (mirror the Dataset Matcher UX):
  - 🟢 **Likely exact matches** — sorted by price ascending; **highlight the cheapest** ("Best price: $X at Merchant").
  - ⚪ **Similar but rejected** (accessories, variants) with scores.
  - Per-card: thumbnail, title, merchant, price, s_text / s_img / fused, link.
- Show a small "why" breakdown (text vs image contribution bar).

### D5. Reproducible demo cache
- [ ] Cache raw SerpAPI JSON per query hash in `data/live_cache/` (gitignored) and commit 3 curated demo cases to `data/live_demo/` (JSON + query image) so the video and Streamlit Cloud work without spending API credits. Toggle "Use cached demo".
- [ ] Respect SerpAPI quota: debounce on button press only (no search on every rerun), `st.cache_data(ttl=3600)` on search calls.

### Acceptance criteria — Phase D
- With a key: uploading a product photo + hint returns real merchant listings with prices; accessories are rejected; cheapest exact match is highlighted.
- Without a key: cached demos fully work.
- No API key is ever logged or rendered.

---

## Phase E — Docs, Report & Deliverables

- [ ] Update [README.md](file:///c:/Users/Rong/dev/productmatching/README.md): real results table (generated from `results.json`), the experiment list from §1, Kaggle run instructions (`scripts/kaggle_run.py`), SerpAPI setup, Git LFS paths for embeddings/models. Remove claims of unimplemented features.
- [ ] Mark `docs/2nd_implementation.md` as superseded or correct its claims (`--benchmark-faiss`, group-aware batching, latency table).
- [ ] Report outline (`docs/final_report_outline.md`): problem; dataset & leakage-free split; methods (TF-IDF, MiniLM, MNRL fine-tune, EfficientNet vs CLIP, weighted fusion grid vs learned); protocol; results & ablation; error analysis; generalisation to unseen products (slide 12); live POC; limitations (Lens needs public URL, price parsing, domain shift Shopee→Google Shopping); module requirement mapping (supervised, ML/DL, hybrid).
- [ ] Video script (10–15 min): problem → data → architecture → results dashboard → Dataset Matcher live (threshold slider, phone-vs-case example) → Shopping Assistant (cached demo) → conclusion.

---

## 2. File Change Map

| File | Action | Phase |
|---|---|---|
| `notebooks/kaggle_pipeline.py` | Major rewrite (fixes + exp_03 CLIP + exp_04b + exports) | A |
| `scripts/kaggle_run.py` | **New** — push/poll/download Kaggle kernel | A |
| `notebooks/kernel-metadata.json` | **New** | A |
| `src/evaluation/protocol.py` | **New** — Top-K + threshold evaluation | A |
| `scripts/run_experiments.py` | Use protocol; add `--benchmark-faiss` | A |
| `.gitattributes`, `.gitignore` | LFS for embeddings + `models/` | A |
| `src/models/multimodal.py` | Add `learned_fusion` | B |
| `src/config.py` / `src/app_config.py` | Load `final_config.json` | B |
| `app/pages/1_Product_Matcher.py`, `app/components/sidebar_controls.py` | Honest model selection, tuned defaults | B |
| `src/retrieval/matching.py` | Include unretrieved misses | B |
| `app/pages/2_Experiment_Results.py` | Remove fakes; data-driven charts & errors | C |
| `src/retrieval/web_search.py` | **New** — SerpAPI client | D |
| `src/pipeline/live_matcher.py` | **New** — encode/fuse/rank live candidates | D |
| `app/pages/3_Shopping_Assistant.py` | Replace mockup | D |
| `data/live_demo/` | **New** — curated cached demos | D |
| `.env.example`, `requirements.txt` | `serpapi_key`, `python-dotenv`, `google-search-results`, `open_clip_torch`/`transformers` | A, D |
| `README.md`, `docs/*` | Truthful docs, report outline | E |

## 3. Gotchas for the implementer
- **Never print or commit secrets** from `.env` (`kaggle_token`, `serpapi_key`, `huggingface_token`, `github_token`).
- Embedding row order **must** match `test_demo_split.csv`; assert `posting_id` alignment by also exporting `test_posting_ids.json` from Kaggle.
- Streamlit Cloud memory is ~1 GB: load only the selected encoders for Phase D, lazily; prefer CPU torch wheels; keep float16 embeddings.
- CLIP and EfficientNet use different normalisation — use each model's own preprocessing.
- The user has `streamlit run app/app.py` running; Streamlit hot-reloads on save.
- Windows dev machine (PowerShell); console encoding is GBK — set `PYTHONIOENCODING=utf-8` when printing non-ASCII titles.
- Shopee titles contain Indonesian; keep the multilingual model in exp_01.

## 4. Definition of Done
1. All Phase A–D acceptance criteria pass.
2. `streamlit run app/app.py` → all 3 pages work from a fresh clone + `git lfs pull` (no Kaggle run needed).
3. Results in the app, README and report are identical and come from `experiments/results.json`.
