# Final Project Report Outline & Academic Alignment Guide

**Module**: Pattern Recognition Systems (NUS ISS Master of Technology in AI Systems)  
**Project**: Multimodal Product Matching for Cross-Merchant Comparison  
**Group**: Group 43 (Lim Choon Rong)  

---

## 1. Executive Summary & Problem Formulation
- **Industry Motivation**: E-commerce catalog fragmentation across sellers and merchants; different product titles, informal abbreviations, languages, and seller photography for identical items.
- **Task Definition**: Pattern-recognition task formulated as multimodal similarity estimation: given two noisy observations $(\text{Title}_A, \text{Image}_A)$ and $(\text{Title}_B, \text{Image}_B)$, predict whether they represent the same underlying physical product identity.
- **Key Contribution**: An end-to-end multimodal matching architecture with strict zero-leakage held-out evaluation, systematic progression across 6 experimental stages, and a consumer-facing price comparison POC.

---

## 2. Dataset & Zero Data Leakage Partitioning
- **Source**: Kaggle Shopee - Price Match Guarantee dataset (~34,250 listings, 11,014 `label_group` identities).
- **Zero-Leakage Splitting Strategy**:
  - Partitioning performed **strictly by `label_group`**, ensuring no single product identity exists in more than one partition.
  - **Train split** (~70%, 23,971 rows, 7,710 groups): Model training, vocabulary fitting, and metric learning.
  - **Validation split** (~15%, 5,141 rows, 1,652 groups): Model selection, threshold calibration ($\tau$), and fusion weight optimization.
  - **Test split** (~15%, 5,138 rows, 1,652 groups): Completely frozen held-out evaluation on unseen products.
  - Label group intersection: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Train Sub-Split for Unbiased Fusion (`train_fit` vs. `train_fusion`)**:
  - `train_fit` (85% of train groups): Supervised text metric learning (`MultipleNegativesRankingLoss`).
  - `train_fusion` (15% of train groups): Unbiased fitting of logistic regression fusion weights, preventing over-optimistic text bias.

---

## 3. Methodology & System Architecture
- **Modality 1: Text Feature Representation**:
  - *Classical ML*: TF-IDF word unigrams and bigrams, sublinear TF scaling, sparse cosine similarity.
  - *Pretrained Dense Transformers*: `all-MiniLM-L6-v2` (384-d) vs. `paraphrase-multilingual-MiniLM-L12-v2` (384-d) to bridge Indonesian/English multilingual vocabulary gaps.
  - *Domain Adaptation*: Metric learning using `MultipleNegativesRankingLoss` with group-aware batch sampling.
- **Modality 2: Visual Feature Representation**:
  - *Pretrained CNN*: `timm` EfficientNet-B0 (1280-d feature trunk, L2-normalized).
  - *Zero-Shot Vision Transformer*: `openai/clip-vit-base-patch32` (512-d visual projection).
- **Modality 3: Multimodal Fusion Strategies**:
  - *Experiment 4a (Grid Calibration)*: Convex score fusion: $\text{Sim}_{\text{fused}} = \alpha \cdot \text{Sim}_{\text{text}} + (1 - \alpha) \cdot \text{Sim}_{\text{img}}$.
  - *Experiment 4b (Learned Multimodal Fusion)*: Logistic regression on hard candidate pairs from `train_fusion`:
    $$P(\text{Match}) = \sigma(w_{\text{text}} \cdot \text{Sim}_{\text{text}} + w_{\text{img}} \cdot \text{Sim}_{\text{img}} + b)$$
- **Candidate Retrieval & Scaling**:
  - FAISS `IndexFlatIP` inner-product search for exact nearest neighbor retrieval at sub-millisecond latency.
  - Evaluation Protocol: Top-K (K=50) candidate generation followed by calibrated threshold filtering.

---

## 4. Experimental Progression & Ablation Results
- **Full Benchmark Table (Exps 0 to 5)**:
  - Exp 0: TF-IDF Classical Baseline
  - Exp 1: Pretrained Text Transformer Selection
  - Exp 2: Domain-Adapted Text Encoder
  - Exp 3: Visual Feature Extraction (EffNet vs. CLIP)
  - Exp 4a & 4b: Multimodal Fusion (Grid Sweep vs. Supervised Learned Weights)
  - Exp 5: Final Optimized Retrieval & Post-Processing
- **Key Metrics Tracked**: Mean Precision, Mean Recall, Mean F1 score across queries.
- **Ablation Discussion**:
  - Step-by-step performance gains from classical to deep learning, and from unimodal to multimodal.
  - Generalization verification: Comparison of Validation F1 vs. Test F1 across unseen product identities.

---

## 5. Qualitative Error Analysis & Sense-Making
- **Case 1: Resolving Text False Positives via Vision**:
  - Smartphone chassis vs. smartphone cases/accessories sharing near-identical titles.
- **Case 2: Resolving Visual False Positives via Text**:
  - Different models/brands with identical product color and shape.
- **Case 3: Multimodal Recovery of False Negatives**:
  - Ambiguous colloquial titles or poor lighting where individual modalities fail, but fused confidence clears the threshold.
- **Case 4: Remaining Hard Edge Cases**:
  - Multi-packs vs. single items, cross-generation identical enclosures (e.g. AirPods Gen 1 vs. Gen 2).

---

## 6. Runnable Prototype & Consumer Application
- **Mode 1: Dataset Product Matcher**:
  - 3-Section inspector: Matched Products, Similar but Rejected (hard negatives), and Missed Matches.
  - Live threshold slider demonstrating the precision-recall trade-off.
  - Side-by-side comparative evaluation of distinct representation models.
- **Mode 2: Visual Shopping Assistant**:
  - Real-world POC connecting user product image/screenshot and title hint to live merchant search via SerpAPI (Google Shopping / Google Lens).
  - Multimodal re-ranking and threshold filtering to isolate exact matches.
  - Automated merchant price comparison highlighting the cheapest available offer.

---

## 7. Module Requirement Mapping (NUS ISS Assessment Matrix)

| Requirement Aspect | Project Implementation & Evidence |
|---|---|
| **1. Supervised / Unsupervised Learning** | Unsupervised FAISS cosine candidate retrieval; Supervised metric learning via Shopee `label_group` ground truth; Supervised logistic regression learned fusion. |
| **2. Machine Learning / Deep Learning** | Scikit-learn TF-IDF bigrams (Classical ML); PyTorch SentenceTransformers, timm EfficientNet-B0, and CLIP ViT-B/32 (Deep Learning). |
| **3. Hybrid / Ensemble Approach** | Multimodal late fusion combining textual semantic embeddings and visual CNN/ViT representations with learned cross-modal weights. |
| **4. Intelligent Sensing / Sense-Making** | Computer vision image understanding resolving ambiguous e-commerce text and false positives in real-world merchant listings. |

---

## 8. Limitations & Future Work
- External visual search dependency on public URLs for Google Lens.
- Extraction of complex merchant price structures (discounts, vouchers, shipping fees).
- Exploration of end-to-end multimodal metric learning (e.g., dual-encoder CLIP fine-tuning).
