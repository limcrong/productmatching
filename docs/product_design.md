# Product: Multimodal Product Matching

## Context

This is an academic Pattern Recognition project based on the Kaggle **Shopee Product Matching** dataset.

The objective is to identify e-commerce listings that represent the same underlying product despite differences in titles and images.

The dataset contains product titles, images, and ground-truth `label_group` values identifying equivalent listings.

The project should demonstrate both:
1. A working product matching prototype.
2. The measurable improvement from baseline approaches to optimized multimodal approaches.

## Core Research Goal

Evaluate whether combining textual and visual information improves product matching compared with simpler approaches.

At minimum compare:

- Classical text baseline (e.g. TF-IDF)
- Text embedding model
- Image embedding model
- Multimodal text + image model
- Final optimized approach

Use appropriate retrieval/matching metrics such as **Precision, Recall and F1**.

All optimizations should be evaluated against previous approaches so their contribution can be demonstrated.

## Core Pipeline

```text
Shopee Listings
      |
      +---- Title ----> Text Encoder ----> Text Embedding
      |
      +---- Image ----> Image Encoder ---> Image Embedding
                                     
                       |
                 Similarity / Fusion
                       |
                Candidate Retrieval
                       |
                 Product Matches
```

Use **FAISS** or an equivalent nearest-neighbour approach for efficient embedding retrieval.

Models and techniques may evolve during experimentation. Keep components modular so encoders, fusion strategies, thresholds and retrieval methods can be replaced or compared easily.

## High-Level Architecture

```text
                        SHOPEE DATASET
                              |
                    Data Preprocessing
                              |
              +---------------+---------------+
              |                               |
           Product Title                  Product Image
              |                               |
              v                               v
         Text Pipeline                  Image Pipeline
              |                               |
         Text Embedding                 Image Embedding
              |                               |
              +---------------+---------------+
                              |
                       Candidate Retrieval
                              |
                            FAISS
                              |
                              v
                     Similarity Scoring
                              |
                  +-----------+-----------+
                  |                       |
             Unimodal Mode          Multimodal Mode
                  |                       |
                  +-----------+-----------+
                              |
                       Match Threshold
                              |
                              v
                    Predicted Product Group
                              |
              +---------------+---------------+
              |                               |
        Academic Evaluation             Streamlit Demo
```
## Experimental Design

The system should be developed modularly.

Each major improvement should be measurable against the previous approach.

The project should demonstrate **what each optimization contributes**.

The experiment progression can approximately follow:

```text
Experiment 0
TF-IDF Baseline

        ↓

Experiment 1
Transformer Text Embeddings

        ↓

Experiment 2
Improved / Fine-Tuned Text Embeddings

        ↓

Experiment 3
Image Embeddings

        ↓

Experiment 4
Text + Image Multimodal Fusion

        ↓

Experiment 5
Retrieval / Threshold Optimizations

        ↓

Final Optimized Model
```

Every experiment should produce reproducible evaluation metrics.

---

## Experiment 0 — Classical Baseline

Implement a simple baseline using product titles.

## Method

```text
Product Title
      |
    TF-IDF
      |
TF-IDF Vector
      |
Cosine Similarity
      |
Similarity Threshold
      |
Predicted Matches
```

Libraries:

- scikit-learn
- Pandas
- NumPy

Purpose:

- Establish a simple classical machine-learning baseline.
- Determine how much improvement subsequent deep-learning approaches provide.
- Provide an interpretable benchmark.

---

## Technology Guidelines

Keep the technical stack simple and appropriate for an academic prototype.

Preferred stack:

- Python
- PyTorch
- Hugging Face / Sentence Transformers
- timm where appropriate
- scikit-learn
- FAISS
- Pandas / NumPy
- Streamlit
- Jupyter/Kaggle for experimentation and training

Avoid unnecessary production infrastructure unless required later.

## Streamlit Application

Streamlit should serve as both:

- A product matching demonstration
- An academic experiment interface

The application should have two product-search modes.

### Mode 1 — Dataset Product Matcher (Core)

This is part of the required prototype.

Allow users to enter/search a product name and select a reference listing from the **held-out test/validation data**.

Run the selected matching model against the dataset and display:

- Reference product
- Predicted matching products
- Similar but non-matching products
- Product images and titles
- Similarity/confidence scores
- Ground-truth match status

Because ground-truth `label_group` is available, clearly indicate:

```text
✓ Correct Match
✗ False Match
✓ Correctly Rejected Similar Product
✗ Missed Match
```

The purpose is to visually demonstrate both what the model retrieves and **how accurate those predictions are against known ground truth**.

Where appropriate, show per-query statistics such as precision and recall alongside the retrieved products.

The UI should make difficult examples especially easy to inspect, such as similar product generations, variants and accessories.

### Mode 2 — Live Web Product Matcher (Optional Extension)

**This mode is strictly optional and must only be developed after the core prototype, experiments and evaluation are complete.**

Allow a user to enter a real-world product query such as:

```text
AirPods Pro 3
```

The system may retrieve current product listings from web search or other external sources, extract available product information such as title, image, merchant and price, and run those listings through the trained matching model.

Example flow:

```text
User Query
    ↓
Web Product Search
    ↓
Candidate Listings
    ↓
Text + Image Processing
    ↓
Trained Matching Model
    ↓
Equivalent Products
    ↓
Cross-Merchant Comparison
```

The purpose of this mode is to demonstrate how the academic model could support a real consumer-facing cross-merchant comparison application.

Since web listings do not have Shopee `label_group` ground truth, do **not** present their predictions as measured accuracy. Display model similarity/confidence scores and predicted matches instead.

This extension must not delay or complicate completion of the academic prototype.

## Academic Experiment Controls

The UI should expose useful settings for demonstrating model behaviour, such as:

- Baseline vs optimized models
- Text-only / image-only / multimodal
- Similarity threshold
- Text/image fusion weighting
- Retrieval Top-K

Do not expose every internal parameter.

Provide a comparison view where the same reference product can be evaluated using different approaches so differences in retrieved results are visually obvious.

Also provide an experiment-results view showing saved validation metrics for the major model configurations.

## Academic Guidelines

Treat reproducibility and evaluation as first-class requirements.

Experiments should record:

- Model/configuration
- Relevant hyperparameters
- Precision
- Recall
- F1
- Other useful retrieval metrics where appropriate

Maintain a clear progression:

```text
Baseline
   ↓
Improvement
   ↓
Measure Result
   ↓
Further Optimization
   ↓
Measure Result
   ↓
Final Model
```

Include qualitative error analysis for difficult cases such as:

- Accessories vs actual products
- Different product generations
- Product variants
- Visually similar products
- Similar titles representing different products

The final project should explain not only **which model performs best**, but **which changes improved performance and why**.

## Implementation Principles

- Start with a simple working baseline before optimization.
- Build the evaluation pipeline early.
- Keep model components modular.
- Precompute embeddings where practical.
- Keep held-out evaluation data separate from training data.
- Separate formal validation metrics from individual Streamlit examples.
- Prefer measurable improvements over unnecessary architectural complexity.
- Do not attempt to reproduce every technique from Kaggle winning solutions.
- Keep the code structured so additional models and experiments can be added easily.
- **Complete the dataset-based prototype before implementing live web retrieval.**
- Treat live web product matching as an optional extension, not a core dependency.

## Definition of Success

The core project should allow us to demonstrate:

1. How a simple baseline performs.
2. Where the baseline fails.
3. How text embeddings improve matching.
4. What visual information contributes.
5. Whether multimodal matching improves results.
6. What additional optimizations contribute.
7. Quantitative evidence through validation metrics.
8. Interactive testing against held-out Shopee products.
9. Correct matches, incorrect matches and difficult non-matches against known ground truth.
10. A clear research story from **baseline → experimentation → optimization → evaluated multimodal product matching system**.

Only after these are complete should the project optionally demonstrate:

> **Real product query → live web listings → trained product matcher → predicted cross-merchant matches.**