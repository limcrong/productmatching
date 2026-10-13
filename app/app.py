import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.config import (
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_DEMO_SPLIT_CSV,
    EXPERIMENTS_RESULTS_JSON,
)

st.set_page_config(
    page_title="Multimodal Product Matching",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛍️ Multimodal Product Matching System")
st.caption(
    "Pattern Recognition Systems Practice Module | NUS ISS Master of Technology in AI Systems"
)

st.markdown(
    """
    ### 🎯 Project Overview
    In e-commerce marketplaces like Shopee, duplicate product listings frequently occur with variations in
    product titles, languages, and photography. This system leverages **multimodal deep learning**
    combining text semantics and visual features to identify and group equivalent product listings.
    """
)

# Top metrics cards
col1, col2, col3, col4 = st.columns(4)

total_listings = 34250  # Shopee dataset baseline
train_count = "~24,000"
val_count = "~5,100"
test_count = "~5,100"

if TEST_DEMO_SPLIT_CSV.exists():
    try:
        test_df = pd.read_csv(TEST_DEMO_SPLIT_CSV)
        test_count = f"{len(test_df):,}"
    except Exception:
        pass

with col1:
    st.metric("Total Shopee Listings", f"{total_listings:,}")
with col2:
    st.metric("Train Split (Model Fitting)", train_count)
with col3:
    st.metric("Val Split (Threshold Tuning)", val_count)
with col4:
    st.metric("Test/Demo Split (Unseen Groups)", test_count)

st.divider()

# System Architecture & Flow
st.subheader("📐 System Architecture")

st.markdown(
    """
```text
                  SHOPEE PRODUCT LISTINGS
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
      Product Title                     Product Image
            │                                 │
     [Text Pipeline]                   [Image Pipeline]
   TF-IDF / MiniLM-L6               timm (EfficientNet-B0)
            │                                 │
     Text Embedding                    Image Embedding
            │                                 │
            └────────────────┬────────────────┘
                             │
                      [Multimodal Fusion]
            Early Concat  or  Late Weighted Score (α)
                             │
                     [FAISS Vector Index]
                   Cosine Similarity Search
                             │
                  [Threshold Filtering (τ)]
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
   MATCHED PRODUCTS     SIMILAR REJECTED    MISSED MATCHES
   (Score ≥ τ)          (Hard Negatives)    (Score < τ)
```
    """
)

st.divider()

# Navigation helper cards
st.subheader("🧭 Explore the System")
n1, n2, n3 = st.columns(3)

with n1:
    with st.container(border=True):
        st.markdown("### 🔍 1. Product Matcher")
        st.write(
            "Interactive demonstration against held-out **unseen product groups**."
            " Test different models and observe the precision/recall trade-off live with the threshold slider."
        )
        st.page_link("pages/1_Product_Matcher.py", label="Open Product Matcher →", icon="🔍")

with n2:
    with st.container(border=True):
        st.markdown("### 📊 2. Experiment Dashboard")
        st.write(
            "Academic evaluation comparing TF-IDF baseline, Text DL, Image DL, and Multimodal Fusion"
            " on held-out test data."
        )
        st.page_link("pages/2_Experiment_Results.py", label="View Experiment Results →", icon="📊")

with n3:
    with st.container(border=True):
        st.markdown("### 🌐 3. Image-Assisted Shopping Matcher")
        st.write(
            "Consumer-facing POC demonstrating how Google Shopping candidate retrieval"
            " combined with our trained multimodal model filters accessories and enables cross-merchant price comparison."
        )
        st.page_link("pages/3_Live_Matcher.py", label="Open Shopping Matcher POC →", icon="🌐")
