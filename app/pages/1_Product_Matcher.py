import sys
from pathlib import Path

# Ensure repository root and app directory are at head of sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    TEST_DEMO_SPLIT_CSV,
    EMBEDDINGS_DIR,
)
from src.data.loader import load_data, get_ground_truth_map
from src.retrieval.matching import categorize_and_filter_candidates
from src.models.text_encoder import TfidfEncoder
from src.models.multimodal import MultimodalFusion

try:
    from app.components.sidebar_controls import render_sidebar_controls
    from app.components.product_card import render_product_card
except (ModuleNotFoundError, ImportError):
    from components.sidebar_controls import render_sidebar_controls
    from components.product_card import render_product_card

st.set_page_config(page_title="Product Matcher | Demo", page_icon="🔍", layout="wide")

st.title("🔍 Dataset Product Matcher")
st.caption(
    "Interactive evaluation on held-out **test/demo split** (unseen product groups with ground truth)"
)

# Load test/demo data
@st.cache_data
def get_demo_dataset():
    if not TEST_DEMO_SPLIT_CSV.exists():
        return None, None
    df = pd.read_csv(TEST_DEMO_SPLIT_CSV)
    gt_map = get_ground_truth_map(df)
    return df, gt_map


demo_df, gt_map = get_demo_dataset()

if demo_df is None:
    st.warning(
        "⚠️ Test/demo split not found. Please run the extraction script: "
        "`python scripts/extract_data.py` to extract CSVs and generate test thumbnails."
    )
    st.stop()

# Sidebar experiment controls
controls = render_sidebar_controls()
model_choice = controls["model_choice"]
threshold = controls["threshold"]
alpha = controls["alpha"]
top_k = controls["top_k"]
enable_comparison = controls["enable_comparison"]
comp_model = controls["comp_model"]

# Pre-load or compute demo embeddings cache
@st.cache_resource
def load_or_compute_demo_embeddings(df_len: int):
    """
    Loads saved test embeddings if available, or computes on-the-fly for demo set.
    """
    text_emb_file = EMBEDDINGS_DIR / "test_text_embeddings.npy"
    image_emb_file = EMBEDDINGS_DIR / "test_image_embeddings.npy"

    text_emb = None
    image_emb = None

    if text_emb_file.exists():
        try:
            text_emb = np.load(text_emb_file)
        except Exception:
            pass

    if image_emb_file.exists():
        try:
            image_emb = np.load(image_emb_file)
        except Exception:
            pass

    return text_emb, image_emb


@st.cache_resource
def get_fitted_tfidf(data_size: int):
    """Fits and caches TF-IDF vectorizer on demo texts (instant cache lookup by size)."""
    encoder = TfidfEncoder(max_features=15000, ngram_range=(1, 2))
    matrix = encoder.fit_transform(demo_df["title"].tolist())
    return encoder, matrix


# Fit TF-IDF on demo dataset once and keep cached
tfidf_encoder, tfidf_matrix = get_fitted_tfidf(len(demo_df))

# Reference product selection
st.subheader("1️⃣ Select Reference Product")

search_term = st.text_input(
    "Search product listings by keyword:",
    value="",
    placeholder="e.g., kemeja, casing iphone, sepatu, cream, masker...",
)

filtered_df = demo_df
if search_term.strip():
    filtered_df = demo_df[
        demo_df["title"].str.contains(search_term.strip(), case=False, na=False)
    ]

if len(filtered_df) == 0:
    st.info("No products match your search keyword. Showing all listings.")
    filtered_df = demo_df

# Group products with multiple matches to make demonstration easy
sample_options = filtered_df.head(100)
option_labels = {
    idx: f"[{row['posting_id']}] {row['title'][:80]}"
    for idx, row in sample_options[["posting_id", "title"]].iterrows()
}

selected_idx = st.selectbox(
    "Choose a reference listing:",
    options=sample_options.index.tolist(),
    format_func=lambda idx: option_labels.get(idx, str(idx)),
    index=0,
)

ref_row = demo_df.loc[selected_idx]
ref_posting_id = ref_row["posting_id"]
ref_label_group = ref_row["label_group"]
ground_truth_postings = set(gt_map.get(ref_posting_id, [ref_posting_id]))

# Display reference product
ref_col1, ref_col2 = st.columns([1, 2])
with ref_col1:
    render_product_card(
        title=ref_row["title"],
        image_name=ref_row["image"],
        label_group=ref_label_group,
        posting_id=ref_posting_id,
        is_reference=True,
    )
with ref_col2:
    st.markdown(f"**Reference Title:** {ref_row['title']}")
    st.markdown(f"**Posting ID:** `{ref_posting_id}`")
    st.markdown(f"**Ground-Truth Label Group:** `{ref_label_group}`")
    st.markdown(
        f"**Total Ground-Truth Equivalents in Demo Split:** **{len(ground_truth_postings)}** listing(s)"
    )
    st.info(
        "💡 **Academic Insight:** Ground-truth matches share the exact same `label_group`."
        " Drag the threshold slider in the sidebar to observe the precision/recall trade-off live!"
    )

st.divider()

# Retrieval Function
def retrieve_candidates(model_name: str, query_idx: int, k: int, current_alpha: float):
    """Retrieves top candidates with similarity scores for a given model."""
    if "TF-IDF" in model_name:
        query_vec = tfidf_matrix[query_idx]
        sims = tfidf_encoder.compute_similarity(query_vec, tfidf_matrix)
    else:
        # Check if saved precomputed embeddings exist
        text_emb, image_emb = load_or_compute_demo_embeddings(len(demo_df))
        if text_emb is not None and "Text Embeddings" in model_name:
            query_emb = text_emb[query_idx : query_idx + 1]
            sims = np.dot(text_emb, query_emb.T).squeeze()
        elif image_emb is not None and "Image Embeddings" in model_name:
            query_emb = image_emb[query_idx : query_idx + 1]
            sims = np.dot(image_emb, query_emb.T).squeeze()
        elif text_emb is not None and image_emb is not None and "Multimodal" in model_name:
            fusion = MultimodalFusion(default_alpha=current_alpha)
            fused = fusion.fuse_embeddings(text_emb, image_emb, alpha=current_alpha)
            query_fused = fused[query_idx : query_idx + 1]
            sims = np.dot(fused, query_fused.T).squeeze()
        else:
            # Fallback when heavy DL embeddings are not yet generated locally:
            # TF-IDF sparse similarity
            query_vec = tfidf_matrix[query_idx]
            sims = tfidf_encoder.compute_similarity(query_vec, tfidf_matrix)

    # Get top-k indices (excluding query self if desired, or including)
    top_indices = np.argsort(sims)[::-1][:k]
    candidate_df = demo_df.iloc[top_indices].copy()
    candidate_df["score"] = sims[top_indices]
    return candidate_df


def display_matching_view(results_model_name: str, current_alpha: float):
    candidate_df = retrieve_candidates(results_model_name, selected_idx, top_k, current_alpha)
    
    res = categorize_and_filter_candidates(
        query_posting_id=ref_posting_id,
        query_label_group=ref_label_group,
        retrieved_df=candidate_df,
        ground_truth_postings=ground_truth_postings,
        threshold=threshold,
    )

    # Live Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Precision", f"{res.metrics['precision']:.1%}")
    with m2:
        st.metric("Recall", f"{res.metrics['recall']:.1%}")
    with m3:
        st.metric("F1 Score", f"{res.metrics['f1']:.1%}")
    with m4:
        st.metric(
            "Matches / Ground Truth",
            f"{res.metrics['tp_count']} / {len(ground_truth_postings)}",
        )

    st.markdown("---")

    # Section 1: MATCHED PRODUCTS (Score >= threshold)
    st.subheader(f"🟢 Matched Products (Score ≥ {threshold:.2f})")
    if res.matched_products:
        cols = st.columns(min(len(res.matched_products), 4))
        for i, match in enumerate(res.matched_products):
            with cols[i % 4]:
                render_product_card(
                    title=match.title,
                    image_name=match.image,
                    label_group=match.label_group,
                    score=match.score,
                    status=match.status,
                    posting_id=match.posting_id,
                )
    else:
        st.caption("No products retrieved above the current similarity threshold.")

    # Section 2: SIMILAR BUT REJECTED (Hard negatives immediately below threshold)
    st.subheader(f"⚪ Similar but Rejected (Hard Negatives < {threshold:.2f})")
    if res.similar_rejected:
        st.caption(
            "Highest-scoring candidates safely rejected because their similarity is below the threshold."
        )
        cols = st.columns(min(len(res.similar_rejected), 4))
        for i, rej in enumerate(res.similar_rejected):
            with cols[i % 4]:
                render_product_card(
                    title=rej.title,
                    image_name=rej.image,
                    label_group=rej.label_group,
                    score=rej.score,
                    status=rej.status,
                    posting_id=rej.posting_id,
                )
    else:
        st.caption("No rejected candidates in the top retrieved window.")

    # Section 3: MISSED TRUE MATCHES
    st.subheader("🟠 Missed True Matches")
    if res.missed_matches:
        st.caption(
            "⚠️ True equivalent products (same label_group) that were missed because similarity fell below the threshold."
        )
        cols = st.columns(min(len(res.missed_matches), 4))
        for i, missed in enumerate(res.missed_matches):
            with cols[i % 4]:
                render_product_card(
                    title=missed.title,
                    image_name=missed.image,
                    label_group=missed.label_group,
                    score=missed.score,
                    status=missed.status,
                    posting_id=missed.posting_id,
                )
    else:
        st.caption("✅ All retrieved ground-truth matches cleared the current threshold!")


if not enable_comparison:
    st.subheader(f"2️⃣ Results for **{model_choice}** (τ = {threshold:.2f})")
    display_matching_view(model_choice, alpha)
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"Model A: **{model_choice}**")
        display_matching_view(model_choice, alpha)
    with col_b:
        st.subheader(f"Model B: **{comp_model}**")
        display_matching_view(comp_model, 0.5)
