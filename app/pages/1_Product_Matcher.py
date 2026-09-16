import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    VAL_SPLIT_CSV,
    TEST_DEMO_SPLIT_CSV,
    EMBEDDINGS_DIR,
)
from src.data.loader import (
    load_data,
    get_ground_truth_map,
    ensure_thumbnails_extracted,
)
from src.retrieval.matching import (
    categorize_and_filter_candidates,
    apply_connected_components,
)
from src.models.text_encoder import TfidfEncoder
from src.models.multimodal import MultimodalFusion
from app.components.sidebar_controls import render_sidebar_controls
from app.components.product_card import render_product_card

st.set_page_config(page_title="Product Matcher | Demo", page_icon="🔍", layout="wide")

st.title("🔍 Dataset Product Matcher")
st.caption(
    "Interactive inspection of product matching representations, threshold trade-offs, and multimodal fusion."
)

# Split selector per academic protocol: validation for tuning, test for presentation
st.sidebar.markdown("### 📁 Dataset Split")
active_split = st.sidebar.radio(
    "Active Inspection Split:",
    options=["Validation Split (Development / Tuning)", "Test Split (Frozen Presentation)"],
    index=0,
    help=(
        "Use Validation Split to explore thresholds and inspect model behavior without test set contamination. "
        "Use Test Split for final demonstration of frozen models on unseen product groups."
    ),
)
split_name = "val" if "Validation" in active_split else "test_demo"

@st.cache_data
def get_active_dataset(s_name: str):
    target_csv = VAL_SPLIT_CSV if s_name == "val" else TEST_DEMO_SPLIT_CSV
    if not target_csv.exists():
        return None, None
    df = pd.read_csv(target_csv)
    gt_map = get_ground_truth_map(df)
    return df, gt_map


active_df, gt_map = get_active_dataset(split_name)
ensure_thumbnails_extracted()

if active_df is None:
    st.warning(
        f"⚠️ Split CSV for '{split_name}' not found. Please run the extraction script: "
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

# Advanced experimental hypotheses
st.sidebar.markdown("### 🧪 Experimental Hypotheses")
enable_cc = st.sidebar.checkbox(
    "Hypothesis 5B: Transitive Graph Grouping (Connected Components)",
    value=False,
    help="Clusters candidates transitively (A~B and B~C => A~C). Evaluates whether graph grouping improves recall or induces false positive chaining.",
)

# Load cached embeddings if present
@st.cache_resource
def load_cached_embeddings(s_name: str):
    prefix = "val" if s_name == "val" else "test"
    text_f = EMBEDDINGS_DIR / f"{prefix}_text_embeddings.npy"
    img_f = EMBEDDINGS_DIR / f"{prefix}_image_embeddings.npy"
    
    t_emb = np.load(text_f) if text_f.exists() else None
    i_emb = np.load(img_f) if img_f.exists() else None
    return t_emb, i_emb


text_emb, image_emb = load_cached_embeddings(split_name)

@st.cache_resource
def get_fitted_tfidf(titles: tuple):
    encoder = TfidfEncoder(max_features=15000, ngram_range=(1, 2))
    matrix = encoder.fit_transform(list(titles))
    return encoder, matrix


tfidf_encoder, tfidf_matrix = get_fitted_tfidf(tuple(active_df["title"].tolist()))

if ("Multimodal" in model_choice or "Embeddings" in model_choice) and (text_emb is None or image_emb is None):
    st.info(
        "💡 **Notice:** Offline deep learning `.npy` embeddings not detected in `data/embeddings/`. "
        "The system is currently running on live **TF-IDF sparse cosine similarity** as a fallback. "
        "Run the Kaggle GPU pipeline to export `test_text_embeddings.npy` and `test_image_embeddings.npy` to activate multimodal vector scoring."
    )

# Reference product selection
st.subheader("1️⃣ Select Reference Product")
search_term = st.text_input(
    "Search product listings by keyword:",
    value="",
    placeholder="e.g., kemeja, casing iphone, sepatu, cream, masker...",
)

filtered_df = active_df
if search_term.strip():
    filtered_df = active_df[
        active_df["title"].str.contains(search_term.strip(), case=False, na=False)
    ]

if len(filtered_df) == 0:
    st.info("No products match your search keyword. Showing all listings.")
    filtered_df = active_df

sample_options = filtered_df.head(100)
format_func = lambda idx: f"[{active_df.loc[idx, 'posting_id']}] {active_df.loc[idx, 'title'][:80]}"

selected_idx = st.selectbox(
    "Choose a reference listing:",
    options=sample_options.index.tolist(),
    format_func=format_func,
    index=0,
)

ref_row = active_df.loc[selected_idx]
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
        f"**Total Ground-Truth Equivalents in Split:** **{len(ground_truth_postings)}** listing(s)"
    )
    st.info(
        f"Active inspection split: **{active_split}** ({len(active_df):,} total listings). "
        "Move the threshold slider in the sidebar to observe the precision/recall trade-off live!"
    )

st.divider()

def retrieve_candidates(model_name: str, query_idx: int, k: int, current_alpha: float):
    if "TF-IDF" in model_name:
        query_vec = tfidf_matrix[query_idx]
        sims = tfidf_encoder.compute_similarity(query_vec, tfidf_matrix)
    else:
        if text_emb is not None and "Text Embeddings" in model_name:
            query_emb = text_emb[query_idx : query_idx + 1]
            sims = np.dot(text_emb, query_emb.T).squeeze()
        elif image_emb is not None and "Image Embeddings" in model_name:
            query_emb = image_emb[query_idx : query_idx + 1]
            sims = np.dot(image_emb, query_emb.T).squeeze()
        elif text_emb is not None and image_emb is not None and "Multimodal" in model_name:
            fused_sims = current_alpha * np.dot(text_emb, text_emb[query_idx : query_idx + 1].T).squeeze() + (1.0 - current_alpha) * np.dot(image_emb, image_emb[query_idx : query_idx + 1].T).squeeze()
            sims = fused_sims
        else:
            query_vec = tfidf_matrix[query_idx]
            sims = tfidf_encoder.compute_similarity(query_vec, tfidf_matrix)

    top_indices = np.argsort(sims)[::-1][:k]
    candidate_df = active_df.iloc[top_indices].copy()
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

    # Section 1: MATCHED PRODUCTS
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

    # Section 2: SIMILAR BUT REJECTED
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
