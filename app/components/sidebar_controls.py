import streamlit as st
from typing import Any, Dict


def render_sidebar_controls() -> Dict[str, Any]:
    """
    Renders academic experiment controls in the Streamlit sidebar.
    Returns a dictionary of active hyperparameter values.
    """
    st.sidebar.header("⚙️ Experiment Controls")
    
    model_choice = st.sidebar.selectbox(
        "Matching Model",
        options=[
            "Multimodal Fusion (Text + Image)",
            "Text Embeddings (SentenceTransformer)",
            "Image Embeddings (Timm EfficientNet)",
            "TF-IDF Baseline (Sparse Cosine)",
        ],
        index=0,
        help="Select the representation model to retrieve and rank candidate products.",
    )
    
    threshold = st.sidebar.slider(
        "Similarity Threshold (τ)",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.01,
        help="Scores ≥ τ are predicted as matching products. Drag to see precision/recall trade-off.",
    )
    
    alpha = 0.60
    if "Multimodal" in model_choice:
        alpha = st.sidebar.slider(
            "Text Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05,
            help="Fused similarity = α * Sim_text + (1 - α) * Sim_image",
        )
        st.sidebar.caption(f"Image Weight (1 - α): **{1.0 - alpha:.2f}**")
        
    top_k = st.sidebar.slider(
        "Candidate Retrieval Top-K",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of nearest neighbors to retrieve before thresholding.",
    )
    
    st.sidebar.divider()
    enable_comparison = st.sidebar.checkbox(
        "Enable Model Comparison",
        value=False,
        help="Compare two different models side-by-side on the same reference product.",
    )
    
    comp_model = None
    if enable_comparison:
        comp_model = st.sidebar.selectbox(
            "Comparison Baseline",
            options=[
                "TF-IDF Baseline (Sparse Cosine)",
                "Text Embeddings (SentenceTransformer)",
                "Image Embeddings (Timm EfficientNet)",
            ],
            index=0,
        )
        
    return {
        "model_choice": model_choice,
        "threshold": threshold,
        "alpha": alpha,
        "top_k": top_k,
        "enable_comparison": enable_comparison,
        "comp_model": comp_model,
    }
