import streamlit as st
from typing import Any, Dict
from src.app_config import get_model_profile


def render_sidebar_controls() -> Dict[str, Any]:
    """
    Renders academic experiment controls in the Streamlit sidebar.
    Pulls validation-calibrated defaults and availability status from app_config.
    """
    st.sidebar.header("⚙️ Experiment Controls")

    model_options = [
        "Multimodal Fusion (Text + Image)",
        "Text Embeddings (SentenceTransformer)",
        "Image Embeddings (Timm EfficientNet / CLIP)",
        "TF-IDF Baseline (Sparse Cosine)",
    ]

    model_choice = st.sidebar.selectbox(
        "Matching Model",
        options=model_options,
        index=0,
        help="Select the representation model to retrieve and rank candidate products.",
    )

    profile = get_model_profile(model_choice)

    if not profile["is_available"] and "TF-IDF" not in model_choice:
        st.sidebar.warning(
            f"⚠️ Dense embeddings for **{model_choice}** not found in `data/embeddings/`.\n\n"
            "Run Phase A pipeline (`python scripts/kaggle_run.py`) to generate embeddings."
        )

    calibrated_th = profile.get("default_threshold", 0.65)
    threshold = float(calibrated_th)
    st.sidebar.markdown(f"**🎯 Similarity Threshold (τ):** `{threshold:.2f}`")
    st.sidebar.caption(f"Validation-calibrated decision threshold (Tuned τ = {threshold:.2f}).")

    alpha = float(profile.get("default_alpha", 0.60))
    fusion_type = profile.get("fusion_type", "grid")

    if "Multimodal" in model_choice:
        if fusion_type == "learned":
            st.sidebar.info("🤖 **Learned Multimodal Fusion:** Scores are computed via trained logistic weights.")
            effective_alpha = profile.get("default_alpha", 0.60)
            st.sidebar.caption(f"Effective Text Weight: **{effective_alpha:.2f}** | Image Weight: **{1.0 - effective_alpha:.2f}**")
        else:
            st.sidebar.markdown(f"**⚖️ Text Weight (α):** `{alpha:.2f}` | **Image Weight:** `{1.0 - alpha:.2f}`")
            st.sidebar.caption("Fixed validation optimum weighting.")

    top_k = 20

    st.sidebar.divider()
    enable_comparison = st.sidebar.checkbox(
        "Enable Model Comparison",
        value=False,
        help="Compare two different models side-by-side on the same reference product.",
    )

    comp_model = None
    comp_profile = None
    if enable_comparison:
        comp_options = [m for m in model_options if m != model_choice]
        comp_model = st.sidebar.selectbox(
            "Comparison Baseline",
            options=comp_options,
            index=0,
        )
        comp_profile = get_model_profile(comp_model)

    return {
        "model_choice": model_choice,
        "profile": profile,
        "threshold": threshold,
        "alpha": alpha,
        "top_k": top_k,
        "enable_comparison": enable_comparison,
        "comp_model": comp_model,
        "comp_profile": comp_profile,
    }
