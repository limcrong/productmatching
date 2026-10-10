import sys
import io
from pathlib import Path
from PIL import Image

# Ensure repository root and app directory are on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.app_config import get_model_profile
from src.retrieval.web_search import (
    get_serpapi_key,
    search_google_shopping,
    search_google_lens,
    load_curated_demo_cases,
    WebCandidate,
)
from src.pipeline.live_matcher import match_and_rank_candidates

load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="Shopping Assistant | Live Matcher",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Visual Shopping Assistant")
st.caption("Consumer-Facing Cross-Merchant Comparison & Price Discovery POC (Slides 13B & 14)")

# Profile and Calibrated Defaults
mm_profile = get_model_profile("Multimodal")
calibrated_th = mm_profile.get("default_threshold", 0.65)
calibrated_alpha = mm_profile.get("default_alpha", 0.60)
fusion_type = mm_profile.get("fusion_type", "grid")

# Sidebar settings
st.sidebar.header("⚙️ Assistant Settings")

search_mode = st.sidebar.radio(
    "Search Mode",
    options=["⚡ Curated Demo Scenarios (Zero Cost)", "🔍 Live Web Search (SerpAPI)"],
    index=0,
    help="Select Curated Demos for instant, quota-free testing or Live Web Search for real web crawling.",
)

api_key = get_serpapi_key()
if "Live Web Search" in search_mode:
    if not api_key:
        api_key = st.sidebar.text_input(
            "SerpAPI Key:",
            type="password",
            help="Enter your SerpAPI key or configure serpapi_key in .env",
        )
        if not api_key:
            st.sidebar.warning("⚠️ API Key required for live web search. Switch to Curated Demos or enter a key.")
    else:
        st.sidebar.success("✓ SerpAPI Key Loaded")

country = st.sidebar.selectbox("Marketplace Country:", ["sg", "us", "id", "my"], index=0)

th_slider = st.sidebar.slider(
    "Match Threshold (τ):",
    min_value=0.0,
    max_value=1.0,
    value=float(calibrated_th),
    step=0.01,
    help=f"Calibrated validation threshold is {calibrated_th:.2f}.",
)

alpha_slider = st.sidebar.slider(
    "Text Weight (α):",
    min_value=0.0,
    max_value=1.0,
    value=float(calibrated_alpha),
    step=0.05,
    help="Similarity = α * Sim_text + (1 - α) * Sim_image",
)

st.sidebar.caption(f"Image Weight (1 - α): **{1.0 - alpha_slider:.2f}**")

# Main Interface
st.subheader("1️⃣ Product Input (Image & Title Hint)")

curated_cases = load_curated_demo_cases()

if "Curated Demo" in search_mode:
    case_names = {
        "sony_headphones": "Sony WH-1000XM5 Noise Cancelling Headphones",
        "iphone_15_pro": "Apple iPhone 15 Pro 128GB Natural Titanium",
    }
    selected_case_key = st.selectbox(
        "Choose an example product scenario:",
        options=list(case_names.keys()),
        format_func=lambda k: case_names.get(k, k),
    )
    active_case = curated_cases.get(selected_case_key, {})
    default_title = active_case.get("query_title", "")
    default_img_url = active_case.get("query_image_url", "")

    col_in1, col_in2 = st.columns([1, 2])
    with col_in1:
        if default_img_url:
            st.image(default_img_url, caption="Query Product Image", use_container_width=True)
    with col_in2:
        query_title = st.text_input("Product Title / Model Hint:", value=default_title)
        st.caption("💡 Curated scenarios include exact models from multiple merchants alongside accessories (cases, screen protectors).")

    execute_search = st.button("🚀 Match & Compare Prices across Merchants", type="primary")

else:
    # Live Search Mode
    col_upload, col_url = st.columns(2)
    with col_upload:
        uploaded_file = st.file_uploader("Upload Product Screenshot / Photo:", type=["jpg", "jpeg", "png", "webp"])
    with col_url:
        image_url = st.text_input("Or Public Image URL (for Google Lens):", value="")

    query_title = st.text_input("Product Title / Search Keyword:", value="Sony WH-1000XM5")

    execute_search = st.button("🔍 Search Real Web & Compare Offers", type="primary")

st.divider()

# Execution and Results
if execute_search:
    with st.spinner("Retrieving candidate listings across merchants and applying multimodal matcher..."):
        candidates = []

        if "Curated Demo" in search_mode:
            raw_cands = active_case.get("candidates", [])
            for c in raw_cands:
                candidates.append(
                    WebCandidate(
                        title=c["title"],
                        merchant=c["merchant"],
                        price=c.get("price"),
                        currency=c.get("currency", "SGD"),
                        link=c.get("link", "#"),
                        thumbnail_url=c.get("thumbnail_url"),
                    )
                )
            query_image = None
        else:
            if not api_key:
                st.error("Please provide a SerpAPI key in .env or the sidebar.")
                st.stop()

            if image_url.strip():
                candidates = search_google_lens(image_url.strip(), api_key=api_key, country=country)
            elif query_title.strip():
                candidates = search_google_shopping(query_title.strip(), api_key=api_key, country=country)
            else:
                st.warning("Please provide a product title query or image.")
                st.stop()

            query_image = None
            if uploaded_file is not None:
                try:
                    query_image = Image.open(uploaded_file).convert("RGB")
                except Exception:
                    pass

        # Match and rank candidates
        matched, rejected = match_and_rank_candidates(
            query_title=query_title,
            query_image=query_image,
            candidates=candidates,
            threshold=th_slider,
            alpha=alpha_slider,
            fusion_type=fusion_type,
            weights=mm_profile.get("weights"),
            bias=mm_profile.get("bias"),
        )

        st.subheader("2️⃣ Cross-Merchant Comparison Results")

        if matched:
            best_offer = matched[0]
            try:
                best_price_str = f"${float(best_offer.price):.2f} {best_offer.currency}"
            except (ValueError, TypeError):
                best_price_str = f"{best_offer.price} {best_offer.currency}" if best_offer.price else "Price upon inquiry"

            st.success(
                f"🏆 **Best Price Found:** **{best_price_str}** at **{best_offer.merchant}** "
                f"(Match Score: `{best_offer.similarity_fused:.2f}`)"
            )
        else:
            st.warning("No listings matched above the current threshold. Try lowering the threshold slider.")

        # Section 1: Exact Product Matches
        st.markdown(f"### 🟢 Verified Exact Product Matches ({len(matched)})")
        st.caption("Listings confirmed as the exact target product across merchants, ordered by price:")

        if matched:
            cols = st.columns(min(len(matched), 3))
            for i, cand in enumerate(matched):
                with cols[i % 3]:
                    with st.container(border=True):
                        if cand.thumbnail_url:
                            st.image(cand.thumbnail_url, use_container_width=True)
                        st.markdown(f"**🛒 {cand.merchant}**")
                        st.write(cand.title)
                        try:
                            price_display = f"${float(cand.price):.2f} {cand.currency}"
                        except (ValueError, TypeError):
                            price_display = f"{cand.price} {cand.currency}" if cand.price else "Price upon inquiry"

                        st.markdown(f"### {price_display}")
                        st.caption(
                            f"Score: `{cand.similarity_fused:.2f}` (Text: `{cand.similarity_text:.2f}` | Img: `{cand.similarity_image:.2f}`)"
                        )
                        st.link_button("View Listing →", cand.link)
        else:
            st.info("No candidates cleared the match threshold.")

        # Section 2: Rejected Accessories / Hard Negatives
        st.markdown(f"### ⚪ Filtered Out Accessories & Variants ({len(rejected)})")
        st.caption("Safely rejected items (cases, covers, cables, distinct models) filtered out by the model:")

        if rejected:
            r_cols = st.columns(min(len(rejected), 3))
            for i, rej in enumerate(rejected[:6]):
                with r_cols[i % 3]:
                    with st.container(border=True):
                        if rej.thumbnail_url:
                            st.image(rej.thumbnail_url, use_container_width=True)
                        st.markdown(f"**Merchant:** {rej.merchant}")
                        st.write(rej.title)
                        try:
                            rej_price_str = f"Price: ${float(rej.price):.2f} {rej.currency}"
                        except (ValueError, TypeError):
                            rej_price_str = f"Price: {rej.price} {rej.currency}" if rej.price else ""

                        if rej_price_str:
                            st.caption(rej_price_str)
                        st.error(f"✗ Rejected (Score: {rej.similarity_fused:.2f} < {th_slider:.2f})")
                        st.caption(f"Text Sim: `{rej.similarity_text:.2f}` | Img Sim: `{rej.similarity_image:.2f}`")
