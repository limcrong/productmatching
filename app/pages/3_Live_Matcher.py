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

# Force reload web_search in long-running Streamlit processes to pick up newly added functions
import importlib
import src.retrieval.web_search
try:
    importlib.reload(src.retrieval.web_search)
except Exception:
    pass

from src.retrieval.web_search import (
    get_serpapi_key,
    search_google_shopping,
    search_google_lens,
    load_curated_demo_cases,
    WebCandidate,
)

try:
    from src.retrieval.web_search import detect_title_from_image, fetch_candidate_image
except ImportError:
    # Bulletproof fallback in case Streamlit process holds stale in-memory module
    def detect_title_from_image(image_url: str, api_key: str, country: str = "sg"):
        return None

    def fetch_candidate_image(url: str, timeout: int = 5):
        if not url:
            return None
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception:
            return None

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

only_show_priced = st.sidebar.checkbox(
    "Only Show Listings With Prices",
    value=False,
    help="Hides catalog cards or entries without an available merchant price.",
)

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
    st.info("🛍️ **Google Shopping Tab Search**: Retrieves live listings across online stores with merchant names, links, and real-time prices.")

    col_title, col_engine = st.columns([3, 1])
    with col_title:
        query_title = st.text_input(
            "Product Title / Search Keyword (Google Shopping):",
            value="Sony WH-1000XM5",
            help="Enter product name/model. This queries Google Shopping tab for verified merchant offers.",
        )
    with col_engine:
        search_engine_choice = st.selectbox(
            "Retrieval Engine:",
            options=["Google Shopping Tab (Prices)", "Google Lens (Visual Matches)"],
            index=0,
            help="Google Shopping tab returns store listings with prices. Google Lens performs open web reverse-image matching.",
        )

    st.markdown("**Optional Query Image (for Multimodal Visual Comparison):**")
    col_upload, col_url = st.columns(2)
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Product Photo / Screenshot:",
            type=["jpg", "jpeg", "png", "webp"],
            help="Visual similarity compares this image with candidate listing thumbnails.",
        )
    with col_url:
        image_url = st.text_input(
            "Or Public Image URL:",
            value="",
            help="Direct image URL to use as visual matching reference.",
        )

    # Preview query image if supplied
    query_image = None
    if uploaded_file is not None:
        try:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Loaded Query Image (File Upload)", width=150)
        except Exception:
            pass
    elif image_url.strip():
        with st.spinner("Loading query image preview..."):
            query_image = fetch_candidate_image(image_url.strip())
            if query_image:
                st.image(query_image, caption="Loaded Query Image (URL)", width=150)

    execute_search = st.button("🔍 Search Google Shopping & Compare Offers", type="primary")

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

            # Ensure query_image is prepared
            if query_image is None:
                if uploaded_file is not None:
                    try:
                        query_image = Image.open(uploaded_file).convert("RGB")
                    except Exception:
                        pass
                elif image_url.strip():
                    query_image = fetch_candidate_image(image_url.strip())

            # Decide retrieval engine
            if "Google Lens" in search_engine_choice:
                if not image_url.strip():
                    st.warning("Google Lens requires a public Image URL.")
                    st.stop()
                candidates = search_google_lens(image_url.strip(), api_key=api_key, country=country)
            else:
                # Default: Google Shopping Tab
                search_query = query_title.strip()
                if not search_query and image_url.strip():
                    # Attempt to detect product name from image URL
                    detected = detect_title_from_image(image_url.strip(), api_key=api_key, country=country)
                    if detected:
                        st.info(f"Recognized product name from image: **{detected}**")
                        search_query = detected

                if not search_query:
                    st.warning("Please provide a product title query (e.g. 'Sony WH-1000XM5') to search Google Shopping.")
                    st.stop()

                candidates = search_google_shopping(search_query, api_key=api_key, country=country)

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

        if only_show_priced:
            matched = [c for c in matched if c.price is not None]
            rejected = [c for c in rejected if c.price is not None]

        st.subheader("2️⃣ Cross-Merchant Comparison Results")

        if matched:
            # Pick lowest price offer among matched products
            offers_with_price = [c for c in matched if c.price is not None]
            best_offer = offers_with_price[0] if offers_with_price else matched[0]
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
