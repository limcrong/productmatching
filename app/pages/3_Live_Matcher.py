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

fusion_model_choice = st.sidebar.selectbox(
    "Matching & Fusion Strategy",
    options=[
        "🎯 Linear Score Fusion (Exp 4a - Adjustable Weights, 90.7% Precision)",
        "🤖 Learned Logistic Fusion (Exp 4b - Supervised Fixed Sigmoid)",
    ],
    index=0,
    help="Linear Score Fusion (Exp 4a) is recommended for variant filtering because it allows tuning the Text Weight (α) to penalize fine variant differences, avoiding sigmoid saturation.",
)

is_learned = "Learned" in fusion_model_choice
fusion_type = "learned" if is_learned else "grid"
default_th = 0.95 if is_learned else 0.70
default_alpha = 0.70 if not is_learned else float(mm_profile.get("default_alpha", 0.55))

th_slider = st.sidebar.slider(
    "Match Threshold (τ):",
    min_value=0.0,
    max_value=1.0,
    value=float(default_th),
    step=0.01,
    help=f"Recommended threshold for filtering variants is {default_th:.2f}.",
)
st.sidebar.caption(f"🎯 Threshold τ: **{th_slider:.2f}**")

if not is_learned:
    alpha_slider = st.sidebar.slider(
        "Text Weight (α):",
        min_value=0.0,
        max_value=1.0,
        value=float(default_alpha),
        step=0.05,
        help="Higher α increases focus on exact text specifications (e.g. stage, capacity, model) over generic visual packaging.",
    )
    st.sidebar.caption(f"Text Weight (α): **{alpha_slider:.2f}** | Image Weight: **{1.0 - alpha_slider:.2f}**")
else:
    alpha_slider = default_alpha
    st.sidebar.info("🤖 **Learned Logistic Weights:** w_text = 9.46, w_image = 7.81, bias = -8.20")
    st.sidebar.caption(f"Sigmoid probability calibration requires high τ (≥ 0.95) because candidate search results already share partial text.")

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
            st.image(default_img_url, caption="Query Product Image", width="stretch")
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
    status_box = st.status("🔍 Searching Google Shopping & Ranking Offers...", expanded=True)
    progress_bar = st.progress(5)

    candidates = []

    if "Curated Demo" in search_mode:
        status_box.write("⚡ Loading curated test scenario listings...")
        progress_bar.progress(30)
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
        progress_bar.progress(60)
    else:
        if not api_key:
            status_box.update(label="❌ Missing SerpAPI Key", state="error")
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
                status_box.update(label="❌ Missing Image URL for Lens", state="error")
                st.warning("Google Lens requires a public Image URL.")
                st.stop()
            status_box.write("📷 Step 1/3: Querying Google Lens reverse-image index...")
            progress_bar.progress(30)
            candidates = search_google_lens(image_url.strip(), api_key=api_key, country=country)
        else:
            # Default: Google Shopping Tab
            search_query = query_title.strip()
            if not search_query and image_url.strip():
                status_box.write("🔍 Detecting product name from query image...")
                detected = detect_title_from_image(image_url.strip(), api_key=api_key, country=country)
                if detected:
                    st.info(f"Recognized product name from image: **{detected}**")
                    search_query = detected

            if not search_query:
                status_box.update(label="❌ Missing Search Query", state="error")
                st.warning("Please provide a product title query (e.g. 'Sony WH-1000XM5') to search Google Shopping.")
                st.stop()

            status_box.write(f"🌐 Step 1/4: Querying Google Shopping Tab (Country: `{country.upper()}`, Query: `{search_query}`)...")
            progress_bar.progress(25)

            def on_progress_update(msg, pct):
                status_box.write(f"📥 {msg}")
                progress_bar.progress(pct)

            candidates = search_google_shopping(
                search_query,
                api_key=api_key,
                country=country,
                progress_callback=on_progress_update,
            )

    status_box.write(f"📦 Step 3/4: Retrieved {len(candidates)} listings across online merchants.")
    progress_bar.progress(75)

    status_box.write("🧠 Step 4/4: Running domain-adapted Transformer model & filtering hard negatives...")
    progress_bar.progress(85)

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

    progress_bar.progress(100)
    status_box.update(
        label=f"✅ Matching complete! Found {len(matched)} verified offers ({len(rejected)} accessories/variants filtered).",
        state="complete",
        expanded=False,
    )

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
                        st.image(cand.thumbnail_url, width="stretch")
                    st.markdown(f"**🛒 {cand.merchant}**")
                    st.write(cand.title)
                    try:
                        price_display = f"${float(cand.price):.2f} {cand.currency}"
                    except (ValueError, TypeError):
                        price_display = f"{cand.price} {cand.currency}" if cand.price else "Price upon inquiry"

                    st.markdown(f"### {price_display}")
                    reason_badge = f" • *{cand.match_reason}*" if cand.match_reason else ""
                    st.caption(
                        f"Score: `{cand.similarity_fused:.2f}` (Text: `{cand.similarity_text:.2f}` | Img: `{cand.similarity_image:.2f}`){reason_badge}"
                    )
                    st.link_button("View Listing →", cand.link)
    else:
        st.info("No candidates cleared the match threshold.")

    # Section 2: Rejected Variants / Non-Matches
    st.markdown(f"### ⚪ Filtered Out Variants & Non-Matches ({len(rejected)})")
    st.caption("Unrelated variants and low-similarity listings safely rejected by the ML model below the decision threshold:")

    if rejected:
        r_cols = st.columns(min(len(rejected), 3))
        display_limit = min(len(rejected), 12)
        for i, rej in enumerate(rejected[:display_limit]):
            with r_cols[i % 3]:
                with st.container(border=True):
                    if rej.thumbnail_url:
                        st.image(rej.thumbnail_url, width="stretch")
                    st.markdown(f"**Merchant:** {rej.merchant}")
                    st.write(rej.title)
                    try:
                        rej_price_str = f"Price: ${float(rej.price):.2f} {rej.currency}"
                    except (ValueError, TypeError):
                        rej_price_str = f"Price: {rej.price} {rej.currency}" if rej.price else ""

                    if rej_price_str:
                        st.caption(rej_price_str)
                    st.error(f"✗ Filtered (Model Score: `{rej.similarity_fused:.3f}` < `{th_slider:.2f}`)")
                    st.caption(f"Text Sim: `{rej.similarity_text:.3f}` | Img Sim: `{rej.similarity_image:.3f}`")
        if len(rejected) > display_limit:
            with st.expander(f"View remaining {len(rejected) - display_limit} rejected items"):
                for rej in rejected[display_limit:]:
                    st.write(f"- **{rej.merchant}**: {rej.title} (Score: `{rej.similarity_fused:.3f}`)")
