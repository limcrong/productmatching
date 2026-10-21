import sys
import io
import re
import base64
from pathlib import Path
from typing import Optional
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

from src.app_config import get_model_profile, get_selected_inference_config

# Force reload modules in long-running Streamlit processes to pick up newly added functions
import importlib
import src.retrieval.web_search
import src.pipeline.live_matcher
try:
    importlib.reload(src.retrieval.web_search)
except Exception:
    pass
try:
    importlib.reload(src.pipeline.live_matcher)
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

from src.pipeline.live_matcher import (
    match_and_rank_candidates,
    categorize_model_disagreements,
    group_candidates_by_currency,
)

load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="Image-Assisted Product Matching | Shopping Assistant",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS for consistent 4-in-a-row carousel cards and uniform image sizing
st.markdown(
    """
    <style>
    /* Consistent listing image container and card presentation across columns */
    div[data-testid="stColumn"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] {
        height: 180px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #ffffff !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        margin: 0 auto 8px auto !important;
    }
    div[data-testid="stColumn"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] > div {
        height: 100% !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stColumn"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] img {
        max-height: 170px !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }

    /* Dedicated Query Preview card image styling */
    div[data-testid="stColumn"]:first-child div[data-testid="stVerticalBlock"] div[data-testid="stImage"] {
        height: 200px !important;
    }
    div[data-testid="stColumn"]:first-child div[data-testid="stVerticalBlock"] div[data-testid="stImage"] img {
        max-height: 190px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_image_display_uri(cand: WebCandidate) -> Optional[str]:
    """Resolves thumbnail URL or local file path to a URI that displays in st.dataframe ImageColumn."""
    if cand.thumbnail_url:
        u = str(cand.thumbnail_url).strip()
        if u.startswith("http://") or u.startswith("https://") or u.startswith("data:"):
            return u
        p = Path(u)
        if not p.is_absolute():
            p = ROOT_DIR / u
        if p.exists() and p.is_file():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                ext = p.suffix.lower().replace(".", "")
                mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
                return f"data:image/{mime};base64,{b64}"
            except Exception:
                pass
    return None


# Refined Title and Intended Academic / Consumer Positioning
st.title("🌐 Image-Assisted Cross-Merchant Product Matching & Price Comparison")
st.caption(
    "Consumer-Facing Proof of Concept (POC) | Pattern Recognition Systems Practice Module | NUS ISS MTech in AI Systems"
)

# Workflow Architecture Callout
with st.container(border=True):
    st.markdown(
        """
        **Process flow:**
        1. **Stage 1 — Candidate Retrieval (Google Shopping / SerpAPI)**
        2. **Stage 2 — Identity Matching & Filtering (Trained Multimodal Model)**
        3. **Stage 3 — Cross-Merchant Price Comparison**
        
        """
    )

# Profile and Calibrated Defaults
mm_profile = get_model_profile("Multimodal")
deployed_cfg = get_selected_inference_config(strict=False)

# Sidebar settings
st.sidebar.header("⚙️ Assistant Settings")

search_mode = st.sidebar.radio(
    "Search Mode",
    options=["⚡ Curated Demo Scenarios", "🔍 Live Web Search (SerpAPI)"],
    index=0,
    help="Select Curated Demos for instant, quota-free offline evaluation or Live Web Search for real web crawling.",
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

# country = st.sidebar.selectbox("Marketplace Country:", ["sg", "us", "id", "my"], index=0)
country = "sg"

# Learned logistic fusion as default (Fix 3)
fusion_model_choice = st.sidebar.text_input(
    "Matching & Fusion Strategy",
    value="🤖 Exp 4b Learned Logistic Fusion",
    disabled=True,
    help="Learned Logistic Fusion (Exp 4b) is the validation-selected model combining text and EfficientNet-B0 embeddings via supervised logistic regression weights calibrated on Kaggle training splits.",
)

is_learned = "Learned" in fusion_model_choice
fusion_type = "learned" if is_learned else "grid"
default_th = deployed_cfg["multimodal_threshold"] if is_learned else deployed_cfg["grid_threshold"]
default_alpha = float(deployed_cfg["effective_alpha"]) if is_learned else float(deployed_cfg["grid_alpha"])

# Authoritative academic evaluation thresholds dynamically loaded from configuration
th_slider = float(default_th)
text_th_slider = float(deployed_cfg["text_threshold"])
alpha_slider = default_alpha

st.sidebar.markdown(f"**🎯 Multimodal Decision τ:** `{th_slider:.2f}`")
st.sidebar.caption("Validation-calibrated decision threshold from academic evaluation.")

st.sidebar.markdown(f"**📝 Text-Only Baseline τ:** `{text_th_slider:.2f}`")
st.sidebar.caption("Calibrated baseline threshold for fine-tuned SentenceTransformer (Exp 2).")

if not is_learned:
    st.sidebar.markdown(f"**⚖️ Text Weight (α):** `{alpha_slider:.2f}` | **Image Weight:** `{1.0 - alpha_slider:.2f}`")
    st.sidebar.caption(f"Fixed validation optimum weighting (α={alpha_slider:.2f}).")
else:
    w_t = deployed_cfg["weights"]["w_text"]
    w_i = deployed_cfg["weights"]["w_image"]
    b = deployed_cfg["bias"]
    st.sidebar.info(f"🤖 **Learned Logistic Weights (Exp 4b):**\nw_text = {w_t:.3f}, w_image = {w_i:.3f}, bias = {b:.3f}")
    st.sidebar.caption("Supervised logistic fusion evaluated on Kaggle held-out split (Test F1 = 0.8629).")

# Main Interface: Input Selection
st.subheader("1️⃣ Product Input (Image & Title Hint)")

curated_cases = load_curated_demo_cases()

if "Curated Demo" in search_mode:
    case_names = {
        "sony_headphones": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "iphone_15_pro": "Apple iPhone 15 Pro 128GB Natural Titanium",
    }
    selected_case_key = st.selectbox(
        "Choose an example product scenario:",
        options=list(case_names.keys()),
        format_func=lambda k: case_names.get(k, k),
    )
    active_case = curated_cases.get(selected_case_key, {})
    default_title = active_case.get("query_title", "")
    default_img_ref = active_case.get("image_path") or active_case.get("query_image_url", "")
    curated_preview_img = fetch_candidate_image(default_img_ref)

    col_in1, col_in2 = st.columns([1, 2])
    with col_in1:
        with st.container(border=True):
            st.caption("📌 **Query Product Photo Reference**")
            img_to_show = curated_preview_img if curated_preview_img is not None else default_img_ref
            if img_to_show is not None:
                st.image(img_to_show, use_container_width=True)
            else:
                st.info("🖼️ No image available")
    with col_in2:
        query_title = st.text_input("Product Title / Model Hint:", value=default_title)
        st.caption("💡 Curated scenarios include exact models from multiple merchants alongside accessories (cases, ear pads).")
        st.info("📌 **Curated Example Notice:** Zero-cost offline mode using cached local thumbnails. Prices and offers are illustrative evaluation samples.")

    execute_search = st.button("🚀 Match & Compare Prices across Merchants", type="primary")

else:
    # Live Search Mode
    st.info("🛍️ **Google Shopping Tab Search**: Retrieves live listings across online stores with merchant names, links, and real-time prices.")

    col_title, col_engine = st.columns([3, 1])
    with col_title:
        query_title = st.text_input(
            "Product Title / Search Keyword (Google Shopping):",
            value="Sony WH-1000XM5",
            help="Enter product name/model. This queries Google Shopping tab for candidate merchant offers.",
        )
    with col_engine:
        search_engine_choice = st.text_input(
            "Retrieval Engine:",
            value="Google Shopping Tab (Prices)",
            disabled=True,
            help="Google Shopping tab returns store listings with prices.",
        )

    st.markdown("**Query Image (for Multimodal Visual Comparison):**")
    # File upload option commented out per user request (Fix 4)
    # uploaded_file = st.file_uploader(
    #     "Upload Product Photo / Screenshot:",
    #     type=["jpg", "jpeg", "png", "webp"],
    #     help="Visual similarity compares this image with candidate listing thumbnails.",
    # )
    uploaded_file = None

    image_url = st.text_input(
        "Public Image URL:",
        value="",
        placeholder="https://example.com/product_photo.jpg",
        help="Direct public image URL to use as visual matching reference.",
    )

    # Preview query image if supplied
    query_image = None
    if image_url.strip():
        with st.spinner("Loading query image preview..."):
            query_image = fetch_candidate_image(image_url.strip())
            if query_image:
                st.image(query_image, caption="Loaded Query Image (URL)", width=150)

    execute_search = st.button("🔍 Search Google Shopping & Compare Offers", type="primary")

st.divider()

# Execution and Results
if execute_search:
    status_box = st.status("🔍 Step 1: Retrieving candidates via Google Shopping...", expanded=True)
    progress_bar = st.progress(5)

    candidates = []

    if "Curated Demo" in search_mode:
        status_box.write("⚡ Loading curated test scenario listings & cached thumbnails...")
        progress_bar.progress(30)
        raw_cands = active_case.get("candidates", [])
        for idx, c in enumerate(raw_cands):
            c_img_ref = c.get("image_path") or c.get("thumbnail_url")
            c_img = fetch_candidate_image(c_img_ref)
            candidates.append(
                WebCandidate(
                    title=c["title"],
                    merchant=c["merchant"],
                    price=c.get("price"),
                    currency=c.get("currency", "SGD"),
                    link=c.get("link", "#"),
                    thumbnail_url=c_img_ref,
                    image=c_img,
                    retrieval_rank=idx + 1,
                )
            )
        q_img_ref = active_case.get("image_path") or active_case.get("query_image_url")
        query_image = fetch_candidate_image(q_img_ref)
        progress_bar.progress(60)
    else:
        if not api_key:
            status_box.update(label="❌ Missing SerpAPI Key", state="error")
            st.error("Please provide a SerpAPI key in .env or the sidebar.")
            st.stop()

        # Ensure query_image is prepared from public URL
        if query_image is None and image_url.strip():
            query_image = fetch_candidate_image(image_url.strip())

        # Decide retrieval engine
        if "Google Lens" in search_engine_choice:
            if not image_url.strip():
                status_box.update(label="❌ Missing Image URL for Lens", state="error")
                st.warning("Google Lens requires a public Image URL.")
                st.stop()
            status_box.write("📷 Step 1/3: Querying Google Lens reverse-image index...")
            progress_bar.progress(30)
            candidates = search_google_lens(image_url.strip(), api_key=api_key, country=country, max_results=40)
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
                max_results=40,
                progress_callback=on_progress_update,
            )

    status_box.write(f"📦 Step 2/3: Retrieved {len(candidates)} raw candidates from Google Shopping.")
    progress_bar.progress(75)

    status_box.write("🧠 Step 3/3: Running dual-model inference (Fine-Tuned Text-Only & Multimodal Fusion)...")
    progress_bar.progress(85)

    # Match and rank candidates across both models
    matched, rejected = match_and_rank_candidates(
        query_title=query_title,
        query_image=query_image,
        candidates=candidates,
        threshold=th_slider,
        alpha=alpha_slider,
        fusion_type=fusion_type,
        weights=deployed_cfg.get("weights"),
        bias=deployed_cfg.get("bias"),
        text_threshold=text_th_slider,
    )

    disagreements = categorize_model_disagreements(candidates)

    progress_bar.progress(100)
    status_box.update(
        label=f"✅ Evaluation complete! Evaluated {len(candidates)} candidates: {len(matched)} likely matches, {len(rejected)} filtered out.",
        state="complete",
        expanded=False,
    )

    # -----------------------------------------------------------------
    # STAGE 1: RAW GOOGLE SHOPPING CANDIDATES (Part 2 Stage A - 4 In A Row Grid)
    # -----------------------------------------------------------------
    st.markdown("## 📦 Stage 1: Google Shopping Candidate Retrieval")
    st.markdown(
        f"Google Shopping retrieved **{len(candidates)} raw candidate listings** using the title query. "
        "These listings reflect initial search retrieval *before* our trained product matching model is applied."
    )
    st.info(
        "💡 **Understanding Raw Retrieval:** Google Shopping uses search indexing to surface candidate listings across stores. "
        "However, keyword queries frequently surface accessories (e.g. ear cushions, cases), related prior models (e.g. XM4 vs XM5), "
        "or third-party parts. Google Shopping has **not** verified these candidates as exact product matches."
    )

    with st.expander(f"👁️ View All {len(candidates)} Raw Candidates (Original Retrieval Order)", expanded=False):
        for row_start in range(0, len(candidates), 4):
            cols = st.columns(4)
            for col_idx, c in enumerate(candidates[row_start : row_start + 4]):
                with cols[col_idx]:
                    with st.container(border=True):
                        st.caption(f"**Order #{c.retrieval_rank} Retrieved**")
                        if c.thumbnail_url:
                            st.image(c.thumbnail_url, width="stretch")
                        else:
                            st.info("🖼️ No thumbnail")
                        st.markdown(f"**🛒 {c.merchant}**")
                        display_title = c.title if len(c.title) <= 60 else c.title[:57] + "..."
                        st.markdown(f"**{display_title}**", help=c.title)
                        if c.price is not None:
                            st.markdown(f"**${float(c.price):.2f} {c.currency}**")
                        else:
                            st.caption("Price upon inquiry / unlisted")
                        st.link_button("View Listing →", c.link, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------
    # STAGE 2: MULTIMODAL MATCHING & MODEL COMPARISON (Part 2 Stage B & Part 3)
    # -----------------------------------------------------------------
    st.markdown("## 🧠 Stage 2: Multimodal Product Matching & Model Comparison")
    st.markdown(
        "Our trained model evaluates candidate listings by comparing the user's title and image against "
        "each candidate. Below, you can explore the primary multimodal filtering results and examine how "
        "the visual modality changes predictions compared to text-only matching."
    )

    # High-level pipeline summary metrics
    text_matches_count = len([c for c in candidates if c.text_only_is_match])
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Raw Candidates Retrieved", len(candidates), help="Google Shopping initial retrieval set")
    with m2:
        st.metric("Text-Only Accepted", text_matches_count, help=f"Score ≥ τ_text ({text_th_slider:.2f})")
    with m3:
        st.metric("Multimodal Accepted", len(matched), help=f"Score ≥ τ_mm ({th_slider:.2f})")
    with m4:
        st.metric("Filtered Out by Multimodal", len(rejected), delta=f"-{len(rejected)}", delta_color="inverse")

    # Tabbed Interface for Primary Results vs Academic Comparison
    tab_mm, tab_comp, tab_table = st.tabs([
        "🎯 Multimodal Matching Results (Primary Model)",
        "⚖️ Academic Model Comparison (Text-Only vs Multimodal)",
        "📋 Full Candidate Inspection Table"
    ])

    # TAB 1: Multimodal Matching Results (4 In A Row Grid Style - Fix 1)
    with tab_mm:
        st.markdown(f"### 🟢 Likely Matching Listings ({len(matched)})")
        st.caption(f"Listings predicted as equivalent product identity across merchants (Multimodal Score ≥ τ={th_slider:.2f}):")

        if matched:
            for row_start in range(0, len(matched), 4):
                cols = st.columns(4)
                for col_idx, cand in enumerate(matched[row_start : row_start + 4]):
                    with cols[col_idx]:
                        with st.container(border=True):
                            st.caption(f"Order #{cand.retrieval_rank} • **{cand.merchant}**")
                            if cand.thumbnail_url:
                                st.image(cand.thumbnail_url, width="stretch")
                            else:
                                st.info("🖼️ No thumbnail")
                            display_title = cand.title if len(cand.title) <= 60 else cand.title[:57] + "..."
                            st.markdown(f"**{display_title}**", help=cand.title)
                            try:
                                price_display = f"${float(cand.price):.2f} {cand.currency}"
                            except (ValueError, TypeError):
                                price_display = f"{cand.price} {cand.currency}" if cand.price else "Price upon inquiry"

                            st.markdown(f"### {price_display}")
                            img_str = f"{cand.similarity_image:.2f}" if cand.similarity_image is not None else "N/A"
                            st.caption(
                                f"**Fused: `{cand.similarity_fused:.2f}`** (Txt: `{cand.similarity_text:.2f}` | Img: `{img_str}`)"
                            )
                            st.link_button("View Listing →", cand.link, use_container_width=True)
        else:
            st.warning("No candidate listings cleared the calibrated multimodal decision threshold.")

        st.markdown(f"### ⚪ Filtered Out Variants & Accessories ({len(rejected)})")
        st.caption(f"Listings safely rejected by the ML model below decision threshold (Score < τ={th_slider:.2f}):")

        if rejected:
            display_limit = min(len(rejected), 16)
            for row_start in range(0, display_limit, 4):
                cols = st.columns(4)
                for col_idx, rej in enumerate(rejected[row_start : min(row_start + 4, display_limit)]):
                    with cols[col_idx]:
                        with st.container(border=True):
                            st.caption(f"Order #{rej.retrieval_rank} • **{rej.merchant}**")
                            if rej.thumbnail_url:
                                st.image(rej.thumbnail_url, width="stretch")
                            else:
                                st.info("🖼️ No thumbnail")
                            display_title = rej.title if len(rej.title) <= 60 else rej.title[:57] + "..."
                            st.markdown(f"**{display_title}**", help=rej.title)
                            try:
                                rej_price_str = f"Price: ${float(rej.price):.2f} {rej.currency}"
                            except (ValueError, TypeError):
                                rej_price_str = f"Price: {rej.price} {rej.currency}" if rej.price else ""
                            if rej_price_str:
                                st.caption(rej_price_str)

                            st.error(f"✗ Filtered (Score: `{rej.similarity_fused:.3f}` < `{th_slider:.2f}`)")
                            rej_img_str = f"{rej.similarity_image:.3f}" if rej.similarity_image is not None else "N/A"
                            st.caption(f"Text: `{rej.similarity_text:.3f}` | Img: `{rej_img_str}`")
                            st.link_button("Inspect Listing →", rej.link, use_container_width=True)
            if len(rejected) > display_limit:
                with st.expander(f"View remaining {len(rejected) - display_limit} rejected items"):
                    for rej in rejected[display_limit:]:
                        st.write(f"- **Order #{rej.retrieval_rank}** ({rej.merchant}): {rej.title} (Score: `{rej.similarity_fused:.3f}`)")

    # TAB 2: Academic Model Comparison (Part 3)
    with tab_comp:
        st.markdown("### ⚖️ Fine-Tuned Text-Only vs. Multimodal Fusion Comparison")
        st.markdown(
            "This view processes the **exact same candidate set** through two distinct models to demonstrate "
            "why adding visual features (`EfficientNet-B0`) resolves text ambiguities and accessory false positives:"
        )

        # Comparison summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Consensus Matches", len(disagreements["consensus_matches"]), help="Both Text-Only and Multimodal accept")
        with c2:
            st.metric(
                "Filtered by Visual Modality",
                len(disagreements["text_only_matches"]),
                help="Text-only accepted due to title overlap, but Multimodal safely rejected due to low visual similarity"
            )
        with c3:
            st.metric("Rescued by Visual Modality", len(disagreements["multimodal_only_matches"]), help="Multimodal accepted, Text-only rejected")
        with c4:
            st.metric("Consensus Rejections", len(disagreements["consensus_rejections"]), help="Both models reject")

        # Key Demonstration: Disagreements (4 In A Row Grid Style)
        disagree_items = disagreements["text_only_matches"]
        if disagree_items:
            st.markdown("#### ⚡ Crucial Differences: Candidates Filtered by Visual Evidence")
            st.markdown(
                "These candidates scored **high text similarity** (overlapping title keywords such as product model name), "
                "leading the **Text-Only model to accept them**. However, because the visual image clearly depicts an "
                "accessory or replacement part rather than the product, **our Multimodal model correctly filtered them out**:"
            )
            for row_start in range(0, len(disagree_items), 4):
                d_cols = st.columns(4)
                for col_idx, d in enumerate(disagree_items[row_start : row_start + 4]):
                    with d_cols[col_idx]:
                        with st.container(border=True):
                            st.caption(f"Order #{d.retrieval_rank} • **{d.merchant}**")
                            if d.thumbnail_url:
                                st.image(d.thumbnail_url, width="stretch")
                            else:
                                st.info("🖼️ No thumbnail")
                            display_title = d.title if len(d.title) <= 60 else d.title[:57] + "..."
                            st.markdown(f"**{display_title}**", help=d.title)
                            st.warning(f"📝 Text-Only: Accepted (`{d.text_only_sim:.3f}` ≥ `{text_th_slider:.2f}`)")
                            d_img_val = f"{d.similarity_image:.3f}" if d.similarity_image is not None else "N/A"
                            st.error(f"🖼️ Multimodal: Filtered (`{d.similarity_fused:.3f}` < `{th_slider:.2f}`)")
                            st.caption(f"Img Sim: `{d_img_val}` • *Visual difference prevented false match.*")
        else:
            st.info("No candidates were accepted by text-only and filtered by multimodal at the selected thresholds.")

        if disagreements["multimodal_only_matches"]:
            st.markdown("#### 🔍 Candidates Rescued by Visual Modality")
            st.markdown("These candidates had lower text similarity but high visual similarity, causing the multimodal model to accept them:")
            for m in disagreements["multimodal_only_matches"]:
                st.write(f"- **Order #{m.retrieval_rank}** ({m.merchant}): {m.title} (Text Sim: `{m.text_only_sim:.3f}`, Img Sim: `{m.similarity_image:.3f}`, Fused: `{m.similarity_fused:.3f}`)")

        with st.expander("👁️ View Consensus Matches (Both Models Agree)", expanded=False):
            for c in disagreements["consensus_matches"]:
                st.write(f"- **Order #{c.retrieval_rank}** ({c.merchant}): {c.title} — Text: `{c.text_only_sim:.3f}`, Img: `{c.similarity_image:.3f}`, Fused: `{c.similarity_fused:.3f}`")

        with st.expander("👁️ View Consensus Rejections (Both Models Reject)", expanded=False):
            for c in disagreements["consensus_rejections"]:
                st.write(f"- **Order #{c.retrieval_rank}** ({c.merchant}): {c.title} — Text: `{c.text_only_sim:.3f}`, Img: `{c.similarity_image:.3f}`, Fused: `{c.similarity_fused:.3f}`")

        st.caption(
            "📌 *Academic Note: In live search without external ground-truth labels, model disagreements are presented "
            "as differences between model hypotheses rather than confirmed error corrections.*"
        )

    # TAB 3: Full Table Inspection
    with tab_table:
        st.markdown("### 📋 Complete Candidate Feature & Decision Matrix")
        table_rows = []
        for c in candidates:
            disagreement_status = "Consensus Match" if (c.text_only_is_match and c.multimodal_is_match) else (
                "Filtered by Vision" if (c.text_only_is_match and not c.multimodal_is_match) else (
                    "Rescued by Vision" if (not c.text_only_is_match and c.multimodal_is_match) else "Consensus Reject"
                )
            )
            table_rows.append({
                "Rank": c.retrieval_rank,
                "Merchant": c.merchant,
                "Title": c.title,
                "Price": f"${c.price:.2f} {c.currency}" if c.price is not None else "Unlisted",
                "Text Sim": round(c.text_only_sim, 3),
                "Img Sim": round(c.similarity_image, 3) if c.similarity_image is not None else "N/A",
                "Fused Score": round(c.similarity_fused, 3),
                "Text Decision": "✓ Match" if c.text_only_is_match else "✗ Filtered",
                "Multimodal Decision": "✓ Match" if c.multimodal_is_match else "✗ Filtered",
                "Agreement": disagreement_status,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------
    # STAGE 3: CROSS-MERCHANT PRICE COMPARISON TABLE (Fix 2)
    # -----------------------------------------------------------------
    st.markdown("## 💰 Stage 3: Cross-Merchant Price Comparison (Among Likely Matches)")
    st.markdown(
        "Price comparison applies **strictly to candidates retrieved and accepted** by the trained model. "
        "All verified offers are displayed in the comparison table below, ordered from lowest to highest price."
    )

    if "Curated Demo" in search_mode:
        st.info("📌 **Curated Evaluation Notice:** Prices and merchant listings are illustrative offline samples for model verification, not real-time live quotes.")

    if matched:
        currency_groups = group_candidates_by_currency(matched)

        if len(currency_groups) > 1:
            st.warning(
                f"⚠️ Note: Retrieved listings span multiple currencies ({', '.join(currency_groups.keys())}). "
                "Offers are presented separately per currency without currency conversion."
            )

        for curr, group_info in currency_groups.items():
            st.markdown(f"### 💵 Offers in {curr}")
            lowest_offer = group_info["lowest_offer"]
            priced_offers = group_info["priced_offers"]
            unpriced_offers = group_info["unpriced_offers"]

            if lowest_offer is not None:
                st.success(
                    f"🏆 **Lowest Price Among Retrieved Matches ({curr}):** "
                    f"**${lowest_offer.price:.2f} {curr}** at **{lowest_offer.merchant}** "
                    f"(Multimodal Match Score: `{lowest_offer.similarity_fused:.2f}`)"
                )

            if priced_offers:
                price_table_rows = []
                for rank_num, po in enumerate(priced_offers, 1):
                    img_uri = get_image_display_uri(po)
                    price_table_rows.append({
                        "Price Rank": f"#{rank_num}",
                        "Thumbnail": img_uri,
                        "Merchant": po.merchant,
                        "Product Title": po.title,
                        "Price": float(po.price),
                        "Currency": po.currency,
                        "Match Score": float(po.similarity_fused),
                        "Listing": po.link,
                        "Google Order": f"#{po.retrieval_rank}",
                    })

                df_price = pd.DataFrame(price_table_rows)
                st.dataframe(
                    df_price,
                    column_config={
                        "Price Rank": st.column_config.TextColumn("Price Rank", width="small"),
                        "Thumbnail": st.column_config.ImageColumn("Image", width="small"),
                        "Merchant": st.column_config.TextColumn("Merchant", width="medium"),
                        "Product Title": st.column_config.TextColumn("Product Title", width="large"),
                        "Price": st.column_config.NumberColumn(
                            f"Price ({curr})",
                            format="$%.2f",
                            width="small",
                        ),
                        "Currency": st.column_config.TextColumn("Currency", width="small"),
                        "Match Score": st.column_config.ProgressColumn(
                            "Match Score",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                            width="medium",
                        ),
                        "Listing": st.column_config.LinkColumn(
                            "Merchant Store",
                            display_text="View Store ↗",
                            width="small",
                        ),
                        "Google Order": st.column_config.TextColumn("Google Order", width="small"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            if unpriced_offers:
                with st.expander(f"Unpriced / Incomplete Listings in {curr} ({len(unpriced_offers)})"):
                    unpriced_rows = []
                    for up in unpriced_offers:
                        unpriced_rows.append({
                            "Thumbnail": get_image_display_uri(up),
                            "Merchant": up.merchant,
                            "Product Title": up.title,
                            "Listing": up.link,
                            "Google Order": f"#{up.retrieval_rank}",
                        })
                    st.dataframe(
                        pd.DataFrame(unpriced_rows),
                        column_config={
                            "Thumbnail": st.column_config.ImageColumn("Image", width="small"),
                            "Listing": st.column_config.LinkColumn("Merchant Store", display_text="Inquire Store ↗"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )

    else:
        st.warning("No candidate listings cleared the matching threshold to compare prices.")
