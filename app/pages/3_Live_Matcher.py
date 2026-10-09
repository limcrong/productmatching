import sys
from pathlib import Path

# Ensure repository root and app directory are at head of sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

st.set_page_config(page_title="Live Web Matcher | Extension", page_icon="🌐", layout="wide")

st.title("🌐 Live Web Product Matcher (Mode 2 — Extension)")
st.caption("Consumer-Facing Cross-Merchant Comparison Concept")

st.info(
    "📌 **Academic Scope Note:** Per project requirements and product design, "
    "this mode is a future consumer-facing extension stub. "
    "The core academic deliverables and quantitative validations are implemented in "
    "**Mode 1 (Dataset Product Matcher)** and **Experiment Results**."
)

st.subheader("💡 Proposed Concept & Architecture")

st.markdown(
    """
    In a production e-commerce aggregator, consumers search for an item across distinct online merchants
    (e.g., Amazon, Shopee, Lazada, eBay).
    
    ```text
    User Query ("Sony WH-1000XM5")
                 │
                 ▼
       External Web Search API
                 │
                 ▼
       Live Candidate Listings (Titles, Images, Prices, Merchants)
                 │
                 ▼
       Trained Multimodal Matching Model (Text + Vision)
                 │
                 ▼
       Grouped Equivalent Offers with Cross-Merchant Price Comparison
    ```
    """
)

st.divider()

st.subheader("🧪 Live Search Mockup")

query = st.text_input("Enter a consumer product query:", value="Sony WH-1000XM5 Wireless Headphones")

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("##### 🛒 Merchant: AudioStore")
        st.caption("Sony WH-1000XM5 Noise Cancelling Headphones - Black")
        st.markdown("**$349.99**")
        st.success("✓ Predicted Match (Score: 0.94)")

with c2:
    with st.container(border=True):
        st.markdown("##### 🛒 Merchant: TechDeals")
        st.caption("Sony WH1000XM5/B Over-Ear BT Headset, Midnight Silver")
        st.markdown("**$329.00**")
        st.success("✓ Predicted Match (Score: 0.91)")

with c3:
    with st.container(border=True):
        st.markdown("##### 🛒 Merchant: CaseWorld")
        st.caption("Hard Travel Case for Sony WH-1000XM5 / XM4 Headphone")
        st.markdown("**$19.99**")
        st.error("✗ Rejected Accessory (Score: 0.42)")

st.caption("Live external crawling integration can be connected here once web APIs are configured.")
