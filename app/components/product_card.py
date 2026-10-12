import streamlit as st
from pathlib import Path
from typing import Optional
from PIL import Image

from src.data.loader import get_image_path


def get_status_badge_html(status: str, score: Optional[float] = None) -> str:
    """Returns styled HTML badge for candidate match status."""
    badge_colors = {
        "Correct Match": ("#10b981", "#ecfdf5", "✓ Correct Match"),
        "False Match": ("#ef4444", "#fef2f2", "✗ False Match"),
        "Correct Rejection": ("#6b7280", "#f9fafb", "✓ Correct Rejection"),
        "Missed Match": ("#f59e0b", "#fffbeb", "⚠ Missed Match"),
    }
    
    color, bg_color, label = badge_colors.get(
        status, ("#6b7280", "#f3f4f6", status)
    )
    
    score_str = f" • {score:.3f}" if score is not None else ""
    
    return f"""
    <span style="
        background-color: {bg_color};
        color: {color};
        border: 1px solid {color};
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
    ">
        {label}{score_str}
    </span>
    """


def render_product_card(
    title: str,
    image_name: str,
    label_group: int,
    score: Optional[float] = None,
    status: Optional[str] = None,
    posting_id: Optional[str] = None,
    is_reference: bool = False,
):
    """
    Renders an e-commerce product card with image thumbnail, title, label group,
    and match classification status.
    """
    img_path = get_image_path(image_name)
    
    with st.container(border=True):
        if is_reference:
            st.caption("📌 **REFERENCE PRODUCT**")
        elif status:
            st.markdown(get_status_badge_html(status, score), unsafe_allow_html=True)
            
        # Display image
        if img_path and img_path.exists():
            try:
                st.image(str(img_path), width="stretch")
            except Exception:
                st.info("🖼️ Image unavailable")
        else:
            st.info("🖼️ No thumbnail")

        # Truncate title for clean grid layout
        display_title = title if len(title) <= 75 else title[:72] + "..."
        st.markdown(f"**{display_title}**", help=title)
        
        caption_parts = [f"Group: `{label_group}`"]
        if posting_id:
            caption_parts.append(f"ID: `{posting_id}`")
            
        st.caption(" • ".join(caption_parts))
