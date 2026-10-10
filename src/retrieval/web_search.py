"""
SerpAPI Integration for Live Web Product Retrieval (Mode 2: Visual Shopping Assistant)
Retrieves real-world product candidates across online merchants via Google Shopping & Google Lens.
"""

import os
import io
import json
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

try:
    from serpapi import GoogleSearch
    HAS_SERPAPI = True
except ImportError:
    HAS_SERPAPI = False

from src.config import ROOT_DIR

LIVE_DEMO_DIR = ROOT_DIR / "data" / "live_demo"
LIVE_DEMO_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WebCandidate:
    title: str
    merchant: str
    price: Optional[float]
    currency: str
    link: str
    thumbnail_url: Optional[str]
    image: Optional[Image.Image] = None
    similarity_text: float = 0.0
    similarity_image: float = 0.0
    similarity_fused: float = 0.0
    is_match: bool = False


def get_serpapi_key() -> Optional[str]:
    """Retrieves SerpAPI key from environment or Streamlit secrets."""
    key = os.getenv("serpapi_key") or os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("serpapi_key")
        except Exception:
            pass
    return key


def fetch_candidate_image(url: str, timeout: int = 5) -> Optional[Image.Image]:
    """Downloads candidate thumbnail safely within timeout."""
    if not url:
        return None
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def search_google_shopping(
    query_text: str,
    api_key: str,
    country: str = "sg",
    max_results: int = 25,
) -> List[WebCandidate]:
    """
    Queries Google Shopping via SerpAPI.
    """
    if not HAS_SERPAPI:
        raise ImportError("google-search-results package is required for live search.")

    params = {
        "engine": "google_shopping",
        "q": query_text,
        "gl": country,
        "hl": "en",
        "num": max_results,
        "api_key": api_key,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    shopping_results = results.get("shopping_results", [])

    candidates = []
    urls_to_fetch = []

    for item in shopping_results:
        title = item.get("title", "")
        merchant = item.get("source", "Online Merchant")
        price = item.get("extracted_price")
        currency = item.get("currency", "SGD" if country == "sg" else "USD")
        link = item.get("product_link") or item.get("link", "#")
        thumb_url = item.get("thumbnail")

        cand = WebCandidate(
            title=title,
            merchant=merchant,
            price=price,
            currency=currency,
            link=link,
            thumbnail_url=thumb_url,
        )
        candidates.append(cand)
        urls_to_fetch.append(thumb_url)

    # Concurrently fetch candidate images
    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(fetch_candidate_image, urls_to_fetch))

    for cand, img in zip(candidates, images):
        cand.image = img

    return candidates


def search_google_lens(
    image_url: str,
    api_key: str,
    country: str = "sg",
    max_results: int = 25,
) -> List[WebCandidate]:
    """
    Queries Google Lens via SerpAPI with a public image URL.
    """
    if not HAS_SERPAPI:
        raise ImportError("google-search-results package is required for live search.")

    params = {
        "engine": "google_lens",
        "url": image_url,
        "gl": country,
        "hl": "en",
        "api_key": api_key,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    visual_matches = results.get("visual_matches", [])

    candidates = []
    urls_to_fetch = []

    for item in visual_matches[:max_results]:
        title = item.get("title", "")
        merchant = item.get("source", "Online Merchant")
        price_obj = item.get("price", {})
        price = price_obj.get("extracted_value") if isinstance(price_obj, dict) else None
        currency = price_obj.get("currency", "USD") if isinstance(price_obj, dict) else "USD"
        link = item.get("link", "#")
        thumb_url = item.get("thumbnail")

        cand = WebCandidate(
            title=title,
            merchant=merchant,
            price=price,
            currency=currency,
            link=link,
            thumbnail_url=thumb_url,
        )
        candidates.append(cand)
        urls_to_fetch.append(thumb_url)

    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(fetch_candidate_image, urls_to_fetch))

    for cand, img in zip(candidates, images):
        cand.image = img

    return candidates


def load_curated_demo_cases() -> Dict[str, Any]:
    """Loads curated demo cases for offline execution without API costs."""
    demo_file = LIVE_DEMO_DIR / "curated_cases.json"
    if demo_file.exists():
        try:
            with open(demo_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
