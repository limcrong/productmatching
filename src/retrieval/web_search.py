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
    retrieval_rank: int = 1
    similarity_text: float = 0.0
    similarity_image: Optional[float] = None
    similarity_fused: float = 0.0
    is_match: bool = False
    match_reason: str = ""
    modality: str = "multimodal"
    # Dual-model evaluation tracking for academic explainability (Text-Only vs Multimodal)
    text_only_sim: float = 0.0
    text_only_is_match: bool = False
    text_only_reason: str = ""
    multimodal_sim: float = 0.0
    multimodal_is_match: bool = False
    multimodal_reason: str = ""


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


def fetch_candidate_image(url_or_path: str, timeout: float = 3.0) -> Optional[Image.Image]:
    """
    Safely loads candidate image from a local file path or downloads thumbnail within timeout.
    """
    if not url_or_path:
        return None

    # Check local filesystem first
    try:
        p = Path(url_or_path)
        if not p.is_absolute():
            p = ROOT_DIR / url_or_path
        if p.exists() and p.is_file():
            return Image.open(p).convert("RGB")
    except Exception:
        pass

    # Fetch from URL if local resolution is not applicable
    try:
        req = urllib.request.Request(
            url_or_path,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None



import re
from typing import List, Optional, Dict, Any, Tuple

def extract_price_and_currency(item: Dict[str, Any], country: str = "sg") -> Tuple[Optional[float], str]:
    """
    Robustly extracts numeric price and currency from SerpAPI item.
    Handles numeric extracted_price, raw price strings (e.g., '$420.37', 'US$190+'),
    and provides accurate fallback currency for the selected market.
    """
    price: Optional[float] = None
    extracted = item.get("extracted_price")
    if extracted is not None:
        try:
            price = float(extracted)
        except (ValueError, TypeError):
            price = None

    raw_price = item.get("price")
    if price is None and raw_price is not None:
        if isinstance(raw_price, (int, float)):
            price = float(raw_price)
        elif isinstance(raw_price, str):
            clean = raw_price.replace(",", "")
            m = re.search(r"[\d]+(?:\.\d+)?", clean)
            if m:
                try:
                    price = float(m.group(0))
                except ValueError:
                    pass
        elif isinstance(raw_price, dict):
            ev = raw_price.get("extracted_value") or raw_price.get("value")
            if ev is not None:
                try:
                    price = float(ev)
                except (ValueError, TypeError):
                    pass

    # Check secondary price fields if still None
    if price is None:
        if item.get("extracted_old_price") is not None:
            try:
                price = float(item["extracted_old_price"])
            except (ValueError, TypeError):
                pass
        elif item.get("old_price") and isinstance(item.get("old_price"), str):
            clean = str(item.get("old_price")).replace(",", "")
            m = re.search(r"[\d]+(?:\.\d+)?", clean)
            if m:
                try:
                    price = float(m.group(0))
                except ValueError:
                    pass

    # Determine currency
    currency: Optional[str] = item.get("currency")
    if not currency and isinstance(raw_price, str):
        u = raw_price.upper()
        if "US$" in u or "USD" in u:
            currency = "USD"
        elif "S$" in u or "SGD" in u:
            currency = "SGD"
        elif "RM" in u or "MYR" in u:
            currency = "MYR"
        elif "RP" in u or "IDR" in u:
            currency = "IDR"
        elif "£" in u or "GBP" in u:
            currency = "GBP"
        elif "€" in u or "EUR" in u:
            currency = "EUR"
        elif "A$" in u or "AUD" in u:
            currency = "AUD"

    if not currency:
        country_curr = {
            "sg": "SGD",
            "us": "USD",
            "id": "IDR",
            "my": "MYR",
            "uk": "GBP",
            "gb": "GBP",
            "au": "AUD",
        }
        currency = country_curr.get(country.lower(), "USD")

    return price, currency


def search_google_shopping(
    query_text: str,
    api_key: str,
    country: str = "sg",
    max_results: int = 40,
    progress_callback: Optional[Any] = None,
) -> List[WebCandidate]:
    """
    Queries Google Shopping Tab via SerpAPI to retrieve authentic merchant offers with prices.
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

    if progress_callback:
        progress_callback(f"Querying Google Shopping (Country: {country.upper()})...", 20)

    search = GoogleSearch(params)
    results = search.get_dict()
    shopping_results = results.get("shopping_results", [])
    if not shopping_results:
        if "inline_shopping_results" in results:
            shopping_results = results["inline_shopping_results"]
        elif "categorized_shopping_results" in results:
            for cat in results["categorized_shopping_results"]:
                shopping_results.extend(cat.get("items", []))

    if progress_callback:
        progress_callback(f"Parsing {min(len(shopping_results), max_results)} merchant offers and prices...", 40)

    candidates = []
    urls_to_fetch = []

    for idx, item in enumerate(shopping_results[:max_results]):
        title = item.get("title", "")
        merchant = item.get("source") or item.get("merchant") or item.get("seller") or "Online Merchant"
        price, currency = extract_price_and_currency(item, country=country)
        link = item.get("product_link") or item.get("link") or "#"
        thumb_url = item.get("thumbnail") or item.get("serpapi_thumbnail")
        if not thumb_url and item.get("thumbnails"):
            thumb_url = item["thumbnails"][0]

        cand = WebCandidate(
            title=title,
            merchant=merchant,
            price=price,
            currency=currency,
            link=link,
            thumbnail_url=thumb_url,
            retrieval_rank=idx + 1,
        )
        candidates.append(cand)
        urls_to_fetch.append(thumb_url)

    if progress_callback:
        progress_callback(f"Loading candidate product thumbnails...", 60)

    # Concurrently fetch candidate images for multimodal matching
    with ThreadPoolExecutor(max_workers=12) as executor:
        images = list(executor.map(fetch_candidate_image, urls_to_fetch))

    for cand, img in zip(candidates, images):
        cand.image = img

    return candidates


def detect_title_from_image(
    image_url: str,
    api_key: str,
    country: str = "sg",
) -> Optional[str]:
    """
    Uses Google Lens to recognize a product name/title from an image URL.
    """
    if not HAS_SERPAPI or not image_url:
        return None

    try:
        params = {
            "engine": "google_lens",
            "url": image_url,
            "gl": country,
            "hl": "en",
            "api_key": api_key,
        }
        res = GoogleSearch(params).get_dict()
        visual_matches = res.get("visual_matches", [])
        for match in visual_matches[:3]:
            t = match.get("title")
            if t and len(t.strip()) > 3:
                return t.strip()
    except Exception:
        pass
    return None


def search_google_lens(
    image_url: str,
    api_key: str,
    country: str = "sg",
    max_results: int = 40,
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

    for idx, item in enumerate(visual_matches[:max_results]):
        title = item.get("title", "")
        merchant = item.get("source") or item.get("merchant") or "Online Merchant"
        price, currency = extract_price_and_currency(item, country=country)
        link = item.get("link", "#")
        thumb_url = item.get("thumbnail") or item.get("serpapi_thumbnail")
        if not thumb_url and item.get("thumbnails"):
            thumb_url = item["thumbnails"][0]

        cand = WebCandidate(
            title=title,
            merchant=merchant,
            price=price,
            currency=currency,
            link=link,
            thumbnail_url=thumb_url,
            retrieval_rank=idx + 1,
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
