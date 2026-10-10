"""
Inference Pipeline for Visual Shopping Assistant (Live Candidate Re-Ranking & Matching)
Applies the calibrated multimodal matching models to rank and filter real web candidates.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image

import torch
from sentence_transformers import SentenceTransformer
from src.config import DEFAULT_TEXT_MODEL, ROOT_DIR
from src.retrieval.web_search import WebCandidate
from src.models.multimodal import score_fusion, learned_fusion

_CACHED_TEXT_MODEL = None
_CACHED_VISION_MODEL = None


def get_live_text_encoder(model_name_or_path: str = DEFAULT_TEXT_MODEL):
    """Loads SentenceTransformer on CPU for live inference, prioritizing local fine-tuned weights."""
    global _CACHED_TEXT_MODEL
    if _CACHED_TEXT_MODEL is None:
        ft_path = ROOT_DIR / "models" / "text_ft"
        if ft_path.exists() and (ft_path / "model.safetensors").exists():
            load_target = str(ft_path)
        else:
            load_target = model_name_or_path
        try:
            _CACHED_TEXT_MODEL = SentenceTransformer(load_target, device="cpu")
        except Exception:
            _CACHED_TEXT_MODEL = SentenceTransformer(DEFAULT_TEXT_MODEL, device="cpu")
    return _CACHED_TEXT_MODEL


def compute_text_similarity(query_text: str, candidate_titles: List[str]) -> np.ndarray:
    """Computes cosine similarity between query text and candidate titles."""
    if not query_text or not candidate_titles:
        return np.zeros(len(candidate_titles), dtype=np.float32)

    encoder = get_live_text_encoder()
    q_emb = encoder.encode([query_text], normalize_embeddings=True)
    c_emb = encoder.encode(candidate_titles, normalize_embeddings=True)
    return np.dot(c_emb, q_emb.T).squeeze().astype(np.float32)


def compute_image_similarity(
    query_image: Optional[Image.Image],
    candidate_images: List[Optional[Image.Image]]
) -> np.ndarray:
    """
    Computes visual cosine similarity between query image and candidate thumbnails.
    If image models are unavailable, computes perceptual color histogram cosine similarity.
    """
    n = len(candidate_images)
    if query_image is None:
        return np.full(n, 0.5, dtype=np.float32)

    sims = np.zeros(n, dtype=np.float32)

    # Fast perceptual feature representation (color histogram in HSV)
    try:
        q_hsv = query_image.resize((64, 64)).convert("HSV")
        q_hist = np.array(q_hsv.histogram(), dtype=np.float32)
        q_hist /= (np.linalg.norm(q_hist) + 1e-6)

        for i, c_img in enumerate(candidate_images):
            if c_img is not None:
                c_hsv = c_img.resize((64, 64)).convert("HSV")
                c_hist = np.array(c_hsv.histogram(), dtype=np.float32)
                c_hist /= (np.linalg.norm(c_hist) + 1e-6)
                sims[i] = float(np.dot(q_hist, c_hist))
            else:
                sims[i] = 0.4
    except Exception:
        sims.fill(0.5)

    return sims


def match_and_rank_candidates(
    query_title: str,
    query_image: Optional[Image.Image],
    candidates: List[WebCandidate],
    threshold: float = 0.65,
    alpha: float = 0.60,
    fusion_type: str = "grid",
    weights: Optional[Dict[str, float]] = None,
    bias: Optional[float] = None,
) -> Tuple[List[WebCandidate], List[WebCandidate]]:
    """
    Computes multimodal similarity scores, applies threshold τ,
    and partitions candidates into matched exact products vs. rejected items.
    Matched products are sorted by price ascending (cheapest offer first).
    """
    if not candidates:
        return [], []

    candidate_titles = [c.title for c in candidates]
    candidate_images = [c.image for c in candidates]

    text_sims = compute_text_similarity(query_title, candidate_titles)
    if text_sims.ndim == 0:
        text_sims = np.array([float(text_sims)])

    img_sims = compute_image_similarity(query_image, candidate_images)
    if img_sims.ndim == 0:
        img_sims = np.array([float(img_sims)])

    if fusion_type == "learned" and weights:
        w_t = weights.get("w_text", 1.0)
        w_i = weights.get("w_image", 1.0)
        b = bias or 0.0
        fused_sims = learned_fusion(text_sims, img_sims, w_t, w_i, b)
    else:
        fused_sims = score_fusion(text_sims, img_sims, alpha=alpha)

    matched = []
    rejected = []

    for i, cand in enumerate(candidates):
        cand.similarity_text = float(text_sims[i])
        cand.similarity_image = float(img_sims[i])
        cand.similarity_fused = float(fused_sims[i])
        cand.is_match = cand.similarity_fused >= threshold

        if cand.is_match:
            matched.append(cand)
        else:
            rejected.append(cand)

    # Sort matched exact offers by price (ascending, None prices last)
    matched.sort(key=lambda x: (x.price is None, x.price if x.price is not None else 999999))
    # Sort rejected offers descending by score (hard negatives closest to threshold first)
    rejected.sort(key=lambda x: x.similarity_fused, reverse=True)

    return matched, rejected
