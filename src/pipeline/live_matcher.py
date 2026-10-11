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


import re

def verify_candidate_compatibility(
    query_title: str,
    candidate_title: str,
    raw_score: float,
    threshold: float,
) -> Tuple[bool, float, str]:
    """
    Applies e-commerce hard negative filtering and exact model code alignment.
    Filters out unwanted accessories, conflicting generations (e.g. XM6 vs XM5),
    and form factor mismatches (e.g. WF earbuds vs WH over-ear).
    Recovers valid exact model matches where the retail title omits the brand.
    """
    q_lower = query_title.lower()
    c_lower = candidate_title.lower()

    # 1. Accessories check: if query does not ask for an accessory, reject accessories
    accessories = ["case", "cover", "ear pad", "earpad", "cushion", "cable", "strap", "stand", "protector", "sleeve", "skin"]
    has_accessory = any(acc in c_lower for acc in accessories)
    query_wants_accessory = any(acc in q_lower for acc in accessories)
    if has_accessory and not query_wants_accessory:
        return False, min(raw_score, threshold - 0.05), "Accessory / Peripheral"

    # 2. Conflicting model generation check (e.g. XM6 / XM4 vs XM5)
    for gen in ["xm5", "xm4", "xm3", "xm6"]:
        if gen in q_lower:
            other_gens = [g for g in ["xm5", "xm4", "xm3", "xm6"] if g != gen]
            if any(g in c_lower for g in other_gens) and gen not in c_lower:
                return False, min(raw_score, threshold - 0.05), f"Model generation mismatch (e.g. {gen.upper()})"

    # Form factor check (Over-ear WH vs in-ear WF)
    if ("wh" in q_lower or "over-ear" in q_lower) and ("wf-" in c_lower or "wf1000" in c_lower or "in-ear" in c_lower):
        return False, min(raw_score, threshold - 0.05), "Form-factor mismatch (Earbuds vs Over-ear)"
    elif ("wf" in q_lower or "in-ear" in q_lower) and ("wh-" in c_lower or "wh1000" in c_lower or "over-ear" in c_lower):
        return False, min(raw_score, threshold - 0.05), "Form-factor mismatch (Over-ear vs Earbuds)"

    # 3. Model number boost: if query contains a distinct alphanumeric model number (e.g., '1000xm5', 'wh1000xm5', '15 pro')
    model_codes = re.findall(r'[a-zA-Z]*\d+[a-zA-Z0-9]*', q_lower)
    model_codes = [c for c in model_codes if len(c) >= 3 and any(char.isdigit() for char in c)]
    final_score = raw_score
    if model_codes and all(code in c_lower for code in model_codes):
        # Confirmed exact model code in candidate title
        final_score = max(final_score, max(threshold, 0.65))
        return True, final_score, "Verified Exact Model"

    is_match = final_score >= threshold
    return is_match, final_score, "Match Verified" if is_match else f"Score {final_score:.2f} < Threshold {threshold:.2f}"


def match_and_rank_candidates(
    query_title: str,
    query_image: Optional[Image.Image],
    candidates: List[WebCandidate],
    threshold: float = 0.55,
    alpha: float = 0.60,
    fusion_type: str = "grid",
    weights: Optional[Dict[str, float]] = None,
    bias: Optional[float] = None,
) -> Tuple[List[WebCandidate], List[WebCandidate]]:
    """
    Computes similarity scores, applies threshold τ and hard negative guards,
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

    # If query image is absent, perform pure text matching (gap plan D4 & Exp 2)
    has_query_image = query_image is not None
    if has_query_image:
        img_sims = compute_image_similarity(query_image, candidate_images)
        if img_sims.ndim == 0:
            img_sims = np.array([float(img_sims)])

        if fusion_type == "learned" and weights:
            w_t = weights.get("w_text", 1.0)
            w_i = weights.get("w_image", 1.0)
            b = bias or 0.0
            fused_sims = learned_fusion(text_sims, img_sims, w_t, w_i, b)
        else:
            # Linear score fusion (Experiment 4a)
            fused_sims = score_fusion(text_sims, img_sims, alpha=alpha)
    else:
        img_sims = np.zeros(len(candidates), dtype=np.float32)
        fused_sims = text_sims.copy()

    matched = []
    rejected = []

    for i, cand in enumerate(candidates):
        cand.similarity_text = float(text_sims[i])
        cand.similarity_image = float(img_sims[i]) if has_query_image else 0.0

        raw_fused = float(fused_sims[i])
        is_match, final_score, reason = verify_candidate_compatibility(
            query_title=query_title,
            candidate_title=cand.title,
            raw_score=raw_fused,
            threshold=threshold,
        )
        cand.similarity_fused = final_score
        cand.is_match = is_match
        cand.match_reason = reason

        if cand.is_match:
            matched.append(cand)
        else:
            rejected.append(cand)

    # Sort matched exact offers by price (ascending, None prices last)
    matched.sort(key=lambda x: (x.price is None, x.price if x.price is not None else 999999))
    # Sort rejected offers descending by score (hard negatives closest to threshold first)
    rejected.sort(key=lambda x: x.similarity_fused, reverse=True)

    return matched, rejected
