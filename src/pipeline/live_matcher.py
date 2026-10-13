"""
Inference Pipeline for Visual Shopping Assistant (Live Candidate Re-Ranking & Matching)
Applies the calibrated multimodal matching models to rank and filter real web candidates.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image

import torch
from sentence_transformers import SentenceTransformer
from src.config import DEFAULT_TEXT_MODEL, DEFAULT_IMAGE_MODEL, ROOT_DIR
from src.retrieval.web_search import WebCandidate
from src.models.image_encoder import ImageEncoder
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


def get_live_image_encoder(model_name: str = DEFAULT_IMAGE_MODEL) -> ImageEncoder:
    """Loads and caches ImageEncoder (EfficientNet-B0) on CPU for live inference."""
    global _CACHED_VISION_MODEL
    if _CACHED_VISION_MODEL is None:
        _CACHED_VISION_MODEL = ImageEncoder(model_name=model_name, pretrained=True, device="cpu")
    return _CACHED_VISION_MODEL


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
    candidate_images: List[Optional[Image.Image]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes visual cosine similarity between query image and candidate thumbnails
    using the academically evaluated EfficientNet-B0 image encoder.
    Returns (similarities, is_valid_mask) where is_valid_mask indicates whether
    visual similarity could be genuinely computed (both query and candidate image valid).
    """
    n = len(candidate_images)
    sims = np.zeros(n, dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    if query_image is None or n == 0:
        return sims, valid_mask

    encoder = get_live_image_encoder()

    # Encode query image
    q_emb, q_mask = encoder.encode_images([query_image], normalize=True, return_mask=True)
    if not q_mask[0]:
        return sims, valid_mask

    # Encode candidate images
    c_embs, c_mask = encoder.encode_images(candidate_images, normalize=True, return_mask=True)

    # Compute dot product (cosine similarity between L2-normalized embeddings)
    raw_dots = np.dot(c_embs, q_emb.T).squeeze()
    if raw_dots.ndim == 0:
        raw_dots = np.array([float(raw_dots)], dtype=np.float32)

    for i in range(n):
        if c_mask[i]:
            sims[i] = float(np.clip(raw_dots[i], 0.0, 1.0))
            valid_mask[i] = True

    return sims, valid_mask


def verify_candidate_compatibility(
    query_title: str,
    candidate_title: str,
    raw_score: float,
    threshold: float,
    text_sim: float = 1.0,
    modality: str = "multimodal",
) -> Tuple[bool, float, str]:
    """
    Evaluates candidate matching strictly via similarity score against decision threshold τ.
    Zero heuristic hardcoding. Explicitly notes fallback mode if image was missing.
    """
    is_match = raw_score >= threshold
    mode_label = "Multimodal" if modality == "multimodal" else "Text-Only Fallback"
    if is_match:
        reason = f"{mode_label} Match ({raw_score:.3f} ≥ τ={threshold:.2f})"
    else:
        reason = f"{mode_label} Filtered Out ({raw_score:.3f} < τ={threshold:.2f})"
    return is_match, raw_score, reason


def match_and_rank_candidates(
    query_title: str,
    query_image: Optional[Image.Image],
    candidates: List[WebCandidate],
    threshold: float = 0.75,
    alpha: float = 0.60,
    fusion_type: str = "learned",
    weights: Optional[Dict[str, float]] = None,
    bias: Optional[float] = None,
) -> Tuple[List[WebCandidate], List[WebCandidate]]:
    """
    Computes text and image similarities using the calibrated offline models,
    applies the validation-selected fusion strategy and threshold τ, and partitions
    candidates into verified exact products vs. rejected items.
    """
    if not candidates:
        return [], []

    candidate_titles = [c.title for c in candidates]
    candidate_images = [c.image for c in candidates]

    text_sims = compute_text_similarity(query_title, candidate_titles)
    if text_sims.ndim == 0:
        text_sims = np.array([float(text_sims)], dtype=np.float32)

    has_query_image = query_image is not None
    if has_query_image:
        img_sims, valid_img_mask = compute_image_similarity(query_image, candidate_images)
    else:
        img_sims = np.zeros(len(candidates), dtype=np.float32)
        valid_img_mask = np.zeros(len(candidates), dtype=bool)

    # Resolve learned parameters from final_config.json
    w_t = float(weights.get("w_text") or 9.463388336669517) if weights else 9.463388336669517
    w_i = float(weights.get("w_image") or 7.808722705879871) if weights else 7.808722705879871
    b = float(bias) if bias is not None else -8.197978989829934

    matched = []
    rejected = []

    for i, cand in enumerate(candidates):
        cand.similarity_text = float(text_sims[i])

        if valid_img_mask[i]:
            # Both text and visual representations are genuinely available
            cand.similarity_image = float(img_sims[i])
            cand.modality = "multimodal"

            if fusion_type == "learned":
                fused_score = float(learned_fusion(
                    np.array([cand.similarity_text]),
                    np.array([cand.similarity_image]),
                    w_t, w_i, b
                )[0])
            else:
                fused_score = float(score_fusion(
                    np.array([cand.similarity_text]),
                    np.array([cand.similarity_image]),
                    alpha=alpha
                )[0])
        else:
            # Visual modality unavailable (query image missing or candidate thumbnail failed)
            # Explicit text-only fallback: do NOT apply learned fusion to fabricated image scores
            cand.similarity_image = None
            cand.modality = "text_only"
            fused_score = float(cand.similarity_text)

        is_match, final_score, reason = verify_candidate_compatibility(
            query_title=query_title,
            candidate_title=cand.title,
            raw_score=fused_score,
            threshold=threshold,
            text_sim=float(cand.similarity_text),
            modality=cand.modality,
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
    # Sort rejected offers descending by score (closest to threshold first)
    rejected.sort(key=lambda x: x.similarity_fused, reverse=True)

    return matched, rejected

