import numpy as np
from typing import Optional, Tuple


def normalize_vectors(v: np.ndarray) -> np.ndarray:
    """L2 normalizes rows of a 2D numpy array."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def concat_fusion(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Early fusion with modality weighting:
    Combined = [sqrt(alpha) * text_emb || sqrt(1 - alpha) * image_emb]
    When both text_emb and image_emb are unit-normalized, the inner product of
    this concatenated representation equals the convex combination of individual similarities:
    <e_fused1, e_fused2> = alpha * <text1, text2> + (1 - alpha) * <img1, img2>.
    """
    assert 0.0 <= alpha <= 1.0, f"alpha must be between 0 and 1, got {alpha}"
    assert len(text_embeddings) == len(
        image_embeddings
    ), "Embedding counts must match"

    w_text = np.sqrt(alpha)
    w_image = np.sqrt(1.0 - alpha)

    scaled_text = text_embeddings * w_text
    scaled_image = image_embeddings * w_image

    concatenated = np.hstack([scaled_text, scaled_image])
    return normalize_vectors(concatenated).astype(np.float32)


def score_fusion(
    text_similarities: np.ndarray,
    image_similarities: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Late fusion: weighted linear combination of similarity scores.
    """
    return alpha * text_similarities + (1.0 - alpha) * image_similarities


def learned_fusion(
    text_similarities: np.ndarray,
    image_similarities: np.ndarray,
    w_text: float,
    w_image: float,
    bias: float,
) -> np.ndarray:
    """
    Supervised late fusion using learned logistic regression parameters:
    P(match) = sigmoid(w_text * s_text + w_image * s_image + bias)
    """
    logits = w_text * text_similarities + w_image * image_similarities + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))


class MultimodalFusion:
    """
    Unified manager for multimodal fusion strategies.
    """

    def __init__(self, default_alpha: float = 0.6):
        self.default_alpha = default_alpha

    def fuse_embeddings(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        current_alpha = self.default_alpha if alpha is None else alpha
        return concat_fusion(text_embeddings, image_embeddings, current_alpha)

    def fuse_scores(
        self,
        text_sims: np.ndarray,
        image_sims: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        current_alpha = self.default_alpha if alpha is None else alpha
        return score_fusion(text_sims, image_sims, current_alpha)

    def fuse_learned(
        self,
        text_sims: np.ndarray,
        image_sims: np.ndarray,
        w_text: float,
        w_image: float,
        bias: float,
    ) -> np.ndarray:
        return learned_fusion(text_sims, image_sims, w_text, w_image, bias)
