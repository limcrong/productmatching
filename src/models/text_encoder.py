import pickle
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from src.config import DEFAULT_TEXT_MODEL


class TfidfEncoder:
    """
    Classical NLP baseline encoder using scikit-learn's TfidfVectorizer.
    Operates on sparse matrices with direct cosine similarity computation.
    """

    def __init__(
        self,
        max_features: int = 25000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words="english",
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Fits the TF-IDF vocabulary on the training split only."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """Transforms text into normalized sparse TF-IDF vectors."""
        if not self.is_fitted:
            raise ValueError("TfidfEncoder must be fitted before transform")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return matrix

    def compute_similarity(
        self, query_vec: csr_matrix, candidate_matrix: csr_matrix
    ) -> np.ndarray:
        """
        Computes sparse cosine similarities between query vector and candidate listings.
        """
        return cosine_similarity(query_vec, candidate_matrix)[0]

    def save(self, filepath: Union[str, Path]):
        with open(filepath, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, filepath: Union[str, Path]):
        with open(filepath, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        return self


class SentenceTransformerEncoder:
    """
    Deep learning text encoder using SentenceTransformers.
    Outputs L2-normalized dense embeddings for fast inner-product (cosine) search.
    """

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str = "cpu"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed")
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Encodes a list of texts into dense numpy float32 embeddings.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
