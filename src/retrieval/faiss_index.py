from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class FAISSIndex:
    """
    Wrapper around FAISS for fast cosine similarity / inner product search.
    Assumes embeddings are L2-normalized float32 vectors.
    """

    def __init__(self, d: int, index_type: str = "flat"):
        if faiss is None:
            raise ImportError("faiss-cpu or faiss-gpu is required for FAISSIndex")

        self.d = d
        self.index_type = index_type

        if index_type == "flat":
            # Inner Product on normalized vectors = Cosine Similarity
            self.index = faiss.IndexFlatIP(d)
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

    def add(self, embeddings: np.ndarray):
        """Adds float32 embeddings (N, D) to the index."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)

    def search(
        self, query_embeddings: np.ndarray, top_k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the nearest neighbors.
        Returns:
            distances: (N, top_k) cosine similarity scores
            indices: (N, top_k) candidate indices in the index
        """
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        if len(query_embeddings.shape) == 1:
            query_embeddings = np.expand_dims(query_embeddings, axis=0)

        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def save(self, filepath: Union[str, Path]):
        faiss.write_index(self.index, str(filepath))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FAISSIndex":
        raw_index = faiss.read_index(str(filepath))
        instance = cls(d=raw_index.d, index_type="flat")
        instance.index = raw_index
        return instance
