import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
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


def benchmark_faiss_efficiency(
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int = 50,
) -> Dict[str, float]:
    """
    Hypothesis 5C: Retrieval Latency & Efficiency Benchmark.
    Separates matching quality (F1) from computational efficiency:
    Compares exact brute-force dot product against faiss.IndexFlatIP.
    """
    n_queries = len(query_embeddings)
    dim = embeddings.shape[1]

    # 1. Brute-force dot product
    start_bf = time.perf_counter()
    bf_sims = np.dot(query_embeddings, embeddings.T)
    # top-k argpartition
    bf_indices = np.argpartition(-bf_sims, top_k, axis=1)[:, :top_k]
    bf_time = time.perf_counter() - start_bf
    bf_latency_ms = (bf_time / n_queries) * 1000.0
    bf_qps = n_queries / bf_time if bf_time > 0 else 0.0

    # 2. FAISS IndexFlatIP (exact search)
    if faiss is not None:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        
        start_faiss = time.perf_counter()
        faiss_distances, faiss_indices = index.search(
            query_embeddings.astype(np.float32), top_k
        )
        faiss_time = time.perf_counter() - start_faiss
        faiss_latency_ms = (faiss_time / n_queries) * 1000.0
        faiss_qps = n_queries / faiss_time if faiss_time > 0 else 0.0
        speedup = bf_time / faiss_time if faiss_time > 0 else 1.0
    else:
        faiss_latency_ms = 0.0
        faiss_qps = 0.0
        speedup = 1.0

    return {
        "num_queries": n_queries,
        "database_size": len(embeddings),
        "embedding_dim": dim,
        "brute_force_latency_ms_per_query": round(bf_latency_ms, 3),
        "brute_force_qps": round(bf_qps, 1),
        "faiss_flat_latency_ms_per_query": round(faiss_latency_ms, 3),
        "faiss_flat_qps": round(faiss_qps, 1),
        "faiss_speedup_factor": round(speedup, 2),
    }
