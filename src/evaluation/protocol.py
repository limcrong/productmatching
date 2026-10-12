import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from src.evaluation.metrics import calculate_query_metrics, evaluate_predictions

try:
    import faiss
except ImportError:
    faiss = None


def search_top_k_dense(
    query_embeddings: np.ndarray,
    index_embeddings: np.ndarray,
    top_k: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves Top-K nearest neighbors using inner product (cosine similarity on L2-normalized embeddings).
    Uses FAISS if available, otherwise chunked numpy dot product.
    Returns:
        scores: (N, top_k) float32
        indices: (N, top_k) int64
    """
    n_queries = len(query_embeddings)
    top_k = min(top_k, len(index_embeddings))

    q_emb = query_embeddings.astype(np.float32)
    idx_emb = index_embeddings.astype(np.float32)

    if faiss is not None:
        index = faiss.IndexFlatIP(idx_emb.shape[1])
        index.add(idx_emb)
        scores, indices = index.search(q_emb, top_k)
        return scores, indices

    # Fallback to numpy
    scores_list = []
    indices_list = []
    chunk_size = 512
    for i in range(0, n_queries, chunk_size):
        chunk = q_emb[i : i + chunk_size]
        sims = np.dot(chunk, idx_emb.T)
        part_idx = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
        row_indices = np.arange(chunk.shape[0])[:, None]
        part_scores = sims[row_indices, part_idx]
        sort_order = np.argsort(-part_scores, axis=1)
        sorted_indices = part_idx[row_indices, sort_order]
        sorted_scores = part_scores[row_indices, sort_order]
        scores_list.append(sorted_scores)
        indices_list.append(sorted_indices)

    return np.vstack(scores_list), np.vstack(indices_list)


def evaluate_top_k_candidates(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    posting_ids: List[str],
    ground_truth_map: Dict[str, List[str]],
    threshold: float,
) -> Dict[str, float]:
    """
    Given precomputed Top-K candidates (indices and scores) for each posting,
    filters by threshold, ensures self-match, and calculates P/R/F1.
    """
    predictions = {}
    for i, pid in enumerate(posting_ids):
        valid_mask = candidate_scores[i] >= threshold
        matched_idx = candidate_indices[i][valid_mask]
        matched_pids = [posting_ids[idx] for idx in matched_idx]
        if pid not in matched_pids:
            matched_pids.append(pid)
        predictions[pid] = matched_pids

    return evaluate_predictions(predictions, ground_truth_map)


def find_best_threshold_from_top_k(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    posting_ids: List[str],
    ground_truth_map: Dict[str, List[str]],
    thresholds: List[float],
) -> Tuple[float, Dict[str, float], List[Dict]]:
    """
    Fast threshold sweep on precomputed Top-K candidates.
    """
    best_th = thresholds[0]
    best_f1 = -1.0
    best_metrics = {}
    history = []

    for th in thresholds:
        metrics = evaluate_top_k_candidates(
            candidate_indices, candidate_scores, posting_ids, ground_truth_map, th
        )
        history.append({
            "threshold": th,
            "mean_precision": metrics["mean_precision"],
            "mean_recall": metrics["mean_recall"],
            "mean_f1": metrics["mean_f1"],
        })
        if metrics["mean_f1"] > best_f1:
            best_f1 = metrics["mean_f1"]
            best_th = th
            best_metrics = metrics

    return best_th, best_metrics, history
