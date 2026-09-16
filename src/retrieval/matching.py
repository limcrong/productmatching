from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class CandidateMatch:
    posting_id: str
    title: str
    image: str
    label_group: int
    score: float
    status: str  # "Correct Match", "False Match", "Correct Rejection", "Missed Match"


@dataclass
class MatchingResults:
    matched_products: List[CandidateMatch]  # score >= threshold
    similar_rejected: List[CandidateMatch]  # score < threshold & diff group (hard negatives)
    missed_matches: List[CandidateMatch]  # same group & score < threshold
    metrics: Dict[str, float]  # precision, recall, f1


def apply_threshold(
    candidate_ids: List[str],
    candidate_scores: List[float],
    threshold: float,
) -> List[str]:
    """Filters candidate posting_ids with similarity >= threshold."""
    return [
        pid
        for pid, score in zip(candidate_ids, candidate_scores)
        if score >= threshold
    ]


def apply_reciprocal_nn(
    query_pid: str,
    retrieved_pids: List[str],
    all_pids: List[str],
    sim_matrix: np.ndarray,
    threshold: float,
    pid_to_idx: Dict[str, int],
) -> List[str]:
    """
    Hypothesis 5A: Reciprocal Nearest Neighbors.
    Requires mutual agreement above threshold:
    Listing B is accepted as a match for A only if:
    Sim(A, B) >= threshold AND Sim(B, A) >= threshold.
    In symmetric similarity matrices, this is equivalent to standard thresholding,
    but in k-NN top-k candidate lists, it enforces that A in TopK(B) and B in TopK(A).
    """
    q_idx = pid_to_idx.get(query_pid)
    if q_idx is None:
        return [query_pid]

    filtered = []
    for cand_pid in retrieved_pids:
        if cand_pid == query_pid:
            filtered.append(cand_pid)
            continue
        c_idx = pid_to_idx.get(cand_pid)
        if c_idx is not None:
            sim_ab = sim_matrix[q_idx, c_idx]
            sim_ba = sim_matrix[c_idx, q_idx]
            if sim_ab >= threshold and sim_ba >= threshold:
                filtered.append(cand_pid)
    return filtered


def apply_connected_components(
    predictions: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Hypothesis 5B: Transitive Graph Grouping (Connected Components).
    If A matches B and B matches C, merges {A, B, C} into a single connected component.
    Warning: May cause false-positive chaining if threshold is too low.
    """
    # Build adjacency list
    adj: Dict[str, Set[str]] = {}
    for pid, neighbors in predictions.items():
        if pid not in adj:
            adj[pid] = set()
        adj[pid].add(pid)
        for n in neighbors:
            if n not in adj:
                adj[n] = set()
            adj[pid].add(n)
            adj[n].add(pid)

    # Find connected components via BFS
    visited = set()
    components: Dict[str, List[str]] = {}

    for node in adj:
        if node not in visited:
            component = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            # Assign component to all members
            for member in component:
                components[member] = component

    # Ensure all original keys are preserved
    for pid in predictions:
        if pid not in components:
            components[pid] = [pid]

    return components


def categorize_and_filter_candidates(
    query_posting_id: str,
    query_label_group: int,
    retrieved_df: pd.DataFrame,
    ground_truth_postings: Set[str],
    threshold: float,
    max_rejected_to_show: int = 6,
) -> MatchingResults:
    """
    Categorizes retrieved candidates relative to the given similarity threshold:
    
    1. MATCHED PRODUCTS (score >= threshold):
       - Correct Match: same label_group
       - False Match: different label_group (false positive)
       
    2. SIMILAR BUT REJECTED (score < threshold, different label_group):
       - Correct Rejection: hard negatives successfully filtered out
       
    3. MISSED TRUE MATCHES (same label_group, but score < threshold or not retrieved):
       - Missed Match: false negative
       
    Provides live academic inspection of the precision-recall trade-off as threshold changes.
    """
    matched = []
    similar_rejected = []
    missed_retrieved = []
    retrieved_pids = set()

    for _, row in retrieved_df.iterrows():
        pid = row["posting_id"]
        retrieved_pids.add(pid)
        score = float(row["score"])
        is_true_match = (row["label_group"] == query_label_group)

        if score >= threshold:
            status = "Correct Match" if is_true_match else "False Match"
            matched.append(
                CandidateMatch(
                    posting_id=pid,
                    title=row["title"],
                    image=row["image"],
                    label_group=row["label_group"],
                    score=score,
                    status=status,
                )
            )
        else:
            if is_true_match:
                missed_retrieved.append(
                    CandidateMatch(
                        posting_id=pid,
                        title=row["title"],
                        image=row["image"],
                        label_group=row["label_group"],
                        score=score,
                        status="Missed Match",
                    )
                )
            else:
                similar_rejected.append(
                    CandidateMatch(
                        posting_id=pid,
                        title=row["title"],
                        image=row["image"],
                        label_group=row["label_group"],
                        score=score,
                        status="Correct Rejection",
                    )
                )

    # Sort similar_rejected descending by score (highest-scoring hard negatives immediately below threshold)
    similar_rejected.sort(key=lambda x: x.score, reverse=True)
    similar_rejected = similar_rejected[:max_rejected_to_show]

    missed_matches = missed_retrieved

    # Compute per-query precision, recall, F1
    predicted_pids = {m.posting_id for m in matched}
    if query_posting_id not in predicted_pids and any(m.score >= threshold for m in matched):
        predicted_pids.add(query_posting_id)

    true_pids = set(ground_truth_postings)
    intersection = len(predicted_pids.intersection(true_pids))
    
    precision = intersection / len(predicted_pids) if predicted_pids else 0.0
    recall = intersection / len(true_pids) if true_pids else 0.0
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return MatchingResults(
        matched_products=matched,
        similar_rejected=similar_rejected,
        missed_matches=missed_matches,
        metrics={
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp_count": intersection,
            "fp_count": len(predicted_pids) - intersection,
            "fn_count": len(true_pids) - intersection,
        },
    )
