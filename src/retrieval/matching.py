from dataclasses import dataclass
from typing import Dict, List, Optional, Set
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


def categorize_and_filter_candidates(
    query_posting_id: str,
    query_label_group: int,
    retrieved_df: pd.DataFrame,
    ground_truth_postings: Set[str],
    threshold: float,
    max_rejected_to_show: int = 6,
    full_df: Optional[pd.DataFrame] = None,
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

    # Include true matches from ground_truth that were not in the retrieved set at all (score 0.0)
    missed_matches = list(missed_retrieved)
    unretrieved_pids = set(ground_truth_postings) - retrieved_pids - {query_posting_id}
    if full_df is not None and unretrieved_pids:
        unretrieved_rows = full_df[full_df["posting_id"].isin(unretrieved_pids)]
        for _, row in unretrieved_rows.iterrows():
            missed_matches.append(
                CandidateMatch(
                    posting_id=row["posting_id"],
                    title=row["title"],
                    image=row["image"],
                    label_group=int(row["label_group"]),
                    score=0.0,
                    status="Missed Match (Below Top-K)",
                )
            )

    # Compute per-query precision, recall, F1
    # Predicted set includes all items in matched_products
    predicted_pids = {m.posting_id for m in matched}
    # If the query itself was not retrieved, add it (self-match)
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
