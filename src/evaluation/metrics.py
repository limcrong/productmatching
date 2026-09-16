import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Union, Callable


def calculate_query_metrics(
    predicted_postings: Union[Set[str], List[str]],
    true_postings: Union[Set[str], List[str]],
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1 for a single product query.
    Both predicted and true sets typically include the query product itself.
    """
    pred_set = set(predicted_postings)
    true_set = set(true_postings)
    
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    intersection = len(pred_set.intersection(true_set))
    precision = intersection / len(pred_set)
    recall = intersection / len(true_set) if true_set else 0.0
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp_count": intersection,
        "fp_count": len(pred_set) - intersection,
        "fn_count": len(true_set) - intersection,
    }


def evaluate_predictions(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Evaluates global mean Precision, Recall, and F1 across all query listings.
    
    predictions: {posting_id: [matched_posting_ids]}
    ground_truth: {posting_id: [true_matched_posting_ids]}
    """
    precisions = []
    recalls = []
    f1s = []
    
    for posting_id, true_matches in ground_truth.items():
        preds = predictions.get(posting_id, [posting_id])
        metrics = calculate_query_metrics(preds, true_matches)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])
        
    return {
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_f1": float(np.mean(f1s)),
        "num_queries": len(ground_truth),
    }


def find_best_threshold(
    eval_func_at_threshold: Callable[[float], Dict[str, float]],
    thresholds: List[float],
) -> Tuple[float, Dict[str, float], List[Dict]]:
    """
    Grid-searches across candidate thresholds to maximize validation F1 score.
    Returns (best_threshold, best_metrics, history).
    """
    best_threshold = thresholds[0]
    best_f1 = -1.0
    best_metrics = {}
    history = []
    
    for th in thresholds:
        metrics = eval_func_at_threshold(th)
        current_f1 = metrics["mean_f1"]
        history.append({
            "threshold": th,
            "mean_precision": metrics["mean_precision"],
            "mean_recall": metrics["mean_recall"],
            "mean_f1": current_f1,
        })
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = th
            best_metrics = metrics
            
    return best_threshold, best_metrics, history
