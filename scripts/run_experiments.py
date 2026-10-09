import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_DEMO_SPLIT_CSV,
    EMBEDDINGS_DIR,
    EXPERIMENTS_RESULTS_JSON,
)
from src.data.loader import load_data, get_ground_truth_map
from src.models.text_encoder import TfidfEncoder
from src.evaluation.metrics import (
    evaluate_predictions,
    find_best_threshold,
)
from src.evaluation.experiment_tracker import ExperimentTracker


def run_experiment_0_tfidf():
    """
    Experiment 0: TF-IDF Classical Baseline.
    - Fits vocabulary on Train split only
    - Searches for optimal threshold on Validation split
    - Reports unbiased final performance on Test/Demo split (unseen product groups)
    - Saves result to experiments/results.json
    """
    print("=" * 60)
    print("Running Experiment 0: TF-IDF Classical Baseline")
    print("=" * 60)

    print("Loading train, val, and test splits...")
    train_df = load_data("train")
    val_df = load_data("val")
    test_df = load_data("test_demo")

    val_gt = get_ground_truth_map(val_df)
    test_gt = get_ground_truth_map(test_df)

    print(f"Fitting TfidfEncoder on {len(train_df)} training product titles...")
    encoder = TfidfEncoder(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
    encoder.fit(train_df["title"].tolist())

    print("Transforming validation and test product titles...")
    val_matrix = encoder.transform(val_df["title"].tolist())
    test_matrix = encoder.transform(test_df["title"].tolist())

    print("Precomputing validation cosine similarities...")
    val_sims = val_matrix.dot(val_matrix.T).toarray()

    def eval_val_at_threshold(th: float):
        preds = {}
        for i, pid in enumerate(val_df["posting_id"]):
            matched_indices = np.where(val_sims[i] >= th)[0]
            matched_pids = val_df.iloc[matched_indices]["posting_id"].tolist()
            if pid not in matched_pids:
                matched_pids.append(pid)
            preds[pid] = matched_pids
        return evaluate_predictions(preds, val_gt)

    candidate_thresholds = [round(x, 2) for x in np.arange(0.40, 0.85, 0.05)]
    print(f"Tuning threshold on validation set across {candidate_thresholds}...")
    best_th, best_val_metrics, _ = find_best_threshold(
        eval_val_at_threshold, candidate_thresholds
    )
    print(
        f"Optimal threshold on Validation Split: {best_th:.2f} "
        f"(Val F1: {best_val_metrics['mean_f1']:.4f})"
    )

    print("Evaluating on held-out Test/Demo Split (unseen product groups)...")
    test_sims = test_matrix.dot(test_matrix.T).toarray()
    test_preds = {}
    for i, pid in enumerate(test_df["posting_id"]):
        matched_indices = np.where(test_sims[i] >= best_th)[0]
        matched_pids = test_df.iloc[matched_indices]["posting_id"].tolist()
        if pid not in matched_pids:
            matched_pids.append(pid)
        test_preds[pid] = matched_pids

    test_metrics = evaluate_predictions(test_preds, test_gt)
    print("=" * 60)
    print("Experiment 0 Test Results (Unbiased Held-Out Evaluation):")
    print(f"Mean Precision: {test_metrics['mean_precision']:.4f}")
    print(f"Mean Recall:    {test_metrics['mean_recall']:.4f}")
    print(f"Mean F1 Score:  {test_metrics['mean_f1']:.4f}")
    print("=" * 60)

    tracker = ExperimentTracker(EXPERIMENTS_RESULTS_JSON)
    tracker.log_experiment(
        experiment_id="exp_00",
        name="Experiment 0 — TF-IDF Classical Baseline",
        approach_type="Classical ML",
        model_name="TfidfVectorizer (bigrams, sublinear TF)",
        hyperparameters={"threshold": best_th, "max_features": 25000},
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        notes="Classical NLP baseline using sparse cosine similarity. Vocabulary fitted strictly on train split.",
    )
    print(f"Saved benchmark to {EXPERIMENTS_RESULTS_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Run product matching experiments")
    parser.add_argument(
        "--experiment",
        type=int,
        default=0,
        help="Experiment number to run (0: TF-IDF, 1: Text DL, etc.)",
    )
    args = parser.parse_args()

    if args.experiment == 0:
        run_experiment_0_tfidf()
    else:
        print(f"Experiment {args.experiment} is configured for GPU execution (Kaggle/Colab).")


if __name__ == "__main__":
    main()
