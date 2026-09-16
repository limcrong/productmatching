import argparse
from pathlib import Path
import json
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
from src.retrieval.faiss_index import FAISSIndex, benchmark_faiss_efficiency
from src.retrieval.matching import apply_connected_components
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
    print("=" * 65)
    print("Running Experiment 0: TF-IDF Classical Baseline")
    print("=" * 65)

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

    candidate_thresholds = [round(x, 2) for x in np.arange(0.35, 0.85, 0.05)]
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
    print("=" * 65)
    print("Experiment 0 Test Results (Unbiased Held-Out Evaluation):")
    print(f"Mean Precision: {test_metrics['mean_precision']:.4f}")
    print(f"Mean Recall:    {test_metrics['mean_recall']:.4f}")
    print(f"Mean F1 Score:  {test_metrics['mean_f1']:.4f}")
    print("=" * 65)

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
    return test_metrics


def run_cached_evaluation(experiment_id: int):
    """
    Evaluates experiments using cached embeddings from data/embeddings/.
    """
    tracker = ExperimentTracker(EXPERIMENTS_RESULTS_JSON)
    rec = tracker.get_by_id(f"exp_{experiment_id:02d}")
    if rec:
        print(f"\nExperiment {experiment_id} record found in {EXPERIMENTS_RESULTS_JSON}:")
        print(f"  Name: {rec.get('name')}")
        print(f"  Model: {rec.get('model_name')}")
        print(f"  Validation F1: {rec.get('validation_metrics', {}).get('mean_f1', 'N/A')}")
        print(f"  Test F1:       {rec.get('test_metrics', {}).get('mean_f1', 'N/A')}")
        return rec
    else:
        print(f"Experiment {experiment_id} has not yet been executed or ingested into {EXPERIMENTS_RESULTS_JSON}.")
        print("To run Experiments 1-5, execute notebooks/kaggle_pipeline.py on Kaggle GPU and download results.json.")
        return None


def run_faiss_benchmark():
    """Benchmarks retrieval latency: brute force vs FAISS IndexFlatIP."""
    print("\n" + "=" * 65)
    print("BENCHMARK: Retrieval Latency & Computational Efficiency")
    print("=" * 65)
    
    test_df = load_data("test_demo")
    n = len(test_df)
    dim = 384
    
    # Check if real embeddings exist, otherwise generate unit vectors
    emb_path = EMBEDDINGS_DIR / "test_text_embeddings.npy"
    if emb_path.exists():
        embs = np.load(emb_path)
        print(f"Using actual cached embeddings from {emb_path} (N={len(embs)}, D={embs.shape[1]})")
    else:
        print(f"Generating synthetic L2-normalized float32 vectors for latency test (N={n}, D={dim})...")
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, dim), dtype=np.float32)
        embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        
    bench_results = benchmark_faiss_efficiency(embs, embs[:100], top_k=50)
    print("\nLatency & Throughput Results (Querying 100 listings against full demo split):")
    print(f"  - Database Size:                   {bench_results['database_size']:,} vectors")
    print(f"  - Embedding Dimension:             {bench_results['embedding_dim']}")
    print(f"  - Brute Force Latency per query:   {bench_results['brute_force_latency_ms_per_query']:.3f} ms ({bench_results['brute_force_qps']:.1f} QPS)")
    print(f"  - FAISS IndexFlatIP Latency:       {bench_results['faiss_flat_latency_ms_per_query']:.3f} ms ({bench_results['faiss_flat_qps']:.1f} QPS)")
    print(f"  - FAISS Exact Search Speedup:      {bench_results['faiss_speedup_factor']:.2f}x")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Run product matching experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="0",
        help="Experiment number (0, 1, 2, 3, 4, 5, or 'all')",
    )
    parser.add_argument(
        "--benchmark-faiss",
        action="store_true",
        help="Run FAISS latency and efficiency benchmark",
    )
    args = parser.parse_args()

    if args.benchmark_faiss:
        run_faiss_benchmark()
        return

    if args.experiment == "0":
        run_experiment_0_tfidf()
    elif args.experiment in ["1", "2", "3", "4", "5"]:
        run_cached_evaluation(int(args.experiment))
    elif args.experiment == "all":
        run_experiment_0_tfidf()
        for i in range(1, 6):
            run_cached_evaluation(i)
    else:
        print(f"Unknown experiment argument: {args.experiment}")


if __name__ == "__main__":
    main()
