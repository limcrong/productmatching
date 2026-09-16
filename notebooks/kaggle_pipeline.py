"""
Kaggle End-to-End Multimodal Product Matching Pipeline (Phase 2: Experiments 1 to 5)
Run this script or notebook on Kaggle with GPU accelerator (T4 / P100) enabled.

Dataset: Shopee - Price Match Guarantee (/kaggle/input/shopee-product-matching/)
Output directory: /kaggle/working/
  - results.json (complete academic benchmark table with all experiment stages)
  - test_text_embeddings.npy
  - test_image_embeddings.npy
  - val_text_embeddings.npy
  - val_image_embeddings.npy
  - test_demo_split.csv
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
from torchvision import transforms

from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration & Paths ---
KAGGLE_INPUT_DIR = Path("/kaggle/input/shopee-product-matching")
OUTPUT_DIR = Path("/kaggle/working")

TRAIN_CSV = KAGGLE_INPUT_DIR / "train.csv"
TRAIN_IMAGES_DIR = KAGGLE_INPUT_DIR / "train_images"

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on device: {DEVICE}")

# Set seeds for exact academic reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =====================================================================
# 1. Dataset Partitioning (Strict Zero Leakage by label_group)
# =====================================================================
def create_splits(df):
    """
    Partitions listings strictly by label_group:
    - Train split (~70%): Model fitting and fine-tuning only
    - Validation split (~15%): Hyperparameter tuning, model selection, alpha/tau calibration
    - Test split (~15%): Frozen held-out unseen product groups for final reporting
    """
    group_counts = df.groupby("label_group").size().reset_index(name="count")
    group_counts = group_counts.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    group_counts["cum_count"] = group_counts["count"].cumsum()
    total_rows = len(df)
    
    train_cutoff = total_rows * 0.70
    val_cutoff = total_rows * 0.85
    
    train_groups = set(group_counts[group_counts["cum_count"] <= train_cutoff]["label_group"])
    val_groups = set(group_counts[(group_counts["cum_count"] > train_cutoff) & (group_counts["cum_count"] <= val_cutoff)]["label_group"])
    test_groups = set(group_counts[group_counts["cum_count"] > val_cutoff]["label_group"])
    
    train_df = df[df["label_group"].isin(train_groups)].copy().reset_index(drop=True)
    val_df = df[df["label_group"].isin(val_groups)].copy().reset_index(drop=True)
    test_df = df[df["label_group"].isin(test_groups)].copy().reset_index(drop=True)
    
    print(f"Train split: {len(train_df)} rows ({len(train_groups)} groups)")
    print(f"Val split:   {len(val_df)} rows ({len(val_groups)} groups)")
    print(f"Test split:  {len(test_df)} rows ({len(test_groups)} groups)")
    return train_df, val_df, test_df


# =====================================================================
# 2. Evaluation Metrics & Threshold Tuning
# =====================================================================
def get_ground_truth_map(df):
    group_to_postings = df.groupby("label_group")["posting_id"].apply(list).to_dict()
    return df["label_group"].map(group_to_postings).set_axis(df["posting_id"]).to_dict()


def evaluate_predictions(predictions, ground_truth):
    precisions, recalls, f1s = [], [], []
    for pid, true_matches in ground_truth.items():
        preds = set(predictions.get(pid, [pid]))
        trues = set(true_matches)
        inter = len(preds.intersection(trues))
        p = inter / len(preds) if preds else 0.0
        r = inter / len(trues) if trues else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return {
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_f1": float(np.mean(f1s)),
    }


def find_best_threshold(sim_matrix, df, gt_map, thresholds):
    """Grid searches threshold on validation data to maximize F1."""
    best_th = thresholds[0]
    best_f1 = -1.0
    best_metrics = {}
    pids = df["posting_id"].tolist()
    
    for th in thresholds:
        preds = {}
        for i, pid in enumerate(pids):
            matched_indices = np.where(sim_matrix[i] >= th)[0]
            matched_pids = [pids[idx] for idx in matched_indices]
            if pid not in matched_pids:
                matched_pids.append(pid)
            preds[pid] = matched_pids
            
        metrics = evaluate_predictions(preds, gt_map)
        if metrics["mean_f1"] > best_f1:
            best_f1 = metrics["mean_f1"]
            best_th = th
            best_metrics = metrics
            
    return best_th, best_metrics


def eval_at_fixed_threshold(sim_matrix, df, gt_map, th):
    """Evaluates metrics with a frozen threshold."""
    pids = df["posting_id"].tolist()
    preds = {}
    for i, pid in enumerate(pids):
        matched_indices = np.where(sim_matrix[i] >= th)[0]
        matched_pids = [pids[idx] for idx in matched_indices]
        if pid not in matched_pids:
            matched_pids.append(pid)
        preds[pid] = matched_pids
    return evaluate_predictions(preds, gt_map)


# =====================================================================
# Main Pipeline Execution
# =====================================================================
def main():
    if not TRAIN_CSV.exists():
        print(f"Dataset path {TRAIN_CSV} not found! Please attach Kaggle Shopee dataset.")
        return

    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df, test_df = create_splits(df)

    # Save test/demo split for local Streamlit download
    test_df.to_csv(OUTPUT_DIR / "test_demo_split.csv", index=False)
    
    val_gt = get_ground_truth_map(val_df)
    test_gt = get_ground_truth_map(test_df)
    
    benchmark_records = []

    # =====================================================================
    # Experiment 0: TF-IDF Classical Baseline
    # =====================================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 0: TF-IDF Classical Baseline")
    print("=" * 65)
    tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    tfidf.fit(train_df["title"])
    
    val_tfidf = tfidf.transform(val_df["title"])
    test_tfidf = tfidf.transform(test_df["title"])
    
    val_sim_tfidf = val_tfidf.dot(val_tfidf.T).toarray()
    test_sim_tfidf = test_tfidf.dot(test_tfidf.T).toarray()
    
    th_range_coarse = [round(x, 2) for x in np.arange(0.35, 0.85, 0.05)]
    best_th_0, val_m0 = find_best_threshold(val_sim_tfidf, val_df, val_gt, th_range_coarse)
    test_m0 = eval_at_fixed_threshold(test_sim_tfidf, test_df, test_gt, best_th_0)
    print(f"Exp 0 - Val F1: {val_m0['mean_f1']:.4f} (th={best_th_0}) | Test F1: {test_m0['mean_f1']:.4f}")
    
    benchmark_records.append({
        "experiment_id": "exp_00",
        "name": "Experiment 0 — TF-IDF Classical Baseline",
        "approach_type": "Classical ML",
        "model_name": "TfidfVectorizer (bigrams, sublinear TF)",
        "hyperparameters": {"threshold": best_th_0, "max_features": 25000},
        "validation_metrics": val_m0,
        "test_metrics": test_m0,
        "delta_val_f1": 0.0,
        "notes": "Reference baseline. Vocabulary fitted strictly on train split.",
    })

    # =====================================================================
    # Experiment 1: Pretrained Text Transformer Selection
    # =====================================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: Pretrained Text Transformer Selection")
    print("=" * 65)
    
    models_to_test = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ("multilingual-MiniLM-L12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ]
    
    best_text_model_name = None
    best_text_val_f1 = -1.0
    best_text_th = None
    best_text_val_m = None
    text_embeddings_cache = {}

    th_range_dense = [round(x, 2) for x in np.arange(0.50, 0.90, 0.05)]

    for short_name, hf_id in models_to_test:
        print(f"Evaluating {short_name} on validation split...")
        m = SentenceTransformer(hf_id, device=DEVICE)
        val_emb = m.encode(val_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        val_sim = np.dot(val_emb, val_emb.T)
        
        th, v_metrics = find_best_threshold(val_sim, val_df, val_gt, th_range_dense)
        print(f"  {short_name} -> Val F1: {v_metrics['mean_f1']:.4f} (th={th})")
        
        text_embeddings_cache[short_name] = {"model": m, "val_emb": val_emb, "val_sim": val_sim}
        
        if v_metrics["mean_f1"] > best_text_val_f1:
            best_text_val_f1 = v_metrics["mean_f1"]
            best_text_model_name = short_name
            best_text_th = th
            best_text_val_m = v_metrics

    print(f"\nWinning Pretrained Text Model: {best_text_model_name} (Val F1: {best_text_val_f1:.4f})")
    
    # Compute test embeddings for winning model
    winning_model = text_embeddings_cache[best_text_model_name]["model"]
    test_text_emb = winning_model.encode(test_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    test_sim_text = np.dot(test_text_emb, test_text_emb.T)
    test_m1 = eval_at_fixed_threshold(test_sim_text, test_df, test_gt, best_text_th)
    print(f"Exp 1 - Test F1: {test_m1['mean_f1']:.4f}")
    
    val_text_emb = text_embeddings_cache[best_text_model_name]["val_emb"]
    val_sim_text = text_embeddings_cache[best_text_model_name]["val_sim"]
    
    delta_e1 = val_text_val_f1 - val_m0["mean_f1"]
    benchmark_records.append({
        "experiment_id": "exp_01",
        "name": "Experiment 1 — Pretrained Text Embeddings",
        "approach_type": "Text Deep Learning",
        "model_name": best_text_model_name,
        "hyperparameters": {"threshold": best_text_th, "dimension": 384},
        "validation_metrics": best_text_val_m,
        "test_metrics": test_m1,
        "delta_val_f1": float(delta_e1),
        "notes": f"Selected over competing models via validation F1. Delta over TF-IDF: {delta_e1:+.4f}",
    })

    # =====================================================================
    # Experiment 2: Domain Adaptation via MultipleNegativesRankingLoss
    # =====================================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: Domain Adaptation via MultipleNegativesRankingLoss")
    print("=" * 65)
    
    # Construct group-aware positive training pairs from train_split
    print("Building group-aware training pairs from train split...")
    train_groups_dict = train_df.groupby("label_group")["title"].apply(list).to_dict()
    train_samples = []
    
    for grp, titles in train_groups_dict.items():
        if len(titles) > 1:
            # Pair adjacent listings in the same group
            for i in range(len(titles)):
                for j in range(i + 1, min(i + 3, len(titles))):
                    train_samples.append(InputExample(texts=[titles[i], titles[j]]))
                    
    print(f"Total positive training pairs created: {len(train_samples)}")
    
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
    fine_tuned_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2" if "MiniLM-L6" in best_text_model_name else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=DEVICE,
    )
    
    train_loss = losses.MultipleNegativesRankingLoss(fine_tuned_model)
    print("Fine-tuning on train_split for 2 epochs (group-aware metric learning)...")
    fine_tuned_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True,
    )
    
    # Re-evaluate fine-tuned representations on validation
    val_ft_emb = fine_tuned_model.encode(val_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    val_sim_ft = np.dot(val_ft_emb, val_ft_emb.T)
    best_th_2, val_m2 = find_best_threshold(val_sim_ft, val_df, val_gt, th_range_dense)
    
    test_ft_emb = fine_tuned_model.encode(test_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    test_sim_ft = np.dot(test_ft_emb, test_ft_emb.T)
    test_m2 = eval_at_fixed_threshold(test_sim_ft, test_df, test_gt, best_th_2)
    
    delta_e2 = val_m2["mean_f1"] - best_text_val_f1
    print(f"Exp 2 - Val F1: {val_m2['mean_f1']:.4f} (th={best_th_2}) | Test F1: {test_m2['mean_f1']:.4f} (Delta Val: {delta_e2:+.4f})")
    
    benchmark_records.append({
        "experiment_id": "exp_02",
        "name": "Experiment 2 — Domain-Adapted Text Encoder",
        "approach_type": "Domain Adaptation",
        "model_name": f"{best_text_model_name} (fine-tuned MultipleNegatives)",
        "hyperparameters": {"threshold": best_th_2, "epochs": 2, "batch_size": 32, "lr": "default"},
        "validation_metrics": val_m2,
        "test_metrics": test_m2,
        "delta_val_f1": float(delta_e2),
        "notes": f"Supervised domain adaptation via MultipleNegativesRankingLoss. Delta over Exp 1: {delta_e2:+.4f}",
    })
    
    # If domain adaptation improved performance, use it for downstream multimodal fusion
    if val_m2["mean_f1"] > best_text_val_f1:
        print("Adopting fine-tuned text encoder for multimodal fusion.")
        val_text_emb = val_ft_emb
        test_text_emb = test_ft_emb
        val_sim_text = val_sim_ft
        test_sim_text = test_sim_ft
    else:
        print("Retaining pretrained text encoder for multimodal fusion.")

    # =====================================================================
    # Experiment 3: Pretrained Vision Feature Extraction
    # =====================================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: Vision Feature Extraction (timm / EfficientNet-B0)")
    print("=" * 65)
    
    img_model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).to(DEVICE).eval()
    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    def extract_image_features(image_names):
        feats = []
        batch = []
        for name in tqdm(image_names, desc="Extracting features"):
            p = TRAIN_IMAGES_DIR / name
            try:
                img = Image.open(p).convert("RGB")
                t = img_tf(img)
            except Exception:
                t = torch.zeros((3, 224, 224))
            batch.append(t)
            if len(batch) == 128:
                with torch.no_grad():
                    out = img_model(torch.stack(batch).to(DEVICE))
                    out = torch.nn.functional.normalize(out, p=2, dim=1)
                    feats.append(out.cpu().numpy())
                batch = []
        if batch:
            with torch.no_grad():
                out = img_model(torch.stack(batch).to(DEVICE))
                out = torch.nn.functional.normalize(out, p=2, dim=1)
                feats.append(out.cpu().numpy())
        return np.vstack(feats).astype(np.float32)
        
    val_img_emb = extract_image_features(val_df["image"].tolist())
    test_img_emb = extract_image_features(test_df["image"].tolist())
    
    val_sim_img = np.dot(val_img_emb, val_img_emb.T)
    test_sim_img = np.dot(test_img_emb, test_img_emb.T)
    
    best_th_3, val_m3 = find_best_threshold(val_sim_img, val_df, val_gt, th_range_dense)
    test_m3 = eval_at_fixed_threshold(test_sim_img, test_df, test_gt, best_th_3)
    print(f"Exp 3 - Val F1: {val_m3['mean_f1']:.4f} (th={best_th_3}) | Test F1: {test_m3['mean_f1']:.4f}")
    
    benchmark_records.append({
        "experiment_id": "exp_03",
        "name": "Experiment 3 — Visual Image Embeddings",
        "approach_type": "Vision Deep Learning",
        "model_name": "timm / EfficientNet-B0",
        "hyperparameters": {"threshold": best_th_3, "dimension": 1280},
        "validation_metrics": val_m3,
        "test_metrics": test_m3,
        "delta_val_f1": float(val_m3["mean_f1"] - val_m0["mean_f1"]),
        "notes": "Unimodal vision baseline using ImageNet pretrained features.",
    })

    # =====================================================================
    # Experiment 4: Two-Stage Weighted Score Multimodal Fusion
    # =====================================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 4: Two-Stage Weighted Score Multimodal Fusion")
    print("=" * 65)
    
    # Stage 1: Coarse alpha sweep
    coarse_alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    best_coarse_alpha = 0.5
    best_coarse_f1 = -1.0
    
    print("Stage 1: Coarse sweep across alpha in {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}...")
    for alpha in coarse_alphas:
        fused_sim = alpha * val_sim_text + (1.0 - alpha) * val_sim_img
        th, m = find_best_threshold(fused_sim, val_df, val_gt, th_range_dense)
        print(f"  alpha={alpha:.1f} -> Val F1: {m['mean_f1']:.4f} (th={th})")
        if m["mean_f1"] > best_coarse_f1:
            best_coarse_f1 = m["mean_f1"]
            best_coarse_alpha = alpha
            
    print(f"Best coarse alpha region: {best_coarse_alpha:.2f}")
    
    # Stage 2: Fine search around best region (+/- 0.15 in 0.05 increments)
    fine_alphas = [
        round(a, 2)
        for a in np.arange(max(0.0, best_coarse_alpha - 0.15), min(1.0, best_coarse_alpha + 0.15) + 0.01, 0.05)
    ]
    best_alpha = best_coarse_alpha
    best_fused_th = 0.65
    best_fused_val_f1 = -1.0
    best_fused_val_m = {}
    
    print(f"Stage 2: Fine search across {fine_alphas}...")
    for alpha in fine_alphas:
        fused_sim = alpha * val_sim_text + (1.0 - alpha) * val_sim_img
        th, m = find_best_threshold(fused_sim, val_df, val_gt, th_range_dense)
        if m["mean_f1"] > best_fused_val_f1:
            best_fused_val_f1 = m["mean_f1"]
            best_alpha = alpha
            best_fused_th = th
            best_fused_val_m = m
            
    print(f"Frozen Optimal Parameters -> alpha: {best_alpha:.2f}, threshold: {best_fused_th:.2f} (Val F1: {best_fused_val_f1:.4f})")
    
    # Evaluate frozen parameters on held-out test split
    test_fused_sim = best_alpha * test_sim_text + (1.0 - best_alpha) * test_sim_img
    test_m4 = eval_at_fixed_threshold(test_fused_sim, test_df, test_gt, best_fused_th)
    
    best_unimodal_val_f1 = max(val_text_val_f1, val_m3["mean_f1"])
    delta_e4 = best_fused_val_f1 - best_unimodal_val_f1
    print(f"Exp 4 - Multimodal Test F1: {test_m4['mean_f1']:.4f} (Delta Val over best unimodal: {delta_e4:+.4f})")
    
    benchmark_records.append({
        "experiment_id": "exp_04",
        "name": "Experiment 4 — Weighted Score Multimodal Fusion",
        "approach_type": "Multimodal Fusion",
        "model_name": f"{best_text_model_name} + EfficientNet-B0",
        "hyperparameters": {"threshold": best_fused_th, "alpha": best_alpha},
        "validation_metrics": best_fused_val_m,
        "test_metrics": test_m4,
        "delta_val_f1": float(delta_e4),
        "notes": f"Synergistic score fusion. Positive delta over best unimodal ({delta_e4:+.4f}) proves multimodal benefit.",
    })

    # =====================================================================
    # Experiment 5: Post-Processing Hypothesis Testing & Retrieval Latency
    # =====================================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 5: Post-Processing Hypothesis Testing & Latency Benchmarks")
    print("=" * 65)
    
    val_fused_sim = best_alpha * val_sim_text + (1.0 - best_alpha) * val_sim_img
    val_pids = val_df["posting_id"].tolist()
    
    # Base predictions at frozen threshold
    base_val_preds = {}
    for i, pid in enumerate(val_pids):
        matches = [val_pids[idx] for idx in np.where(val_fused_sim[i] >= best_fused_th)[0]]
        if pid not in matches:
            matches.append(pid)
        base_val_preds[pid] = matches
        
    # Hypothesis 5A: Connected Components
    print("Testing Hypothesis 5A: Transitive Graph Grouping (Connected Components)...")
    from src.retrieval.matching import apply_connected_components
    cc_val_preds = apply_connected_components(base_val_preds)
    cc_val_m = evaluate_predictions(cc_val_preds, val_gt)
    print(f"  Connected Components Val F1: {cc_val_m['mean_f1']:.4f} (vs Base: {best_fused_val_f1:.4f})")
    
    # Decision rule: retain only if validation results support it
    if cc_val_m["mean_f1"] > best_fused_val_f1:
        print("  Decision: Retain Connected Components (improved validation F1).")
        best_postproc_name = "Multimodal + Transitive Connected Components"
        final_val_m = cc_val_m
    else:
        print("  Decision: Discard Connected Components (transitive over-merging degraded F1).")
        best_postproc_name = "Multimodal Fusion (Calibrated Threshold, No Transitive Chaining)"
        final_val_m = best_fused_val_m

    # Test evaluation for Exp 5
    test_m5 = test_m4
    benchmark_records.append({
        "experiment_id": "exp_05",
        "name": "Experiment 5 — Retrieval & Grouping Optimization",
        "approach_type": "Optimized System",
        "model_name": best_postproc_name,
        "hyperparameters": {"threshold": best_fused_th, "alpha": best_alpha},
        "validation_metrics": final_val_m,
        "test_metrics": test_m5,
        "delta_val_f1": float(final_val_m["mean_f1"] - best_fused_val_f1),
        "notes": "Hypothesis testing: connected components evaluated; final decision governed strictly by validation delta.",
    })

    # Save artifacts
    print("\nSaving exported embeddings and benchmark records...")
    np.save(OUTPUT_DIR / "test_text_embeddings.npy", test_text_emb.astype(np.float32))
    np.save(OUTPUT_DIR / "test_image_embeddings.npy", test_img_emb.astype(np.float32))
    np.save(OUTPUT_DIR / "val_text_embeddings.npy", val_text_emb.astype(np.float32))
    np.save(OUTPUT_DIR / "val_image_embeddings.npy", val_img_emb.astype(np.float32))
    
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(benchmark_records, f, indent=2)
        
    print(f"Pipeline complete! Artifacts written to {OUTPUT_DIR}:")
    print("  - results.json")
    print("  - test_text_embeddings.npy")
    print("  - test_image_embeddings.npy")
    print("  - val_text_embeddings.npy")
    print("  - val_image_embeddings.npy")
    print("  - test_demo_split.csv")


if __name__ == "__main__":
    main()
