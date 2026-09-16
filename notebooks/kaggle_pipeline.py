"""
Kaggle End-to-End Multimodal Product Matching Pipeline
Run this script or notebook in Kaggle with GPU accelerator enabled.

Dataset: Shopee - Price Match Guarantee (/kaggle/input/shopee-product-matching/)
Output:
  - test_demo_split.csv
  - test_text_embeddings.npy
  - test_image_embeddings.npy
  - results.json
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import timm
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Paths ---
KAGGLE_INPUT_DIR = Path("/kaggle/input/shopee-product-matching")
OUTPUT_DIR = Path("/kaggle/working")

TRAIN_CSV = KAGGLE_INPUT_DIR / "train.csv"
TRAIN_IMAGES_DIR = KAGGLE_INPUT_DIR / "train_images"

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- 1. Split Data by label_group (Zero Leakage) ---
def create_splits(df):
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
    
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")
    return train_df, val_df, test_df

# --- 2. Evaluation Metrics ---
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
    pids = df["posting_id"].tolist()
    preds = {}
    for i, pid in enumerate(pids):
        matched_indices = np.where(sim_matrix[i] >= th)[0]
        matched_pids = [pids[idx] for idx in matched_indices]
        if pid not in matched_pids:
            matched_pids.append(pid)
        preds[pid] = matched_pids
    return evaluate_predictions(preds, gt_map)

# --- Main Pipeline ---
def main():
    if not TRAIN_CSV.exists():
        print(f"Kaggle path {TRAIN_CSV} not found. Please ensure Shopee dataset is attached.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df, test_df = create_splits(df)
    
    # Save test_demo_split.csv for download
    test_df.to_csv(OUTPUT_DIR / "test_demo_split.csv", index=False)
    
    val_gt = get_ground_truth_map(val_df)
    test_gt = get_ground_truth_map(test_df)
    
    results = []
    
    # ==========================================
    # Experiment 0: TF-IDF Classical Baseline
    # ==========================================
    print("\n--- Experiment 0: TF-IDF Baseline ---")
    tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    tfidf.fit(train_df["title"])
    
    val_tfidf = tfidf.transform(val_df["title"])
    test_tfidf = tfidf.transform(test_df["title"])
    
    val_sim_tfidf = val_tfidf.dot(val_tfidf.T).toarray()
    test_sim_tfidf = test_tfidf.dot(test_tfidf.T).toarray()
    
    th_range = [round(x, 2) for x in np.arange(0.40, 0.85, 0.05)]
    best_th_0, val_m0 = find_best_threshold(val_sim_tfidf, val_df, val_gt, th_range)
    test_m0 = eval_at_fixed_threshold(test_sim_tfidf, test_df, test_gt, best_th_0)
    print(f"Exp 0 - Val F1: {val_m0['mean_f1']:.4f} (th={best_th_0}) | Test F1: {test_m0['mean_f1']:.4f}")
    
    results.append({
        "experiment_id": "exp_00",
        "name": "Experiment 0 — TF-IDF Classical Baseline",
        "approach_type": "Classical ML",
        "model_name": "TfidfVectorizer (bigrams, sublinear TF)",
        "hyperparameters": {"threshold": best_th_0, "max_features": 25000},
        "validation_metrics": val_m0,
        "test_metrics": test_m0,
    })
    
    # ==========================================
    # Experiment 1: SentenceTransformers Text DL
    # ==========================================
    print("\n--- Experiment 1: SentenceTransformers (all-MiniLM-L6-v2) ---")
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    val_text_emb = text_model.encode(val_df["title"].tolist(), batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    test_text_emb = text_model.encode(test_df["title"].tolist(), batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    
    # Save test text embeddings for local demo
    np.save(OUTPUT_DIR / "test_text_embeddings.npy", test_text_emb.astype(np.float32))
    
    val_sim_text = np.dot(val_text_emb, val_text_emb.T)
    test_sim_text = np.dot(test_text_emb, test_text_emb.T)
    
    th_range_dl = [round(x, 2) for x in np.arange(0.50, 0.90, 0.05)]
    best_th_1, val_m1 = find_best_threshold(val_sim_text, val_df, val_gt, th_range_dl)
    test_m1 = eval_at_fixed_threshold(test_sim_text, test_df, test_gt, best_th_1)
    print(f"Exp 1 - Val F1: {val_m1['mean_f1']:.4f} (th={best_th_1}) | Test F1: {test_m1['mean_f1']:.4f}")
    
    results.append({
        "experiment_id": "exp_01",
        "name": "Experiment 1 — Transformer Text Embeddings",
        "approach_type": "Text Deep Learning",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "hyperparameters": {"threshold": best_th_1, "dimension": 384},
        "validation_metrics": val_m1,
        "test_metrics": test_m1,
    })
    
    # ==========================================
    # Experiment 3: Visual Image Embeddings
    # ==========================================
    print("\n--- Experiment 3: Visual Features (timm / EfficientNet-B0) ---")
    img_model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).to(DEVICE).eval()
    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    def extract_image_features(image_names):
        feats = []
        batch = []
        for name in tqdm(image_names, desc="Extracting images"):
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
    
    # Save test image embeddings for local demo
    np.save(OUTPUT_DIR / "test_image_embeddings.npy", test_img_emb)
    
    val_sim_img = np.dot(val_img_emb, val_img_emb.T)
    test_sim_img = np.dot(test_img_emb, test_img_emb.T)
    
    best_th_3, val_m3 = find_best_threshold(val_sim_img, val_df, val_gt, th_range_dl)
    test_m3 = eval_at_fixed_threshold(test_sim_img, test_df, test_gt, best_th_3)
    print(f"Exp 3 - Val F1: {val_m3['mean_f1']:.4f} (th={best_th_3}) | Test F1: {test_m3['mean_f1']:.4f}")
    
    results.append({
        "experiment_id": "exp_03",
        "name": "Experiment 3 — Visual Image Embeddings",
        "approach_type": "Vision Deep Learning",
        "model_name": "timm / EfficientNet-B0",
        "hyperparameters": {"threshold": best_th_3, "dimension": 1280},
        "validation_metrics": val_m3,
        "test_metrics": test_m3,
    })
    
    # ==========================================
    # Experiment 4: Multimodal Fusion
    # ==========================================
    print("\n--- Experiment 4: Multimodal Fusion (Tuning Alpha & Threshold) ---")
    best_alpha = 0.60
    best_fused_th = 0.65
    best_fused_val_f1 = -1.0
    best_fused_val_m = {}
    
    for alpha in [0.40, 0.50, 0.60, 0.70]:
        val_fused_sim = alpha * val_sim_text + (1.0 - alpha) * val_sim_img
        th, m = find_best_threshold(val_fused_sim, val_df, val_gt, th_range_dl)
        if m["mean_f1"] > best_fused_val_f1:
            best_fused_val_f1 = m["mean_f1"]
            best_alpha = alpha
            best_fused_th = th
            best_fused_val_m = m
            
    print(f"Optimal Alpha on Validation Split: {best_alpha:.2f}, Threshold: {best_fused_th:.2f} (Val F1: {best_fused_val_f1:.4f})")
    
    test_fused_sim = best_alpha * test_sim_text + (1.0 - best_alpha) * test_sim_img
    test_m4 = eval_at_fixed_threshold(test_fused_sim, test_df, test_gt, best_fused_th)
    print(f"Exp 4 - Multimodal Test F1: {test_m4['mean_f1']:.4f} (Precision: {test_m4['mean_precision']:.4f}, Recall: {test_m4['mean_recall']:.4f})")
    
    results.append({
        "experiment_id": "exp_04",
        "name": "Experiment 4 — Multimodal Early/Late Fusion",
        "approach_type": "Multimodal Fusion",
        "model_name": "MiniLM-L6 + EfficientNet-B0",
        "hyperparameters": {"threshold": best_fused_th, "alpha": best_alpha},
        "validation_metrics": best_fused_val_m,
        "test_metrics": test_m4,
    })
    
    # Save results.json
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nArtifacts successfully written to {OUTPUT_DIR}:")
    print(" - test_demo_split.csv")
    print(" - test_text_embeddings.npy")
    print(" - test_image_embeddings.npy")
    print(" - results.json")
    print("Done! Download these files to your local repository.")

if __name__ == "__main__":
    main()
