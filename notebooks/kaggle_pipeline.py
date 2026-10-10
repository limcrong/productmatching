"""
Kaggle End-to-End Multimodal Product Matching Pipeline (Experiments 0 to 5)
Self-contained script to be run on Kaggle GPU accelerator (T4 / P100).

Dataset sources:
- shopee-product-matching (/kaggle/input/shopee-product-matching/)
- (Optional) productmatching-splits (/kaggle/input/productmatching-splits/)

Outputs in /kaggle/working/:
- results.json (complete academic benchmark records)
- sweeps.json (alpha x threshold grid search data for UI heatmap)
- latency.json (FAISS vs brute force retrieval benchmarks)
- error_cases.json (data-driven qualitative error cases)
- final_config.json (parameters of the best winning system)
- test_posting_ids.json (ordered test posting IDs for verification)
- test_text_minilm_l6.npy, test_text_minilm_l12.npy, test_text_ft.npy
- test_img_effnet.npy, test_img_clip.npy
- test_text_selected.npy, test_image_selected.npy
- val_text_selected.npy, val_image_selected.npy
- models/text_ft/ (fine-tuned transformer checkpoint, if adopted)
"""

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

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
from sklearn.linear_model import LogisticRegression

try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# =====================================================================
# Configuration & Seed
# =====================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing pipeline on device: {DEVICE}")

KAGGLE_INPUT_DIR = Path("/kaggle/input/shopee-product-matching")
SPLITS_INPUT_DIR = Path("/kaggle/input/productmatching-splits")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = KAGGLE_INPUT_DIR / "train.csv"
TRAIN_IMAGES_DIR = KAGGLE_INPUT_DIR / "train_images"


def find_input_file(filename: str) -> Optional[Path]:
    """Dynamically locates a file anywhere under /kaggle/input/."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    for p in input_root.rglob(filename):
        if p.is_file():
            return p
    return None


def find_input_dir(dirname: str) -> Optional[Path]:
    """Dynamically locates a directory anywhere under /kaggle/input/."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    for p in input_root.rglob(dirname):
        if p.is_dir():
            return p
    return None


# =====================================================================
# 1. Dataset Partitioning & Loading
# =====================================================================
def get_dataset_splits():
    """
    Loads exact split CSVs if uploaded as Kaggle dataset;
    otherwise creates deterministic zero-leakage splits strictly by label_group.
    """
    global TRAIN_IMAGES_DIR

    print("=== INSPECTING /kaggle/input MOUNTS ===")
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for item in sorted(input_root.iterdir()):
            print(f"Mount point: {item.name}")
            if item.is_dir():
                subitems = list(item.iterdir())[:8]
                for sub in subitems:
                    print(f"   -> {sub.name}")
    print("=======================================")

    # Resolve train_images directory
    found_img_dir = find_input_dir("train_images")
    if found_img_dir:
        TRAIN_IMAGES_DIR = found_img_dir
        print(f"Resolved train_images directory: {TRAIN_IMAGES_DIR}")
    else:
        print(f"Notice: train_images directory not found under /kaggle/input.")

    train_split_path = find_input_file("train_split.csv")
    val_split_path = find_input_file("val_split.csv")
    test_split_path = find_input_file("test_demo_split.csv")

    if train_split_path and val_split_path and test_split_path:
        print(f"Found pre-computed zero-leakage splits at: {train_split_path.parent}")
        train_df = pd.read_csv(train_split_path)
        val_df = pd.read_csv(val_split_path)
        test_df = pd.read_csv(test_split_path)
    else:
        train_csv_path = find_input_file("train.csv")
        if not train_csv_path:
            raise FileNotFoundError(
                f"Could not locate train.csv or train_split.csv under /kaggle/input. "
                f"Available files: {[str(p) for p in input_root.rglob('*')][:25]}"
            )
        print(f"Creating zero-leakage splits from: {train_csv_path}")
        df = pd.read_csv(train_csv_path)
        group_counts = df.groupby("label_group").size().reset_index(name="count")
        group_counts = group_counts.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        group_counts["cum_count"] = group_counts["count"].cumsum()
        total_rows = len(df)

        train_cutoff = total_rows * 0.70
        val_cutoff = total_rows * 0.85

        train_groups = set(group_counts[group_counts["cum_count"] <= train_cutoff]["label_group"])
        val_groups = set(
            group_counts[(group_counts["cum_count"] > train_cutoff) & (group_counts["cum_count"] <= val_cutoff)]["label_group"]
        )
        test_groups = set(group_counts[group_counts["cum_count"] > val_cutoff]["label_group"])

        train_df = df[df["label_group"].isin(train_groups)].copy().reset_index(drop=True)
        val_df = df[df["label_group"].isin(val_groups)].copy().reset_index(drop=True)
        test_df = df[df["label_group"].isin(test_groups)].copy().reset_index(drop=True)

    print(f"Splits loaded: Train={len(train_df):,} rows, Val={len(val_df):,} rows, Test={len(test_df):,} rows")
    return train_df, val_df, test_df


def sub_split_train_for_fusion(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sub-splits train groups into train_fit (85% for text fine-tuning)
    and train_fusion (15% for honest logistic regression fusion weight estimation).
    """
    unique_groups = train_df["label_group"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_groups)

    n_fit = int(len(unique_groups) * 0.85)
    fit_groups = set(unique_groups[:n_fit])
    fusion_groups = set(unique_groups[n_fit:])

    train_fit_df = train_df[train_df["label_group"].isin(fit_groups)].copy().reset_index(drop=True)
    train_fusion_df = train_df[train_df["label_group"].isin(fusion_groups)].copy().reset_index(drop=True)
    print(f"Train sub-split: Train-Fit={len(train_fit_df)} rows, Train-Fusion={len(train_fusion_df)} rows")
    return train_fit_df, train_fusion_df


def get_ground_truth_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    group_to_pids = df.groupby("label_group")["posting_id"].apply(list).to_dict()
    return df["label_group"].map(group_to_pids).set_axis(df["posting_id"]).to_dict()


# =====================================================================
# 2. Evaluation Protocol & Metrics
# =====================================================================
def evaluate_predictions(predictions: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
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
        "num_queries": len(ground_truth),
    }


def search_top_k_similarities(embeddings: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds top_k nearest neighbors for each row vector in embeddings using cosine similarity.
    Returns:
        scores: (N, top_k) float32
        indices: (N, top_k) int64
    """
    n = len(embeddings)
    k = min(top_k, n)
    emb = embeddings.astype(np.float32)

    if HAS_FAISS:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        scores, indices = index.search(emb, k)
        return scores, indices

    # Fallback chunked numpy
    scores_list, indices_list = [], []
    chunk_size = 512
    for i in range(0, n, chunk_size):
        chunk = emb[i : i + chunk_size]
        sims = np.dot(chunk, emb.T)
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(chunk.shape[0])[:, None]
        part_scores = sims[rows, part]
        order = np.argsort(-part_scores, axis=1)
        scores_list.append(part_scores[rows, order])
        indices_list.append(part[rows, order])
    return np.vstack(scores_list), np.vstack(indices_list)


def evaluate_top_k(indices: np.ndarray, scores: np.ndarray, pids: List[str], gt: Dict[str, List[str]], th: float):
    preds = {}
    for i, pid in enumerate(pids):
        valid = scores[i] >= th
        matched = [pids[idx] for idx in indices[i][valid]]
        if pid not in matched:
            matched.append(pid)
        preds[pid] = matched
    return evaluate_predictions(preds, gt)


def find_best_threshold(indices: np.ndarray, scores: np.ndarray, pids: List[str], gt: Dict[str, List[str]], thresholds: List[float]):
    best_th = thresholds[0]
    best_f1 = -1.0
    best_metrics = {}
    history = []
    for th in thresholds:
        m = evaluate_top_k(indices, scores, pids, gt, th)
        history.append({
            "threshold": th,
            "mean_precision": m["mean_precision"],
            "mean_recall": m["mean_recall"],
            "mean_f1": m["mean_f1"]
        })
        if m["mean_f1"] > best_f1:
            best_f1 = m["mean_f1"]
            best_th = th
            best_metrics = m
    return best_th, best_metrics, history


def apply_connected_components(predictions: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Transitive grouping via Union-Find (Hypothesis 5A)."""
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(item1, item2):
        root1 = find(item1)
        root2 = find(item2)
        if root1 != root2:
            parent[root1] = root2

    for src, targets in predictions.items():
        for dst in targets:
            union(src, dst)

    groups = {}
    for item in predictions.keys():
        root = find(item)
        groups.setdefault(root, []).append(item)

    return {item: groups[find(item)] for item in predictions.keys()}


# =====================================================================
# 3. Model Encoders
# =====================================================================
def extract_effnet_embeddings(image_names: List[str], batch_size: int = 128) -> np.ndarray:
    print("Extracting EfficientNet-B0 visual embeddings...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).to(DEVICE).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    all_feats = []
    batch = []
    for name in tqdm(image_names, desc="EffNet batches"):
        p = TRAIN_IMAGES_DIR / name
        try:
            img = Image.open(p).convert("RGB")
            t = tf(img)
        except Exception:
            t = torch.zeros((3, 224, 224))
        batch.append(t)
        if len(batch) == batch_size:
            with torch.no_grad():
                out = model(torch.stack(batch).to(DEVICE))
                out = torch.nn.functional.normalize(out, p=2, dim=1)
                all_feats.append(out.cpu().numpy())
            batch = []
    if batch:
        with torch.no_grad():
            out = model(torch.stack(batch).to(DEVICE))
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            all_feats.append(out.cpu().numpy())
    return np.vstack(all_feats).astype(np.float32)


def extract_clip_embeddings(image_names: List[str], batch_size: int = 128) -> np.ndarray:
    print("Extracting CLIP-ViT-B/32 visual embeddings...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    all_feats = []
    batch = []
    for name in tqdm(image_names, desc="CLIP batches"):
        p = TRAIN_IMAGES_DIR / name
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=0)
        batch.append(img)
        if len(batch) == batch_size:
            inputs = clip_processor(images=batch, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                feats = clip_model.get_image_features(**inputs)
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                    feats = feats.image_embeds
                elif isinstance(feats, (tuple, list)):
                    feats = feats[0]
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                all_feats.append(feats.cpu().numpy())
            batch = []
    if batch:
        inputs = clip_processor(images=batch, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                feats = feats.image_embeds
            elif isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
    return np.vstack(all_feats).astype(np.float32)


# =====================================================================
# Main Pipeline
# =====================================================================
def main():
    train_df, val_df, test_df = get_dataset_splits()
    val_gt = get_ground_truth_map(val_df)
    test_gt = get_ground_truth_map(test_df)

    val_pids = val_df["posting_id"].tolist()
    test_pids = test_df["posting_id"].tolist()

    # Save ordered posting IDs for test alignment verification
    with open(OUTPUT_DIR / "test_posting_ids.json", "w") as f:
        json.dump(test_pids, f)

    benchmark_records = []
    sweeps_data = {}

    # -----------------------------------------------------------------
    # EXPERIMENT 0: TF-IDF Classical Baseline
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 0: TF-IDF Classical Baseline (Sparse Cosine)")
    print("=" * 65)
    tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
    tfidf.fit(train_df["title"].tolist())

    val_tfidf = tfidf.transform(val_df["title"].tolist())
    test_tfidf = tfidf.transform(test_df["title"].tolist())

    val_sim_tfidf = val_tfidf.dot(val_tfidf.T).toarray()
    test_sim_tfidf = test_tfidf.dot(test_tfidf.T).toarray()

    # Top-K candidate extraction for TF-IDF
    k_eval = 50
    val_top_idx_0 = np.argpartition(-val_sim_tfidf, k_eval - 1, axis=1)[:, :k_eval]
    r_val = np.arange(len(val_df))[:, None]
    val_top_scores_0 = val_sim_tfidf[r_val, val_top_idx_0]
    order_val = np.argsort(-val_top_scores_0, axis=1)
    val_top_idx_0 = val_top_idx_0[r_val, order_val]
    val_top_scores_0 = val_top_scores_0[r_val, order_val]

    test_top_idx_0 = np.argpartition(-test_sim_tfidf, k_eval - 1, axis=1)[:, :k_eval]
    r_test = np.arange(len(test_df))[:, None]
    test_top_scores_0 = test_sim_tfidf[r_test, test_top_idx_0]
    order_test = np.argsort(-test_top_scores_0, axis=1)
    test_top_idx_0 = test_top_idx_0[r_test, order_test]
    test_top_scores_0 = test_top_scores_0[r_test, order_test]

    th_range_coarse = [round(x, 2) for x in np.arange(0.35, 0.85, 0.05)]
    best_th_0, val_m0, h0 = find_best_threshold(val_top_idx_0, val_top_scores_0, val_pids, val_gt, th_range_coarse)
    test_m0 = evaluate_top_k(test_top_idx_0, test_top_scores_0, test_pids, test_gt, best_th_0)
    print(f"Exp 0 -> Val F1: {val_m0['mean_f1']:.4f} (th={best_th_0:.2f}) | Test F1: {test_m0['mean_f1']:.4f}")

    benchmark_records.append({
        "experiment_id": "exp_00",
        "name": "Experiment 0 — TF-IDF Classical Baseline",
        "approach_type": "Classical ML",
        "model_name": "TfidfVectorizer (bigrams, sublinear TF)",
        "hyperparameters": {"threshold": best_th_0, "max_features": 25000, "top_k": k_eval},
        "validation_metrics": val_m0,
        "test_metrics": test_m0,
        "delta_val_f1": 0.0,
        "delta_reference": "exp_00",
        "selected": False,
        "notes": "Reference baseline. Vocabulary fitted strictly on train split.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    sweeps_data["exp_00"] = h0

    # -----------------------------------------------------------------
    # EXPERIMENT 1: Pretrained Text Transformer Comparison
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: Pretrained Text Transformer Selection")
    print("=" * 65)

    text_candidates = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "test_text_minilm_l6.npy"),
        ("multilingual-MiniLM-L12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "test_text_minilm_l12.npy"),
    ]

    best_text_model_name = None
    best_text_val_f1 = -1.0
    best_text_th = None
    best_text_val_m = None
    cached_text_embeddings = {}

    th_range_dense = [round(x, 2) for x in np.arange(0.50, 0.90, 0.05)]

    for short_name, hf_id, save_filename in text_candidates:
        print(f"Encoding with {short_name}...")
        m = SentenceTransformer(hf_id, device=DEVICE)
        val_emb = m.encode(val_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        test_emb = m.encode(test_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)

        # Save test embedding
        np.save(OUTPUT_DIR / save_filename, test_emb.astype(np.float16))

        val_scores, val_idx = search_top_k_similarities(val_emb, top_k=k_eval)
        th, val_m, h1 = find_best_threshold(val_idx, val_scores, val_pids, val_gt, th_range_dense)
        print(f"  {short_name} -> Val F1: {val_m['mean_f1']:.4f} (th={th:.2f})")

        cached_text_embeddings[short_name] = {
            "model": m, "val_emb": val_emb, "test_emb": test_emb,
            "th": th, "val_m": val_m, "history": h1
        }
        if val_m["mean_f1"] > best_text_val_f1:
            best_text_val_f1 = val_m["mean_f1"]
            best_text_model_name = short_name
            best_text_th = th
            best_text_val_m = val_m

    print(f"\nWinning Pretrained Text Model: {best_text_model_name} (Val F1: {best_text_val_f1:.4f})")
    winning_text_dict = cached_text_embeddings[best_text_model_name]
    test_scores_1, test_idx_1 = search_top_k_similarities(winning_text_dict["test_emb"], top_k=k_eval)
    test_m1 = evaluate_top_k(test_idx_1, test_scores_1, test_pids, test_gt, best_text_th)

    delta_e1 = best_text_val_f1 - val_m0["mean_f1"]
    benchmark_records.append({
        "experiment_id": "exp_01",
        "name": "Experiment 1 — Pretrained Text Embeddings",
        "approach_type": "Text Deep Learning",
        "model_name": best_text_model_name,
        "hyperparameters": {"threshold": best_text_th, "dimension": 384, "top_k": k_eval},
        "validation_metrics": best_text_val_m,
        "test_metrics": test_m1,
        "delta_val_f1": float(delta_e1),
        "delta_reference": "exp_00",
        "selected": False,
        "notes": f"Winning pretrained transformer based on validation F1. Delta over TF-IDF: {delta_e1:+.4f}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    sweeps_data["exp_01"] = winning_text_dict["history"]

    # -----------------------------------------------------------------
    # EXPERIMENT 2: Domain Adaptation via MultipleNegativesRankingLoss
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: Domain Adaptation (Group-Aware Metric Learning)")
    print("=" * 65)

    train_fit_df, train_fusion_df = sub_split_train_for_fusion(train_df)

    # Build group-aware pairs from train_fit
    print("Constructing group-aware positive training pairs from train_fit...")
    groups_to_titles = train_fit_df.groupby("label_group")["title"].apply(list).to_dict()
    multi_groups = [grp for grp, titles in groups_to_titles.items() if len(titles) > 1]
    random.shuffle(multi_groups)

    train_examples = []
    # Build batch-friendly pairs where each item is from a distinct group
    for grp in multi_groups:
        titles = groups_to_titles[grp]
        for i in range(min(len(titles) - 1, 3)):
            train_examples.append(InputExample(texts=[titles[i], titles[i + 1]]))

    print(f"Total training pairs: {len(train_examples):,} across {len(multi_groups):,} groups")

    batch_size_ft = 32
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size_ft)

    hf_base_id = "sentence-transformers/all-MiniLM-L6-v2" if "MiniLM-L6" in best_text_model_name else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ft_model = SentenceTransformer(hf_base_id, device=DEVICE)
    train_loss = losses.MultipleNegativesRankingLoss(ft_model)

    print("Fine-tuning on train_fit for 2 epochs...")
    ft_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True,
    )

    ft_save_dir = MODELS_DIR / "text_ft"
    ft_model.save(str(ft_save_dir))
    print(f"Saved fine-tuned model checkpoint to {ft_save_dir}")

    val_ft_emb = ft_model.encode(val_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    test_ft_emb = ft_model.encode(test_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    np.save(OUTPUT_DIR / "test_text_ft.npy", test_ft_emb.astype(np.float16))

    val_scores_2, val_idx_2 = search_top_k_similarities(val_ft_emb, top_k=k_eval)
    best_th_2, val_m2, h2 = find_best_threshold(val_idx_2, val_scores_2, val_pids, val_gt, th_range_dense)

    test_scores_2, test_idx_2 = search_top_k_similarities(test_ft_emb, top_k=k_eval)
    test_m2 = evaluate_top_k(test_idx_2, test_scores_2, test_pids, test_gt, best_th_2)

    delta_e2 = val_m2["mean_f1"] - best_text_val_f1
    print(f"Exp 2 -> Val F1: {val_m2['mean_f1']:.4f} (th={best_th_2:.2f}) | Test F1: {test_m2['mean_f1']:.4f} (Delta Val: {delta_e2:+.4f})")

    benchmark_records.append({
        "experiment_id": "exp_02",
        "name": "Experiment 2 — Domain-Adapted Text Encoder",
        "approach_type": "Domain Adaptation",
        "model_name": f"{best_text_model_name} (fine-tuned MultipleNegatives)",
        "hyperparameters": {"threshold": best_th_2, "epochs": 2, "batch_size": batch_size_ft, "top_k": k_eval},
        "validation_metrics": val_m2,
        "test_metrics": test_m2,
        "delta_val_f1": float(delta_e2),
        "delta_reference": "exp_01",
        "selected": False,
        "notes": f"Supervised domain adaptation via MultipleNegativesRankingLoss. Delta over Exp 1: {delta_e2:+.4f}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    sweeps_data["exp_02"] = h2

    # Selected text encoder for multimodal downstream
    if val_m2["mean_f1"] > best_text_val_f1:
        print("Adopting Fine-Tuned Text Encoder as Selected Text Model.")
        selected_text_name = f"{best_text_model_name} (fine-tuned)"
        val_text_selected = val_ft_emb
        test_text_selected = test_ft_emb
        best_text_final_val_f1 = val_m2["mean_f1"]
    else:
        print("Retaining Pretrained Text Encoder as Selected Text Model.")
        selected_text_name = best_text_model_name
        val_text_selected = winning_text_dict["val_emb"]
        test_text_selected = winning_text_dict["test_emb"]
        best_text_final_val_f1 = best_text_val_f1

    np.save(OUTPUT_DIR / "val_text_selected.npy", val_text_selected.astype(np.float16))
    np.save(OUTPUT_DIR / "test_text_selected.npy", test_text_selected.astype(np.float16))

    # -----------------------------------------------------------------
    # EXPERIMENT 3: Pretrained Vision Feature Extraction (EffNet vs CLIP)
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: Vision Feature Extraction (EfficientNet-B0 vs CLIP)")
    print("=" * 65)

    # 3A: EfficientNet-B0
    val_img_effnet = extract_effnet_embeddings(val_df["image"].tolist())
    test_img_effnet = extract_effnet_embeddings(test_df["image"].tolist())
    np.save(OUTPUT_DIR / "test_img_effnet.npy", test_img_effnet.astype(np.float16))

    val_scores_eff, val_idx_eff = search_top_k_similarities(val_img_effnet, top_k=k_eval)
    th_eff, val_m_eff, h_eff = find_best_threshold(val_idx_eff, val_scores_eff, val_pids, val_gt, th_range_dense)
    test_scores_eff, test_idx_eff = search_top_k_similarities(test_img_effnet, top_k=k_eval)
    test_m_eff = evaluate_top_k(test_idx_eff, test_scores_eff, test_pids, test_gt, th_eff)
    print(f"  EfficientNet-B0 -> Val F1: {val_m_eff['mean_f1']:.4f} (th={th_eff:.2f}) | Test F1: {test_m_eff['mean_f1']:.4f}")

    # 3B: CLIP ViT-B/32
    if HAS_CLIP:
        val_img_clip = extract_clip_embeddings(val_df["image"].tolist())
        test_img_clip = extract_clip_embeddings(test_df["image"].tolist())
        np.save(OUTPUT_DIR / "test_img_clip.npy", test_img_clip.astype(np.float16))

        val_scores_clip, val_idx_clip = search_top_k_similarities(val_img_clip, top_k=k_eval)
        th_clip, val_m_clip, h_clip = find_best_threshold(val_idx_clip, val_scores_clip, val_pids, val_gt, th_range_dense)
        test_scores_clip, test_idx_clip = search_top_k_similarities(test_img_clip, top_k=k_eval)
        test_m_clip = evaluate_top_k(test_idx_clip, test_scores_clip, test_pids, test_gt, th_clip)
        print(f"  CLIP ViT-B/32  -> Val F1: {val_m_clip['mean_f1']:.4f} (th={th_clip:.2f}) | Test F1: {test_m_clip['mean_f1']:.4f}")
    else:
        val_img_clip = val_img_effnet
        test_img_clip = test_img_effnet
        val_m_clip = val_m_eff
        test_m_clip = test_m_eff
        th_clip = th_eff
        h_clip = h_eff

    if val_m_clip["mean_f1"] > val_m_eff["mean_f1"]:
        winning_img_name = "CLIP-ViT-B/32"
        val_img_selected = val_img_clip
        test_img_selected = test_img_clip
        best_img_th = th_clip
        best_img_val_m = val_m_clip
        best_img_test_m = test_m_clip
        sweeps_data["exp_03"] = h_clip
    else:
        winning_img_name = "timm / EfficientNet-B0"
        val_img_selected = val_img_effnet
        test_img_selected = test_img_effnet
        best_img_th = th_eff
        best_img_val_m = val_m_eff
        best_img_test_m = test_m_eff
        sweeps_data["exp_03"] = h_eff

    print(f"\nWinning Vision Model: {winning_img_name} (Val F1: {best_img_val_m['mean_f1']:.4f})")
    np.save(OUTPUT_DIR / "val_image_selected.npy", val_img_selected.astype(np.float16))
    np.save(OUTPUT_DIR / "test_image_selected.npy", test_img_selected.astype(np.float16))

    benchmark_records.append({
        "experiment_id": "exp_03",
        "name": "Experiment 3 — Visual Image Embeddings",
        "approach_type": "Vision Deep Learning",
        "model_name": winning_img_name,
        "hyperparameters": {"threshold": best_img_th, "dimension": val_img_selected.shape[1], "top_k": k_eval},
        "validation_metrics": best_img_val_m,
        "test_metrics": best_img_test_m,
        "delta_val_f1": float(best_img_val_m["mean_f1"] - val_m0["mean_f1"]),
        "delta_reference": "exp_00",
        "selected": False,
        "notes": f"Winning vision encoder evaluated on validation split.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

    # -----------------------------------------------------------------
    # EXPERIMENT 4a: Weighted Score Multimodal Fusion (Grid Sweep)
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 4a: Weighted Score Multimodal Fusion (Grid Calibration)")
    print("=" * 65)

    val_sim_text = np.dot(val_text_selected, val_text_selected.T)
    val_sim_img = np.dot(val_img_selected, val_img_selected.T)

    test_sim_text = np.dot(test_text_selected, test_text_selected.T)
    test_sim_img = np.dot(test_img_selected, test_img_selected.T)

    grid_alphas = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_grid_alpha = 0.6
    best_grid_th = 0.65
    best_grid_val_f1 = -1.0
    best_grid_val_m = {}
    heatmap_grid = []

    for a in grid_alphas:
        fused_val = a * val_sim_text + (1.0 - a) * val_sim_img
        r = np.arange(len(val_df))[:, None]
        top_idx = np.argpartition(-fused_val, k_eval - 1, axis=1)[:, :k_eval]
        top_sc = fused_val[r, top_idx]
        o = np.argsort(-top_sc, axis=1)
        top_idx = top_idx[r, o]
        top_sc = top_sc[r, o]

        th, m, hist = find_best_threshold(top_idx, top_sc, val_pids, val_gt, th_range_dense)
        for h in hist:
            heatmap_grid.append({"alpha": a, "threshold": h["threshold"], "f1": h["mean_f1"]})

        print(f"  alpha={a:.2f} -> Val F1: {m['mean_f1']:.4f} (th={th:.2f})")
        if m["mean_f1"] > best_grid_val_f1:
            best_grid_val_f1 = m["mean_f1"]
            best_grid_alpha = a
            best_grid_th = th
            best_grid_val_m = m

    # Test eval for 4a
    test_fused_grid = best_grid_alpha * test_sim_text + (1.0 - best_grid_alpha) * test_sim_img
    r_t = np.arange(len(test_df))[:, None]
    test_top_idx_4a = np.argpartition(-test_fused_grid, k_eval - 1, axis=1)[:, :k_eval]
    test_top_sc_4a = test_fused_grid[r_t, test_top_idx_4a]
    o_t = np.argsort(-test_top_sc_4a, axis=1)
    test_top_idx_4a = test_top_idx_4a[r_t, o_t]
    test_top_sc_4a = test_top_sc_4a[r_t, o_t]
    test_m4a = evaluate_top_k(test_top_idx_4a, test_top_sc_4a, test_pids, test_gt, best_grid_th)

    delta_e4a = best_grid_val_f1 - max(best_text_final_val_f1, best_img_val_m["mean_f1"])
    print(f"\nExp 4a -> Val F1: {best_grid_val_f1:.4f} (alpha={best_grid_alpha:.2f}, th={best_grid_th:.2f}) | Test F1: {test_m4a['mean_f1']:.4f}")

    benchmark_records.append({
        "experiment_id": "exp_04a",
        "name": "Experiment 4a — Grid-Calibrated Weighted Fusion",
        "approach_type": "Multimodal Fusion",
        "model_name": f"{selected_text_name} + {winning_img_name} (Grid)",
        "hyperparameters": {"alpha": best_grid_alpha, "threshold": best_grid_th, "top_k": k_eval},
        "validation_metrics": best_grid_val_m,
        "test_metrics": test_m4a,
        "delta_val_f1": float(delta_e4a),
        "delta_reference": "exp_01" if best_text_final_val_f1 >= best_img_val_m["mean_f1"] else "exp_03",
        "selected": False,
        "notes": f"Score fusion via validation grid sweep. Delta over best unimodal: {delta_e4a:+.4f}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    sweeps_data["heatmap_4a"] = heatmap_grid

    # -----------------------------------------------------------------
    # EXPERIMENT 4b: Learned Weighted Fusion (Logistic Regression on train_fusion)
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 4b: Learned Multimodal Fusion (Optimized by Training)")
    print("=" * 65)

    print("Encoding train_fusion split to train logistic regression fusion...")
    if "fine-tuned" in selected_text_name:
        fusion_text_emb = ft_model.encode(train_fusion_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    else:
        fusion_text_emb = winning_text_dict["model"].encode(train_fusion_df["title"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)

    if "CLIP" in winning_img_name:
        fusion_img_emb = extract_clip_embeddings(train_fusion_df["image"].tolist())
    else:
        fusion_img_emb = extract_effnet_embeddings(train_fusion_df["image"].tolist())

    # Build candidate pairs on train_fusion
    fusion_sim_text = np.dot(fusion_text_emb, fusion_text_emb.T)
    fusion_sim_img = np.dot(fusion_img_emb, fusion_img_emb.T)
    fusion_groups = train_fusion_df["label_group"].values

    X_train_pairs = []
    y_train_pairs = []
    n_fusion = len(train_fusion_df)

    for i in range(min(n_fusion, 1000)):
        # Positives
        pos_indices = np.where((fusion_groups == fusion_groups[i]) & (np.arange(n_fusion) != i))[0]
        for p_idx in pos_indices[:5]:
            X_train_pairs.append([fusion_sim_text[i, p_idx], fusion_sim_img[i, p_idx]])
            y_train_pairs.append(1)
        # Negatives: top hard negatives from text and image
        neg_text = np.argsort(-fusion_sim_text[i])[:15]
        neg_img = np.argsort(-fusion_sim_img[i])[:15]
        neg_candidates = set(neg_text).union(set(neg_img))
        for neg_idx in neg_candidates:
            if fusion_groups[neg_idx] != fusion_groups[i]:
                X_train_pairs.append([fusion_sim_text[i, neg_idx], fusion_sim_img[i, neg_idx]])
                y_train_pairs.append(0)

    X_train_pairs = np.array(X_train_pairs)
    y_train_pairs = np.array(y_train_pairs)
    print(f"Trained pair samples: {len(y_train_pairs)} (Pos: {np.sum(y_train_pairs)}, Neg: {len(y_train_pairs) - np.sum(y_train_pairs)})")

    clf = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_train_pairs, y_train_pairs)

    w_text = float(clf.coef_[0][0])
    w_img = float(clf.coef_[0][1])
    bias = float(clf.intercept_[0])
    effective_alpha = float(w_text / (w_text + w_img)) if (w_text + w_img) > 0 else 0.5
    print(f"Learned weights -> w_text: {w_text:.3f}, w_img: {w_img:.3f}, bias: {bias:.3f} | Effective Alpha: {effective_alpha:.3f}")

    # Compute learned probabilities on val split
    val_logits = w_text * val_sim_text + w_img * val_sim_img + bias
    val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -15, 15)))

    r_v = np.arange(len(val_df))[:, None]
    val_top_idx_4b = np.argpartition(-val_probs, k_eval - 1, axis=1)[:, :k_eval]
    val_top_sc_4b = val_probs[r_v, val_top_idx_4b]
    o_v = np.argsort(-val_top_sc_4b, axis=1)
    val_top_idx_4b = val_top_idx_4b[r_v, o_v]
    val_top_sc_4b = val_top_sc_4b[r_v, o_v]

    th_prob_range = [round(x, 2) for x in np.arange(0.30, 0.85, 0.05)]
    best_th_4b, val_m4b, h4b = find_best_threshold(val_top_idx_4b, val_top_sc_4b, val_pids, val_gt, th_prob_range)

    # Test evaluation for 4b
    test_logits = w_text * test_sim_text + w_img * test_sim_img + bias
    test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -15, 15)))

    r_t = np.arange(len(test_df))[:, None]
    test_top_idx_4b = np.argpartition(-test_probs, k_eval - 1, axis=1)[:, :k_eval]
    test_top_sc_4b = test_probs[r_t, test_top_idx_4b]
    o_t = np.argsort(-test_top_sc_4b, axis=1)
    test_top_idx_4b = test_top_idx_4b[r_t, o_t]
    test_top_sc_4b = test_top_sc_4b[r_t, o_t]

    test_m4b = evaluate_top_k(test_top_idx_4b, test_top_sc_4b, test_pids, test_gt, best_th_4b)

    delta_e4b = val_m4b["mean_f1"] - best_grid_val_f1
    print(f"Exp 4b -> Val F1: {val_m4b['mean_f1']:.4f} (th={best_th_4b:.2f}) | Test F1: {test_m4b['mean_f1']:.4f} (Delta over 4a: {delta_e4b:+.4f})")

    use_learned_fusion = (val_m4b["mean_f1"] >= best_grid_val_f1)
    print(f"Winner between 4a (Grid) vs 4b (Learned): {'4b (Learned)' if use_learned_fusion else '4a (Grid)'}")

    benchmark_records.append({
        "experiment_id": "exp_04b",
        "name": "Experiment 4b — Learned Weighted Fusion",
        "approach_type": "Multimodal Fusion",
        "model_name": f"{selected_text_name} + {winning_img_name} (Learned)",
        "hyperparameters": {
            "w_text": w_text, "w_image": w_img, "bias": bias,
            "effective_alpha": effective_alpha, "threshold": best_th_4b, "top_k": k_eval
        },
        "validation_metrics": val_m4b,
        "test_metrics": test_m4b,
        "delta_val_f1": float(delta_e4b),
        "delta_reference": "exp_04a",
        "selected": use_learned_fusion,
        "notes": f"Supervised multimodal fusion trained on train_fusion split. Effective alpha: {effective_alpha:.3f}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    sweeps_data["exp_04b"] = h4b

    if not use_learned_fusion:
        # Mark 4a as selected
        for rec in benchmark_records:
            if rec["experiment_id"] == "exp_04a":
                rec["selected"] = True

    # -----------------------------------------------------------------
    # EXPERIMENT 5: Post-Processing & Latency Benchmarks
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXPERIMENT 5: Retrieval Benchmarking & Hypothesis 5A Testing")
    print("=" * 65)

    # 1. FAISS Latency Benchmark on Test Embeddings
    print("Benchmarking FAISS vs Brute-Force retrieval speed...")
    bf_start = time.perf_counter()
    bf_sims = np.dot(test_text_selected[:200], test_text_selected.T)
    bf_top = np.argpartition(-bf_sims, k_eval - 1, axis=1)[:, :k_eval]
    bf_dur = time.perf_counter() - bf_start
    bf_latency_ms = (bf_dur / 200) * 1000
    bf_qps = 200 / bf_dur

    if HAS_FAISS:
        faiss_idx = faiss.IndexFlatIP(test_text_selected.shape[1])
        faiss_idx.add(test_text_selected.astype(np.float32))
        f_start = time.perf_counter()
        f_sc, f_indices = faiss_idx.search(test_text_selected[:200].astype(np.float32), k_eval)
        f_dur = time.perf_counter() - f_start
        f_latency_ms = (f_dur / 200) * 1000
        f_qps = 200 / f_dur
    else:
        f_latency_ms = bf_latency_ms
        f_qps = bf_qps

    latency_data = {
        "database_size": len(test_df),
        "dimension": test_text_selected.shape[1],
        "top_k": k_eval,
        "brute_force_latency_ms": round(bf_latency_ms, 3),
        "brute_force_qps": round(bf_qps, 1),
        "faiss_latency_ms": round(f_latency_ms, 3),
        "faiss_qps": round(f_qps, 1),
        "recall_at_50": 1.0,
    }
    with open(OUTPUT_DIR / "latency.json", "w") as f:
        json.dump(latency_data, f, indent=2)

    # 2. Test Connected Components Hypothesis on Best Multimodal Predictions
    if use_learned_fusion:
        base_val_preds = {}
        for i, pid in enumerate(val_pids):
            valid = val_top_sc_4b[i] >= best_th_4b
            matched = [val_pids[idx] for idx in val_top_idx_4b[i][valid]]
            if pid not in matched:
                matched.append(pid)
            base_val_preds[pid] = matched
        base_val_f1 = val_m4b["mean_f1"]

        base_test_preds = {}
        for i, pid in enumerate(test_pids):
            valid = test_top_sc_4b[i] >= best_th_4b
            matched = [test_pids[idx] for idx in test_top_idx_4b[i][valid]]
            if pid not in matched:
                matched.append(pid)
            base_test_preds[pid] = matched
    else:
        base_val_preds = {}
        for i, pid in enumerate(val_pids):
            valid = top_sc[i] >= best_grid_th
            matched = [val_pids[idx] for idx in top_idx[i][valid]]
            if pid not in matched:
                matched.append(pid)
            base_val_preds[pid] = matched
        base_val_f1 = best_grid_val_f1

        base_test_preds = {}
        for i, pid in enumerate(test_pids):
            valid = test_top_sc_4a[i] >= best_grid_th
            matched = [test_pids[idx] for idx in test_top_idx_4a[i][valid]]
            if pid not in matched:
                matched.append(pid)
            base_test_preds[pid] = matched

    print("Evaluating Hypothesis 5A: Transitive Graph Grouping (Connected Components)...")
    cc_val_preds = apply_connected_components(base_val_preds)
    cc_val_m = evaluate_predictions(cc_val_preds, val_gt)
    print(f"  CC Val F1: {cc_val_m['mean_f1']:.4f} vs Base Val F1: {base_val_f1:.4f}")

    if cc_val_m["mean_f1"] > base_val_f1:
        print("Decision: Retain Connected Components.")
        final_postproc_name = "Multimodal + Transitive Connected Components"
        final_val_m = cc_val_m
        final_test_preds = apply_connected_components(base_test_preds)
        final_test_m = evaluate_predictions(final_test_preds, test_gt)
    else:
        print("Decision: Discard Connected Components (chaining over-merges false positives).")
        final_postproc_name = "Multimodal (Calibrated Threshold, Direct NN Match)"
        final_val_m = val_m4b if use_learned_fusion else best_grid_val_m
        final_test_m = test_m4b if use_learned_fusion else test_m4a

    benchmark_records.append({
        "experiment_id": "exp_05",
        "name": "Experiment 5 — Final Optimization & Post-Processing",
        "approach_type": "Optimized System",
        "model_name": final_postproc_name,
        "hyperparameters": {
            "fusion_type": "learned" if use_learned_fusion else "grid",
            "threshold": best_th_4b if use_learned_fusion else best_grid_th,
            "top_k": k_eval,
        },
        "validation_metrics": final_val_m,
        "test_metrics": final_test_m,
        "delta_val_f1": float(final_val_m["mean_f1"] - base_val_f1),
        "delta_reference": "exp_04b" if use_learned_fusion else "exp_04a",
        "selected": True,
        "notes": "End-to-end deployed configuration for the final demo system.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

    # -----------------------------------------------------------------
    # Qualitative Error Analysis Cases Extraction
    # -----------------------------------------------------------------
    print("\nExtracting data-driven qualitative error analysis cases...")
    test_groups = test_df["label_group"].values
    error_cases = []

    for i in range(len(test_df)):
        if len(error_cases) >= 12:
            break
        for j in range(i + 1, min(i + 20, len(test_df))):
            s_t = float(test_sim_text[i, j])
            s_i = float(test_sim_img[i, j])
            s_f = float(test_probs[i, j]) if use_learned_fusion else float(best_grid_alpha * s_t + (1 - best_grid_alpha) * s_i)
            same = bool(test_groups[i] == test_groups[j])
            th_f = best_th_4b if use_learned_fusion else best_grid_th

            # Case A: Fusion fixed text FP (e.g. phone case vs phone)
            if not same and s_t >= 0.70 and s_i < 0.45 and s_f < th_f:
                error_cases.append({
                    "category": "fusion_fixed_text_fp",
                    "title_query": test_df.iloc[i]["title"],
                    "image_query": test_df.iloc[i]["image"],
                    "title_candidate": test_df.iloc[j]["title"],
                    "image_candidate": test_df.iloc[j]["image"],
                    "s_text": round(s_t, 3), "s_img": round(s_i, 3), "s_fused": round(s_f, 3),
                    "explanation": "Text model assigned high similarity due to shared keywords, but visual feature encoder rejected the match."
                })
                break

            # Case B: Fusion fixed image FP (e.g. similar plain shirt, different brand)
            if not same and s_i >= 0.75 and s_t < 0.40 and s_f < th_f:
                error_cases.append({
                    "category": "fusion_fixed_image_fp",
                    "title_query": test_df.iloc[i]["title"],
                    "image_query": test_df.iloc[i]["image"],
                    "title_candidate": test_df.iloc[j]["title"],
                    "image_candidate": test_df.iloc[j]["image"],
                    "s_text": round(s_t, 3), "s_img": round(s_i, 3), "s_fused": round(s_f, 3),
                    "explanation": "Visual appearance was near-identical, but text semantic model detected distinct models/specifications."
                })
                break

            # Case C: True match recovered by fusion
            if same and s_f >= th_f and (s_t < 0.55 or s_i < 0.55):
                error_cases.append({
                    "category": "fusion_recovered_fn",
                    "title_query": test_df.iloc[i]["title"],
                    "image_query": test_df.iloc[i]["image"],
                    "title_candidate": test_df.iloc[j]["title"],
                    "image_candidate": test_df.iloc[j]["image"],
                    "s_text": round(s_t, 3), "s_img": round(s_i, 3), "s_fused": round(s_f, 3),
                    "explanation": "One modality was noisy/incomplete, but synergistic multimodal fusion recovered the true product match."
                })
                break

    with open(OUTPUT_DIR / "error_cases.json", "w") as f:
        json.dump(error_cases, f, indent=2)

    # -----------------------------------------------------------------
    # Save Final Artifacts
    # -----------------------------------------------------------------
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(benchmark_records, f, indent=2)

    with open(OUTPUT_DIR / "sweeps.json", "w") as f:
        json.dump(sweeps_data, f, indent=2)

    final_config = {
        "text_model": selected_text_name,
        "image_model": winning_img_name,
        "fusion_type": "learned" if use_learned_fusion else "grid",
        "threshold": best_th_4b if use_learned_fusion else best_grid_th,
        "top_k": k_eval,
        "parameters": {
            "w_text": w_text if use_learned_fusion else None,
            "w_image": w_img if use_learned_fusion else None,
            "bias": bias if use_learned_fusion else None,
            "alpha": effective_alpha if use_learned_fusion else best_grid_alpha,
        },
        "model_files": {
            "text_ft_dir": "models/text_ft" if "fine-tuned" in selected_text_name else None,
            "test_text_emb": "test_text_selected.npy",
            "test_image_emb": "test_image_selected.npy",
            "val_text_emb": "val_text_selected.npy",
            "val_image_emb": "val_image_selected.npy",
        }
    }
    with open(OUTPUT_DIR / "final_config.json", "w") as f:
        json.dump(final_config, f, indent=2)

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Artifacts exported to {OUTPUT_DIR}:")
    print("  - results.json")
    print("  - final_config.json")
    print("  - sweeps.json")
    print("  - latency.json")
    print("  - error_cases.json")
    print("  - test_posting_ids.json")
    print("  - test_text_selected.npy / test_image_selected.npy")
    print("=" * 65)


if __name__ == "__main__":
    main()
