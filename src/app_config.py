"""
Application Configuration and Benchmark Loader
Single source of truth for Streamlit UI and local inference pipelines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import (
    ROOT_DIR,
    EMBEDDINGS_DIR,
    EXPERIMENTS_DIR,
    EXPERIMENTS_RESULTS_JSON,
)

FINAL_CONFIG_PATH = EXPERIMENTS_DIR / "final_config.json"
SWEEPS_DATA_PATH = EXPERIMENTS_DIR / "sweeps.json"
LATENCY_DATA_PATH = EXPERIMENTS_DIR / "latency.json"
ERROR_CASES_PATH = EXPERIMENTS_DIR / "error_cases.json"


def load_final_config() -> Optional[Dict[str, Any]]:
    """Loads final_config.json if generated from Phase A Kaggle pipeline."""
    if FINAL_CONFIG_PATH.exists():
        try:
            with open(FINAL_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def load_results_records() -> List[Dict[str, Any]]:
    """Loads experiments/results.json."""
    if EXPERIMENTS_RESULTS_JSON.exists():
        try:
            with open(EXPERIMENTS_RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def load_latency_benchmark() -> Optional[Dict[str, Any]]:
    """Loads latency.json benchmarks."""
    if LATENCY_DATA_PATH.exists():
        try:
            with open(LATENCY_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def load_error_cases() -> List[Dict[str, Any]]:
    """Loads error_cases.json qualitative analysis cases."""
    if ERROR_CASES_PATH.exists():
        try:
            with open(ERROR_CASES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def load_sweeps_data() -> Dict[str, Any]:
    """Loads sweeps.json grid sweep and alpha heatmap data."""
    if SWEEPS_DATA_PATH.exists():
        try:
            with open(SWEEPS_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_model_profile(model_choice: str) -> Dict[str, Any]:
    """
    Returns hyperparameter profile, embedding paths, and availability status
    for the chosen model in the Dataset Matcher.
    """
    final_cfg = load_final_config() or {}
    results = load_results_records()
    records_by_id = {r.get("experiment_id"): r for r in results}

    profile = {
        "is_available": False,
        "default_threshold": 0.65,
        "default_alpha": 0.60,
        "weights": None,
        "bias": None,
        "text_emb_file": None,
        "image_emb_file": None,
        "description": "",
    }

    if "TF-IDF" in model_choice:
        r0 = records_by_id.get("exp_00", {})
        profile["is_available"] = True
        profile["default_threshold"] = r0.get("hyperparameters", {}).get("threshold", 0.45)
        profile["description"] = "Sparse word n-gram cosine similarity fitted strictly on training data."

    elif "Text Embeddings" in model_choice:
        # Check text embeddings files in priority: selected, ft, minilm
        candidates = [
            ("test_text_selected.npy", "Selected Transformer"),
            ("test_text_ft.npy", "Domain-Adapted MiniLM"),
            ("test_text_minilm_l12.npy", "Multilingual MiniLM-L12"),
            ("test_text_minilm_l6.npy", "MiniLM-L6"),
            ("test_text_embeddings.npy", "Precomputed Text Embeddings"),
        ]
        r1 = records_by_id.get("exp_01", {})
        r2 = records_by_id.get("exp_02", {})
        th = r2.get("hyperparameters", {}).get("threshold") or r1.get("hyperparameters", {}).get("threshold", 0.62)
        profile["default_threshold"] = float(th)

        for filename, desc in candidates:
            p = EMBEDDINGS_DIR / filename
            if p.exists():
                profile["is_available"] = True
                profile["text_emb_file"] = p
                profile["description"] = f"Dense semantic text embeddings ({desc})."
                break

    elif "Image Embeddings" in model_choice:
        candidates = [
            ("test_image_selected.npy", "Selected Vision Encoder"),
            ("test_img_clip.npy", "CLIP-ViT-B/32"),
            ("test_img_effnet.npy", "timm EfficientNet-B0"),
            ("test_image_embeddings.npy", "Precomputed Image Embeddings"),
        ]
        r3 = records_by_id.get("exp_03", {})
        th = r3.get("hyperparameters", {}).get("threshold", 0.68)
        profile["default_threshold"] = float(th)

        for filename, desc in candidates:
            p = EMBEDDINGS_DIR / filename
            if p.exists():
                profile["is_available"] = True
                profile["image_emb_file"] = p
                profile["description"] = f"Visual deep learning features ({desc})."
                break

    elif "Multimodal" in model_choice:
        # Check if both text and image are available
        t_cand = [EMBEDDINGS_DIR / f for f in ["test_text_selected.npy", "test_text_ft.npy", "test_text_minilm_l6.npy", "test_text_embeddings.npy"]]
        i_cand = [EMBEDDINGS_DIR / f for f in ["test_image_selected.npy", "test_img_clip.npy", "test_img_effnet.npy", "test_image_embeddings.npy"]]

        t_path = next((p for p in t_cand if p.exists()), None)
        i_path = next((p for p in i_cand if p.exists()), None)

        if t_path and i_path:
            profile["is_available"] = True
            profile["text_emb_file"] = t_path
            profile["image_emb_file"] = i_path

        params = final_cfg.get("parameters", {})
        profile["default_threshold"] = float(final_cfg.get("threshold", 0.65))
        profile["default_alpha"] = float(params.get("alpha", 0.60))
        profile["weights"] = {
            "w_text": params.get("w_text"),
            "w_image": params.get("w_image"),
        }
        profile["bias"] = params.get("bias")
        profile["fusion_type"] = final_cfg.get("fusion_type", "grid")
        profile["description"] = "Synergistic multimodal fusion combining textual and visual feature representations."

    return profile
