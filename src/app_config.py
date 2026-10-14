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


class ConfigurationError(RuntimeError):
    """Raised when critical model or experiment configuration is missing or malformed."""
    pass


def load_final_config(strict: bool = False) -> Optional[Dict[str, Any]]:
    """Loads final_config.json if generated from Phase A Kaggle pipeline."""
    if not FINAL_CONFIG_PATH.exists():
        if strict:
            raise ConfigurationError(
                f"Authoritative model configuration missing: '{FINAL_CONFIG_PATH}' does not exist."
            )
        return None
    try:
        with open(FINAL_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        if strict:
            raise ConfigurationError(
                f"Failed to parse authoritative model configuration '{FINAL_CONFIG_PATH}': {e}"
            )
        return None


def load_results_records(strict: bool = False) -> List[Dict[str, Any]]:
    """Loads experiments/results.json."""
    if not EXPERIMENTS_RESULTS_JSON.exists():
        if strict:
            raise ConfigurationError(
                f"Authoritative experiment results missing: '{EXPERIMENTS_RESULTS_JSON}' does not exist."
            )
        return []
    try:
        with open(EXPERIMENTS_RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        if strict:
            raise ConfigurationError(
                f"Failed to parse experiment results '{EXPERIMENTS_RESULTS_JSON}': {e}"
            )
        return []


def get_selected_inference_config(strict: bool = True) -> Dict[str, Any]:
    """
    Returns the authoritative configuration for deployed model inference,
    synchronizing validation-selected parameters between academic benchmark and live POC.
    """
    final_cfg = load_final_config(strict=strict)
    results = load_results_records(strict=strict)
    records_by_id = {r.get("experiment_id"): r for r in results}

    if strict and not final_cfg:
        raise ConfigurationError("Authoritative model configuration final_config.json could not be loaded.")

    cfg = final_cfg or {}
    params = cfg.get("parameters", {})

    if strict:
        required_keys = ["threshold", "parameters", "text_model", "image_model", "fusion_type"]
        missing_keys = [k for k in required_keys if k not in cfg]
        if missing_keys:
            raise ConfigurationError(
                f"final_config.json is malformed: missing required keys {missing_keys}"
            )
        for param_key in ["w_text", "w_image", "bias"]:
            if param_key not in params or params[param_key] is None:
                raise ConfigurationError(
                    f"final_config.json is malformed: missing parameter '{param_key}' in 'parameters'"
                )

    # Resolve text-only baseline configuration from exp_02 (fine-tuned) or fallback exp_01
    r2 = records_by_id.get("exp_02", {})
    r1 = records_by_id.get("exp_01", {})
    text_th = None
    if r2 and "hyperparameters" in r2:
        text_th = r2["hyperparameters"].get("threshold")
    elif r1 and "hyperparameters" in r1:
        text_th = r1["hyperparameters"].get("threshold")

    if text_th is None:
        if strict:
            raise ConfigurationError(
                "Failed to resolve text model threshold: neither exp_02 nor exp_01 threshold found in results.json"
            )
        text_th = 0.60

    # Resolve grid fallback parameters from exp_04a
    r4a = records_by_id.get("exp_04a", {})
    grid_th = r4a.get("hyperparameters", {}).get("threshold", 0.55)
    grid_alpha = r4a.get("hyperparameters", {}).get("alpha", 0.60)

    # Weights and bias
    w_text = float(params.get("w_text", 9.463388336669517))
    w_image = float(params.get("w_image", 7.808722705879871))
    bias = float(params.get("bias", -8.197978989829934))
    effective_alpha = float(params.get("alpha", w_text / (w_text + w_image) if (w_text + w_image) != 0 else 0.5))

    model_files = cfg.get("model_files", {})
    text_ft_dir = model_files.get("text_ft_dir", "models/text_ft")
    text_model_path = ROOT_DIR / text_ft_dir

    return {
        "text_model": cfg.get("text_model", "all-MiniLM-L6-v2 (fine-tuned)"),
        "text_model_path": text_model_path,
        "image_model": cfg.get("image_model", "timm / EfficientNet-B0"),
        "image_model_name": "efficientnet_b0",
        "fusion_type": cfg.get("fusion_type", "learned"),
        "multimodal_threshold": float(cfg.get("threshold", 0.75)),
        "text_threshold": float(text_th),
        "weights": {
            "w_text": w_text,
            "w_image": w_image,
        },
        "bias": bias,
        "effective_alpha": effective_alpha,
        "grid_threshold": float(grid_th),
        "grid_alpha": float(grid_alpha),
        "top_k": int(cfg.get("top_k", 50)),
    }


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

        sel_cfg = get_selected_inference_config(strict=False)
        profile["default_threshold"] = sel_cfg["multimodal_threshold"]
        profile["default_alpha"] = sel_cfg["effective_alpha"]
        profile["weights"] = sel_cfg["weights"]
        profile["bias"] = sel_cfg["bias"]
        profile["fusion_type"] = sel_cfg["fusion_type"]
        profile["description"] = "Synergistic multimodal fusion combining textual and visual feature representations."

    return profile
