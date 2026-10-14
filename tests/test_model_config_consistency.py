import sys
import json
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT_DIR, EXPERIMENTS_RESULTS_JSON
from src.app_config import (
    get_selected_inference_config,
    load_final_config,
    load_results_records,
    get_model_profile,
    ConfigurationError,
)
from src.models.multimodal import (
    learned_fusion as academic_learned_fusion,
    score_fusion as academic_score_fusion,
)
from src.pipeline.live_matcher import (
    learned_fusion as live_learned_fusion,
    score_fusion as live_score_fusion,
    match_and_rank_candidates,
    WebCandidate,
)


def test_selected_inference_config_integrity():
    """Verify get_selected_inference_config loads authoritative parameters."""
    cfg = get_selected_inference_config(strict=True)

    assert cfg["text_model"] == "all-MiniLM-L6-v2 (fine-tuned)"
    assert cfg["image_model"] == "timm / EfficientNet-B0"
    assert cfg["image_model_name"] == "efficientnet_b0"
    assert cfg["fusion_type"] == "learned"
    assert np.isclose(cfg["multimodal_threshold"], 0.75)
    assert np.isclose(cfg["text_threshold"], 0.60)
    assert np.isclose(cfg["weights"]["w_text"], 9.463388336669517)
    assert np.isclose(cfg["weights"]["w_image"], 7.808722705879871)
    assert np.isclose(cfg["bias"], -8.197978989829934)
    assert np.isclose(cfg["effective_alpha"], 0.5478999245290115)
    assert np.isclose(cfg["grid_threshold"], 0.55)
    assert np.isclose(cfg["grid_alpha"], 0.60)


def test_live_text_threshold_matches_validation_selected_text_model():
    """Verify live text threshold matches validation-selected model (Exp 2)."""
    cfg = get_selected_inference_config(strict=True)
    results = load_results_records(strict=True)

    exp2 = next((r for r in results if r["experiment_id"] == "exp_02"), None)
    assert exp2 is not None, "exp_02 record must exist in experiments/results.json"

    val_th = exp2["hyperparameters"]["threshold"]
    assert np.isclose(cfg["text_threshold"], val_th), (
        f"Live text threshold {cfg['text_threshold']} does not match validation {val_th}"
    )
    assert cfg["text_threshold"] == 0.60


def test_live_multimodal_threshold_and_weights_match_validation_experiment():
    """Verify live multimodal threshold and weights match validation-selected model (Exp 4b)."""
    cfg = get_selected_inference_config(strict=True)
    results = load_results_records(strict=True)

    exp4b = next((r for r in results if r["experiment_id"] == "exp_04b"), None)
    assert exp4b is not None, "exp_04b record must exist in experiments/results.json"

    val_th = exp4b["hyperparameters"]["threshold"]
    assert np.isclose(cfg["multimodal_threshold"], val_th), (
        f"Live multimodal threshold {cfg['multimodal_threshold']} does not match validation {val_th}"
    )
    assert cfg["multimodal_threshold"] == 0.75

    exp_w_t = exp4b["hyperparameters"]["w_text"]
    exp_w_i = exp4b["hyperparameters"]["w_image"]
    exp_b = exp4b["hyperparameters"]["bias"]
    exp_alpha = exp4b["hyperparameters"]["effective_alpha"]

    assert np.isclose(cfg["weights"]["w_text"], exp_w_t)
    assert np.isclose(cfg["weights"]["w_image"], exp_w_i)
    assert np.isclose(cfg["bias"], exp_b)
    assert np.isclose(cfg["effective_alpha"], exp_alpha)


def test_mathematical_fusion_identity_across_pipelines():
    """Verify learned_fusion and score_fusion implementations are identical."""
    cfg = get_selected_inference_config(strict=True)
    w_t = cfg["weights"]["w_text"]
    w_i = cfg["weights"]["w_image"]
    bias = cfg["bias"]

    # Test across grid of text and image similarity values
    text_sims = np.array([0.0, 0.2, 0.5, 0.6, 0.75, 0.9, 1.0, 1.2, -0.5], dtype=np.float32)
    img_sims = np.array([0.0, 0.1, 0.4, 0.65, 0.75, 0.85, 1.0, 1.5, -0.2], dtype=np.float32)

    academic_out = academic_learned_fusion(text_sims, img_sims, w_t, w_i, bias)
    live_out = live_learned_fusion(text_sims, img_sims, w_t, w_i, bias)

    np.testing.assert_allclose(
        live_out,
        academic_out,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Mathematical discrepancy detected between academic and live learned fusion outputs!",
    )

    academic_score = academic_score_fusion(text_sims, img_sims, alpha=0.6)
    live_score = live_score_fusion(text_sims, img_sims, alpha=0.6)

    np.testing.assert_allclose(
        live_score,
        academic_score,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Discrepancy detected between academic and live score fusion outputs!",
    )


def test_strict_error_handling_on_missing_or_malformed_config():
    """Verify that missing or malformed configuration raises ConfigurationError rather than silent fallbacks."""
    import src.app_config as ac
    orig_path = ac.FINAL_CONFIG_PATH

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            # 1. Missing final_config.json
            ac.FINAL_CONFIG_PATH = tmp_path / "non_existent.json"
            error_raised = False
            try:
                get_selected_inference_config(strict=True)
            except ConfigurationError as e:
                error_raised = True
                assert "missing" in str(e) or "Authoritative" in str(e)
            assert error_raised, "Expected ConfigurationError for missing config file"

            # 2. Corrupted JSON file
            corrupt_file = tmp_path / "corrupt.json"
            corrupt_file.write_text("{ broken_json: ", encoding="utf-8")
            ac.FINAL_CONFIG_PATH = corrupt_file
            error_raised = False
            try:
                get_selected_inference_config(strict=True)
            except ConfigurationError:
                error_raised = True
            assert error_raised, "Expected ConfigurationError for corrupt JSON file"

            # 3. Missing required keys
            incomplete_file = tmp_path / "incomplete.json"
            incomplete_file.write_text(json.dumps({"text_model": "test"}), encoding="utf-8")
            ac.FINAL_CONFIG_PATH = incomplete_file
            error_raised = False
            try:
                get_selected_inference_config(strict=True)
            except ConfigurationError as e:
                error_raised = True
                assert "missing required keys" in str(e)
            assert error_raised, "Expected ConfigurationError for incomplete config file"
        finally:
            ac.FINAL_CONFIG_PATH = orig_path


def test_live_matcher_defaults_to_authoritative_config():
    """Verify match_and_rank_candidates defaults to authoritative config when parameters are None."""
    candidates = [
        WebCandidate(
            title="Sony WH-1000XM5 Wireless Noise-Cancelling Headphones",
            merchant="Amazon SG",
            price=399.0,
            currency="SGD",
            link="#",
            thumbnail_url=None,
            image=None,
            retrieval_rank=1,
        )
    ]

    matched, rejected = match_and_rank_candidates(
        query_title="Sony WH-1000XM5 Wireless Noise-Cancelling Headphones",
        query_image=None,
        candidates=candidates,
        threshold=None,       # None -> should use 0.75 from config
        text_threshold=None,  # None -> should use 0.60 from config
        weights=None,         # None -> should use authoritative weights
        bias=None,            # None -> should use authoritative bias
    )

    all_cands = matched + rejected
    cand = all_cands[0]

    # Verify decision reasoning reflects authoritative thresholds (τ=0.75 and τ_text=0.60)
    assert "0.75" in cand.match_reason
    assert "0.60" in cand.text_only_reason


if __name__ == "__main__":
    tests = [
        test_selected_inference_config_integrity,
        test_live_text_threshold_matches_validation_selected_text_model,
        test_live_multimodal_threshold_and_weights_match_validation_experiment,
        test_mathematical_fusion_identity_across_pipelines,
        test_strict_error_handling_on_missing_or_malformed_config,
        test_live_matcher_defaults_to_authoritative_config,
    ]

    print(f"Executing {len(tests)} model configuration consistency tests...")
    passed = 0
    failed = 0

    for test_fn in tests:
        test_name = test_fn.__name__
        try:
            test_fn()
            print(f"  [PASS] {test_name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_name}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Test Summary: {passed} passed, {failed} failed.")
    print("=" * 50)
    if failed > 0:
        exit(1)
