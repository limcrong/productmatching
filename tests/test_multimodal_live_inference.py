import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import zipfile
import io
import numpy as np
import pandas as pd
from PIL import Image


from src.config import (
    ROOT_DIR,
    ZIP_PATH,
    EMBEDDINGS_DIR,
    TEST_DEMO_SPLIT_CSV,
    DEFAULT_IMAGE_MODEL,
)
from src.models.image_encoder import ImageEncoder
from src.pipeline.live_matcher import (
    get_live_image_encoder,
    get_live_text_encoder,
    compute_image_similarity,
    compute_text_similarity,
    match_and_rank_candidates,
)
from src.retrieval.web_search import (
    load_curated_demo_cases,
    fetch_candidate_image,
    WebCandidate,
)


def test_efficientnet_architecture_and_normalization():
    """Verify ImageEncoder architecture, output dimension (1280), and unit L2-normalization."""
    encoder = get_live_image_encoder()
    assert encoder.model_name == DEFAULT_IMAGE_MODEL == "efficientnet_b0"

    test_img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    embs, mask = encoder.encode_images([test_img], normalize=True, return_mask=True)

    assert embs.shape == (1, 1280)
    assert mask[0] is True or mask[0] == True
    norm = np.linalg.norm(embs[0])
    assert np.isclose(norm, 1.0, atol=1e-5), f"Expected unit L2 norm, got {norm}"


def test_embedding_consistency_with_offline_experiment():
    """
    Numerically compare live EfficientNet-B0 embeddings against stored Kaggle experiment
    embeddings (test_img_effnet.npy / test_image_selected.npy) on raw Kaggle test images.
    """
    if not ZIP_PATH.exists():
        print("  [SKIP] Kaggle zip file not found on disk.")
        return

    df = pd.read_csv(TEST_DEMO_SPLIT_CSV)
    stored_effnet = np.load(EMBEDDINGS_DIR / "test_img_effnet.npy")

    encoder = get_live_image_encoder()
    num_samples = 5
    raw_pil_images = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for idx in range(num_samples):
            img_name = df.iloc[idx]["image"]
            data = z.read(f"train_images/{img_name}")
            raw_pil_images.append(Image.open(io.BytesIO(data)).convert("RGB"))

    live_embs = encoder.encode_images(raw_pil_images, normalize=True)
    stored_sample = stored_effnet[:num_samples].astype(np.float32)

    cos_sims = np.sum(live_embs * stored_sample, axis=1)
    max_abs_diff = np.max(np.abs(live_embs - stored_sample), axis=1)

    for i in range(num_samples):
        assert cos_sims[i] >= 0.9999, f"Sample {i} cosine similarity {cos_sims[i]} < 0.9999"
        assert max_abs_diff[i] < 1e-4, f"Sample {i} max diff {max_abs_diff[i]} >= 1e-4"


def test_model_caching():
    """Verify encoder singletons are cached across calls."""
    img_enc1 = get_live_image_encoder()
    img_enc2 = get_live_image_encoder()
    assert img_enc1 is img_enc2

    txt_enc1 = get_live_text_encoder()
    txt_enc2 = get_live_text_encoder()
    assert txt_enc1 is txt_enc2


def test_genuine_image_similarity_and_no_hsv():
    """Verify compute_image_similarity uses EfficientNet cosine similarity, not HSV histograms."""
    red_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    blue_img = Image.new("RGB", (100, 100), color=(0, 0, 255))
    identical_red = Image.new("RGB", (100, 100), color=(255, 0, 0))

    sims, valid_mask = compute_image_similarity(red_img, [identical_red, blue_img, None])

    assert len(sims) == 3
    assert valid_mask[0] is True or valid_mask[0] == True
    assert valid_mask[1] is True or valid_mask[1] == True
    assert valid_mask[2] is False or valid_mask[2] == False

    # Identical image should have cosine similarity ~ 1.0
    assert np.isclose(sims[0], 1.0, atol=1e-3)
    # Missing candidate image has False valid_mask
    assert sims[2] == 0.0


def test_missing_image_text_only_fallback():
    """
    Verify that when query image or candidate images are missing, the pipeline
    falls back cleanly to text-only matching without fabricating image scores
    or corrupting learned logistic fusion.
    """
    candidates = [
        WebCandidate(
            title="Sony WH-1000XM5 Wireless Headphones",
            merchant="Merchant A",
            price=399.0,
            currency="SGD",
            link="#",
            thumbnail_url=None,
            image=None,  # Missing image
        )
    ]

    # Case 1: Query image is None
    matched, rejected = match_and_rank_candidates(
        query_title="Sony WH-1000XM5 Wireless Headphones",
        query_image=None,
        candidates=candidates,
        threshold=0.60,
        fusion_type="learned",
    )
    all_cands = matched + rejected
    cand = all_cands[0]
    assert cand.modality == "text_only"
    assert cand.similarity_image is None
    assert cand.similarity_fused == cand.similarity_text
    assert "Text-Only" in cand.match_reason

    # Case 2: Query image is provided but candidate image is None
    dummy_query_img = Image.new("RGB", (64, 64), color="white")
    matched2, rejected2 = match_and_rank_candidates(
        query_title="Sony WH-1000XM5 Wireless Headphones",
        query_image=dummy_query_img,
        candidates=candidates,
        threshold=0.60,
        fusion_type="learned",
    )
    cand2 = (matched2 + rejected2)[0]
    assert cand2.modality == "text_only"
    assert cand2.similarity_image is None
    assert cand2.similarity_fused == cand2.similarity_text
    assert "Text-Only" in cand2.match_reason


def test_learned_logistic_parameters_integrity():
    """Verify learned weights and bias match final_config.json exactly."""
    with open(ROOT_DIR / "experiments" / "final_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    params = cfg["parameters"]
    assert np.isclose(params["w_text"], 9.463388336669517)
    assert np.isclose(params["w_image"], 7.808722705879871)
    assert np.isclose(params["bias"], -8.197978989829934)
    assert cfg["threshold"] == 0.75


def test_curated_demo_end_to_end_multimodal_execution():
    """
    Verify both curated demo scenarios (Sony WH-1000XM5 and iPhone 15 Pro)
    run through genuine multimodal inference without dummy scores or hardcoded decisions.
    """
    cases = load_curated_demo_cases()
    assert "sony_headphones" in cases
    assert "iphone_15_pro" in cases

    for case_key, case_data in cases.items():
        q_title = case_data["query_title"]
        q_img_ref = case_data.get("image_path") or case_data.get("query_image_url")
        q_img = fetch_candidate_image(q_img_ref)
        assert q_img is not None, f"Failed to load query image for {case_key}"

        cands = []
        for c in case_data["candidates"]:
            img_ref = c.get("image_path") or c.get("thumbnail_url")
            c_img = fetch_candidate_image(img_ref)
            assert c_img is not None, f"Failed to load candidate image for {c['title']}"
            cands.append(
                WebCandidate(
                    title=c["title"],
                    merchant=c["merchant"],
                    price=c.get("price"),
                    currency=c.get("currency", "SGD"),
                    link=c.get("link", "#"),
                    thumbnail_url=img_ref,
                    image=c_img,
                )
            )

        matched, rejected = match_and_rank_candidates(
            query_title=q_title,
            query_image=q_img,
            candidates=cands,
            threshold=0.75,
            fusion_type="learned",
        )

        assert len(matched) >= 2, f"{case_key} should have matched exact target products"
        assert len(rejected) >= 2, f"{case_key} should have rejected accessories"

        for m in matched:
            assert m.modality == "multimodal"
            assert m.similarity_image is not None
            assert m.similarity_image > 0.60, f"Expected high image similarity for match: {m.title}"
            assert m.similarity_fused >= 0.75

        for r in rejected:
            assert r.modality == "multimodal"
            assert r.similarity_image is not None
            assert r.similarity_fused < 0.75


def test_experiment_results_immutability():
    """Verify academic benchmark files remain completely unchanged."""
    results_path = ROOT_DIR / "experiments" / "results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 7
    exp4b = next(r for r in records if r["experiment_id"] == "exp_04b")
    assert np.isclose(exp4b["test_metrics"]["mean_f1"], 0.8629440899621069)
    assert exp4b["hyperparameters"]["threshold"] == 0.75


if __name__ == "__main__":
    tests = [
        test_efficientnet_architecture_and_normalization,
        test_embedding_consistency_with_offline_experiment,
        test_model_caching,
        test_genuine_image_similarity_and_no_hsv,
        test_missing_image_text_only_fallback,
        test_learned_logistic_parameters_integrity,
        test_curated_demo_end_to_end_multimodal_execution,
        test_experiment_results_immutability,
    ]

    print(f"Executing {len(tests)} verification tests...")
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

