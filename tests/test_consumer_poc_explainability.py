import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
from PIL import Image

from src.config import ROOT_DIR
from src.retrieval.web_search import (
    WebCandidate,
    load_curated_demo_cases,
    fetch_candidate_image,
)
from src.pipeline.live_matcher import (
    match_and_rank_candidates,
    categorize_model_disagreements,
    group_candidates_by_currency,
)


def test_retrieval_rank_preservation():
    """Verify WebCandidate preserves 1-indexed retrieval order."""
    c1 = WebCandidate(title="Item 1", merchant="Store A", price=100.0, currency="SGD", link="#", thumbnail_url=None, retrieval_rank=1)
    c2 = WebCandidate(title="Item 2", merchant="Store B", price=90.0, currency="SGD", link="#", thumbnail_url=None, retrieval_rank=2)
    assert c1.retrieval_rank == 1
    assert c2.retrieval_rank == 2


def test_dual_model_single_candidate_evaluation_and_disagreements():
    """
    Verify both Fine-Tuned Text-Only and Multimodal Fusion evaluate the EXACT SAME candidate set,
    and verify that accessory hard negatives (e.g. Sony WH-1000XM5 ear pads) are accepted by
    Text-Only (due to title overlap) but safely rejected by Multimodal (due to low visual similarity).
    """
    cases = load_curated_demo_cases()
    assert "sony_headphones" in cases
    sony = cases["sony_headphones"]

    q_title = sony["query_title"]
    q_img = fetch_candidate_image(sony["image_path"])
    assert q_img is not None

    candidates = []
    for idx, c in enumerate(sony["candidates"]):
        img = fetch_candidate_image(c["image_path"])
        candidates.append(
            WebCandidate(
                title=c["title"],
                merchant=c["merchant"],
                price=c.get("price"),
                currency=c.get("currency", "SGD"),
                link=c.get("link", "#"),
                thumbnail_url=c["thumbnail_url"],
                image=img,
                retrieval_rank=idx + 1,
            )
        )

    matched, rejected = match_and_rank_candidates(
        query_title=q_title,
        query_image=q_img,
        candidates=candidates,
        threshold=0.75,
        alpha=0.60,
        fusion_type="learned",
        text_threshold=0.60,
    )

    # Verify all candidates have both text_only and multimodal fields populated
    for c in candidates:
        assert hasattr(c, "text_only_sim")
        assert hasattr(c, "text_only_is_match")
        assert hasattr(c, "multimodal_sim")
        assert hasattr(c, "multimodal_is_match")

    disagreements = categorize_model_disagreements(candidates)
    consensus_matches = disagreements["consensus_matches"]
    text_only_matches = disagreements["text_only_matches"]  # Disagreements filtered by vision
    consensus_rejections = disagreements["consensus_rejections"]

    # The 2 exact headphone candidates (Amazon SG, Shopee Mall) must be consensus matches
    assert len(consensus_matches) >= 2
    consensus_merchants = [c.merchant for c in consensus_matches]
    assert "Amazon SG" in consensus_merchants
    assert "Shopee Mall" in consensus_merchants

    # Candidate: "Replacement Ear Pads Cushions for Sony WH-1000XM5 / XM4 Headphone" (AudioAccessories)
    # Text similarity contains "Sony WH-1000XM5", giving text_sim >= 0.60
    # But image similarity is low (~0.227), giving multimodal score ~0.327 < 0.75
    # Therefore, it MUST be identified as a disagreement: accepted by text, rejected by vision!
    ear_pad_cand = next((c for c in candidates if "Ear Pads" in c.title), None)
    assert ear_pad_cand is not None
    assert ear_pad_cand.text_only_is_match is True, f"Expected text match on ear pads, got sim={ear_pad_cand.text_only_sim}"
    assert ear_pad_cand.multimodal_is_match is False, f"Expected multimodal rejection on ear pads, got score={ear_pad_cand.multimodal_sim}"
    assert ear_pad_cand in text_only_matches, "Ear pads must be categorized under text_only_matches (filtered by vision)"


def test_currency_aware_grouping():
    """Verify group_candidates_by_currency correctly segregates offers by currency without cross-currency price ranking."""
    c_sgd_1 = WebCandidate(title="Sony XM5", merchant="Shop SG 1", price=399.0, currency="SGD", link="#", thumbnail_url=None)
    c_sgd_2 = WebCandidate(title="Sony XM5", merchant="Shop SG 2", price=369.0, currency="SGD", link="#", thumbnail_url=None)
    c_usd_1 = WebCandidate(title="Sony XM5", merchant="Shop US 1", price=349.0, currency="USD", link="#", thumbnail_url=None)
    c_sgd_unpriced = WebCandidate(title="Sony XM5", merchant="Catalog SG", price=None, currency="SGD", link="#", thumbnail_url=None)

    candidates = [c_sgd_1, c_sgd_2, c_usd_1, c_sgd_unpriced]
    grouped = group_candidates_by_currency(candidates)

    assert "SGD" in grouped
    assert "USD" in grouped

    # In SGD group:
    sgd_info = grouped["SGD"]
    assert len(sgd_info["priced_offers"]) == 2
    assert len(sgd_info["unpriced_offers"]) == 1
    assert sgd_info["lowest_offer"] is c_sgd_2
    assert sgd_info["lowest_offer"].price == 369.0

    # In USD group:
    usd_info = grouped["USD"]
    assert len(usd_info["priced_offers"]) == 1
    assert usd_info["lowest_offer"] is c_usd_1
    assert usd_info["lowest_offer"].price == 349.0


def test_no_misleading_claims_in_ui_and_docs():
    """Verify that forbidden misleading phrases are not present in key user-facing files."""
    forbidden_terms = [
        "Verified exact product matches",
        "Confirmed identical products",
        "Best price on the internet",
        "Best Price Found",
    ]

    files_to_check = [
        ROOT_DIR / "app" / "pages" / "3_Live_Matcher.py",
        ROOT_DIR / "app" / "app.py",
        ROOT_DIR / "README.md",
    ]

    for fpath in files_to_check:
        content = fpath.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term '{term}' found in {fpath.name}"


if __name__ == "__main__":
    tests = [
        test_retrieval_rank_preservation,
        test_dual_model_single_candidate_evaluation_and_disagreements,
        test_currency_aware_grouping,
        test_no_misleading_claims_in_ui_and_docs,
    ]

    print(f"Executing {len(tests)} consumer POC explainability tests...")
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
