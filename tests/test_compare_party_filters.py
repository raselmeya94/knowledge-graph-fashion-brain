"""Multi-brand compare must stay side-by-side, not one merged total."""

from __future__ import annotations

from fashion_brain.agents.answer_card import AnswerCard, SourceResult, safe_fallback
from fashion_brain.ask import ensure_compare_party_filters


def test_ensure_compare_recovers_from_question():
    out = ensure_compare_party_filters({}, "Compare Primark and Zara assortment profiles")
    assert "primark" in out.get("brand", "").casefold()
    assert "zara" in out.get("brand", "").casefold()
    assert "|" in out["brand"]


def test_ensure_compare_keeps_existing_pipe():
    out = ensure_compare_party_filters({"brand": "Primark | Zara"}, "anything")
    assert out["brand"] == "Primark | Zara"


def test_safe_fallback_side_by_side_not_single_line():
    card = AnswerCard(
        question="Compare Primark and Zara assortment profiles",
        intent="analysis",
        depth="full",
        filters={"brand": "Primark | Zara"},
        results=[SourceResult(source_id="ecom_products", total=9862)],
        context=[
            "### Primark",
            "Product analysis for Primark: **4058** products.",
            "Leading value: Clothes (1464, 36.1%).",
            "### Zara",
            "Product analysis for Zara: **5804** products.",
            "Leading value: Clothing (2000, 34.0%).",
        ],
    )
    text = safe_fallback(card)
    assert "Primark" in text and "Zara" in text
    assert "4058" in text.replace(",", "") or "4,058" in text
    assert "Leading attributes:" not in text  # old single-line collapse
