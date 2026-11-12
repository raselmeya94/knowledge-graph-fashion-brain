"""Post-LLM guard: adverbs like 'frequently' must never become brand/buyer filters."""

from __future__ import annotations

from fashion_brain.ask import _ground_filters, _is_grammar_party_name, resolve_catalog_plan, sanitize_party_filters


def test_frequently_is_grammar_party_name():
    assert _is_grammar_party_name("frequently") is True
    assert _is_grammar_party_name("often") is True
    assert _is_grammar_party_name("primark") is False


def test_sanitize_drops_buyer_frequently():
    cleaned = sanitize_party_filters({"brand": "primark", "buyer": "frequently"})
    assert cleaned.get("brand") == "primark"
    assert "buyer" not in cleaned


def test_ground_filters_drops_buyer_frequently_even_if_in_question():
    kept = _ground_filters(
        {"brand": "primark", "buyer": "frequently"},
        ["decide from past analysis, which type of products are frequently make PRIMARK?"],
    )
    assert kept.get("brand") == "primark"
    assert "buyer" not in kept


def test_keyword_plan_does_not_set_buyer_frequently():
    plan = resolve_catalog_plan(
        "decide from past analysis, which type of products are frequently make PRIMARK?",
        None,
        carry=False,
    )
    filters = plan.get("filters") or {}
    assert filters.get("buyer") != "frequently"
    assert "frequently" not in str(filters.get("buyer") or "").casefold()
    assert "frequently" not in str(filters.get("brand") or "").casefold()
    party = " ".join(
        [
            str(filters.get("brand") or ""),
            str(filters.get("buyer") or ""),
        ]
    ).casefold()
    assert "primark" in party
