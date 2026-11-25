"""Conversational verbs like 'tell me' must never become brand/buyer filters."""

from __future__ import annotations

from fashion_brain.ask import (
    _is_grammar_party_name,
    _is_party_token,
    ground_filters_from_live_brands,
    resolve_catalog_plan,
    sanitize_party_filters,
)


def test_tell_is_grammar_not_party():
    assert _is_grammar_party_name("tell") is True
    assert _is_party_token("tell") is False
    assert _is_grammar_party_name("primark") is False


def test_sanitize_drops_brand_tell():
    cleaned = sanitize_party_filters(
        {"brand": "tell", "buyer": "tell", "garment": "sweater", "color": "red"}
    )
    assert "brand" not in cleaned
    assert "buyer" not in cleaned
    assert cleaned.get("garment") == "sweater"
    assert cleaned.get("color") == "red"


def test_keyword_plan_tell_me_red_sweater_has_no_tell_party():
    plan = resolve_catalog_plan(
        "now tell me how many sweater of them are red color?",
        None,
        carry=False,
    )
    filters = plan.get("filters") or {}
    assert filters.get("garment") == "sweater"
    assert filters.get("color") == "red"
    assert "tell" not in str(filters.get("brand") or "").casefold()
    assert "tell" not in str(filters.get("buyer") or "").casefold()


def test_ground_live_brands_drops_tell_keeps_primark_from_question():
    grounded = ground_filters_from_live_brands(
        "now tell me how many sweater of them are red color?",
        {"brand": "tell", "buyer": "tell", "garment": "sweater", "color": "red"},
        ["Primark", "Zara"],
    )
    assert "tell" not in str(grounded.get("brand") or "").casefold()
    assert "tell" not in str(grounded.get("buyer") or "").casefold()
    # No Primark in this question text → leave brand empty for prior-chat carry.
    assert not grounded.get("brand")
    assert not grounded.get("buyer")
