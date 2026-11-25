"""Dynamic live-brand scope enrichment — no hardcoded retailer tables in the assert path."""

from __future__ import annotations

from fashion_brain.ask import (
    enrich_scope_from_question,
    ground_filters_from_live_brands,
    live_brands_mentioned,
    question_has_party_deixis,
)


LIVE = ["Primark", "Zara", "Matalan", "C&A", "Walmart"]


def test_live_brands_mentioned_is_dynamic():
    assert live_brands_mentioned("How many products does Primark have?", LIVE) == ["Primark"]
    assert set(live_brands_mentioned("Primark or Zara?", LIVE)) == {"Primark", "Zara"}
    assert live_brands_mentioned("how many sweaters?", LIVE) == []


def test_named_brand_count_grounds_party():
    filters, group_by = enrich_scope_from_question(
        "How many products does Primark have?",
        {},
        brand_names=LIVE,
    )
    assert filters.get("brand") == "Primark"
    assert group_by is None


def test_explicit_buyer_stays_a_buyer_filter():
    filters, _ = enrich_scope_from_question(
        "How many products does Primark buyer have?",
        {},
        brand_names=LIVE,
    )
    assert filters.get("buyer") == "Primark"
    assert "brand" not in filters


def test_single_party_replaces_multi_party_sticky():
    filters, _ = enrich_scope_from_question(
        "analysis primark",
        {"brand": "Primark | Zara", "color": "red"},
        brand_names=LIVE,
    )
    assert filters.get("brand") == "Primark"
    assert "Zara" not in str(filters.get("brand"))
    assert "color" not in filters


def test_of_them_keeps_prior_party_and_adds_garment():
    assert question_has_party_deixis("And how many of them are sweaters?")
    filters, _ = enrich_scope_from_question(
        "And how many of them are sweaters?",
        {"brand": "Primark", "buyer": "Primark"},
        brand_names=LIVE,
    )
    assert filters.get("brand") == "Primark"
    assert "sweater" in str(filters.get("garment") or "").casefold()


def test_tell_me_red_sweater_no_tell_party():
    filters, _ = enrich_scope_from_question(
        "now tell me how many sweater of them are red color?",
        {"brand": "Primark"},
        brand_names=LIVE,
    )
    assert filters.get("brand") == "Primark"
    assert "tell" not in str(filters.get("brand") or "").casefold()
    assert "red" in str(filters.get("color") or "").casefold()


def test_explore_category_sets_group_by_and_keeps_party():
    filters, group_by = enrich_scope_from_question(
        "explore category",
        {"brand": "Primark", "garment": "sweater", "color": "red"},
        brand_names=LIVE,
    )
    assert filters.get("brand") == "Primark"
    assert group_by == "category"
    assert "garment" not in filters
    assert "color" not in filters


def test_zara_count_replaces_primark_and_clears_stale_facets():
    filters, _ = enrich_scope_from_question(
        "How many products does Zara have?",
        {"brand": "Primark | Zara", "garment": "sweater", "color": "red"},
        brand_names=LIVE,
    )
    assert filters.get("brand") == "Zara"
    assert "garment" not in filters
    assert "color" not in filters


def test_ground_replaces_even_when_brand_already_set():
    out = ground_filters_from_live_brands(
        "analysis primark",
        {"brand": "Primark | Zara"},
        LIVE,
    )
    assert out.get("brand") == "Primark"
