"""Deterministic QU helpers: classify_reply, resolve_entity, budget."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fashion_brain.agents.qu_fast import (
    budget_exhausted,
    cached,
    classify_reply,
    clear_cache,
    resolve_entity,
)


def test_classify_reply_yes_no_option_bangla():
    assert classify_reply("yes")[0] == "yes"
    assert classify_reply("হ্যাঁ")[0] == "yes"
    assert classify_reply("nope")[0] == "no"
    assert classify_reply("2", ["a", "b", "c"]) == ("option", "b")
    assert classify_reply("yes, but only sweaters")[0] == "edit"
    assert classify_reply("hello there")[0] == "other"


def test_budget_exhausted():
    assert budget_exhausted(SimpleNamespace(turn_count=2)) is False
    assert budget_exhausted(SimpleNamespace(turn_count=3)) is True


def test_cached_ttl():
    clear_cache()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "x"

    assert cached("k", fn, ttl=60) == "x"
    assert cached("k", fn, ttl=60) == "x"
    assert calls["n"] == 1
    clear_cache()


def test_resolve_entity_exact_and_typo():
    names = ["Zara", "Primark", "H&M"]
    with (
        patch("fashion_brain.ask.live_brand_names", return_value=names),
        patch("fashion_brain.ask.live_brands_mentioned", side_effect=lambda text, ns: [
            n for n in ns if n.casefold() in (text or "").casefold()
        ]),
    ):
        clear_cache()
        brand, sugs = resolve_entity(SimpleNamespace(), "analyze Zara")
        assert brand == "Zara"
        assert sugs == []
        clear_cache()
        brand, sugs = resolve_entity(SimpleNamespace(), "zarra")
        assert brand == "Zara" or (brand is None and "Zara" in sugs)


def test_resolve_entity_does_not_map_price_to_prince():
    """Measure word 'price' must not fuzzy-match brand Prince (ratio≈0.91)."""

    names = ["Prince", "Zara", "Primark"]
    with (
        patch("fashion_brain.ask.live_brand_names", return_value=names),
        patch("fashion_brain.ask.live_brands_mentioned", return_value=[]),
    ):
        clear_cache()
        brand, sugs = resolve_entity(
            SimpleNamespace(), "Which products had the biggest price drops?"
        )
        assert brand is None
        assert "Prince" not in sugs
