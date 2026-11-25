"""'Which product is trending' ranks product activity — not brand currency quiz."""

from __future__ import annotations

from fashion_brain.ask import is_price_trend_analysis_ask, is_trending_product_ask
from fashion_brain.agents.turn_planner import plan_turn
from knowledge_graph.retrieval.ecom import (
    _canonical_product_url,
    _human_product_label,
    _title_from_product_url,
    price_history_mode,
)


def test_trending_product_ask_detected():
    q = "which product present time more trending on primark brand"
    assert is_trending_product_ask(q) is True
    assert is_price_trend_analysis_ask(q) is False
    assert price_history_mode(q) == "trending_products"


def test_brand_price_trend_still_average_trend():
    q = "What is the price trend for Primark?"
    assert is_trending_product_ask(q) is False
    assert is_price_trend_analysis_ask(q) is True
    assert price_history_mode(q) == "trend"


def test_trending_analysis_phrase_stays_trend():
    q = "make trending analysis for Primark"
    assert is_trending_product_ask(q) is False
    assert price_history_mode(q) == "trend"


def test_turn_planner_locks_trending_products_without_llm():
    q = "which product present time more trending on primark brand"
    plan = plan_turn(None, q, [], {}, catalog=None)
    assert plan.llm_ok is True
    assert plan.needs_clarification is False
    assert plan.metric == "price"
    assert plan.group_by is None
    assert "average price change trends" not in (plan.rewritten_question or "").casefold()
    assert "trending" in (plan.rewritten_question or "").casefold()
    assert price_history_mode(plan.rewritten_question) == "trending_products"


def test_product_title_from_url_when_name_null():
    url = (
        "https://www.primark.com/en-us/r/baby/explore-by-product/en-us/p/"
        "0-18mths-colorblock-borg-newborn-romper-multi-991180786935"
    )
    assert "Colorblock Borg Newborn Romper" in _title_from_product_url(url)
    assert "0-18mths" in _title_from_product_url(url)
    label = _human_product_label(
        "null", "ecom:primark:primark:991180786935", product_url=url
    )
    assert "991180786935" not in label
    assert "Romper" in label


def test_color_most_running_is_facet_not_price_trend():
    from fashion_brain.ask import (
        dominant_facet_from_question,
        is_dominant_facet_ask,
        is_trending_product_ask,
    )
    from fashion_brain.agents.turn_planner import plan_turn

    q = "which color is most running in current month for primark brand"
    assert is_dominant_facet_ask(q) is True
    assert dominant_facet_from_question(q) == "color"
    assert is_trending_product_ask(q) is False
    plan = plan_turn(None, q, [], {}, catalog=None)
    assert plan.metric == "count"
    assert plan.group_by == "color"
    assert price_history_mode(plan.rewritten_question) is None


def test_canonical_primark_pdp_url_strips_browse_path():
    browse = (
        "https://www.primark.com/en-us/r/baby/explore-by-product/en-us/p/"
        "0-18mths-2pk-disneys-bambi-long-sleeve-newborn-t-shirts-pink-991183621108"
    )
    assert _canonical_product_url(browse) == (
        "https://www.primark.com/en-us/p/"
        "0-18mths-2pk-disneys-bambi-long-sleeve-newborn-t-shirts-pink-991183621108"
    )
    already = (
        "https://www.primark.com/en-us/p/"
        "0-18mths-2pk-disneys-bambi-long-sleeve-newborn-t-shirts-pink-991183621108"
    )
    assert _canonical_product_url(already) == already
