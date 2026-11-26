"""D-section price-trend ask modes + product line chart."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from knowledge_graph.retrieval.ecom import (
    _price_pct_threshold,
    append_neo4j_price_history_answer,
    price_history_mode,
)


def test_d_section_modes():
    assert price_history_mode("How has the price of Zara style 04786296 changed?") == "product"
    assert price_history_mode("Which products had the biggest price drops?") == "movers_down"
    assert price_history_mode("Which products had the biggest price increases?") == "movers_up"
    assert price_history_mode("Overall, how are Zara prices moving?") == "trend"
    assert price_history_mode("Which categories are getting cheaper?") == "trend_by_category"
    assert price_history_mode("Which brands have the most price changes?") == "events_by_brand"
    assert price_history_mode("Which products are swinging a lot on price?") == "volatility"
    assert (
        price_history_mode(
            "For similar Zara and Primark items in the same category, "
            "how do historical price moves compare?"
        )
        == "brand_compare"
    )
    assert price_history_mode("Show recent same-currency price drops of more than 20%") == "movers_down"
    assert price_history_mode("How many price change events do we have recorded?") == "count_events"
    # Long merchant brief: "price-trend analysis" stays aggregate trend, not movers_down.
    assert (
        price_history_mode(
            "I already reviewed Primark's assortment mix. What I need now is a "
            "price-trend analysis for Primark on e-commerce: how recorded "
            "price-change events are moving, by currency if needed, and which "
            "products look most active — not another attribute-wise product-mix profile."
        )
        == "trend"
    )


def test_pct_threshold_parser():
    assert _price_pct_threshold("Show recent same-currency price drops of more than 20%") == 20.0
    assert _price_pct_threshold("drops over 15.5%") == 15.5
    assert _price_pct_threshold("biggest price drops") is None


def test_product_history_emits_line_chart():
    rows = [
        {
            "at": "2026-11-01T10:00:00",
            "old_price": 80.0,
            "new_price": 60.0,
            "delta_pct": -25.0,
            "direction": "down",
            "currency": "GBP",
            "currency_changed": False,
            "name": "SATIN SHORTS",
            "product_id": "ecom:zara:05919713",
        },
        {
            "at": "2026-11-05T23:05:59",
            "old_price": 60.0,
            "new_price": 39.95,
            "delta_pct": -33.4,
            "direction": "down",
            "currency": "GBP",
            "currency_changed": False,
            "name": "SATIN SHORTS",
            "product_id": "ecom:zara:05919713",
        },
    ]
    lines: list[str] = []
    evidence: list = []
    caveats: list[str] = []
    with patch(
        "knowledge_graph.retrieval.ecom.cypher_price_history",
        return_value=rows,
    ), patch(
        "knowledge_graph.retrieval.ecom.price_history_mode",
        return_value="product",
    ), patch(
        "knowledge_graph.retrieval.ecom.cypher_count_nodes",
        return_value=2,
    ):
        ok = append_neo4j_price_history_answer(
            SimpleNamespace(),
            SimpleNamespace(),
            "ecom_products",
            {"product_id": "05919713", "brand": "zara"},
            "For Zara Satin Shorts with Lace Trim (05919713), what was the old price?",
            10,
            lines,
            evidence,
            caveats,
        )
    assert ok is True
    text = "\n".join(lines)
    assert "```chart" in text
    payload = json.loads(text.split("```chart")[1].split("```")[0].strip())
    assert payload["type"] == "line"
    assert len(payload["values"]) >= 2
