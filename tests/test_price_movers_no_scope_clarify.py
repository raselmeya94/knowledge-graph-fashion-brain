"""Unscoped biggest-drop asks must not quiz for brand — run catalog-wide."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fashion_brain.agents.turn_planner import TurnPlan, _maybe_clarify_price_trend_scope


def test_biggest_price_drops_skips_brand_clarify():
    plan = TurnPlan(
        llm_ok=True,
        turn_type="new_question",
        confidence=0.9,
        rewritten_question="Which products had the biggest price drops in e-commerce?",
        filters={},
        metric="price",
        preferred_graph="ecom_products",
    )
    with patch(
        "knowledge_graph.retrieval.ecom.price_history_mode",
        return_value="movers_down",
    ):
        _maybe_clarify_price_trend_scope(
            SimpleNamespace(),
            "Which products had the biggest price drops in e-commerce?",
            plan,
            {},
            None,
        )
    assert plan.needs_clarification is False
    assert not plan.clarification_question


def test_overall_trend_still_asks_for_brand():
    plan = TurnPlan(
        llm_ok=True,
        turn_type="new_question",
        confidence=0.9,
        rewritten_question="Overall, how are prices moving?",
        filters={},
        metric="price",
        preferred_graph="ecom_products",
    )
    with (
        patch(
            "knowledge_graph.retrieval.ecom.price_history_mode",
            return_value="trend",
        ),
        patch(
            "knowledge_graph.retrieval.ecom._price_trend_scope_options",
            return_value=[
                {
                    "field": "brand",
                    "value": "zara",
                    "label": "Brand: zara",
                    "filters": {"brand": "zara"},
                }
            ],
        ),
        patch(
            "knowledge_graph.retrieval.ecom.registry_product_graphs",
            return_value=["ecom_products"],
        ),
    ):
        _maybe_clarify_price_trend_scope(
            SimpleNamespace(),
            "Overall, how are prices moving?",
            plan,
            {},
            None,
        )
    assert plan.needs_clarification is True
    assert "brand" in (plan.clarification_question or "").casefold()
