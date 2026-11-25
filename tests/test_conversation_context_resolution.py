"""Conversation context: current ask wins; history is not permanent filters."""

from __future__ import annotations

from types import SimpleNamespace

from fashion_brain.agents.conversation_context import ConversationContextAgent
from fashion_brain.ask import enrich_scope_from_question


def _platform():
    return SimpleNamespace(settings=SimpleNamespace(answer_mode="template"))


def test_deixis_continuation_carries_filters():
    agent = ConversationContextAgent()
    decision = agent.analyze(
        "And how many of them are sweaters?",
        history=[{"role": "user", "content": "How many products does Primark have?"}],
        conversation_state={
            "last_graph_id": "ecom_products",
            "last_result_id": "r1",
            "active_filters": {"brand": "primark"},
        },
        platform=_platform(),
    )
    assert decision.intent_type == "continuation"
    assert decision.is_contextual is True
    assert decision.active_filters.get("brand") == "primark"


def test_named_party_is_new_independent():
    agent = ConversationContextAgent()
    decision = agent.analyze(
        "How many products does Zara have?",
        history=[
            {"role": "user", "content": "How many products does Primark have?"},
            {"role": "user", "content": "now tell me how many sweater of them are red color?"},
        ],
        conversation_state={
            "last_graph_id": "ecom_products",
            "last_result_id": "r1",
            "active_filters": {"brand": "primark", "garment": "sweater", "color": "red"},
        },
        platform=_platform(),
    )
    assert decision.intent_type == "new_independent_question"
    assert decision.is_contextual is False
    assert decision.active_filters == {}
    assert "brand" in decision.removed_filters or decision.inheritance_reason


def test_full_analysis_drops_restrictive_facets():
    agent = ConversationContextAgent()
    decision = agent.analyze(
        "Give me a full analysis of Primark",
        history=[
            {"role": "user", "content": "Of those Primark sweaters, how many are red?"},
        ],
        conversation_state={
            "last_graph_id": "ecom_products",
            "last_result_id": "r1",
            "active_filters": {"brand": "primark", "garment": "sweater", "color": "red"},
        },
        platform=_platform(),
    )
    assert decision.intent_type == "partial_context_reuse"
    assert decision.active_filters.get("brand") == "primark"
    assert "garment" not in decision.active_filters
    assert "color" not in decision.active_filters
    assert decision.removed_filters.get("color") == "red" or "color" in decision.removed_filters


def test_enrich_scope_analysis_clears_sticky_color():
    filters, _ = enrich_scope_from_question(
        "Give me a full analysis of Primark",
        {"brand": "primark", "garment": "sweater", "color": "red"},
        brand_names=["Primark", "Zara"],
    )
    assert str(filters.get("brand") or "").casefold() == "primark"
    assert "color" not in filters
    assert "garment" not in filters


def test_enrich_scope_overall_compare_clears_color():
    filters, _ = enrich_scope_from_question(
        "Overall, who has more products — Primark or Zara?",
        {"brand": "primark", "color": "red", "garment": "sweater"},
        brand_names=["Primark", "Zara"],
    )
    assert "primark" in str(filters.get("brand") or "").casefold()
    assert "zara" in str(filters.get("brand") or "").casefold()
    assert "color" not in filters


def test_ecom_dataset_total_clears_sticky_primark():
    agent = ConversationContextAgent()
    decision = agent.analyze(
        "how many products are in the e-comm data?",
        history=[
            {"role": "user", "content": "How many products does Primark have?"},
            {"role": "assistant", "content": "There are 4,058 products with brand primark."},
        ],
        conversation_state={
            "last_graph_id": "ecom_products",
            "last_result_id": "r1",
            "active_filters": {"brand": "primark"},
        },
        platform=_platform(),
    )
    assert decision.intent_type == "new_independent_question"
    assert decision.is_contextual is False
    assert decision.active_filters == {}
    assert "brand" not in decision.active_filters


def test_enrich_scope_dataset_total_drops_prior_brand():
    filters, _ = enrich_scope_from_question(
        "How many products are in the e-comm data?",
        {"brand": "primark"},
        brand_names=["Primark", "Zara"],
    )
    assert "brand" not in filters
