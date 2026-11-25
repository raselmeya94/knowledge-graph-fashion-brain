"""Modal 'possible' must never become brand/buyer; capability asks keep prior party."""

from __future__ import annotations

from fashion_brain.ask import (
    _is_grammar_party_name,
    _is_price_history_question,
    _reliable_filter,
    is_capability_followup_ask,
    is_price_trend_analysis_ask,
    sanitize_party_filters,
)
from fashion_brain.agents.query_understanding import _extract_entity_candidate
from fashion_brain.agents.turn_planner import (
    TurnPlan,
    _action_from_confirmation_text,
    resolve_scope,
)
from knowledge_graph.retrieval.ecom import price_history_mode


def test_possible_is_grammar_not_party():
    assert _is_grammar_party_name("possible") is True
    assert _reliable_filter("possible") is False
    assert sanitize_party_filters({"brand": "possible", "buyer": "possible"}) == {}


def test_usd_is_not_a_buyer_or_brand():
    assert _is_grammar_party_name("USD") is True
    assert _is_grammar_party_name("gbp") is True
    assert sanitize_party_filters({"brand": "primark", "buyer": "USD"}) == {"brand": "primark"}
    assert sanitize_party_filters({"buyer": "usd"}) == {}


def test_capability_followup_detector():
    assert is_capability_followup_ask("is that possible price trend analysis?") is True
    assert is_capability_followup_ask("How many products does Primark have?") is False


def test_make_trending_analysis_is_price_change_trend():
    """'make trending analysis' must NOT become a static Primark product-mix profile."""
    q = "make trending analysis"
    assert is_price_trend_analysis_ask(q) is True
    assert _is_price_history_question(q) is True
    assert price_history_mode(q) == "trend"
    ask = _action_from_confirmation_text(
        "Do you want a trending analysis for Primark?",
        {"brand": "primark"},
    )
    assert "price" in ask.casefold()
    assert "trend" in ask.casefold()
    assert "primark" in ask.casefold()


def test_extract_entity_ignores_possible_price_trend():
    assert _extract_entity_candidate("is that possible price trend analysis?") is None
    assert _extract_entity_candidate("analyze Primark") is not None


def test_resolve_scope_followup_keeps_primark_not_possible():
    plan = TurnPlan(
        llm_ok=True,
        turn_type="follow_up",
        filters={"brand": "possible"},
        carry_filters=True,
    )
    resolve_scope(plan, {"active_filters": {"brand": "primark"}})
    # sanitize drops brand=possible; prior primark remains unless overwritten by empty
    assert plan.effective_filters.get("brand") == "primark"
