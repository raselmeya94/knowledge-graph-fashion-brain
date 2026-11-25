"""Clarification verbs like 'explore category' must not invent brand=explore."""

from __future__ import annotations

from fashion_brain.agents._impl import IntentSummary, _apply_session_scope
from fashion_brain.ask import _is_grammar_party_name, _is_party_token, resolve_catalog_plan, sanitize_party_filters
from fashion_brain.api.app import _graph_from_routes


def test_explore_is_grammar_not_party():
    assert _is_grammar_party_name("explore") is True
    assert _is_party_token("explore") is False


def test_explore_category_keyword_plan_has_no_explore_party():
    plan = resolve_catalog_plan("explore category", None, carry=False)
    filters = plan.get("filters") or {}
    assert "explore" not in str(filters.get("brand") or "").casefold()
    assert "explore" not in str(filters.get("buyer") or "").casefold()


def test_session_scope_keeps_primark_for_explore_category():
    summary = IntentSummary(
        actual_intent="Calculate the product count",
        text="",
        filters={"brand": "explore", "buyer": "explore"},
    )
    state = {
        "active_filters": {"brand": "primark", "buyer": "primark"},
        "clarification": {
            "pending_slot": "scope_or_direction",
            "confirmed": {"entity": "primark", "filters": {"brand": "primark"}},
        },
    }
    out = _apply_session_scope(summary, state, "explore category")
    assert out.filters.get("brand") == "primark"
    assert "explore" not in str(out.filters.get("brand") or "").casefold()
    assert out.group_by == "category"
    assert out.references_prior_chat is True


def test_sanitize_drops_explore_party():
    cleaned = sanitize_party_filters({"brand": "explore", "buyer": "explore", "garment": "sweater"})
    assert "brand" not in cleaned
    assert "buyer" not in cleaned
    assert cleaned.get("garment") == "sweater"


def test_graph_from_routes_ignores_clarification_slots():
    assert _graph_from_routes(["catalog", "clarification", "scope_or_direction"]) is None
    assert _graph_from_routes(["breakdown", "scope_or_direction"]) is None
    assert _graph_from_routes(["breakdown", "ecom_products"]) == "ecom_products"
    assert _graph_from_routes(["clarification", "final_confirm", "plmex_products"]) == "plmex_products"
