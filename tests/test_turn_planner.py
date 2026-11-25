"""Turn planner: scope resolve, interruptions, next_state (no live LLM)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fashion_brain.agents.turn_planner import (
    TurnPlan,
    interruption_reply,
    next_state,
    plan_turn,
    resolve_scope,
    scope_note,
)


def test_new_question_drops_prior_brand():
    plan = TurnPlan(llm_ok=True, turn_type="new_question", filters={})
    resolve_scope(plan, {"active_filters": {"brand": "primark"}})
    assert plan.effective_filters == {}


def test_brand_compare_does_not_carry_prior_color():
    """LLM often marks Primark vs Zara as follow_up after a color count — code must drop color."""
    plan = TurnPlan(
        llm_ok=True,
        turn_type="follow_up",
        rewritten_question="Who has more products — Primark or Zara?",
        filters={"brand": "Primark | Zara"},
        carry_filters=True,
    )
    resolve_scope(plan, {"active_filters": {"color": "white"}})
    assert plan.turn_type == "new_question"
    assert plan.carry_filters is False
    assert plan.effective_filters.get("brand")
    assert "color" not in plan.effective_filters


def test_follow_up_carries_and_merges():
    plan = TurnPlan(
        llm_ok=True,
        turn_type="follow_up",
        rewritten_question="And how many of them are red?",
        filters={"color": "red"},
        carry_filters=True,
    )
    resolve_scope(plan, {"active_filters": {"brand": "primark"}})
    assert plan.effective_filters == {"brand": "primark", "color": "red"}


def test_refinement_drop_filters():
    plan = TurnPlan(
        llm_ok=True,
        turn_type="refinement",
        filters={"brand": "zara"},
        drop_filters=["color"],
        carry_filters=True,
    )
    resolve_scope(plan, {"active_filters": {"brand": "primark", "color": "red"}})
    assert plan.effective_filters == {"brand": "zara"}


def test_clarification_interruption_and_streak():
    plan = TurnPlan(
        llm_ok=True,
        needs_clarification=True,
        clarification_question="Which brand?",
        clarification_options=["Primark", "Zara"],
    )
    text = interruption_reply(plan)
    assert text.startswith("Which brand?")
    assert "1. Primark" in text
    state = next_state(plan, {})
    assert state["clarification"]["pending_slot"] == "llm"
    assert state["clarify_streak"] == 1


def test_confirmation_interruption():
    plan = TurnPlan(
        llm_ok=True,
        needs_confirmation=True,
        confirmation_text="Run a full analysis of Primark?",
        effective_filters={"brand": "primark"},
    )
    assert interruption_reply(plan) == "Run a full analysis of Primark?"
    state = next_state(plan, {})
    assert state["confirmation"]["question"].startswith("Run a full")
    assert state["confirmation"]["filters"]["brand"] == "primark"


def test_answers_confirmation_uses_proposed_filters():
    plan = TurnPlan(llm_ok=True, turn_type="answers_confirmation", filters={})
    resolve_scope(
        plan,
        {
            "active_filters": {"brand": "old"},
            "confirmation": {"filters": {"brand": "primark"}, "question": "Analyze Primark?"},
        },
    )
    assert plan.effective_filters == {"brand": "primark"}


def test_confirmation_persists_price_trend_action():
    plan = TurnPlan(
        llm_ok=True,
        needs_confirmation=True,
        confirmation_text="Do you want a price trend analysis for Primark?",
        rewritten_question="Summarize average price change trends for Primark",
        metric="price",
        analysis_depth="focused",
        effective_filters={"brand": "primark"},
    )
    state = next_state(plan, {"active_filters": {"brand": "primark"}})
    conf = state["confirmation"]
    assert conf["metric"] == "price"
    assert "price change trends" in conf["rewritten_question"].casefold()
    assert conf["filters"]["brand"] == "primark"


def test_currency_clarify_pick_usd():
    from fashion_brain.agents.turn_planner import (
        _plan_from_pending_clarification,
        _recover_currency_clarification_into_state,
    )

    state = {
        "active_filters": {"brand": "primark"},
        "clarification": {
            "pending_slot": "price_trend_currency",
            "question": "Which currency should I trend?",
            "options": ["USD only", "GBP only", "Both currencies"],
            "pending_metric": "price",
            "filters": {"brand": "primark"},
            "interpretations": [
                {
                    "restated_question": "Summarize average price change trends for primark in USD",
                    "filters": {"brand": "primark"},
                    "currency": "USD",
                },
                {
                    "restated_question": "Summarize average price change trends for primark in GBP",
                    "filters": {"brand": "primark"},
                    "currency": "GBP",
                },
                {
                    "restated_question": "Summarize average price change trends for primark in both currencies",
                    "filters": {"brand": "primark"},
                    "currency": "BOTH",
                },
            ],
        },
    }
    plan = _plan_from_pending_clarification("USD", state)
    assert plan is not None
    assert plan.metric == "price"
    assert "usd" in plan.rewritten_question.casefold()
    assert plan.filters.get("brand", "").casefold() == "primark"
    assert "buyer" not in plan.filters
    plan_both = _plan_from_pending_clarification("both", state)
    assert plan_both is not None
    assert "both" in plan_both.rewritten_question.casefold()

    # Session clarification lost — recover from history when user says USD.
    empty = {"active_filters": {"brand": "primark"}}
    history = [
        {"role": "user", "content": "price trend for primark"},
        {
            "role": "assistant",
            "content": (
                "Which currency should I trend?\n1. USD only\n2. GBP only\n3. Both currencies"
            ),
        },
    ]
    _recover_currency_clarification_into_state("USD", history, empty)
    assert empty["clarification"]["pending_slot"] == "price_trend_currency"
    recovered = _plan_from_pending_clarification("USD", empty)
    assert recovered is not None
    assert "usd" in recovered.rewritten_question.casefold()
    assert recovered.filters.get("brand", "").casefold() == "primark"


def test_yes_recovers_price_trend_confirm_from_history():
    """Even if session confirmation was lost, 'yes' after the confirm ask must not count products."""
    from fashion_brain.agents.turn_planner import _plan_from_pending_confirmation

    state = {"active_filters": {"brand": "primark"}}
    history = [
        {"role": "user", "content": "make trend analysis for this brand"},
        {
            "role": "assistant",
            "content": "Do you want a price trend analysis for Primark?",
        },
    ]
    plan = _plan_from_pending_confirmation("yes", state, history)
    assert plan is not None
    assert plan.turn_type == "answers_confirmation"
    assert plan.metric == "price"
    assert plan.group_by is None
    assert "price change trends" in plan.rewritten_question.casefold()
    assert "primark" in plan.rewritten_question.casefold()
    resolve_scope(plan, state)
    assert plan.effective_filters.get("brand", "").casefold() == "primark"


def test_price_trend_clarify_options_pick_by_number():
    from fashion_brain.agents.turn_planner import _plan_from_pending_clarification

    state = {
        "clarification": {
            "question": "Which brand?",
            "options": ["Brand: Primark", "Category: Clothes"],
            "pending_metric": "price",
            "interpretations": [
                {
                    "restated_question": "Summarize average price change trends for Primark",
                    "filters": {"brand": "Primark"},
                },
                {
                    "restated_question": "Summarize average price change trends for Clothes",
                    "filters": {"category": "Clothes"},
                },
            ],
        }
    }
    plan = _plan_from_pending_clarification("1", state)
    assert plan is not None
    assert plan.turn_type == "answers_clarification"
    assert plan.filters["brand"] == "Primark"
    assert "price" in plan.rewritten_question.casefold()
    resolve_scope(plan, state)
    assert plan.effective_filters["brand"] == "Primark"


def test_named_brand_price_trend_skips_confirmation(monkeypatch):
    """'make price trend analysis for Zara' must set metric=price and not re-confirm."""
    fake = {
        "turn_type": "new_question",
        "confidence": 0.9,
        "rewritten_question": "Analyze Zara by brand",
        "filters": {"brand": "zara"},
        "carry_filters": False,
        "drop_filters": [],
        "group_by": "brand",
        "metric": "analysis",
        "analysis_depth": "full",
        "needs_clarification": False,
        "needs_confirmation": True,
        "confirmation_text": "Do you want a full analysis of Zara?",
        "assumptions": [],
    }
    # plan_turn imports build_model_provider from fashion_brain.llm inside the fn.
    monkeypatch.setattr(
        "fashion_brain.llm.build_model_provider",
        lambda settings: SimpleNamespace(
            complete=lambda *a, **k: json.dumps(fake),
        ),
    )
    plan = plan_turn(
        SimpleNamespace(settings=SimpleNamespace(request_timeout_seconds=5)),
        "make price trend analysis for Zara brand",
        history=None,
        state={},
        catalog=None,
    )
    assert plan.llm_ok is True
    assert plan.metric == "price"
    assert plan.needs_confirmation is False
    assert plan.group_by is None
    assert "price" in plan.rewritten_question.casefold()
    assert "zara" in plan.rewritten_question.casefold()
    assert interruption_reply(plan) is None


def test_yes_restores_price_trend_not_count():
    """Bug: 'yes' after price-trend confirm must not become a product count."""
    from fashion_brain.agents.turn_planner import (
        _action_from_confirmation_text,
        _is_affirmative_reply,
        _plan_from_pending_confirmation,
    )

    assert _is_affirmative_reply("yes")
    ask = _action_from_confirmation_text(
        "Do you want a price trend analysis for Primark?",
        {"brand": "primark"},
    )
    assert "price" in ask.casefold()
    assert "trend" in ask.casefold()
    assert "primark" in ask.casefold()

    state = {
        "active_filters": {"brand": "primark"},
        "confirmation": {
            "question": "Do you want a price trend analysis for Primark?",
            "filters": {"brand": "primark"},
            "rewritten_question": "Summarize average price change trends for Primark",
            "metric": "price",
            "analysis_depth": "focused",
        },
    }
    plan = _plan_from_pending_confirmation("yes", state)
    assert plan is not None
    assert plan.turn_type == "answers_confirmation"
    assert plan.metric == "price"
    assert "price change trends" in plan.rewritten_question.casefold()
    resolve_scope(plan, state)
    assert plan.effective_filters == {"brand": "primark"}
    assert "count" not in plan.rewritten_question.casefold()
    assert interruption_reply(plan) is None  # proceed to retrieval, do not re-ask


def test_scope_note_for_follow_up():
    plan = TurnPlan(
        llm_ok=True,
        turn_type="follow_up",
        effective_filters={"brand": "primark"},
        assumptions=["kept prior brand"],
    )
    note = scope_note(plan)
    assert "brand=primark" in note
    assert "Assumption: kept prior brand" in note
