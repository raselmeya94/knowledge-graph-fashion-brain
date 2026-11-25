"""ConversationContext REUSE when prior findings can answer the follow-up."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fashion_brain.agents.conversation_context import ConversationContextAgent


def test_reuse_when_model_says_reuse_and_prior_text_exists():
    agent = ConversationContextAgent()
    state = {
        "last_result_id": "result_abc",
        "last_retrieved_text": "Primark 4058\nZara 5804\n| Brand | Count |\n| Primark | 4058 |",
        "last_graph_id": "ecom_products",
        "active_filters": {"brand": "primark"},
    }
    with patch.object(
        ConversationContextAgent,
        "_from_model",
        return_value={
            "is_contextual": True,
            "decision": "REUSE",
            "carry_filters": True,
            "refined_question": "Summarize those brand counts",
            "group_by": None,
        },
    ):
        decision = agent.analyze(
            "Summarize that",
            history=[{"role": "user", "content": "How many products by brand?"}],
            conversation_state=state,
            platform=SimpleNamespace(settings=SimpleNamespace(answer_mode="llm")),
        )
    assert decision.decision == "REUSE"
    assert decision.reuse_previous_results is True
    assert decision.additional_retrieval_required is False


def test_refine_without_prior_text_becomes_retrieve():
    agent = ConversationContextAgent()
    with patch.object(
        ConversationContextAgent,
        "_from_model",
        return_value={
            "is_contextual": True,
            "decision": "REFINE",
            "carry_filters": True,
            "refined_question": "Only Primark",
            "group_by": None,
        },
    ):
        decision = agent.analyze(
            "Only Primark",
            history=[{"role": "user", "content": "Brand split?"}],
            conversation_state={},
            platform=SimpleNamespace(settings=SimpleNamespace(answer_mode="llm")),
        )
    assert decision.decision == "RETRIEVE"


def test_explicit_brand_comparison_does_not_carry_old_facets():
    agent = ConversationContextAgent()
    state = {
        "last_result_id": "result_abc",
        "last_retrieved_text": "Primark products by color",
        "last_graph_id": "ecom_products",
        "active_filters": {"brand": "primark", "color": "red"},
    }
    with patch.object(
        ConversationContextAgent,
        "_from_model",
        return_value={
            "is_contextual": True,
            "decision": "REFINE",
            "carry_filters": True,
            "refined_question": "Compare Primark and Zara product counts",
        },
    ):
        decision = agent.analyze(
            "Who has more products — Primark or Zara?",
            history=[{"role": "user", "content": "Show Primark products in red"}],
            conversation_state=state,
            platform=SimpleNamespace(settings=SimpleNamespace(answer_mode="llm")),
        )
    assert decision.decision == "RETRIEVE"
    assert decision.is_contextual is False
    assert decision.active_filters == {}


def test_parse_plans_caps_at_four():
    from fashion_brain.agents.text2cypher import _parse_plans

    raw = (
        '{"queries":['
        + ",".join(
            [
                f'{{"label":"q{i}","cypher":"MATCH (p:Product) RETURN count(p) AS c","params":{{"g":"ecom_products"}}}}'
                for i in range(6)
            ]
        )
        + "]}"
    )
    plans = _parse_plans(raw, default_graph="ecom_products")
    assert len(plans) == 4
