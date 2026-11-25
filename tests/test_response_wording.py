from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fashion_brain.agents.query_understanding import _basic_entity_brief
from fashion_brain.agents.contracts import Retrieved
from fashion_brain.ask import answer_from_catalog
from knowledge_graph.access import Principal
from knowledge_graph.models import EvidenceItem
from knowledge_graph.retrieval.ecom import _pluralize_label


def test_analysis_brief_uses_user_facing_wording():
    platform = SimpleNamespace(
        settings=SimpleNamespace(generated=None),
        kg_registry=None,
    )
    with (
        patch(
            "knowledge_graph.retrieval.kg_stats.stats_available",
            return_value=True,
        ),
        patch(
            "knowledge_graph.retrieval.kg_stats.answer_simple_count",
            return_value={"kind": "value_count", "n": 4058},
        ),
    ):
        answer = _basic_entity_brief(
            platform,
            entity="Primark",
            filters={"brand": "Primark"},
            graph_id="ecom_products",
        )
    assert answer == "Primark currently has 4,058 products."
    assert "catalog" not in answer.casefold()


def test_analysis_distinct_value_labels_are_pluralized_correctly():
    assert _pluralize_label("family") == "families"
    assert _pluralize_label("category") == "categories"


def test_metadata_count_uses_answer_card_not_response_agent():
    platform = SimpleNamespace(
        settings=SimpleNamespace(answer_mode="llm", model_provider="none"),
        products=None,
    )
    retrieved = Retrieved(
        text="There are 800 products in category Sweater.",
        evidence=[
            EvidenceItem(
                source_id="ecom_products",
                record_id="kg_stats",
                excerpt="filter: category=Sweater -> 800",
                graph="ecom_products",
                verification_status="observed",
            )
        ],
        caveats=[],
        unavailable=[],
        response_type="stats",
        retrieve_mode="template",
    )
    with (
        patch("fashion_brain.ask.draft_catalog", return_value=(retrieved, ["product_count", "kg_stats"], None)),
        patch("fashion_brain.agents.ResponseAgent.compose") as compose,
    ):
        answer = answer_from_catalog(
            platform,
            "How many sweater products are available?",
            Principal(subject="test", clearance="internal"),
        )

    # Card writer (safe fallback without a real model) — ResponseAgent stays skipped.
    assert "800" in answer.answer
    assert answer.routes == ["product_count", "kg_stats"]
    assert answer.evidence[0].record_id == "kg_stats"
    compose.assert_not_called()
