"""Metadata-first simple counts — kg_stats only; Cypher for multi-filter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from knowledge_graph.access import Principal
from knowledge_graph.models import EvidenceItem
from knowledge_graph.retrieval import kg_stats
from knowledge_graph.retrieval.ecom import append_neo4j_count_answer, clean_product_filters
from fashion_brain.agents.query_understanding import QueryUnderstandingAgent
from fashion_brain.agents.contracts import ReadyResult


INTERNAL = Principal(subject="t", clearance="internal")


def _write_stats(tmp_path: Path, graph_id: str = "plmex_products") -> Path:
    payload = {
        "version": 2,
        "graph_id": graph_id,
        "graph_name": "PLMEX Knowledge Graph",
        "description": "PLM product catalog",
        "source": "test",
        "updated_at": "2026-11-18T00:00:00Z",
        "totals": {"products": 27694, "brands": 2, "columns_indexed": 2},
        "unique_counts": {"brand": 2, "category": 2},
        "value_counts": {
            "brand": {"primark": 1397, "zara": 419},
            "category": {"Sweater": 100, "Polo": 50},
        },
        "columns": [
            {
                "key": "brand",
                "definition": "Retail brand on the product.",
                "unique_count": 2,
                "value_counts": {"primark": 1397, "zara": 419},
            },
            {
                "key": "category",
                "definition": "Product category.",
                "unique_count": 2,
                "value_counts": {"Sweater": 100, "Polo": 50},
            },
        ],
    }
    root = tmp_path / "kg_stats"
    root.mkdir(parents=True)
    path = root / f"{graph_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    # kg_stats readers resolve under DATA_ROOT/generated — patch stats_root via generated=.
    return tmp_path


def test_enrich_stats_payload_adds_columns_and_name():
    platform = SimpleNamespace(kg_registry=None, settings=SimpleNamespace())
    payload = {
        "version": 1,
        "graph_id": "plmex_products",
        "totals": {"products": 10},
        "unique_counts": {"brand": 1},
        "value_counts": {"brand": {"primark": 10}},
    }
    out = kg_stats.enrich_stats_payload(platform, "plmex_products", payload)
    assert out["graph_name"]
    assert out["columns"]
    assert out["columns"][0]["key"] == "brand"
    assert "definition" in out["columns"][0]
    assert out["columns"][0]["value_counts"]["primark"] == 10


def test_answer_simple_count_product_total(tmp_path):
    generated = _write_stats(tmp_path)
    result = kg_stats.answer_simple_count("plmex_products", filters={}, generated=generated)
    assert result is not None
    assert result["kind"] == "product_total"
    assert result["n"] == 27694
    assert result["source"] == "kg_stats"


def test_answer_simple_count_one_filter(tmp_path):
    generated = _write_stats(tmp_path)
    result = kg_stats.answer_simple_count(
        "plmex_products",
        filters={"brand": "primark"},
        generated=generated,
    )
    assert result is not None
    assert result["kind"] == "value_count"
    assert result["n"] == 1397
    assert result["filter"] == "filter: brand=primark"
    assert result["applied_filters"] == {"brand": "primark"}
    assert result["requested_filters"] == {"brand": "primark"}


def test_answer_simple_count_rejects_multi_filter(tmp_path):
    generated = _write_stats(tmp_path)
    result = kg_stats.answer_simple_count(
        "plmex_products",
        filters={"brand": "primark", "category": "Sweater"},
        generated=generated,
    )
    assert result is None


def test_clean_product_filters_drops_plm_brand():
    cleaned = clean_product_filters({"brand": "are in plm"})
    assert "brand" not in cleaned
    cleaned2 = clean_product_filters({"brand": "plm"})
    assert "brand" not in cleaned2
    cleaned3 = clean_product_filters({"brand": "primark"})
    assert cleaned3["brand"] == "primark"


def test_append_count_uses_meta_not_cypher(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    lines: list[str] = []
    evidence: list[EvidenceItem] = []
    with patch("knowledge_graph.retrieval.ecom.cypher_count_products") as cypher:
        cypher.side_effect = AssertionError("Cypher must not run for simple meta count")
        append_neo4j_count_answer(
            platform,
            INTERNAL,
            "plmex_products",
            {},
            None,
            None,
            lines,
            evidence,
            [],
            question="How many products are in the PLM data?",
        )
    assert lines
    assert "27,694" in lines[0]
    assert evidence and evidence[0].record_id == "kg_stats"
    cypher.assert_not_called()


def test_append_one_filter_meta(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    lines: list[str] = []
    evidence: list[EvidenceItem] = []
    with patch("knowledge_graph.retrieval.ecom.cypher_count_products") as cypher:
        append_neo4j_count_answer(
            platform,
            INTERNAL,
            "plmex_products",
            {"brand": "primark"},
            None,
            None,
            lines,
            evidence,
            [],
            question="How many Primark products?",
        )
    assert "1,397" in lines[0]
    assert evidence[0].record_id == "kg_stats"
    assert "filter: brand=primark" in evidence[0].excerpt
    assert evidence[0].applied_filters.get("brand") == "primark"
    cypher.assert_not_called()


def test_append_multi_filter_uses_cypher(tmp_path):
    generated = _write_stats(tmp_path)
    platform = SimpleNamespace(settings=SimpleNamespace(generated=generated), kg_registry=None)
    lines: list[str] = []
    evidence: list[EvidenceItem] = []

    def _fake_count(_platform, _gid, filters):
        assert filters.get("brand") and filters.get("color")
        return 12

    with (
        patch("knowledge_graph.retrieval.ecom.cypher_count_products", side_effect=_fake_count),
        patch("knowledge_graph.retrieval.ecom.cypher_group_by", return_value=None),
        patch(
            "knowledge_graph.retrieval.ecom.search_neo4j_products",
            return_value=([], 12),
        ),
    ):
        append_neo4j_count_answer(
            platform,
            INTERNAL,
            "plmex_products",
            {"brand": "primark", "color": "red"},
            None,
            None,
            lines,
            evidence,
            [],
            question="How many red Primark products?",
        )
    assert lines
    assert "12" in lines[0]


def test_qu_plm_question_not_brand():
    agent = QueryUnderstandingAgent()
    result = agent.evaluate(
        SimpleNamespace(settings=SimpleNamespace(generated=None)),
        "How many products are in the PLM data?",
        conversation_state={},
    )
    assert isinstance(result, ReadyResult)
    assert "brand" not in (result.intent.filters or {})
    assert result.intent.source == "plmex_products"


def test_meta_overview_shape(tmp_path):
    generated = _write_stats(tmp_path)
    overview = kg_stats.meta_overview("plmex_products", generated)
    assert overview is not None
    assert overview["graph_name"] == "PLMEX Knowledge Graph"
    assert overview["columns"][0]["definition"]
    assert "primark" in overview["columns"][0]["value_counts"]
