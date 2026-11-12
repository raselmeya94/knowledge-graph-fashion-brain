"""Distinct facet counts ('how many <facet>s') — schema/stats driven, not a fixed noun list."""

from __future__ import annotations

from knowledge_graph.models import EvidenceItem
from knowledge_graph.retrieval.ecom import _fold_facet_token, append_neo4j_count_answer


def test_fold_facet_token_is_generic():
    assert _fold_facet_token("buyers") == "buyer"
    assert _fold_facet_token("Buyer") == "buyer"
    assert _fold_facet_token("categories") == "category"
    assert _fold_facet_token("products") == "product"
    assert _fold_facet_token("plm_main_categories") == "plm_main_category"


def test_facet_entity_count_buyers_question():
    from fashion_brain.ask import _facet_entity_count

    assert _facet_entity_count("how many buyers are in the plmex") == "buyer"


def test_facet_entity_count_from_schema_any_key():
    from fashion_brain.agents.entity_extract import facet_entity_count_from_schema

    schema = type(
        "S",
        (),
        {
            "facet_types": ["Buyer", "Merchandiser"],
            "filter_keys": ["buyer", "merchandiser"],
            "product_properties": [],
        },
    )()
    assert facet_entity_count_from_schema(
        "how many merchandisers in plmex", schema
    ).casefold() == "merchandiser"
    assert facet_entity_count_from_schema("how many buyers in plmex", schema).casefold() == "buyer"


def test_brand_product_list_is_group_by_not_distinct_count():
    from fashion_brain.agents.entity_extract import (
        detect_group_by_from_schema,
        facet_entity_count_from_schema,
        looks_like_facet_product_breakdown,
    )
    from fashion_brain.agents._impl import _prefer_product_breakdown
    from fashion_brain.ask import _facet_entity_count

    schema = type(
        "S",
        (),
        {
            "facet_types": ["Brand"],
            "filter_keys": ["brand"],
            "product_properties": [],
            "entity_types": ["Brand", "Product"],
        },
    )()
    q = "make list for all brand and total products count"
    assert detect_group_by_from_schema(q, schema) == "brand"
    assert looks_like_facet_product_breakdown(q, schema)
    assert facet_entity_count_from_schema(q, schema) is None
    assert _facet_entity_count(q, schema) is None
    count_entity, group_by = _prefer_product_breakdown(q, schema, "brand", None)
    assert count_entity is None
    assert group_by == "brand"
    # Distinct-facet ask still works.
    assert _facet_entity_count("how many brands are in e-comm data", schema) in {"brand", "Brand"}


def test_append_count_uses_distinct_facet_not_product_total(monkeypatch):
    lines: list[str] = []
    evidence: list[EvidenceItem] = []
    caveats: list[str] = []

    monkeypatch.setattr(
        "knowledge_graph.retrieval.ecom._graph_label",
        lambda _platform, _gid: "PLMEX",
    )
    monkeypatch.setattr(
        "knowledge_graph.retrieval.ecom.resolve_count_facet",
        lambda *_a, **_k: "buyer",
    )
    monkeypatch.setattr(
        "knowledge_graph.retrieval.kg_stats.unique_count",
        lambda *_a, **_k: 42,
    )

    class _P:
        settings = type("S", (), {"generated": None})()

    append_neo4j_count_answer(
        _P(),
        principal=None,  # type: ignore[arg-type]
        graph_id="plmex_products",
        filters={},
        group_by=None,
        limit=None,
        lines=lines,
        evidence=evidence,
        caveats=caveats,
        question="how many buyers are in the plmex",
        count_entity="Buyer",
    )
    assert lines == ["There are 42 distinct buyers in PLMEX."]
    assert evidence and "buyers=42" in (evidence[0].excerpt or "")


def test_stats_attributes_come_from_mapping_not_allowlist():
    from knowledge_graph.pipeline.mapping_sql_sync import _stats_attributes_from_mapping

    mapping = {
        "product_properties": {
            "brand": {},
            "buyer": {},
            "weird_custom_facet": {},
            "created_at": {},
            "shipment_quantity": {},
        },
        "relationships": [
            {"rel_type": "HAS_BUYER", "end_type": "Buyer", "from_column": "buyer"},
            {"rel_type": "HAS_MERCHANDISER", "end_type": "Merchandiser", "from_column": "merchandiser"},
        ],
    }
    attrs = _stats_attributes_from_mapping(mapping)
    assert "buyer" in attrs
    assert "brand" in attrs
    assert "weird_custom_facet" in attrs
    assert "merchandiser" in attrs
    # Timestamp-shaped columns are dropped by is_analysis_facet_attr.
    assert "created_at" not in attrs


def test_query_planner_uses_count_entity_not_noun_list():
    from fashion_brain.agents.contracts import Intention
    from fashion_brain.planning.query_planner import plan_retrieve

    intention = Intention("product_count", count_entity="merchandiser")
    planned = plan_retrieve(intention, "how many merchandisers in the catalog")
    assert planned.metrics == ["facet_count"]
