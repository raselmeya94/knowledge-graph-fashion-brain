"""Unit tests for live schema pruning (no Neo4j required)."""

from __future__ import annotations

from types import SimpleNamespace

from fashion_brain.planning.domain_schema_filter import (
    classify_question_domains,
    prune_live_schema_for_question,
    pruned_schema_card,
)
from knowledge_graph.retrieval.live_schema import LiveSchemaSnapshot


def _snap(**overrides) -> LiveSchemaSnapshot:
    base = LiveSchemaSnapshot(
        labels=[
            "Product",
            "Brand",
            "Material",
            "Entity",
            "Canonical",
            "SourceEntity",
            "PriceChangeEvent",
            "PriceSnapshot",
            "AvailabilitySnapshot",
            "Shipment",
            "Gauge",
            "Machine",
        ],
        relationship_types=[
            "HAS_BRAND",
            "HAS_MATERIAL",
            "RESOLVES_TO",
            "HAS_PRICE_CHANGE",
            "HAS_PRICE_SNAPSHOT",
            "HAS_AVAILABILITY",
        ],
        node_properties={
            "Product": ["id", "name", "graph", "price_num"],
            "Brand": ["id", "name", "normalized_name"],
            "PriceChangeEvent": ["delta_pct", "direction", "detected_at"],
        },
        rel_properties={},
        graph_values=["ecom_products", "plmex_products"],
        domain_names=["Merchandising", "PLM"],
        built_at=1.0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_classify_prefers_live_graph_tokens(monkeypatch):
    platform = SimpleNamespace(settings=SimpleNamespace(schema_ttl_seconds=600), kg_registry=None)
    monkeypatch.setattr(
        "knowledge_graph.retrieval.live_schema.get_live_schema",
        lambda *_a, **_k: _snap(),
    )
    domains = classify_question_domains(
        platform,
        "How did ecom product prices change last week?",
        preferred_graphs=["ecom_products"],
    )
    assert "ecom_products" in domains


def test_prune_keeps_price_labels_for_price_questions(monkeypatch):
    platform = SimpleNamespace(settings=SimpleNamespace(schema_ttl_seconds=600), kg_registry=None)
    monkeypatch.setattr(
        "knowledge_graph.retrieval.live_schema.get_live_schema",
        lambda *_a, **_k: _snap(),
    )
    pruned = prune_live_schema_for_question(
        platform,
        "Show average price movers for Zara",
        preferred_graphs=["ecom_products"],
    )
    assert pruned["source"] == "live"
    assert "Product" in pruned["labels"]
    assert "PriceChangeEvent" in pruned["labels"] or "PriceSnapshot" in pruned["labels"]
    assert "RESOLVES_TO" in pruned["relationship_types"] or "HAS_PRICE_CHANGE" in pruned[
        "relationship_types"
    ]
    assert "Live Neo4j schema" in pruned["card"]


def test_pruned_schema_card_fail_open_without_live(monkeypatch):
    platform = SimpleNamespace(settings=SimpleNamespace(schema_ttl_seconds=600), kg_registry=None)
    monkeypatch.setattr(
        "knowledge_graph.retrieval.live_schema.get_live_schema",
        lambda *_a, **_k: None,
    )
    card = pruned_schema_card(
        platform,
        "How many products?",
        preferred_graphs=["ecom_products"],
        fallback_card="FALLBACK CARD",
    )
    assert card == "FALLBACK CARD"


def test_compact_card_respects_keep_sets():
    snap = _snap()
    card = snap.compact_card(
        keep_labels={"Product", "Brand"},
        keep_rels={"HAS_BRAND"},
    )
    assert "Product" in card
    assert "Brand" in card
    assert "HAS_BRAND" in card
    labels_line = next(line for line in card.splitlines() if line.startswith("Node labels:"))
    assert "Shipment" not in labels_line
