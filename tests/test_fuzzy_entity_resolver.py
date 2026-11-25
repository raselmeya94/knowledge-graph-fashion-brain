"""Unit tests for FuzzyEntityResolver (no live Neo4j)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_graph.linking.fuzzy_entity_resolver import (
    FuzzyEntityResolver,
    resolve_fuzzy_entities,
)
from knowledge_graph.ontology_registry import (
    build_ontology_registry,
    invalidate_ontology_registry_cache,
)


@pytest.fixture()
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "ontologies"
    root.mkdir()
    payload = {
        "name": "test-ontology",
        "version": "1",
        "description": "test brands and materials",
        "entity_types": ["Brand", "Material", "Product"],
        "relationship_allowlist": {"HAS_BRAND": [["Product", "Brand"]]},
        "canonical_entities": [
            {
                "id": "brand:zara",
                "type": "Brand",
                "name": "Zara",
                "aliases": ["ZARA", "Zara Home"],
            },
            {
                "id": "mat:polyester",
                "type": "Material",
                "name": "Polyester",
                "aliases": ["PES", "poly"],
            },
        ],
    }
    (root / "test.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("FASHIONGPT_ONTOLOGY_GLOB", str(root / "*.json"))
    invalidate_ontology_registry_cache()
    return build_ontology_registry(None)


def test_exact_alias_match(registry):
    result = FuzzyEntityResolver(registry=registry, threshold=85).resolve(
        "How many ZARA products are cotton?"
    )
    assert "brand:zara" in {h.canonical_id for h in result.hits}


def test_fuzzy_near_miss(registry):
    result = FuzzyEntityResolver(registry=registry, threshold=80).resolve("Show polyster content")
    assert "mat:polyester" in {h.canonical_id for h in result.hits}


def test_does_not_invent_ids(registry):
    result = resolve_fuzzy_entities(
        "completely unknown brandnamexyz",
        registry=registry,
        threshold=95,
    )
    assert not any(h.canonical_id.startswith("invented") for h in result.hits)
