"""Unit tests for relationship_allowlist Cypher validation (no live Neo4j)."""

from __future__ import annotations

import pytest

from knowledge_graph.retrieval.cypher_safe import (
    assert_allowlisted_cypher,
    extract_relationship_types,
)


ALLOWLIST = {
    "HAS_BRAND": [["Product", "Brand"]],
    "HAS_MATERIAL": [["Product", "Material"]],
}


def test_extract_relationship_types():
    cypher = (
        "MATCH (p:Product)-[:HAS_BRAND]->(b:Brand) "
        "OPTIONAL MATCH (p)-[:HAS_MATERIAL]->(m:Material) "
        "RETURN p.id LIMIT 5"
    )
    types = extract_relationship_types(cypher)
    assert "HAS_BRAND" in types
    assert "HAS_MATERIAL" in types


def test_unknown_rel_rejected():
    with pytest.raises(ValueError, match="MADE_OF"):
        assert_allowlisted_cypher(
            "MATCH (p:Product)-[:MADE_OF]->(m:Material) RETURN p.id LIMIT 1",
            ALLOWLIST,
        )


def test_allowed_rel_passes():
    out = assert_allowlisted_cypher(
        "MATCH (p:Product)-[:HAS_BRAND]->(b:Brand) WHERE coalesce(p.graph,'') = $g RETURN count(p) AS c",
        ALLOWLIST,
    )
    assert "HAS_BRAND" in out


def test_disallowed_direction_rejected():
    with pytest.raises(ValueError, match="Direction"):
        assert_allowlisted_cypher(
            "MATCH (b:Brand)-[:HAS_BRAND]->(p:Product) RETURN p.id LIMIT 1",
            ALLOWLIST,
        )


def test_resolves_to_always_allowed():
    out = assert_allowlisted_cypher(
        "MATCH (s:SourceEntity)-[:RESOLVES_TO]->(c:Canonical) RETURN c.id LIMIT 5",
        ALLOWLIST,
    )
    assert "RESOLVES_TO" in out


def test_empty_allowlist_fail_open():
    cypher = "MATCH (p:Product)-[:ANYTHING]->(x) RETURN p.id LIMIT 1"
    assert "ANYTHING" in assert_allowlisted_cypher(cypher, None)
    assert "ANYTHING" in assert_allowlisted_cypher(cypher, {})
