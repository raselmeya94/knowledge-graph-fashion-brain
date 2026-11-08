"""Neo4j constraints and indexes. Memory repositories record the same statements."""

from __future__ import annotations

import re

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX entity_graph IF NOT EXISTS FOR (n:Entity) ON (n.graph)",
    "CREATE INDEX entity_source IF NOT EXISTS FOR (n:Entity) ON (n.source_id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX entity_lifecycle IF NOT EXISTS FOR (n:Entity) ON (n.lifecycle)",
]

# Baseline facet indexes for common retail ontologies; dynamic labels are added via
# ``facet_index_statements`` / ``ensure_facet_fulltext_indexes`` from RuntimeSchemaContext.
_BASELINE_FACETS = ("GarmentCategory", "Material", "Color", "Brand", "Availability")

INDEXES = INDEXES + [
    f"CREATE INDEX {re.sub(r'[^A-Za-z0-9_]', '', label).lower()}_nn IF NOT EXISTS "
    f"FOR (n:{label}) ON (n.normalized_name)"
    for label in _BASELINE_FACETS
]

FULLTEXT_INDEXES = [
    f"CREATE FULLTEXT INDEX {re.sub(r'[^A-Za-z0-9_]', '', label).lower()}_name_ft IF NOT EXISTS "
    f"FOR (n:{label}) ON EACH [n.name, n.normalized_name]"
    for label in _BASELINE_FACETS
]

SCHEMA_STATEMENTS = CONSTRAINTS + INDEXES + FULLTEXT_INDEXES


def facet_index_statements(labels: list[str]) -> list[str]:
    """RANGE(normalized_name) + FULLTEXT(name, normalized_name) for each ontology facet label."""

    out: list[str] = []
    for label in labels:
        safe = re.sub(r"[^A-Za-z0-9_]", "", label)
        if not safe or safe != label:
            continue
        out.append(
            f"CREATE INDEX {safe.lower()}_nn IF NOT EXISTS FOR (n:{label}) ON (n.normalized_name)"
        )
        out.append(
            f"CREATE FULLTEXT INDEX {safe.lower()}_name_ft IF NOT EXISTS "
            f"FOR (n:{label}) ON EACH [n.name, n.normalized_name]"
        )
    return out


def apply_schema_statements(session, statements: list[str] | None = None) -> list[str]:
    """Run CREATE … IF NOT EXISTS statements; return those that succeeded."""

    applied: list[str] = []
    for stmt in statements or SCHEMA_STATEMENTS:
        try:
            session.run(stmt)
            applied.append(stmt)
        except Exception:
            pass
    return applied
