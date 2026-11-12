"""Hooks after ontology/graph register or source sync — Neo4j indexes + schema cache only."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("knowledge_graph.schema_hooks")


def on_schema_changed(platform, graph_id: str | None = None, *, ontology_id: str | None = None) -> dict[str, Any]:
    """Refresh union ontology, Neo4j full-text facet indexes, and RuntimeSchema cache."""

    report: dict[str, Any] = {"graph_id": graph_id, "ontology_id": ontology_id}
    try:
        platform.refresh_union_ontology()
        report["union_ontology"] = "refreshed"
    except Exception as exc:
        report["union_ontology"] = f"error:{exc}"
        LOGGER.warning("refresh_union_ontology failed: %s", exc)

    try:
        from knowledge_graph.retrieval.runtime_schema import (
            build_runtime_schema,
            ensure_facet_fulltext_indexes,
            invalidate_runtime_schema_cache,
        )

        invalidate_runtime_schema_cache()
        report["schema_cache"] = "invalidated"
        ctx = build_runtime_schema(platform, graph_id)
        facets = list(ctx.facet_types or [])
        if facets:
            applied = ensure_facet_fulltext_indexes(platform, facets)
            report["facet_indexes"] = len(applied)
            report["facet_types"] = facets
        else:
            report["facet_indexes"] = 0
    except Exception as exc:
        report["facet_indexes"] = f"error:{exc}"
        LOGGER.warning("facet index ensure failed: %s", exc)

    return report
