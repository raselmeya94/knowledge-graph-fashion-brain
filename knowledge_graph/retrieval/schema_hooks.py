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

    try:
        from knowledge_graph.retrieval.live_schema import invalidate_live_schema_cache, warm_live_schema

        invalidate_live_schema_cache()
        report["live_schema"] = warm_live_schema(platform)
    except Exception as exc:
        report["live_schema"] = f"error:{exc}"
        LOGGER.warning("live schema warm failed: %s", exc)

    try:
        from knowledge_graph.ontology_registry import (
            invalidate_ontology_registry_cache,
            warm_ontology_registry,
        )

        invalidate_ontology_registry_cache()
        report["ontology_registry"] = warm_ontology_registry(platform)
    except Exception as exc:
        report["ontology_registry"] = f"error:{exc}"
        LOGGER.warning("ontology registry warm failed: %s", exc)

    try:
        from knowledge_graph.retrieval.runtime_catalog import (
            invalidate_runtime_catalog,
            warm_runtime_catalog,
        )

        invalidate_runtime_catalog()
        report["runtime_catalog"] = warm_runtime_catalog(platform, force=True)
    except Exception as exc:
        report["runtime_catalog"] = f"error:{exc}"
        LOGGER.warning("runtime catalog warm failed: %s", exc)

    return report
