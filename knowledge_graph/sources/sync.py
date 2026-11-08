"""Unified sync dispatch for registered sources."""

from __future__ import annotations

from typing import Any

from knowledge_graph.errors import NotFoundError, ValidationFailed
from knowledge_graph.graph.snapshots import save_master_snapshot, save_source_snapshots
from knowledge_graph.logging_config import get_logger

LOGGER = get_logger("knowledge_graph.sources.sync")


def sync_source(platform, source_id: str, *, brand: str | None = None) -> dict[str, Any]:
    """Sync one source into its graph. Dispatches e-com SQL vs file/adapter rebuild."""

    source = _require_source(platform, source_id)
    graph_id = _graph_for_source(platform, source_id)

    if source_id == "ecom_products" or source.source_type == "ecom_products":
        from knowledge_graph.pipeline.ecom_sync import sync_ecom_to_neo4j

        # sync_ecom_to_neo4j persists ecom_sync_state.json during the run and
        # refreshes snapshots + kg_system_meta when finished.
        result = sync_ecom_to_neo4j(platform, brand=brand, refresh_generated=True)
        return {
            "source_id": source_id,
            "graph_id": graph_id,
            "action": "ecom_sync",
            "result": result if isinstance(result, dict) else {"detail": result},
        }

    reports = platform.rebuild([source_id])
    save_source_snapshots(platform.graph, platform.settings.generated, [graph_id])
    save_master_snapshot(platform.graph, platform.settings.generated)
    return {
        "source_id": source_id,
        "graph_id": graph_id,
        "action": "rebuild",
        "reports": [report.model_dump() for report in reports],
    }


def _require_source(platform, source_id: str):
    for source in platform.registry.sources:
        if source.source_id == source_id:
            return source
    raise NotFoundError(f"Source '{source_id}' was not found.")


def _graph_for_source(platform, source_id: str) -> str:
    for graph in platform.kg_registry.list_graphs():
        if source_id in graph.source_ids or graph.id == source_id:
            return graph.id
    raise ValidationFailed(f"No knowledge graph is registered for source '{source_id}'.")
