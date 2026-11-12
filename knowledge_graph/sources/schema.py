"""Static / mapping-backed schema views for registered sources."""

from __future__ import annotations

from typing import Any

from knowledge_graph.errors import NotFoundError
from knowledge_graph.sources.mapping_store import get_mapping, mapping_exists


def source_schema(platform, source_id: str) -> dict[str, Any]:
    """Return schema metadata for a source (ontology + optional column mapping)."""

    source = None
    for item in platform.registry.sources:
        if item.source_id == source_id:
            source = item
            break
    if source is None:
        raise NotFoundError(f"Source '{source_id}' was not found.")

    graph = None
    for item in platform.kg_registry.list_graphs():
        if source_id in item.source_ids or item.id == source_id:
            graph = item
            break

    ontology = None
    ontology_id = graph.ontology_id if graph else None
    if ontology_id:
        try:
            ontology = platform.kg_registry.load_ontology_object(ontology_id).model_dump()
        except Exception:
            ontology = None

    mapping = None
    mapping_path = None
    if mapping_exists(source_id):
        try:
            payload = get_mapping(source_id)
            mapping = payload.get("mapping")
            mapping_path = payload.get("path")
        except Exception:
            mapping = None

    return {
        "source_id": source_id,
        "source": source.model_dump(),
        "graph_id": graph.id if graph else None,
        "ontology_id": ontology_id,
        "ontology": ontology,
        "mapping": mapping,
        "mapping_path": mapping_path,
        "mapping_present": mapping is not None,
        "note": "Upload mapping via POST /api/sources/{id}/mapping. Use GET /api/metadata/values for value discovery.",
    }
