"""Static / mapping-backed schema views for registered sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT
from knowledge_graph.errors import NotFoundError


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

    mapping = _load_mapping(source_id)
    return {
        "source_id": source_id,
        "source": source.model_dump(),
        "graph_id": graph.id if graph else None,
        "ontology_id": ontology_id,
        "ontology": ontology,
        "mapping": mapping,
        "note": "Schema is registry/mapping-backed. Use GET /api/metadata/values for paginated attribute discovery.",
    }


def _load_mapping(source_id: str) -> dict[str, Any] | None:
    candidates = {
        "ecom_products": DATA_ROOT / "schemas" / "ecom_master_products_mapping.json",
        "own_products": DATA_ROOT / "schemas" / "product-knowledge-ontology.json",
    }
    path = candidates.get(source_id)
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"path": str(path), "error": "invalid_json"}
