"""Fashion Brain access to RuntimeSchemaContext (local Platform or KG HTTP)."""

from __future__ import annotations

from typing import Any

from knowledge_graph.retrieval.runtime_schema import (
    RuntimeSchemaContext,
    build_runtime_schema,
    default_product_graph_id,
    invalidate_runtime_schema_cache,
)


def get_runtime_schema(platform, graph_id: str | None = None) -> RuntimeSchemaContext:
    """Load schema context via KG API when configured, else local registry."""

    from fashion_brain.execution.kg_client import kg_request, uses_kg_api

    if uses_kg_api(platform.settings):
        params: dict[str, Any] = {}
        if graph_id:
            params["graph_id"] = graph_id
        try:
            payload = kg_request(
                platform.settings,
                "GET",
                "/brain/runtime_schema",
                clearance="internal",
                params=params or None,
            )
            return _from_payload(payload or {})
        except Exception:
            pass
    return build_runtime_schema(platform, graph_id)


def resolve_product_graph_id(platform, preferred: str | None = None) -> str | None:
    ctx = get_runtime_schema(platform, preferred)
    if preferred and any(g.graph_id == preferred for g in ctx.graphs):
        return preferred
    return default_product_graph_id(platform) or ctx.graph_id


def _from_payload(payload: dict[str, Any]) -> RuntimeSchemaContext:
    from knowledge_graph.retrieval.runtime_schema import GraphRuntimeInfo

    graphs = []
    for row in payload.get("graphs") or []:
        if not isinstance(row, dict):
            continue
        graphs.append(
            GraphRuntimeInfo(
                graph_id=str(row.get("graph_id") or ""),
                name=str(row.get("name") or ""),
                kind=str(row.get("kind") or ""),
                ontology_id=str(row.get("ontology_id") or ""),
                description=str(row.get("description") or ""),
                status=str(row.get("status") or "active"),
                entities=[str(x) for x in (row.get("entities") or [])],
                relationships=[str(x) for x in (row.get("relationships") or [])],
                capabilities=[str(x) for x in (row.get("capabilities") or [])],
                source_ids=[str(x) for x in (row.get("source_ids") or [])],
            )
        )
    return RuntimeSchemaContext(
        graphs=graphs,
        graph_id=payload.get("graph_id"),
        ontology_id=payload.get("ontology_id"),
        ontology_name=str(payload.get("ontology_name") or ""),
        entity_types=[str(x) for x in (payload.get("entity_types") or [])],
        relationship_allowlist=dict(payload.get("relationship_allowlist") or {}),
        product_label=str(payload.get("product_label") or "Product"),
        facet_types=[str(x) for x in (payload.get("facet_types") or [])],
        filter_keys=[str(x) for x in (payload.get("filter_keys") or [])],
        scope_tokens=[str(x) for x in (payload.get("scope_tokens") or [])],
        canonical_hints=list(payload.get("canonical_hints") or []),
        brand_label=payload.get("brand_label"),
        product_properties=[str(x) for x in (payload.get("product_properties") or [])],
    )


__all__ = [
    "RuntimeSchemaContext",
    "get_runtime_schema",
    "resolve_product_graph_id",
    "invalidate_runtime_schema_cache",
    "build_runtime_schema",
    "default_product_graph_id",
]
