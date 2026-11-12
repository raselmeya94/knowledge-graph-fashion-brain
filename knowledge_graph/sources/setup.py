"""Bootstrap + setup checklist for UI-driven KG control."""

from __future__ import annotations

from typing import Any

from knowledge_graph.errors import FashionGPTError, NotFoundError
from knowledge_graph.pipeline.ecom_sync import load_sync_state, require_ecom_dsn
from knowledge_graph.sources.db_connections import get_connection
from knowledge_graph.sources.mapping_store import get_mapping, mapping_exists


def bootstrap_overview(platform, jobs=None) -> dict[str, Any]:
    """One-shot home payload for an admin UI."""

    from knowledge_graph.retrieval.graph_counts import graph_node_edge_counts

    ontologies = [item.model_dump() for item in platform.kg_registry.list_ontologies()]
    sources = [item.model_dump() for item in platform.registry.sources]
    graphs = []
    for item in platform.kg_registry.list_graphs():
        nodes, edges = graph_node_edge_counts(platform, item.id)
        graphs.append({**item.model_dump(), "node_count": nodes, "edge_count": edges})

    health = {"service": "knowledge_graph", "graph_backend": platform.settings.graph_backend}
    try:
        platform.graph.ping()
        health["graph"] = "ok"
    except Exception as exc:
        health["graph"] = str(exc)

    active_jobs = []
    if jobs is not None:
        active_jobs = [job for job in jobs.list(limit=20) if job.get("status") in {"queued", "running"}]

    return {
        "health": health,
        "counts": {
            "ontologies": len(ontologies),
            "sources": len(sources),
            "graphs": len(graphs),
            "active_jobs": len(active_jobs),
        },
        "ontologies": ontologies,
        "sources": sources,
        "graphs": graphs,
        "active_jobs": active_jobs,
        "checklist": setup_checklist(platform, graph_id="ecom_products", source_id="ecom_products"),
    }


def setup_checklist(
    platform,
    *,
    graph_id: str = "ecom_products",
    source_id: str = "ecom_products",
    ontology_id: str | None = None,
) -> dict[str, Any]:
    """Return which pieces exist and whether build is allowed."""

    ontology_id = ontology_id or _ontology_for_graph(platform, graph_id) or "ecom"

    checks: dict[str, bool] = {}
    try:
        platform.graph.ping()
        checks["health"] = True
    except Exception:
        checks["health"] = False

    try:
        platform.kg_registry.get_ontology(ontology_id)
        path = platform.kg_registry.ontology_file(ontology_id)
        checks["ontology"] = path.is_file()
    except Exception:
        checks["ontology"] = False

    checks["mapping"] = mapping_exists(source_id)

    checks["db"] = _database_ready(platform, source_id)

    checks["source"] = any(item.source_id == source_id for item in platform.registry.sources)

    try:
        platform.kg_registry.get_graph(graph_id)
        checks["graph"] = True
    except Exception:
        checks["graph"] = False

    required = ("health", "ontology", "mapping", "db", "source", "graph")
    missing = [key for key in required if not checks.get(key)]
    return {
        "graph_id": graph_id,
        "source_id": source_id,
        "ontology_id": ontology_id,
        "ready_to_build": not missing,
        "missing": missing,
        "checks": checks,
        "sync_state": load_sync_state(platform.settings) or None,
    }


def _ontology_for_graph(platform, graph_id: str) -> str | None:
    try:
        return platform.kg_registry.get_graph(graph_id).ontology_id
    except Exception:
        return None


def _database_ready(platform, source_id: str) -> bool:
    if mapping_exists(source_id):
        mapping = get_mapping(source_id)["mapping"]
        connection_id = str(mapping.get("connection_id") or "").strip()
        if connection_id:
            try:
                get_connection(platform.settings, connection_id)
            except NotFoundError:
                return False
            return True
    try:
        require_ecom_dsn(platform.settings)
    except FashionGPTError:
        return False
    return True
