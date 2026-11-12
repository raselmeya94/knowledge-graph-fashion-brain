"""Building status for UI progress (products / nodes / edges / errors)."""

from __future__ import annotations

from typing import Any

from knowledge_graph.pipeline.ecom_sync import load_sync_state
from knowledge_graph.sources.sync import sync_source


def build_graph(
    platform,
    graph_id: str,
    *,
    source_id: str | None = None,
    brand: str | None = None,
    full: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the registered source sync into an existing graph (no silent scaffolding)."""

    from knowledge_graph.sources.setup import setup_checklist

    record = platform.kg_registry.get_graph(graph_id)
    sid = source_id or (record.source_ids[0] if record.source_ids else graph_id)
    checklist = setup_checklist(
        platform,
        graph_id=graph_id,
        source_id=sid,
        ontology_id=record.ontology_id,
    )
    if not checklist["ready_to_build"]:
        from knowledge_graph.errors import ValidationFailed

        raise ValidationFailed(
            "Cannot build yet. Missing: " + ", ".join(checklist["missing"]) + ". "
            "Complete ontology, mapping, DB, source, and graph registration first."
        )

    if sid == "ecom_products" or (record.kind == "product" and "ecom" in sid):
        from knowledge_graph.pipeline.ecom_sync import sync_ecom_to_neo4j

        result = sync_ecom_to_neo4j(
            platform,
            brand=brand,
            full=full,
            limit=limit,
            refresh_generated=True,
        )
        return {
            "graph_id": graph_id,
            "source_id": sid,
            "action": "build",
            "result": result,
            "status": build_status(platform, graph_id, jobs=None),
        }

    # Dynamic path: mapping + DB connection for any authorized_sql / product KG.
    from knowledge_graph.sources.mapping_store import mapping_exists

    source = next((s for s in platform.registry.sources if s.source_id == sid), None)
    if source and source.connector == "authorized_sql" and mapping_exists(sid):
        from knowledge_graph.pipeline.mapping_sql_sync import sync_mapped_sql_to_neo4j

        result = sync_mapped_sql_to_neo4j(
            platform,
            sid,
            full=full,
            limit=limit,
            refresh_generated=True,
        )
        return {
            "graph_id": graph_id,
            "source_id": sid,
            "action": "build",
            "result": result,
            "status": build_status(platform, graph_id, jobs=None),
        }

    result = sync_source(platform, sid)
    return {
        "graph_id": graph_id,
        "source_id": sid,
        "action": "build",
        "result": result,
        "status": build_status(platform, graph_id, jobs=None),
    }


def build_status(platform, graph_id: str, *, jobs=None, job_id: str | None = None) -> dict[str, Any]:
    """Friendly progress payload for the UI."""

    from knowledge_graph.retrieval.graph_counts import graph_node_edge_counts

    nodes, edges = graph_node_edge_counts(platform, graph_id)
    if graph_id == "ecom_products":
        sync_state = load_sync_state(platform.settings)
    else:
        from knowledge_graph.pipeline.mapping_sql_sync import load_sync_state as load_mapped_state

        record = None
        try:
            record = platform.kg_registry.get_graph(graph_id)
        except Exception:
            record = None
        sid = (record.source_ids[0] if record and record.source_ids else graph_id)
        sync_state = load_mapped_state(platform.settings, sid)
    job = None
    if jobs is not None:
        if job_id:
            try:
                job = jobs.get(job_id)
            except Exception:
                job = None
        if job is None:
            # Prefer active build/sync for this graph.
            for candidate in jobs.list(limit=50):
                params = candidate.get("params") or {}
                if params.get("graph_id") == graph_id or params.get("target") == graph_id:
                    job = candidate
                    break
                if candidate.get("kind") in {"build", "sync", "kg_build"} and graph_id in str(
                    params.get("source_id") or params.get("profile") or ""
                ):
                    job = candidate
                    break

    products = int(sync_state.get("last_products") or 0)
    error_count = int(sync_state.get("errors") or 0)
    phase = "idle"
    status = sync_state.get("status") or "unknown"
    if job:
        status = job.get("status") or status
        if status in {"queued", "running"}:
            phase = "syncing" if status == "running" else "queued"
        elif status == "done":
            phase = "done"
        elif status == "failed":
            phase = "failed"
    elif status == "running":
        phase = "syncing"
    elif status == "ok":
        phase = "done"
    elif status == "failed":
        phase = "failed"

    progress = None
    if job and job.get("progress") is not None:
        progress = job.get("progress")
    elif phase == "done":
        progress = 100
    elif phase == "syncing" and products:
        # Soft estimate only — unknown total until finished.
        progress = min(95, max(5, products // 50))

    return {
        "graph_id": graph_id,
        "job_id": (job or {}).get("id"),
        "phase": phase,
        "status": status,
        "progress_percent": progress,
        "products": products,
        "node_count": nodes,
        "edge_count": edges,
        "node_writes": sync_state.get("node_writes"),
        "edge_writes": sync_state.get("edge_writes"),
        "errors": error_count,
        "brands": sync_state.get("brands") or {},
        "message": (job or {}).get("message") or sync_state.get("status"),
        "finished_at": (job or {}).get("finished_at") or sync_state.get("finished_at"),
        "last_run_at": sync_state.get("last_run_at"),
        "sync_state": sync_state or None,
        "job": job,
    }
