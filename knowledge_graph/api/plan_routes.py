"""Plan-aligned /api/* aliases over existing KG handlers."""

from __future__ import annotations

from typing import Any

import anyio
from fastapi import Depends
from pydantic import BaseModel, Field

from knowledge_graph.errors import RequestTimeout, ValidationFailed
from knowledge_graph.global_kg import global_info
from knowledge_graph.graph.templates import run_template
from knowledge_graph.jobs import JobStore
from knowledge_graph.retrieval.brain import BrainRetrieveRequest, retrieve
from knowledge_graph.retrieval.metadata import discover_values, metadata_overview
from knowledge_graph.sources.registry_ops import SourceRegisterBody, get_source, list_sources, register_source
from knowledge_graph.sources.schema import source_schema
from knowledge_graph.sources.sync import sync_source


class GraphQueryBody(BaseModel):
    template: str
    params: dict[str, str] = Field(default_factory=dict)
    limit: int = 20
    offset: int = 0
    graph_id: str | None = None
    graph_ids: list[str] | None = None

    def bounded(self) -> None:
        if not 1 <= self.limit <= 100:
            raise ValidationFailed("limit must be between 1 and 100.")
        if self.offset < 0:
            raise ValidationFailed("offset must be zero or greater.")


class CrossQueryBody(BaseModel):
    intent: str = "product_count"
    question: str = ""
    filters: dict[str, str] = Field(default_factory=dict)
    group_by: str | None = None
    graph_ids: list[str] = Field(default_factory=list)
    use_global_kg: bool = False
    limit: int | None = None
    suggest: bool = False
    analysis: bool = False


class RebuildBody(BaseModel):
    graph_id: str | None = None
    source_id: str | None = None


def register_plan_api_routes(app, current_platform, current_principal, current_jobs) -> None:
    @app.get("/api/sources")
    def api_list_sources(platform=Depends(current_platform), _principal=Depends(current_principal)):
        return {"sources": list_sources(platform)}

    @app.post("/api/sources")
    def api_register_source(
        body: SourceRegisterBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        return {"source": register_source(platform, body)}

    @app.get("/api/sources/{source_id}")
    def api_get_source(source_id: str, platform=Depends(current_platform), _principal=Depends(current_principal)):
        return {"source": get_source(platform, source_id)}

    @app.get("/api/sources/{source_id}/schema")
    def api_source_schema(source_id: str, platform=Depends(current_platform), _principal=Depends(current_principal)):
        return source_schema(platform, source_id)

    @app.post("/api/sources/{source_id}/sync")
    async def api_sync_source(
        source_id: str,
        platform=Depends(current_platform),
        jobs: JobStore = Depends(current_jobs),
        _principal=Depends(current_principal),
    ):
        def _runner() -> dict[str, Any]:
            return sync_source(platform, source_id)

        job, created = await jobs.start(
            kind="sync",
            runner=_runner,
            params={"source_id": source_id},
            exclusive_key=f"sync:{source_id}",
            message=f"Sync queued for {source_id}",
        )
        return {"status": "started" if created else "already_running", "job": job}

    @app.get("/api/graphs")
    def api_list_graphs(platform=Depends(current_platform), _principal=Depends(current_principal)):
        from knowledge_graph.kg_registry.store import enrich_graph_record
        from knowledge_graph.retrieval.graph_counts import graph_node_edge_counts

        # Reuse the richer list from registry routes by calling the same logic inline.
        rows = []
        for item in platform.kg_registry.list_graphs():
            try:
                ontology = platform.kg_registry.load_ontology_object(item.ontology_id)
            except Exception:
                ontology = None
            enriched = enrich_graph_record(item, ontology)
            nodes, edges = graph_node_edge_counts(platform, item.id)
            rows.append({**enriched.model_dump(), "node_count": nodes, "edge_count": edges})
        known = {item["id"] for item in rows}
        for name, kind, description in (
            ("shared", "interconnect", "Canonical entities and RESOLVES_TO links across individual KGs"),
            ("master", "interconnect", "Persisted shared interconnect snapshot for warm query load"),
        ):
            if name in known:
                continue
            nodes, edges = graph_node_edge_counts(platform, name)
            rows.append(
                {
                    "id": name,
                    "name": name,
                    "ontology_id": None,
                    "kind": kind,
                    "description": description,
                    "builtin": True,
                    "status": "active",
                    "source_ids": [],
                    "capabilities": ["canonical_materials", "products_sharing_material"],
                    "entities": [],
                    "relationships": ["RESOLVES_TO"],
                    "node_count": nodes,
                    "edge_count": edges,
                }
            )
        return {"graphs": rows}

    @app.get("/api/graphs/registry")
    def api_graphs_registry(platform=Depends(current_platform), _principal=Depends(current_principal)):
        from knowledge_graph.retrieval.graph_counts import registry_count_map

        counts = registry_count_map(platform)
        entries = platform.kg_registry.unified_registry(node_counts=counts)
        return {"registry": [item.model_dump() for item in entries]}

    @app.get("/api/graphs/global")
    def api_graphs_global(platform=Depends(current_platform), _principal=Depends(current_principal)):
        return global_info(platform)

    @app.get("/api/graphs/{graph_id}/schema")
    def api_graph_schema(graph_id: str, platform=Depends(current_platform), _principal=Depends(current_principal)):
        return metadata_overview(platform, graph_id)

    @app.post("/api/graphs/query")
    async def api_graphs_query(
        body: GraphQueryBody,
        platform=Depends(current_platform),
        principal=Depends(current_principal),
    ):
        body.bounded()
        graph_name = body.graph_id or "ecom_products"
        return await _run(
            platform,
            run_template,
            platform,
            graph_name,
            body.template,
            principal,
            limit=body.limit,
            offset=body.offset,
            params=body.params,
        )

    @app.post("/api/graphs/cross-query")
    async def api_cross_query(
        body: CrossQueryBody,
        platform=Depends(current_platform),
        principal=Depends(current_principal),
    ):
        graph_ids = list(body.graph_ids)
        if body.use_global_kg and "shared" not in graph_ids:
            graph_ids = [*graph_ids, "shared"]
        request = BrainRetrieveRequest(
            question=body.question,
            intent=body.intent,
            filters=dict(body.filters),
            group_by=body.group_by,
            limit=body.limit,
            suggest=body.suggest,
            analysis=body.analysis,
            graph_ids=graph_ids or None,
        )
        return await _run(platform, retrieve, platform, principal, request)

    @app.post("/api/graphs/rebuild")
    async def api_graphs_rebuild(
        body: RebuildBody,
        platform=Depends(current_platform),
        jobs: JobStore = Depends(current_jobs),
        _principal=Depends(current_principal),
    ):
        target = body.source_id or body.graph_id
        if not target:
            raise ValidationFailed("Provide graph_id or source_id.")

        def _runner() -> dict[str, Any]:
            if body.source_id:
                return sync_source(platform, body.source_id)
            from knowledge_graph.graph.snapshots import restore_snapshots, save_master_snapshot, save_source_snapshots

            record = platform.kg_registry.get_graph(target)
            if record.kind == "product" or record.source_ids:
                sources = record.source_ids or [target]
                # Prefer unified sync for ecom; rebuild for file sources.
                if "ecom_products" in sources:
                    return sync_source(platform, "ecom_products")
                reports = [report.model_dump() for report in platform.rebuild(sources)]
                save_source_snapshots(platform.graph, platform.settings.generated, [target])
                save_master_snapshot(platform.graph, platform.settings.generated)
                return {"graph_id": target, "action": "rebuild", "reports": reports}
            restored = restore_snapshots(platform.graph, platform.settings.generated, [target])
            return {"graph_id": target, "action": "rebuild", "restored": restored}

        job, created = await jobs.start(
            kind="rebuild",
            runner=_runner,
            params={"target": target},
            exclusive_key=f"build:{target}",
            message=f"Rebuild queued for {target}",
        )
        return {"status": "started" if created else "already_running", "job": job}

    @app.get("/api/metadata")
    def api_metadata(
        graph_id: str | None = None,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        return metadata_overview(platform, graph_id)

    @app.get("/api/metadata/values")
    def api_metadata_values(
        attribute: str,
        graph_id: str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        return discover_values(
            platform,
            graph_id=graph_id,
            attribute=attribute,
            q=q,
            limit=limit,
            offset=offset,
        )


async def _run(platform, func, *args, **kwargs):
    try:
        with anyio.fail_after(platform.settings.request_timeout_seconds):
            return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))
    except TimeoutError as exc:
        raise RequestTimeout("The request exceeded the configured timeout.") from exc
