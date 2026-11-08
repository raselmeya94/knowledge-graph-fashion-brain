"""Knowledge-graph HTTP service. Construction routes only."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

import anyio
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from knowledge_graph.access import CLEARANCE_RANK, Principal
from knowledge_graph.config import Settings, load_settings
from knowledge_graph.construction.api import register_construction_routes
from knowledge_graph.construction.ollama import OllamaClient
from knowledge_graph.errors import FashionGPTError, PermissionDenied, RequestTimeout, Unauthorized, ValidationFailed
from knowledge_graph.graph.templates import run_template
from knowledge_graph.jobs import JobStore
from knowledge_graph.kg_registry.api import register_registry_routes
from knowledge_graph.logging_config import configure_logging, get_logger
from knowledge_graph.pipeline.service import Platform
from knowledge_graph.retrieval.brain import BrainRetrieveRequest, retrieve

LOGGER = get_logger("knowledge_graph.api")


class IngestRequest(BaseModel):
    sources: list[str] | None = None


class GraphQueryRequest(BaseModel):
    template: str
    params: dict[str, str] = Field(default_factory=dict)
    limit: int = 20
    offset: int = 0

    def bounded(self) -> None:
        if not 1 <= self.limit <= 100:
            raise ValidationFailed("limit must be between 1 and 100.")
        if self.offset < 0:
            raise ValidationFailed("offset must be zero or greater.")


def open_platform(
    settings: Settings | None = None,
    *,
    load_snapshot: bool = True,
    sources: list[str] | None = None,
) -> Platform:
    """Open the graph. Load a saved snapshot when a later write must keep existing nodes."""

    platform = Platform(settings or load_settings())
    if load_snapshot:
        platform._restore_memory_snapshot(sources)
    return platform


def load_query_data(platform: Platform) -> None:
    """Warm the query path: Neo4j keeps the graph; JSON snapshots are the reload source.

    If Neo4j already has source graphs, only refresh registry + catalog (do not wipe).
    Empty DB (or memory backend) loads snapshots via ``reload_query_data``.
    """

    from knowledge_graph.graph.snapshots import loaded_source_ids

    if platform.settings.graph_backend == "neo4j" and loaded_source_ids(platform.graph):
        platform.kg_registry.register_all_identities()
        platform.refresh_union_ontology()
        catalog_n = platform.load_catalog()
        tallies = platform.graph.counts() if hasattr(platform.graph, "counts") else {}
        stats = {
            "loaded": 1,
            "backend": "neo4j",
            "mode": "reuse",
            "registry_graphs": len(platform.kg_registry.list_graphs()),
            "nodes": int(tallies.get("nodes", 0)),
            "edges": int(tallies.get("edges", 0)),
            "catalog_rows": catalog_n,
        }
        LOGGER.info("knowledge graph query data ready %s", stats)
        return
    stats = platform.reload_query_data()
    LOGGER.info("knowledge graph query data ready %s", stats)


def create_app(platform: Platform | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.platform is None:
            settings = load_settings()
            configure_logging(settings.log_level, settings.log_json)
            app.state.platform = Platform(settings)
            load_query_data(app.state.platform)
        if getattr(app.state, "jobs", None) is None:
            app.state.jobs = JobStore()
        try:
            yield
        finally:
            jobs = getattr(app.state, "jobs", None)
            if jobs is not None:
                await jobs.shutdown(wait_seconds=2.0)
            current = getattr(app.state, "platform", None)
            graph = getattr(current, "graph", None) if current is not None else None
            closer = getattr(graph, "close", None)
            if callable(closer):
                try:
                    closer()
                    LOGGER.info("graph backend closed")
                except Exception:
                    LOGGER.exception("graph backend close failed")

    app = FastAPI(
        title="Knowledge Graph",
        version="0.1.0",
        description="Build and query the knowledge graph. The chatbot is a separate service.",
        lifespan=lifespan,
    )
    app.state.platform = platform
    app.state.jobs = JobStore()

    @app.exception_handler(FashionGPTError)
    async def handle_error(_request: Request, exc: FashionGPTError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "message": exc.message})

    def current_platform(request: Request) -> Platform:
        current = request.app.state.platform
        if current is None:
            raise FashionGPTError("Knowledge graph platform is not initialized.")
        return current

    def current_jobs(request: Request) -> JobStore:
        jobs = getattr(request.app.state, "jobs", None)
        if jobs is None:
            request.app.state.jobs = JobStore()
            jobs = request.app.state.jobs
        return jobs

    def current_principal(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        x_clearance: Annotated[str | None, Header()] = None,
    ) -> Principal:
        return principal_for(current_platform(request), authorization, x_clearance)

    @app.get("/health")
    def health(platform: Platform = Depends(current_platform)):
        graph_status = "ok"
        try:
            platform.graph.ping()
        except FashionGPTError as exc:
            graph_status = exc.message
        ollama = OllamaClient(platform.settings).health()
        return {
            "status": "degraded" if graph_status != "ok" else "ok",
            "service": "knowledge_graph",
            "graph_backend": platform.settings.graph_backend,
            "graph": graph_status,
            "ollama": ollama.get("status", "unchecked"),
        }

    @app.get("/sources")
    def sources(platform: Platform = Depends(current_platform), _principal: Principal = Depends(current_principal)):
        return {"sources": [source.model_dump() for source in platform.registry.sources]}

    @app.get("/jobs")
    def list_jobs(
        kind: str | None = None,
        limit: int = 30,
        jobs: JobStore = Depends(current_jobs),
        _principal: Principal = Depends(current_principal),
    ):
        return {"jobs": jobs.list(limit=limit, kind=kind)}

    @app.get("/jobs/{job_id}")
    def get_job(
        job_id: str,
        jobs: JobStore = Depends(current_jobs),
        _principal: Principal = Depends(current_principal),
    ):
        return {"job": jobs.get(job_id)}

    @app.get("/ingestion/status")
    def ingestion_status(
        platform: Platform = Depends(current_platform),
        jobs: JobStore = Depends(current_jobs),
        _principal: Principal = Depends(current_principal),
    ):
        return {
            "job": jobs.latest_for_kind("ingest"),
            "records": [record.model_dump() for record in platform.state.records.values()],
            "reports": [report.model_dump() for report in platform.last_reports],
            "shared": platform.shared_counts,
            "degraded": platform.degraded_reason,
        }

    @app.post("/ingestion/run")
    async def ingestion_run(
        body: IngestRequest,
        platform: Platform = Depends(current_platform),
        jobs: JobStore = Depends(current_jobs),
        _principal: Principal = Depends(current_principal),
    ):
        """Start PLM data ingest as a background job (large CSVs exceed HTTP timeouts)."""

        if platform.settings.env != "development":
            raise PermissionDenied("Ingestion trigger is available only in development.")

        sources = list(body.sources) if body.sources else None
        exclusive = "ingest:" + (",".join(sources) if sources else "all")

        def _runner() -> dict[str, Any]:
            reports = platform.rebuild(sources)
            return {
                "message": "ingest finished",
                "reports": [report.model_dump() for report in reports],
            }

        job, created = await jobs.start(
            kind="ingest",
            runner=_runner,
            params={"sources": sources},
            exclusive_key=exclusive,
            message="Ingest queued",
        )
        return {"status": "started" if created else "already_running", "job": job}

    @app.post("/graphs/{graph_name}/query")
    async def graph_query(
        graph_name: str,
        body: GraphQueryRequest,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
    ):
        body.bounded()
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

    @app.get("/products")
    async def products(
        q: str | None = None,
        material: str | None = None,
        season: str | None = None,
        category: str | None = None,
        limit: int = 20,
        offset: int = 0,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
    ):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValidationFailed("limit must be 1..100 and offset must be >= 0.")

        def _products():
            rows, total = platform.products.search(
                principal, limit=limit, offset=offset, q=q, material=material, season=season, category=category
            )
            return {
                "items": [
                    {
                        "product_id": row.record_id,
                        "title": row.title,
                        "product": row.metadata.get("product"),
                        "content_hash": row.content_hash,
                        "source_path": row.source_path,
                        "synthetic": row.synthetic,
                    }
                    for row in rows
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
                "note": "Counts come from the synthetic file repository, not a live PLMEX database.",
            }

        return await _run(platform, _products)

    @app.post("/brain/retrieve")
    async def brain_retrieve(
        body: BrainRetrieveRequest,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
    ):
        """Evidence pack for Fashion Brain. Intention stays in the chat service."""

        return await _run(platform, retrieve, platform, principal, body)

    @app.post("/brain/cypher")
    async def brain_cypher(
        body: dict,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
    ):
        """Execute one read-only Cypher plan from Fashion Brain Text2CypherAgent."""

        from knowledge_graph.retrieval.cypher_safe import run_read_cypher

        cypher = str((body or {}).get("cypher") or "")
        params = (body or {}).get("params") if isinstance((body or {}).get("params"), dict) else {}
        limit_rows = int((body or {}).get("limit_rows") or 200)

        def _run_cypher():
            try:
                rows = run_read_cypher(platform, cypher, params, limit_rows=limit_rows)
                return {"rows": rows, "count": len(rows)}
            except Exception as exc:
                return {"rows": [], "count": 0, "error": str(exc)}

        _ = principal  # clearance already enforced by dependency
        return await _run(platform, _run_cypher)

    @app.post("/brain/link_entities")
    async def brain_link_entities(
        body: dict,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
    ):
        """Resolve extracted terms to real Neo4j facet node ids."""

        from knowledge_graph.linking.entity_linker import link_entities, resolved_payload

        graph_id = str((body or {}).get("graph_id") or "ecom_products")
        entities = (body or {}).get("entities") if isinstance((body or {}).get("entities"), dict) else {}

        def _link():
            results = link_entities(platform, entities, graph_id=graph_id)
            return {
                "results": [r.to_dict() for r in results],
                "resolved": resolved_payload(results),
            }

        _ = principal
        return await _run(platform, _link)

    @app.get("/brain/brands")
    async def brain_brands(
        graph_id: str | None = None,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
    ):
        from knowledge_graph.linking.entity_linker import list_live_brands
        from knowledge_graph.retrieval.runtime_schema import default_product_graph_id

        def _brands():
            gid = graph_id or default_product_graph_id(platform)
            return {"brands": list_live_brands(platform, gid) if gid else [], "graph_id": gid}

        _ = principal
        return await _run(platform, _brands)

    @app.get("/brain/runtime_schema")
    async def brain_runtime_schema(
        graph_id: str | None = None,
        platform: Platform = Depends(current_platform),
        principal: Principal = Depends(current_principal),
    ):
        from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

        def _schema():
            return build_runtime_schema(platform, graph_id).to_dict()

        _ = principal
        return await _run(platform, _schema)

    register_registry_routes(app, current_platform, current_principal, current_jobs)
    register_construction_routes(app, current_platform, current_principal)
    from knowledge_graph.api.plan_routes import register_plan_api_routes

    register_plan_api_routes(app, current_platform, current_principal, current_jobs)
    return app


def principal_for(platform: Platform, authorization: str | None, clearance: str | None) -> Principal:
    settings = platform.settings
    if settings.api_token:
        if authorization != f"Bearer {settings.api_token}":
            raise Unauthorized("Authentication is required.")
    elif settings.env not in {"development", "test"}:
        raise Unauthorized("An API token is required outside development.")
    level = clearance or "public"
    if level not in CLEARANCE_RANK:
        raise ValidationFailed("Clearance must be public, internal, or confidential.")
    return Principal(subject="knowledge-graph", clearance=level)


async def _run(platform, func, *args, **kwargs):
    try:
        with anyio.fail_after(platform.settings.request_timeout_seconds):
            return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))
    except TimeoutError as exc:
        raise RequestTimeout("The request exceeded the configured timeout.") from exc
