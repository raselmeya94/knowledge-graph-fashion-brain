"""HTTP routes for ontology and graph registry management."""

from __future__ import annotations

from typing import Any

import anyio
from fastapi import Depends
from pydantic import BaseModel, Field

from knowledge_graph.errors import NotFoundError, RequestTimeout, ValidationFailed
from knowledge_graph.graph.snapshots import restore_snapshots, save_master_snapshot, save_source_snapshots
from knowledge_graph.jobs import JobStore
from knowledge_graph.kg_registry.models import GraphCreate, OntologyCreate, OntologyPairBody, OntologyReplace


class EdgePatchBody(BaseModel):
    properties: dict = Field(default_factory=dict)


class GraphDeleteBody(BaseModel):
    force: bool = False


def register_registry_routes(app, current_platform, current_principal, current_jobs) -> None:
    @app.get("/ontologies")
    def list_ontologies(platform=Depends(current_platform), _principal=Depends(current_principal)):
        rows = platform.kg_registry.list_ontologies()
        return {"ontologies": [item.model_dump() for item in rows]}

    @app.post("/ontologies")
    async def create_ontology(
        body: OntologyCreate,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _create():
            record, ontology = platform.kg_registry.create_ontology(body)
            from knowledge_graph.retrieval.schema_hooks import on_schema_changed

            hooks = on_schema_changed(platform, ontology_id=record.id)
            return {
                "ontology": record.model_dump(),
                "definition": ontology.model_dump(),
                "schema_hooks": hooks,
            }

        return await _run(platform, _create)

    @app.get("/ontologies/{ontology_id}")
    def get_ontology(ontology_id: str, platform=Depends(current_platform), _principal=Depends(current_principal)):
        record = platform.kg_registry.get_ontology(ontology_id)
        try:
            definition = platform.kg_registry.load_ontology_object(ontology_id)
        except FileNotFoundError as exc:
            raise NotFoundError(
                f"Ontology '{ontology_id}' is registered but its schema file is missing "
                f"({platform.kg_registry.ontology_file(ontology_id)})."
            ) from exc
        return {"ontology": record.model_dump(), "definition": definition.model_dump()}

    @app.put("/ontologies/{ontology_id}")
    async def replace_ontology(
        ontology_id: str,
        body: OntologyReplace,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _replace():
            definition = platform.kg_registry.replace_ontology(ontology_id, body)
            from knowledge_graph.retrieval.schema_hooks import on_schema_changed

            hooks = on_schema_changed(platform, ontology_id=ontology_id)
            record = platform.kg_registry.get_ontology(ontology_id)
            return {
                "ontology": record.model_dump(),
                "definition": definition.model_dump(),
                "schema_hooks": hooks,
            }

        return await _run(platform, _replace)

    @app.post("/ontologies/{ontology_id}/pairs")
    async def add_pair(
        ontology_id: str,
        body: OntologyPairBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _add():
            definition = platform.kg_registry.add_pair(ontology_id, body.type, body.start_type, body.end_type)
            from knowledge_graph.retrieval.schema_hooks import on_schema_changed

            hooks = on_schema_changed(platform, ontology_id=ontology_id)
            return {
                "ontology_id": ontology_id,
                "definition": definition.model_dump(),
                "schema_hooks": hooks,
            }

        return await _run(platform, _add)

    @app.delete("/ontologies/{ontology_id}/pairs")
    async def remove_pair(
        ontology_id: str,
        body: OntologyPairBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _remove():
            definition = platform.kg_registry.remove_pair(ontology_id, body.type, body.start_type, body.end_type)
            from knowledge_graph.retrieval.schema_hooks import on_schema_changed

            hooks = on_schema_changed(platform, ontology_id=ontology_id)
            return {
                "ontology_id": ontology_id,
                "definition": definition.model_dump(),
                "schema_hooks": hooks,
            }

        return await _run(platform, _remove)

    @app.get("/graphs")
    def list_graphs(platform=Depends(current_platform), _principal=Depends(current_principal)):
        from knowledge_graph.kg_registry.store import enrich_graph_record
        from knowledge_graph.retrieval.graph_counts import graph_node_edge_counts

        rows = []
        for item in platform.kg_registry.list_graphs():
            try:
                ontology = platform.kg_registry.load_ontology_object(item.ontology_id)
            except Exception:
                ontology = None
            enriched = enrich_graph_record(item, ontology)
            nodes, edges = graph_node_edge_counts(platform, item.id)
            rows.append(
                {
                    **enriched.model_dump(),
                    "node_count": nodes,
                    "edge_count": edges,
                }
            )
        # shared/master = interconnect overlay (not individual chat corpora).
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
                    "node_count": nodes,
                    "edge_count": edges,
                }
            )
        return {"graphs": rows}

    @app.post("/graphs/reload")
    async def reload_graphs(
        platform=Depends(current_platform),
        jobs: JobStore = Depends(current_jobs),
        _principal=Depends(current_principal),
    ):
        """Hot-reload snapshots + master/shared + PLM index as a background job."""

        def _runner() -> dict[str, Any]:
            stats = platform.reload_query_data()
            return {"message": "reload finished", "stats": stats}

        job, created = await jobs.start(
            kind="reload",
            runner=_runner,
            params={},
            exclusive_key="reload:all",
            message="Reload queued",
        )
        return {"status": "started" if created else "already_running", "job": job}

    @app.post("/graphs")
    async def create_graph(
        body: GraphCreate,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _create():
            record = platform.kg_registry.create_graph(body)
            from knowledge_graph.retrieval.schema_hooks import on_schema_changed

            hooks = on_schema_changed(platform, graph_id=record.id, ontology_id=record.ontology_id)
            return {"graph": record.model_dump(), "schema_hooks": hooks}

        return await _run(platform, _create)

    @app.get("/graphs/{graph_id}")
    def get_graph(graph_id: str, platform=Depends(current_platform), _principal=Depends(current_principal)):
        from knowledge_graph.retrieval.graph_counts import graph_node_edge_counts

        nodes, edges = graph_node_edge_counts(platform, graph_id)
        if graph_id in {"shared", "master"}:
            return {
                "graph": {
                    "id": graph_id,
                    "name": graph_id,
                    "kind": graph_id,
                    "builtin": True,
                    "status": "active",
                },
                "node_count": nodes,
                "edge_count": edges,
            }
        record = platform.kg_registry.get_graph(graph_id)
        return {
            "graph": record.model_dump(),
            "node_count": nodes,
            "edge_count": edges,
            "ontology": platform.ontology_for(graph_id).model_dump(),
        }

    @app.post("/graphs/{graph_id}/sync")
    async def sync_graph(
        graph_id: str,
        platform=Depends(current_platform),
        jobs: JobStore = Depends(current_jobs),
        _principal=Depends(current_principal),
    ):
        def _sync() -> dict[str, Any]:
            from knowledge_graph.sources.sync import sync_source
            from knowledge_graph.retrieval.schema_hooks import on_schema_changed

            record = platform.kg_registry.get_graph(graph_id)
            if record.kind == "product" or record.source_ids:
                sources = record.source_ids or [graph_id]
                # Unified dispatch (ecom SQL sync or file rebuild).
                payloads = [sync_source(platform, source_id) for source_id in sources]
                hooks = on_schema_changed(platform, graph_id=graph_id, ontology_id=record.ontology_id)
                return {
                    "message": f"sync finished for {graph_id}",
                    "graph_id": graph_id,
                    "action": "sync",
                    "results": payloads,
                    "schema_hooks": hooks,
                    "edge_count": len(platform.graph.find_edges(graph=graph_id, include_deleted=False)),
                }
            save_source_snapshots(platform.graph, platform.settings.generated, [graph_id])
            save_master_snapshot(platform.graph, platform.settings.generated)
            hooks = on_schema_changed(platform, graph_id=graph_id)
            return {
                "message": f"sync finished for {graph_id}",
                "graph_id": graph_id,
                "action": "sync",
                "reports": [],
                "schema_hooks": hooks,
                "edge_count": len(platform.graph.find_edges(graph=graph_id, include_deleted=False)),
            }

        job, created = await jobs.start(
            kind="sync",
            runner=_sync,
            params={"graph_id": graph_id},
            exclusive_key=f"build:{graph_id}",
            message=f"Sync queued for {graph_id}",
        )
        return {"status": "started" if created else "already_running", "job": job}

    @app.post("/graphs/{graph_id}/rebuild")
    async def rebuild_graph(
        graph_id: str,
        platform=Depends(current_platform),
        jobs: JobStore = Depends(current_jobs),
        _principal=Depends(current_principal),
    ):
        def _rebuild() -> dict[str, Any]:
            record = platform.kg_registry.get_graph(graph_id)
            if record.kind == "product" or record.source_ids:
                sources = record.source_ids or [graph_id]
                reports = [report.model_dump() for report in platform.rebuild(sources)]
                return {
                    "message": f"rebuild finished for {graph_id}",
                    "graph_id": graph_id,
                    "action": "rebuild",
                    "reports": reports,
                }
            restored = restore_snapshots(platform.graph, platform.settings.generated, [graph_id])
            return {
                "message": f"rebuild finished for {graph_id}",
                "graph_id": graph_id,
                "action": "rebuild",
                "restored": restored,
                "edge_count": len(platform.graph.find_edges(graph=graph_id, include_deleted=False)),
            }

        job, created = await jobs.start(
            kind="rebuild",
            runner=_rebuild,
            params={"graph_id": graph_id},
            exclusive_key=f"build:{graph_id}",
            message=f"Rebuild queued for {graph_id}",
        )
        return {"status": "started" if created else "already_running", "job": job}

    @app.delete("/graphs/{graph_id}")
    async def delete_graph(
        graph_id: str,
        body: GraphDeleteBody | None = None,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        force = bool(body.force) if body else False

        def _delete():
            record = platform.kg_registry.soft_delete_graph(graph_id, force=force)
            deleted = 0
            if hasattr(platform.graph, "soft_delete_graph"):
                deleted = platform.graph.soft_delete_graph(graph_id)
            save_source_snapshots(platform.graph, platform.settings.generated, [graph_id])
            save_master_snapshot(platform.graph, platform.settings.generated)
            return {"graph": record.model_dump(), "edges_marked_deleted": deleted}

        return await _run(platform, _delete)

    @app.delete("/graphs/{graph_id}/edges/{edge_id}")
    async def delete_edge(
        graph_id: str,
        edge_id: str,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _delete():
            if graph_id not in {"shared", "master"}:
                platform.kg_registry.get_graph(graph_id)
            edge = platform.graph.get_edge(edge_id, include_deleted=True)
            if edge is None:
                raise NotFoundError(f"Edge '{edge_id}' was not found.")
            if edge.graph != graph_id:
                raise ValidationFailed(f"Edge '{edge_id}' belongs to graph '{edge.graph}', not '{graph_id}'.")
            deleted = platform.graph.soft_delete_edge(edge_id)
            save_source_snapshots(platform.graph, platform.settings.generated, [graph_id])
            save_master_snapshot(platform.graph, platform.settings.generated)
            return {"edge": deleted.model_dump()}

        return await _run(platform, _delete)

    @app.patch("/graphs/{graph_id}/edges/{edge_id}")
    async def patch_edge(
        graph_id: str,
        edge_id: str,
        body: EdgePatchBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _patch():
            if graph_id not in {"shared", "master"}:
                platform.kg_registry.get_graph(graph_id)
            edge = platform.graph.get_edge(edge_id, include_deleted=True)
            if edge is None:
                raise NotFoundError(f"Edge '{edge_id}' was not found.")
            if edge.graph != graph_id:
                raise ValidationFailed(f"Edge '{edge_id}' belongs to graph '{edge.graph}', not '{graph_id}'.")
            patched = platform.graph.patch_edge(edge_id, body.properties)
            save_source_snapshots(platform.graph, platform.settings.generated, [graph_id])
            save_master_snapshot(platform.graph, platform.settings.generated)
            return {"edge": patched.model_dump()}

        return await _run(platform, _patch)


async def _run(platform, func, *args):
    try:
        with anyio.fail_after(platform.settings.request_timeout_seconds):
            return await anyio.to_thread.run_sync(lambda: func(*args))
    except TimeoutError as exc:
        raise RequestTimeout("The request exceeded the configured timeout.") from exc
