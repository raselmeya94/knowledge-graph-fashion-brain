"""HTTP routes for the knowledge-graph construction tools."""

from __future__ import annotations

import anyio

from fastapi import Depends, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from knowledge_graph.construction.detect import detect_kind
from knowledge_graph.construction.pipeline import pipeline_for
from knowledge_graph.construction import scraping as scraping_flow
from knowledge_graph.errors import RequestTimeout, ValidationFailed
from knowledge_graph.kg_registry import default_graph_for_kind


class PipelineBody(BaseModel):
    text: str | None = None
    filename: str | None = None
    kind: str | None = None
    media_type: str | None = None
    json_data: dict | list | None = None
    data_base64: str | None = None
    use_model: bool = True
    clean: bool = False
    auto_commit: bool = False
    graph_id: str | None = None
    ontology_id: str | None = None


class ExtractBody(BaseModel):
    text: str | None = None
    filename: str | None = None
    kind: str | None = None
    media_type: str | None = None
    json_data: dict | list | None = None
    data_base64: str | None = None
    clean: bool = False


class ProposeBody(BaseModel):
    text: str = ""
    title: str | None = None
    kind: str | None = None
    graph_id: str | None = None
    use_model: bool = True
    structured: dict | list | None = None
    filename: str | None = None


class ScrapingChunkBody(BaseModel):
    file_id: str | None = None
    text: str | None = None
    filename: str | None = None
    kind: str | None = None


class ScrapingProposeBody(BaseModel):
    file_id: str
    graph_id: str | None = None
    ontology_id: str | None = None
    use_model: bool = True


class ScrapingQueueBody(BaseModel):
    file_id: str
    graph_id: str | None = None
    ontology_id: str | None = None
    relationships: list[dict] | None = None
    warnings: list[str] = Field(default_factory=list)


class ScrapingCommitBody(BaseModel):
    review_id: str
    ontology: dict = Field(default_factory=dict)
    graph: dict = Field(default_factory=dict)
    graph_id: str | None = None


class ToolBody(BaseModel):
    """Loose tool payload. Known fields stay typed; other tool keys pass through."""

    model_config = ConfigDict(extra="allow")

    text: str | None = None
    filename: str | None = None
    kind: str | None = None
    media_type: str | None = None
    json_data: dict | list | None = None
    data_base64: str | None = None
    prompt: str | None = None
    system: str | None = None
    action: str | None = None
    json_mode: bool = False
    clean: bool = False
    use_model: bool = True
    title: str | None = None
    review_id: str | None = None
    decision: str | None = None
    relationship_ids: list[str] = Field(default_factory=list)
    edits: list[dict] = Field(default_factory=list)
    structured: dict | list | None = None
    relationships: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_kind: str | None = None
    metadata: dict = Field(default_factory=dict)
    graph_id: str | None = None
    ontology_id: str | None = None
    remove_ids: list[str] = Field(default_factory=list)
    hard: bool = False
    relationship_id: str | None = None


_EXTRACTORS = {
    "html": "extract_html",
    "json": "parse_json",
    "image": "understand_image",
}


def register_construction_routes(app, current_platform, current_principal) -> None:
    @app.get("/kg/tools")
    def construction_tools(platform=Depends(current_platform), _principal=Depends(current_principal)):
        return {"tools": pipeline_for(platform).list_tools()}

    @app.get("/kg/ollama")
    def ollama_status(platform=Depends(current_platform), _principal=Depends(current_principal)):
        return pipeline_for(platform).ollama.health()

    @app.get("/kg/relationships")
    async def relationships(
        graph: str | None = None,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        payload = {"graph": graph} if graph else {}
        return await _run(platform, pipeline_for(platform).run_tool, "list_relationships", payload)

    @app.get("/kg/reviews")
    def reviews(
        graph_id: str | None = None,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        items = [item.model_dump() for item in pipeline_for(platform).store.list()]
        if graph_id:
            items = [item for item in items if item.get("graph_id") == graph_id]
        return {"reviews": items}

    @app.get("/kg/reviews/{review_id}")
    def review(review_id: str, platform=Depends(current_platform), _principal=Depends(current_principal)):
        return pipeline_for(platform).store.get(review_id).model_dump()

    @app.post("/kg/tools/{tool_name}")
    async def run_tool(
        tool_name: str,
        body: ToolBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        return await _run(platform, pipeline_for(platform).run_tool, tool_name, body.model_dump(exclude_none=True))

    @app.post("/extract")
    async def extract(
        body: ExtractBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _extract():
            payload = body.model_dump(exclude_none=True)
            kind = detect_kind(payload)
            tool = _EXTRACTORS.get(kind, "extract_text")
            result = pipeline_for(platform).run_tool(tool, payload)
            result["kind"] = kind
            result["extractor"] = tool
            return result

        return await _run(platform, _extract)

    @app.post("/relationships/propose")
    async def propose_relationships(
        body: ProposeBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _propose():
            kind = body.kind or "text"
            graph_id = body.graph_id or default_graph_for_kind(kind)
            pipeline = pipeline_for(platform)
            existing = pipeline.run_tool("list_relationships", {"graph": graph_id})
            return pipeline.run_tool(
                "identify_relationships",
                {
                    "title": body.title or body.filename or "Untitled document",
                    "text": body.text,
                    "kind": kind,
                    "graph_id": graph_id,
                    "use_model": body.use_model,
                    "structured": body.structured,
                    "existing": existing["relationships"],
                },
            )

        return await _run(platform, _propose)

    @app.post("/kg/pipeline")
    async def run_pipeline(
        body: PipelineBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        return await _run(platform, pipeline_for(platform).run, body.model_dump(exclude_none=True))

    @app.post("/kg/pipeline/upload")
    async def upload_pipeline(
        file: UploadFile = File(...),
        kind: str | None = Form(None),
        use_model: bool = Form(True),
        clean: bool = Form(False),
        auto_commit: bool = Form(False),
        graph_id: str | None = Form(None),
        ontology_id: str | None = Form(None),
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        data = await file.read()
        if not data:
            raise ValidationFailed("Uploaded file is empty.")
        payload = {
            "filename": file.filename or "",
            "media_type": file.content_type or "",
            "kind": kind,
            "data": data,
            "use_model": use_model,
            "clean": clean,
            "auto_commit": auto_commit,
            "graph_id": graph_id,
            "ontology_id": ontology_id,
        }
        return await _run(platform, pipeline_for(platform).run, payload)

    @app.post("/kg/reviews/{review_id}")
    async def update_review(
        review_id: str,
        body: ToolBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        payload = body.model_dump(exclude_none=True)
        payload["review_id"] = review_id
        return await _run(platform, pipeline_for(platform).run_tool, "update_review", payload)

    @app.delete("/kg/reviews/{review_id}/relationships/{rel_id}")
    async def remove_review_relationship(
        review_id: str,
        rel_id: str,
        hard: bool = False,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        return await _run(
            platform,
            pipeline_for(platform).run_tool,
            "remove_review_relationship",
            {"review_id": review_id, "relationship_id": rel_id, "hard": hard},
        )

    @app.post("/kg/reviews/{review_id}/commit")
    async def commit_review(
        review_id: str,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        return await _run(platform, pipeline_for(platform).run_tool, "ingest_graph", {"review_id": review_id})

    @app.post("/scraping/extract")
    async def scraping_extract(
        body: ExtractBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _extract():
            return scraping_flow.extract_file(
                platform, pipeline_for(platform), body.model_dump(exclude_none=True)
            )

        return await _run(platform, _extract)

    @app.post("/scraping/extract/upload")
    async def scraping_extract_upload(
        file: UploadFile = File(...),
        kind: str | None = Form(None),
        clean: bool = Form(False),
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        data = await file.read()
        if not data:
            raise ValidationFailed("Uploaded file is empty.")
        payload = {
            "filename": file.filename or "",
            "media_type": file.content_type or "",
            "kind": kind,
            "data": data,
            "clean": clean,
        }

        def _extract():
            return scraping_flow.extract_file(platform, pipeline_for(platform), payload)

        return await _run(platform, _extract)

    @app.post("/scraping/chunk")
    async def scraping_chunk(
        body: ScrapingChunkBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        return await _run(platform, scraping_flow.chunk_file, platform, body.model_dump(exclude_none=True))

    @app.post("/scraping/propose")
    async def scraping_propose(
        body: ScrapingProposeBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _propose():
            return scraping_flow.propose_chunks(
                platform, pipeline_for(platform), body.model_dump(exclude_none=True)
            )

        return await _run(platform, _propose)

    @app.post("/scraping/queue")
    async def scraping_queue(
        body: ScrapingQueueBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _queue():
            return scraping_flow.queue_scraping_review(
                platform, pipeline_for(platform), body.model_dump(exclude_none=True)
            )

        return await _run(platform, _queue)

    @app.post("/scraping/commit")
    async def scraping_commit(
        body: ScrapingCommitBody,
        platform=Depends(current_platform),
        _principal=Depends(current_principal),
    ):
        def _commit():
            return scraping_flow.commit_scraping(
                platform, pipeline_for(platform), body.model_dump(exclude_none=True)
            )

        return await _run(platform, _commit)


async def _run(platform, func, *args):
    try:
        with anyio.fail_after(platform.settings.request_timeout_seconds):
            return await anyio.to_thread.run_sync(lambda: func(*args))
    except TimeoutError as exc:
        raise RequestTimeout("The request exceeded the configured timeout.") from exc
