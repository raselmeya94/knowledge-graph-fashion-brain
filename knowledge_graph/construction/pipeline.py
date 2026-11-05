"""Orchestrates the construction tools. Each tool remains callable on its own."""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_graph.construction.detect import detect_kind
from knowledge_graph.construction.identity import NEWS_KINDS
from knowledge_graph.construction.news_facts import segment_articles
from knowledge_graph.construction.ollama import OllamaClient
from knowledge_graph.construction.store import ReviewStore
from knowledge_graph.construction.tools import tool_registry
from knowledge_graph.errors import NotFoundError, ValidationFailed
from knowledge_graph.kg_registry import default_graph_for_kind
from knowledge_graph.pipeline.service import Platform

_EXTRACTORS = {
    "html": "extract_html",
    "json": "parse_json",
    "image": "understand_image",
}


@dataclass
class ToolContext:
    platform: Platform
    settings: object
    ontology: object
    news_ontology: object
    graph: object
    store: ReviewStore
    ollama: OllamaClient

    def ontology_for(self, graph_id: str):
        return self.platform.ontology_for(graph_id)

    def ontology_path_for(self, graph_id: str):
        return self.platform.ontology_path_for(graph_id)


class ConstructionPipeline:
    def __init__(self, platform: Platform):
        self.platform = platform
        self.store = ReviewStore(platform.settings.generated / "construction" / "reviews.json")
        self.ollama = OllamaClient(platform.settings)
        self.tools = tool_registry()

    def context(self) -> ToolContext:
        return ToolContext(
            platform=self.platform,
            settings=self.platform.settings,
            ontology=self.platform.ontology,
            news_ontology=self.platform.news_ontology,
            graph=self.platform.graph,
            store=self.store,
            ollama=self.ollama,
        )

    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": tool.name, "description": tool.description} for tool in self.tools.values()]

    def run_tool(self, name: str, payload: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self.tools))
            raise NotFoundError(f"Unknown construction tool '{name}'. Tools: {known}.")
        return tool.run(payload, self.context())

    def run(self, payload: dict) -> dict:
        if not any(payload.get(key) is not None for key in ("text", "data", "data_base64", "json_data")):
            raise ValidationFailed("Provide text, json_data, a file, or data_base64.")
        kind = detect_kind(payload)
        ctx = self.context()
        graph_id = str(payload.get("graph_id") or "").strip() or default_graph_for_kind(kind)
        graph = self.platform.kg_registry.get_graph(graph_id)
        ontology_id = str(payload.get("ontology_id") or "").strip() or graph.ontology_id
        if ontology_id != graph.ontology_id:
            raise ValidationFailed(
                f"ontology_id '{ontology_id}' does not match graph '{graph_id}' ontology '{graph.ontology_id}'."
            )
        extractor = _EXTRACTORS.get(kind, "extract_text")
        extracted = self.tools[extractor].run(payload, ctx)
        warnings = list(extracted.get("warnings") or [])
        text = str(extracted.get("text") or "")
        metadata = self.tools["extract_metadata"].run(
            {
                "text": text,
                "title": extracted.get("title") or "",
                "filename": payload.get("filename") or "",
                "kind": kind,
                "document_type": extracted.get("document_type") or kind,
                "links": extracted.get("links") or [],
            },
            ctx,
        )
        identify_text = text
        if kind in NEWS_KINDS:
            articles = segment_articles(text)
            metadata["articles"] = articles
            if len(articles) > 1:
                kept = [article for article in articles if article["domain"] == "garment"]
                for article in articles:
                    if article["domain"] != "garment":
                        warnings.append(f"Out of domain: {article['title']}")
                identify_text = "\n\n".join(article["text"] for article in kept)
        existing = self.tools["list_relationships"].run({"graph": graph_id}, ctx)
        identified = self.tools["identify_relationships"].run(
            {
                "title": metadata["title"],
                "text": identify_text,
                "kind": kind,
                "graph_id": graph_id,
                "structured": extracted.get("structured"),
                "use_model": bool(payload.get("use_model", True)),
                "existing": existing["relationships"],
            },
            ctx,
        )
        warnings.extend(identified.get("warnings") or [])
        structured = self.tools["structure_data"].run(
            {
                "source_kind": kind,
                "filename": payload.get("filename") or "",
                "title": metadata["title"],
                "text": text,
                "metadata": metadata,
                "graph_id": graph_id,
                "ontology_id": ontology_id,
                "relationships": identified["relationships"],
                "warnings": warnings,
            },
            ctx,
        )
        queued = self.tools["queue_review"].run(structured, ctx)
        review = queued["review"]
        committed = None
        if payload.get("auto_commit") and review["relationships"]:
            ready_ids = [
                item["id"]
                for item in review["relationships"]
                if not item.get("needs_ontology_approval")
            ]
            pending_ontology = [
                item for item in review["relationships"] if item.get("needs_ontology_approval")
            ]
            if pending_ontology:
                warnings.append(
                    f"{len(pending_ontology)} relationship(s) need ontology approval before commit."
                )
            if ready_ids:
                self.tools["update_review"].run(
                    {"review_id": review["id"], "decision": "approved", "relationship_ids": ready_ids},
                    ctx,
                )
                committed = self.tools["ingest_graph"].run({"review_id": review["id"]}, ctx)
                review = self.store.get(review["id"]).model_dump()
        return {
            "kind": kind,
            "graph_id": graph_id,
            "ontology_id": ontology_id,
            "steps": [
                extractor,
                "extract_metadata",
                "list_relationships",
                "identify_relationships",
                "structure_data",
                "queue_review",
                *(["update_review", "ingest_graph"] if committed else []),
            ],
            "metadata": metadata,
            "entities": structured["entities"],
            "relationships": review["relationships"],
            "review": review,
            "committed": committed,
            "warnings": warnings,
        }


def pipeline_for(platform: Platform) -> ConstructionPipeline:
    current = getattr(platform, "construction", None)
    if current is None:
        current = ConstructionPipeline(platform)
        platform.construction = current
    return current
