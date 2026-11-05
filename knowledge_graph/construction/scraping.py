"""Scraping construction: extract → chunk → propose → queue → commit."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from knowledge_graph.construction.chunking import build_chunks
from knowledge_graph.construction.detect import detect_kind
from knowledge_graph.errors import NotFoundError, ValidationFailed
from knowledge_graph.kg_registry.models import GraphCreate, OntologyCreate
from knowledge_graph.ontology import normalize_name
from knowledge_graph.pipeline.service import utc_now

SCRAPING_KINDS = frozenset({"pdf", "doc", "docx", "json", "text"})
MAX_CHUNKS = 80
_LOCK = threading.Lock()


class ScrapingStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, file_id: str) -> Path:
        return self.root / f"{file_id}.json"

    def save(self, record: dict) -> dict:
        file_id = record["file_id"]
        path = self.path_for(file_id)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        return record

    def get(self, file_id: str) -> dict:
        path = self.path_for(file_id)
        if not path.exists():
            raise NotFoundError(f"Scraping file '{file_id}' was not found.")
        return json.loads(path.read_text(encoding="utf-8"))


def scraping_store_for(platform) -> ScrapingStore:
    return ScrapingStore(platform.settings.generated / "scraping" / "files")


def extract_file(platform, pipeline, payload: dict) -> dict:
    kind = detect_kind(payload)
    if kind not in SCRAPING_KINDS:
        raise ValidationFailed(
            f"Scraping extract supports pdf, doc, docx, json, text. Got '{kind}'."
        )
    tool = "parse_json" if kind == "json" else "extract_text"
    ctx = pipeline.context()
    extracted = pipeline.tools[tool].run(payload, ctx)
    text = str(extracted.get("text") or "").strip()
    if not text:
        raise ValidationFailed("No text could be extracted from this source.")
    file_id = uuid.uuid4().hex
    filename = str(payload.get("filename") or extracted.get("title") or f"scraped.{kind}")
    title = str(extracted.get("title") or filename)
    record = {
        "file_id": file_id,
        "filename": filename,
        "kind": kind,
        "title": title,
        "text": text,
        "char_count": len(text),
        "warnings": list(extracted.get("warnings") or []),
        "chunks": [],
        "proposals": [],
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    scraping_store_for(platform).save(record)
    return {
        "file_id": file_id,
        "filename": filename,
        "kind": kind,
        "title": title,
        "text": text,
        "char_count": len(text),
        "warnings": record["warnings"],
    }


def chunk_file(platform, payload: dict) -> dict:
    store = scraping_store_for(platform)
    file_id = str(payload.get("file_id") or "").strip()
    if file_id:
        record = store.get(file_id)
        text = record["text"]
        filename = record["filename"]
    else:
        text = str(payload.get("text") or "")
        if not text.strip():
            raise ValidationFailed("Provide file_id or text.")
        file_id = uuid.uuid4().hex
        filename = str(payload.get("filename") or "inline.txt")
        record = {
            "file_id": file_id,
            "filename": filename,
            "kind": str(payload.get("kind") or "text"),
            "title": filename,
            "text": text,
            "char_count": len(text),
            "warnings": [],
            "chunks": [],
            "proposals": [],
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    chunks = build_chunks(file_id, text)
    record["chunks"] = chunks
    record["updated_at"] = utc_now()
    store.save(record)
    return {"file_id": file_id, "filename": filename, "chunk_count": len(chunks), "chunks": chunks}


def propose_chunks(platform, pipeline, payload: dict) -> dict:
    from knowledge_graph.construction.tools.relationships import propose_scraping_chunk

    store = scraping_store_for(platform)
    file_id = str(payload.get("file_id") or "").strip()
    if not file_id:
        raise ValidationFailed("file_id is required.")
    record = store.get(file_id)
    chunks = record.get("chunks") or []
    if not chunks:
        chunks = build_chunks(file_id, record["text"])
        record["chunks"] = chunks
    ontology_id = str(payload.get("ontology_id") or "").strip() or "news"
    graph_id = str(payload.get("graph_id") or "").strip()
    if not graph_id:
        # Prefer a graph bound to the ontology; fall back to news.
        matches = [g for g in platform.kg_registry.list_graphs() if g.ontology_id == ontology_id]
        graph_id = matches[0].id if matches else "news"
    platform.kg_registry.get_ontology(ontology_id)
    platform.kg_registry.get_graph(graph_id)
    use_model = bool(payload.get("use_model", True))
    ctx = pipeline.context()
    ontology = platform.kg_registry.load_ontology_object(ontology_id)
    warnings: list[str] = []
    selected = chunks[:MAX_CHUNKS]
    if len(chunks) > MAX_CHUNKS:
        warnings.append(f"Truncated propose to first {MAX_CHUNKS} of {len(chunks)} paragraphs.")
    relationships: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for chunk in selected:
        found, chunk_warnings = propose_scraping_chunk(
            ctx,
            title=record.get("title") or record["filename"],
            chunk=chunk,
            file_id=file_id,
            ontology=ontology,
            ontology_id=ontology_id,
            graph_id=graph_id,
            use_model=use_model,
        )
        warnings.extend(chunk_warnings)
        for item in found:
            key = (
                item["type"],
                normalize_name(item["start_name"]),
                normalize_name(item["end_name"]),
                item.get("chunk_id") or "",
            )
            if key in seen:
                continue
            seen.add(key)
            relationships.append(item)
    record["proposals"] = relationships
    record["ontology_id"] = ontology_id
    record["graph_id"] = graph_id
    record["updated_at"] = utc_now()
    store.save(record)
    return {
        "file_id": file_id,
        "filename": record["filename"],
        "graph_id": graph_id,
        "ontology_id": ontology_id,
        "chunk_count": len(selected),
        "relationship_count": len(relationships),
        "relationships": relationships,
        "warnings": warnings,
    }


def queue_scraping_review(platform, pipeline, payload: dict) -> dict:
    store = scraping_store_for(platform)
    file_id = str(payload.get("file_id") or "").strip()
    if not file_id:
        raise ValidationFailed("file_id is required.")
    record = store.get(file_id)
    relationships = payload.get("relationships")
    if not relationships:
        relationships = record.get("proposals") or []
    if not relationships:
        raise ValidationFailed("No relationships to queue. Run propose first.")
    ontology_id = str(payload.get("ontology_id") or record.get("ontology_id") or "news")
    graph_id = str(payload.get("graph_id") or record.get("graph_id") or "news")
    for item in relationships:
        props = dict(item.get("properties") or {})
        props.setdefault("file_id", file_id)
        if item.get("chunk_id"):
            props.setdefault("chunk_id", item["chunk_id"])
        if item.get("novelty"):
            props.setdefault("novelty", item["novelty"])
        item["properties"] = props
        # Ensure ProposedRelationship validation fields
        item.setdefault("method", item.get("method") or "ollama")
        item.setdefault("status", "new")
        item.setdefault("review_state", "pending")
        item.setdefault("needs_ontology_approval", item.get("novelty") == "new" or item.get("needs_ontology_approval", False))
        if not item.get("id"):
            item["id"] = f"rel_{uuid.uuid4().hex[:16]}"
    queued = pipeline.run_tool(
        "queue_review",
        {
            "source_kind": record.get("kind") or "text",
            "filename": record.get("filename") or "",
            "title": record.get("title") or record.get("filename") or "Scraped document",
            "text": record.get("text") or "",
            "graph_id": graph_id,
            "ontology_id": ontology_id,
            "metadata": {
                "file_id": file_id,
                "scraping": True,
                "chunk_count": len(record.get("chunks") or []),
            },
            "relationships": relationships,
            "warnings": list(payload.get("warnings") or record.get("warnings") or []),
        },
    )
    review = queued["review"]
    record["review_id"] = review["id"]
    record["updated_at"] = utc_now()
    store.save(record)
    return queued


def commit_scraping(platform, pipeline, payload: dict) -> dict:
    review_id = str(payload.get("review_id") or "").strip()
    if not review_id:
        raise ValidationFailed("review_id is required.")
    record = pipeline.store.get(review_id)
    ontology_spec = payload.get("ontology") or {"mode": "existing", "ontology_id": record.ontology_id}
    graph_spec = payload.get("graph") or {}
    mode = str(ontology_spec.get("mode") or "existing").strip().lower()

    if mode == "new":
        create = OntologyCreate(
            id=ontology_spec.get("id"),
            name=str(ontology_spec.get("name") or ontology_spec.get("id") or "scraped-ontology"),
            description=str(ontology_spec.get("description") or "Created from scraping commit"),
            clone_from=str(ontology_spec.get("clone_from") or "news"),
            entity_types=list(ontology_spec.get("entity_types") or []),
            relationship_allowlist=dict(ontology_spec.get("relationship_allowlist") or {}),
        )
        ont_record, _definition = platform.kg_registry.create_ontology(create)
        ontology_id = ont_record.id
        graph_id = str(graph_spec.get("id") or ontology_id)
        try:
            platform.kg_registry.get_graph(graph_id)
        except NotFoundError:
            platform.kg_registry.create_graph(
                GraphCreate(
                    id=graph_id,
                    name=str(graph_spec.get("name") or f"{ont_record.name} graph"),
                    ontology_id=ontology_id,
                    kind="custom",
                    description=str(graph_spec.get("description") or ""),
                )
            )
    else:
        ontology_id = str(ontology_spec.get("ontology_id") or record.ontology_id or "news")
        platform.kg_registry.get_ontology(ontology_id)
        graph_id = str(graph_spec.get("id") or payload.get("graph_id") or record.graph_id or "")
        if not graph_id:
            matches = [g for g in platform.kg_registry.list_graphs() if g.ontology_id == ontology_id]
            if matches:
                graph_id = matches[0].id
            else:
                graph_id = ontology_id
                platform.kg_registry.create_graph(
                    GraphCreate(
                        id=graph_id,
                        name=f"{ontology_id} graph",
                        ontology_id=ontology_id,
                        kind="custom",
                    )
                )
        else:
            try:
                platform.kg_registry.get_graph(graph_id)
            except NotFoundError:
                platform.kg_registry.create_graph(
                    GraphCreate(
                        id=graph_id,
                        name=str(graph_spec.get("name") or graph_id),
                        ontology_id=ontology_id,
                        kind="custom",
                    )
                )

    # Point the review at the resolved graph/ontology before ingest.
    record.graph_id = graph_id
    record.ontology_id = ontology_id
    pipeline.store.save(record)
    platform.refresh_union_ontology()

    # Ensure approved edges exist; if caller passed decision earlier they should already be approved.
    approved = [rel for rel in record.relationships if rel.review_state == "approved"]
    if not approved:
        raise ValidationFailed("Approve at least one relationship before scraping commit.")

    committed = pipeline.run_tool("ingest_graph", {"review_id": review_id})
    return {
        **committed,
        "ontology_id": ontology_id,
        "graph_id": graph_id,
        "mode": mode,
    }
