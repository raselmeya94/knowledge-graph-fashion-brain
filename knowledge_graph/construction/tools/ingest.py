"""Write approved relationships into the knowledge graph."""

from __future__ import annotations

import knowledge_graph.construction  # noqa: F401  (register the documents source)
from knowledge_graph.construction.identity import build_documents_graph, build_news_graph
from knowledge_graph.construction.models import ProposedRelationship, ReviewRecord
from knowledge_graph.errors import ValidationFailed
from knowledge_graph.graph.build import build_envelopes, get_builder
from knowledge_graph.graph.snapshots import save_master_snapshot, save_source_snapshots
from knowledge_graph.hashing import sha256_text
from knowledge_graph.models import ExtractedEntity, ExtractedRel, RecordEnvelope
from knowledge_graph.ontology import allow_relationship, normalize_name, register_pair
from knowledge_graph.pipeline.service import utc_now


class IngestGraphTool:
    name = "ingest_graph"
    description = "Save approved relationships from one review into the selected knowledge graph."

    def run(self, payload: dict, ctx) -> dict:
        review_id = str(payload.get("review_id") or "")
        record = ctx.store.get(review_id)
        approved = [rel for rel in record.relationships if rel.review_state == "approved"]
        if not approved:
            raise ValidationFailed("Approve at least one relationship before updating the graph.")
        record_id = record.committed_record_id or sha256_text(record.text or record.title)[:32]
        graph_id = record.graph_id or "news"
        ontology = ctx.ontology_for(graph_id)
        path = ctx.ontology_path_for(graph_id)
        for rel in approved:
            allow_relationship(ontology, path, rel.type, rel.start_type, rel.end_type)
            register_pair(ctx.graph.ontology, rel.type, rel.start_type, rel.end_type)
        ctx.platform.refresh_union_ontology()
        envelope = _envelope(record, approved, record_id, graph_id)
        builder = _builder_for(graph_id)
        errors = builder([envelope], ctx.graph, ctx.graph.ontology)
        committed_ids = {rel.id for rel in approved}
        if errors:
            record.warnings.extend(errors)
        else:
            for rel in record.relationships:
                if rel.id in committed_ids:
                    rel.review_state = "committed"
        record.committed_record_id = record_id
        if all(rel.review_state in {"committed", "rejected"} for rel in record.relationships):
            record.status = "committed"
        ctx.store.save(record)
        if not errors:
            _write_snapshot(ctx, graph_id)
        written = [rel.model_dump() for rel in record.relationships if rel.review_state == "committed"]
        return {
            "review_id": record.id,
            "record_id": record_id,
            "graph_id": graph_id,
            "ontology_id": record.ontology_id,
            "committed": written,
            "warnings": errors,
        }


def _builder_for(graph_id: str):
    if graph_id == "news":
        return build_news_graph
    if graph_id == "documents":
        return build_documents_graph
    try:
        return get_builder(graph_id)
    except ValueError:
        return lambda envelopes, repo, ontology, expected=graph_id: build_envelopes(
            envelopes, repo, ontology, expected
        )


def _write_snapshot(ctx, source_id: str) -> None:
    save_source_snapshots(ctx.graph, ctx.settings.generated, [source_id])
    # Refresh shared interconnect when similar entities can link across sources.
    try:
        ctx.platform.build_shared()
    except Exception:
        pass
    save_master_snapshot(ctx.graph, ctx.settings.generated)


def _envelope(
    record: ReviewRecord,
    approved: list[ProposedRelationship],
    record_id: str,
    graph_id: str,
) -> RecordEnvelope:
    entities: list[ExtractedEntity] = []
    rels: list[ExtractedRel] = []
    local: dict[tuple[str, str], str] = {}
    title_key = normalize_name(record.title)

    def endpoint(entity_type: str, name: str) -> str:
        if entity_type == "Document" and normalize_name(name) == title_key:
            return "record"
        key = (entity_type, normalize_name(name))
        found = local.get(key)
        if found:
            return found
        local_id = f"e{len(local) + 1}"
        local[key] = local_id
        entities.append(
            ExtractedEntity(
                local_id=local_id,
                entity_type=entity_type,
                name=name,
                evidence_text=name,
            )
        )
        return local_id

    for rel in approved:
        start = "record" if rel.start_type == "Document" else endpoint(rel.start_type, rel.start_name)
        end = endpoint(rel.end_type, rel.end_name)
        if start == end:
            continue
        rels.append(
            ExtractedRel(
                type=rel.type,
                start_local_id=start,
                end_local_id=end,
                evidence_text=rel.evidence or None,
                field_ref=rel.id,
                verification_status="reviewed" if rel.method == "deterministic" else "model",
                properties=dict(rel.properties or {}),
            )
        )
    text = record.text or record.title
    return RecordEnvelope(
        source_id=graph_id,
        source_type=graph_id,
        record_id=record_id,
        title=record.title,
        content=text,
        content_hash=sha256_text(text),
        synthetic=False,
        classification="internal",
        source_path=record.filename or "inline",
        ingested_at=utc_now(),
        document_type=str(record.metadata.get("document_type") or record.source_kind),
        metadata={
            "source_kind": record.source_kind,
            "review_id": record.id,
            "graph_id": graph_id,
            "ontology_id": record.ontology_id,
        },
        entities=entities,
        relationships=rels,
    )
