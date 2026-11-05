"""Review queue: hold proposed relationships until a person approves them."""

from __future__ import annotations

import uuid

from knowledge_graph.construction.models import ProposedRelationship, ReviewRecord
from knowledge_graph.errors import NotFoundError, ValidationFailed
from knowledge_graph.kg_registry import default_graph_for_kind
from knowledge_graph.ontology import normalize_name
from knowledge_graph.pipeline.service import utc_now


class QueueReviewTool:
    name = "queue_review"
    description = "Store proposed relationships so they can be reviewed or edited before graph ingestion."

    def run(self, payload: dict, ctx) -> dict:
        relationships = [ProposedRelationship.model_validate(item) for item in payload.get("relationships") or []]
        source_kind = str(payload.get("source_kind") or "text")
        graph_id = str(payload.get("graph_id") or "").strip() or default_graph_for_kind(source_kind)
        graph = ctx.platform.kg_registry.get_graph(graph_id)
        ontology_id = str(payload.get("ontology_id") or "").strip() or graph.ontology_id
        record = ReviewRecord(
            id=uuid.uuid4().hex,
            created_at=utc_now(),
            source_kind=source_kind,
            filename=str(payload.get("filename") or ""),
            title=str(payload.get("title") or "Untitled document"),
            text=str(payload.get("text") or ""),
            graph_id=graph_id,
            ontology_id=ontology_id,
            metadata=dict(payload.get("metadata") or {}),
            relationships=relationships,
            warnings=list(payload.get("warnings") or []),
        )
        ctx.store.save(record)
        return {"review": record.model_dump()}


class UpdateReviewTool:
    name = "update_review"
    description = "Approve, reject, or manually edit relationships in one review."

    def run(self, payload: dict, ctx) -> dict:
        review_id = str(payload.get("review_id") or payload.get("id") or "")
        record = ctx.store.get(review_id)
        selected = set(payload.get("relationship_ids") or [])
        edits = {item.get("id"): item for item in payload.get("edits") or [] if isinstance(item, dict)}
        decision = payload.get("decision")
        if decision not in {None, "approved", "rejected", "pending"}:
            raise ValidationFailed("decision must be approved, rejected, or pending.")
        remove_ids = set(payload.get("remove_ids") or [])
        updated: list[ProposedRelationship] = []
        for rel in record.relationships:
            if rel.id in remove_ids and rel.review_state != "committed":
                continue
            if rel.review_state == "committed":
                updated.append(rel)
                continue
            current = rel.model_copy(deep=True)
            change = edits.get(current.id)
            if change:
                for field in ("type", "start_name", "start_type", "end_name", "end_type", "evidence"):
                    if change.get(field):
                        setattr(current, field, str(change[field]).strip())
                if current.type:
                    current.type = current.type.upper()
                _require_allowed(ctx, current, record)
            if decision and (not selected or current.id in selected):
                current.review_state = decision
            updated.append(current)
        record.relationships = updated
        record.status = "pending"
        ctx.store.save(record)
        return {"review": record.model_dump()}


class RemoveReviewRelationshipTool:
    name = "remove_review_relationship"
    description = "Reject or drop one proposed relationship from a review before commit."

    def run(self, payload: dict, ctx) -> dict:
        review_id = str(payload.get("review_id") or "")
        rel_id = str(payload.get("relationship_id") or "")
        record = ctx.store.get(review_id)
        found = None
        kept: list[ProposedRelationship] = []
        for rel in record.relationships:
            if rel.id != rel_id:
                kept.append(rel)
                continue
            found = rel
            if rel.review_state == "committed":
                raise ValidationFailed("Committed relationships must be deleted from the graph, not the review.")
            if payload.get("hard"):
                continue
            current = rel.model_copy(deep=True)
            current.review_state = "rejected"
            kept.append(current)
        if found is None:
            raise NotFoundError(f"Relationship '{rel_id}' was not found in review '{review_id}'.")
        record.relationships = kept
        ctx.store.save(record)
        return {"review": record.model_dump(), "removed": found.model_dump()}


def _require_allowed(ctx, rel: ProposedRelationship, record: ReviewRecord) -> None:
    if not normalize_name(rel.start_name) or not normalize_name(rel.end_name):
        raise ValidationFailed("Relationship names cannot be empty.")
    ontology = ctx.ontology_for(record.graph_id)
    known = set(ontology.entity_types)
    if record.ontology_id == "news" or "Publisher" in known:
        if rel.start_type not in known or rel.end_type not in known:
            raise ValidationFailed(
                f"{rel.start_type} -> {rel.end_type} is not a news entity pair."
            )
    allowed = ontology.relationship_allowlist.get(rel.type, [])
    rel.needs_ontology_approval = [rel.start_type, rel.end_type] not in allowed
