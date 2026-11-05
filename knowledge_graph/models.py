"""Typed records shared by ingestion, graphs, and the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ExtractedEntity(BaseModel):
    local_id: str
    entity_type: str
    name: str
    external_id: str | None = None
    evidence_text: str | None = None
    field_ref: str | None = None
    section_ref: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedRel(BaseModel):
    type: str
    start_local_id: str
    end_local_id: str | None = None
    end_record_id: str | None = None
    evidence_text: str | None = None
    field_ref: str | None = None
    section_ref: str | None = None
    verification_status: str = "observed"
    properties: dict[str, Any] = Field(default_factory=dict)


class RecordEnvelope(BaseModel):
    source_id: str
    source_type: str
    record_id: str
    title: str
    content: str
    content_hash: str
    synthetic: bool
    classification: str
    source_path: str
    source_url: str | None = None
    source_timestamp: str | None = None
    ingested_at: str
    version: str | None = None
    owner: str | None = None
    document_type: str | None = None
    publisher: str | None = None
    license: str | None = None
    topics: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRel] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    lifecycle: Literal["active", "deleted"] = "active"

    @field_validator("content_hash")
    @classmethod
    def _hash_len(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("content_hash must be a sha256 hex digest.")
        return value


class GraphNode(BaseModel):
    id: str
    labels: list[str]
    graph: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    type: str
    start_id: str
    end_id: str
    graph: str
    properties: dict[str, Any] = Field(default_factory=dict)


class SourceConfig(BaseModel):
    source_id: str
    source_type: str
    connector: str
    supported_formats: list[str]
    ingestion_mode: str
    update_strategy: str
    permissions_model: str
    enabled: bool
    description: str = ""
    future_connectors: list[str] = Field(default_factory=list)


class SourceRegistry(BaseModel):
    version: int
    sources: list[SourceConfig]

    def by_id(self, source_id: str) -> SourceConfig:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(source_id)

    def enabled_sources(self) -> list[SourceConfig]:
        return [source for source in self.sources if source.enabled]


class ManifestRecord(BaseModel):
    record_id: str
    path: str
    format: str
    hash_method: Literal["file", "canonical_product"]
    content_hash: str
    source_timestamp: str | None = None
    ingestion_status: str
    synthetic: bool


class SourceManifest(BaseModel):
    source_id: str
    records: list[ManifestRecord]


class RecordState(BaseModel):
    source_id: str
    record_id: str
    content_hash: str = ""
    status: str
    graph_status: str = "pending"
    ingested_at: str | None = None
    title: str = ""
    error: str | None = None
    duplicate_of: str | None = None


class IngestionReport(BaseModel):
    source_id: str
    ingested: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    duplicates: list[dict[str, str]] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    source_id: str
    record_id: str
    title: str | None = None
    excerpt: str
    field_ref: str | None = None
    section_ref: str | None = None
    source_path: str | None = None
    url: str | None = None
    content_hash: str | None = None
    verification_status: str
    graph: str | None = None
    node_id: str | None = None
    relationship: str | None = None
    synthetic: bool = True


class GroundedAnswer(BaseModel):
    question: str
    routes: list[str]
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    synthetic_data: bool = True


class ReviewDecision(BaseModel):
    start_name: str
    start_type: str
    canonical_id: str
    status: Literal["approved", "rejected"]
    note: str | None = None


class MatchDecision(BaseModel):
    status: Literal["verified", "review", "rejected", "unresolved"]
    method: str | None = None
    canonical_id: str | None = None
    similarity: float | None = None
    evidence: str
    entity_type: str
    source_name: str
