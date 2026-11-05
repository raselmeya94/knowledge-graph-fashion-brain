"""Named ontologies and graphs managed by the knowledge-graph API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OntologyRecord(BaseModel):
    id: str
    name: str
    description: str = ""
    path: str
    builtin: bool = False
    status: Literal["active", "deleted"] = "active"


class GraphRecord(BaseModel):
    id: str
    name: str
    ontology_id: str
    kind: Literal["product", "documents", "custom", "system"] = "custom"
    description: str = ""
    builtin: bool = False
    status: Literal["active", "deleted"] = "active"
    source_ids: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)


class UnifiedRegistryEntry(BaseModel):
    """Source + graph + ontology view for the Graph Router Agent."""

    graph_id: str
    source_ids: list[str] = Field(default_factory=list)
    graph_name: str = ""
    description: str = ""
    kind: str = "custom"
    ontology_id: str | None = None
    entities: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    status: str = "active"
    node_count: int = 0
    edge_count: int = 0


class KgRegistry(BaseModel):
    version: int = 1
    ontologies: list[OntologyRecord] = Field(default_factory=list)
    graphs: list[GraphRecord] = Field(default_factory=list)


class OntologyCreate(BaseModel):
    id: str | None = None
    name: str
    description: str = ""
    entity_types: list[str] = Field(default_factory=list)
    relationship_allowlist: dict[str, list[list[str]]] = Field(default_factory=dict)
    clone_from: str | None = None
    version: str = "1.0.0"
    canonical_entities: list[dict[str, Any]] = Field(default_factory=list)


class OntologyReplace(BaseModel):
    name: str | None = None
    description: str | None = None
    entity_types: list[str] | None = None
    relationship_allowlist: dict[str, list[list[str]]] | None = None
    version: str | None = None
    canonical_entities: list[dict[str, Any]] | None = None


class OntologyPairBody(BaseModel):
    type: str
    start_type: str
    end_type: str


class GraphCreate(BaseModel):
    id: str | None = None
    name: str
    ontology_id: str
    kind: Literal["product", "documents", "custom", "system"] = "custom"
    description: str = ""
    source_ids: list[str] = Field(default_factory=list)
