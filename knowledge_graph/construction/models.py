"""Records passed between construction tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProposedRelationship(BaseModel):
    id: str
    type: str
    start_name: str
    start_type: str
    end_name: str
    end_type: str
    evidence: str = ""
    status: Literal["existing", "new"] = "new"
    method: Literal["deterministic", "ollama"] = "deterministic"
    review_state: Literal["pending", "approved", "rejected", "committed"] = "pending"
    needs_ontology_approval: bool = False
    properties: dict[str, Any] = Field(default_factory=dict)


class ReviewRecord(BaseModel):
    id: str
    created_at: str
    status: Literal["pending", "committed"] = "pending"
    source_kind: str
    filename: str = ""
    title: str
    text: str
    graph_id: str = "news"
    ontology_id: str = "product"
    metadata: dict[str, Any] = Field(default_factory=dict)
    relationships: list[ProposedRelationship] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    committed_record_id: str | None = None
