"""Register or update sources in data/raw/source_registry.json."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from knowledge_graph.errors import ValidationFailed
from knowledge_graph.models import SourceConfig, SourceRegistry
from knowledge_graph.registry import load_registry, validate_registry


class SourceRegisterBody(BaseModel):
    source_id: str
    source_type: str
    connector: str = "file"
    supported_formats: list[str] = Field(default_factory=lambda: ["json"])
    ingestion_mode: str = "batch"
    update_strategy: str = "content_hash"
    permissions_model: str = "classification"
    enabled: bool = False
    description: str = ""
    future_connectors: list[str] = Field(default_factory=list)


def list_sources(platform) -> list[dict[str, Any]]:
    return [source.model_dump() for source in platform.registry.sources]


def get_source(platform, source_id: str) -> dict[str, Any]:
    for source in platform.registry.sources:
        if source.source_id == source_id:
            return source.model_dump()
    from knowledge_graph.errors import NotFoundError

    raise NotFoundError(f"Source '{source_id}' was not found.")


def register_source(platform, body: SourceRegisterBody) -> dict[str, Any]:
    """Append or replace a source row and reload the in-memory registry."""

    path = platform.settings.raw_dir / "source_registry.json"
    registry = load_registry(path) if path.exists() else SourceRegistry(version=1, sources=[])
    if any(item.source_id == body.source_id for item in registry.sources):
        raise ValidationFailed(f"Source '{body.source_id}' already exists.")
    record = SourceConfig(**body.model_dump())
    registry.sources.append(record)
    validate_registry(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry.model_dump(), indent=2) + "\n", encoding="utf-8")
    platform.registry = load_registry(path)
    return record.model_dump()
