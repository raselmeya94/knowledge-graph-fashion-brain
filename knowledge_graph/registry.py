"""Source registry and manifest loading. Raw manifests are read, not rewritten."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_graph.errors import ValidationFailed
from knowledge_graph.models import SourceManifest, SourceRegistry


def load_registry(path: Path) -> SourceRegistry:
    data = json.loads(path.read_text(encoding="utf-8"))
    registry = SourceRegistry.model_validate(data)
    validate_registry(registry)
    return registry


def validate_registry(registry: SourceRegistry) -> None:
    if registry.version < 1:
        raise ValidationFailed("Registry version must be >= 1.")
    seen: set[str] = set()
    for source in registry.sources:
        if source.source_id in seen:
            raise ValidationFailed(f"Duplicate source_id {source.source_id}.")
        seen.add(source.source_id)
        if not source.source_type or not source.connector:
            raise ValidationFailed(f"Source {source.source_id} needs a type and connector.")
        if not source.supported_formats:
            raise ValidationFailed(f"Source {source.source_id} must declare supported formats.")
        if source.ingestion_mode not in {"batch", "incremental"}:
            raise ValidationFailed(f"Unsupported ingestion mode for {source.source_id}.")
        if source.update_strategy not in {"content_hash"}:
            raise ValidationFailed(f"Unsupported update strategy for {source.source_id}.")
        if source.permissions_model not in {"public", "classification"}:
            raise ValidationFailed(f"Unsupported permissions model for {source.source_id}.")


def load_manifest(path: Path) -> SourceManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = SourceManifest.model_validate(data)
    ids = [record.record_id for record in manifest.records]
    if len(ids) != len(set(ids)):
        raise ValidationFailed(f"Duplicate record ids in {path}.")
    return manifest


def write_registry_schema_snapshot(registry: SourceRegistry) -> dict:
    return registry.model_json_schema()
