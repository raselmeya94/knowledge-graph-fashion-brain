"""Common source-adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from knowledge_graph.models import RecordEnvelope, SourceConfig


@dataclass
class DiscoveredRecord:
    source_id: str
    record_id: str
    path: Path
    relative_path: str
    format: str
    hash_method: str
    manifest_hash: str
    source_timestamp: str | None
    synthetic_declared: bool
    raw_bytes: bytes
    parsed: dict


@dataclass
class DiscoveryResult:
    records: list[DiscoveredRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_ids: list[str] = field(default_factory=list)


class SourceAdapter(ABC):
    source_type: str

    def __init__(self, source: SourceConfig, source_root: Path):
        self.source = source
        self.source_root = source_root

    @abstractmethod
    def discover(self) -> DiscoveryResult:
        raise NotImplementedError

    @abstractmethod
    def validate(self, record: DiscoveredRecord) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, record: DiscoveredRecord, ingested_at: str) -> RecordEnvelope:
        raise NotImplementedError

    @abstractmethod
    def extract(self, envelope: RecordEnvelope) -> tuple[RecordEnvelope, list[str]]:
        raise NotImplementedError


def synthetic_label_errors(title: str, synthetic: bool, disclaimer: str) -> list[str]:
    errors: list[str] = []
    if synthetic and "SYNTHETIC" not in title.upper():
        errors.append("Synthetic records must include SYNTHETIC in the title.")
    if synthetic and "SYNTHETIC" not in disclaimer.upper():
        errors.append("Synthetic records must include a disclaimer that contains SYNTHETIC.")
    return errors


_ADAPTERS: dict[str, type[SourceAdapter]] = {}


def register_adapter(source_type: str, adapter_cls: type[SourceAdapter]) -> None:
    _ADAPTERS[source_type] = adapter_cls


def create_adapter(source: SourceConfig, source_root: Path) -> SourceAdapter:
    from knowledge_graph.errors import UnknownSourceType

    adapter_cls = _ADAPTERS.get(source.source_type)
    if adapter_cls is None:
        raise UnknownSourceType(
            f"No adapter is registered for source type '{source.source_type}'. "
            "Add an adapter module and call register_adapter. The core pipeline does not need to change."
        )
    return adapter_cls(source, source_root)


def registered_adapter_types() -> list[str]:
    return sorted(_ADAPTERS)
