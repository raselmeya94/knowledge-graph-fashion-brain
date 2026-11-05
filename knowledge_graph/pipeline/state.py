"""Ingestion state lives under generated data, never inside Raw_Data."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_graph.models import RecordState


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, RecordState] = {}
        self.load()

    @staticmethod
    def key(source_id: str, record_id: str) -> str:
        return f"{source_id}:{record_id}"

    def load(self) -> None:
        if not self.path.exists():
            self.records = {}
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = {key: RecordState.model_validate(value) for key, value in data.get("records", {}).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": {key: value.model_dump() for key, value in sorted(self.records.items())}}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self, source_id: str, record_id: str) -> RecordState | None:
        return self.records.get(self.key(source_id, record_id))

    def put(self, record: RecordState) -> None:
        self.records[self.key(record.source_id, record.record_id)] = record

    def for_source(self, source_id: str) -> list[RecordState]:
        return [record for record in self.records.values() if record.source_id == source_id]
