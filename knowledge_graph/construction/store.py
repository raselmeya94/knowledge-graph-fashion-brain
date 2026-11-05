"""File-backed queue of relationship reviews."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from knowledge_graph.construction.models import ReviewRecord
from knowledge_graph.errors import NotFoundError


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def list(self) -> list[ReviewRecord]:
        with self._lock:
            return list(self._load().values())

    def get(self, review_id: str) -> ReviewRecord:
        with self._lock:
            found = self._load().get(review_id)
        if found is None:
            raise NotFoundError(f"Review {review_id} was not found.")
        return found

    def save(self, record: ReviewRecord) -> ReviewRecord:
        with self._lock:
            items = self._load()
            items[record.id] = record
            self._write(items)
        return record

    def _load(self) -> dict[str, ReviewRecord]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {item["id"]: ReviewRecord.model_validate(item) for item in data.get("items", [])}

    def _write(self, items: dict[str, ReviewRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": [item.model_dump() for item in items.values()]}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)
