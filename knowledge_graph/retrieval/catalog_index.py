"""Compact product list for catalog questions. The graph snapshot stays on disk."""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_graph.logging_config import get_logger

LOGGER = get_logger("knowledge_graph.catalog_index")

_DROP = {"source_columns", "composition_parts", "source_permissions"}


class CatalogRow:
    """Fields the catalog search reads. Not a full ingest envelope."""

    __slots__ = (
        "source_id",
        "source_type",
        "record_id",
        "title",
        "classification",
        "source_path",
        "content_hash",
        "synthetic",
        "lifecycle",
        "metadata",
    )

    def __init__(self, row: dict):
        self.source_id = "own_products"
        self.source_type = "own_products"
        self.record_id = row["record_id"]
        self.title = row.get("title") or ""
        self.classification = row.get("classification") or "internal"
        self.source_path = row.get("source_path") or ""
        self.content_hash = row.get("content_hash") or ""
        self.synthetic = bool(row.get("synthetic"))
        self.lifecycle = "active"
        self.metadata = {"product": row.get("product") or {}}


def index_path(generated: Path) -> Path:
    return generated / "catalog_index.jsonl"


def ensure_catalog_index(generated: Path) -> Path:
    """Write one line per product when the index is missing or older than the last ingest."""

    path = index_path(generated)
    state = generated / "ingestion_state.json"
    if path.is_file() and path.stat().st_size > 0 and (not state.is_file() or path.stat().st_mtime >= state.stat().st_mtime):
        return path
    folder = generated / "envelopes" / "own_products"
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if folder.is_dir():
            for envelope_path in folder.glob("*.json"):
                raw = json.loads(envelope_path.read_text(encoding="utf-8"))
                if raw.get("lifecycle") == "deleted" or raw.get("source_type") not in {None, "own_products"}:
                    continue
                product = dict((raw.get("metadata") or {}).get("product") or {})
                for key in _DROP:
                    product.pop(key, None)
                handle.write(
                    json.dumps(
                        {
                            "record_id": raw.get("record_id"),
                            "title": raw.get("title") or "",
                            "classification": raw.get("classification") or "internal",
                            "source_path": raw.get("source_path") or "",
                            "content_hash": raw.get("content_hash") or "",
                            "synthetic": bool(raw.get("synthetic")),
                            "product": product,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
    temporary.replace(path)
    LOGGER.info("wrote catalog index records=%s", count)
    return path


def load_catalog_rows(generated: Path) -> list[CatalogRow]:
    path = ensure_catalog_index(generated)
    if not path.is_file():
        return []
    rows: list[CatalogRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(CatalogRow(json.loads(line)))
    LOGGER.info("loaded catalog index records=%s", len(rows))
    return rows
