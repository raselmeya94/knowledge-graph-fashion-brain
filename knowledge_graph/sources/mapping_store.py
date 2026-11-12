"""API-backed source column mappings (no manual file copy).

Mappings live under ``data/schemas/``. ``ecom_products`` keeps the legacy
filename so sync / ``graph_sketch`` keep working.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT
from knowledge_graph.errors import NotFoundError, ValidationFailed

_SAFE_ID = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")

# Legacy filenames used by sync adapters.
_LEGACY_FILES = {
    "ecom_products": "ecom_master_products_mapping.json",
    "plmex": "plmex_mapping.json",
    "plmex_products": "plmex_mapping.json",
}


def mapping_path(source_id: str) -> Path:
    sid = _require_source_id(source_id)
    name = _LEGACY_FILES.get(sid, f"{sid}_mapping.json")
    return DATA_ROOT / "schemas" / name


def mapping_exists(source_id: str) -> bool:
    return mapping_path(source_id).is_file()


def get_mapping(source_id: str) -> dict[str, Any]:
    path = mapping_path(source_id)
    if not path.is_file():
        raise NotFoundError(f"Mapping for source '{source_id}' was not found.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationFailed(f"Mapping file for '{source_id}' is invalid JSON.") from exc
    if not isinstance(data, dict):
        raise ValidationFailed(f"Mapping for '{source_id}' must be a JSON object.")
    return {
        "source_id": source_id,
        "path": str(path),
        "mapping": data,
    }


def put_mapping(source_id: str, mapping: dict[str, Any], *, replace: bool = True) -> dict[str, Any]:
    """Create or replace a mapping document for ``source_id``."""

    if not isinstance(mapping, dict) or not mapping:
        raise ValidationFailed("mapping body must be a non-empty JSON object.")
    path = mapping_path(source_id)
    existed = path.is_file()
    if existed and not replace:
        raise ValidationFailed(f"Mapping for source '{source_id}' already exists. Use PUT to replace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp source_id / graph_id when missing so docs stay self-describing.
    payload = dict(mapping)
    payload.setdefault("source_id", source_id)
    if source_id == "ecom_products":
        payload.setdefault("graph_id", "ecom_products")
        payload.setdefault("source_table", "master_products")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "source_id": source_id,
        "path": str(path),
        "status": "replaced" if existed else "created",
        "mapping": payload,
    }


def _require_source_id(source_id: str) -> str:
    text = (source_id or "").strip().casefold()
    if not text or not _SAFE_ID.match(text):
        raise ValidationFailed(
            f"Invalid source_id {source_id!r}. Use letters, numbers, hyphens, or underscores."
        )
    return text
