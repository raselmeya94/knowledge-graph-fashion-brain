"""TTL cache for runtime schema cards and brand lists (dynamic, per graph)."""

from __future__ import annotations

import time
from typing import Any

_CACHE: dict[str, tuple[float, Any]] = {}


def _ttl(platform) -> float:
    return float(getattr(platform.settings, "schema_ttl_seconds", 600) or 600)


def get_cached(key: str, platform) -> Any | None:
    row = _CACHE.get(key)
    if not row:
        return None
    ts, value = row
    if time.monotonic() - ts > _ttl(platform):
        _CACHE.pop(key, None)
        return None
    return value


def set_cached(key: str, value: Any) -> None:
    _CACHE[key] = (time.monotonic(), value)


def clear_schema_cache(graph_id: str | None = None) -> None:
    if graph_id is None:
        _CACHE.clear()
        return
    prefix = f"{graph_id}:"
    for key in list(_CACHE):
        if key.startswith(prefix) or key == graph_id:
            _CACHE.pop(key, None)


def get_schema_card(platform, graph_id: str, *, brands: list[dict] | None = None) -> str:
    key = f"{graph_id}:schema_card"
    hit = get_cached(key, platform)
    if hit is not None and brands is None:
        return str(hit)
    from fashion_brain.agents.text2cypher import Text2CypherAgent

    card = Text2CypherAgent().schema_card(platform, graph_id, brands=brands)
    set_cached(key, card)
    return card


def get_brands(platform, graph_id: str, principal=None) -> list[dict]:
    key = f"{graph_id}:brands"
    hit = get_cached(key, platform)
    if hit is not None:
        return list(hit)
    from fashion_brain.execution.kg_client import kg_request, uses_kg_api
    from knowledge_graph.linking.entity_linker import list_live_brands

    if uses_kg_api(platform.settings):
        payload = kg_request(
            platform.settings,
            "GET",
            "/brain/brands",
            clearance=getattr(principal, "clearance", None) or "internal",
            params={"graph_id": graph_id},
        )
        brands = list((payload or {}).get("brands") or [])
    else:
        brands = list_live_brands(platform, graph_id)
    set_cached(key, brands)
    return brands
