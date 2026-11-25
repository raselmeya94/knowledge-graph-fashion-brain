"""Always-on terminal logging for Cypher that hits Neo4j."""

from __future__ import annotations

import json
from typing import Any

from knowledge_graph.logging_config import get_logger

LOGGER = get_logger("knowledge_graph.cypher")

_MAX_QUERY = 4000
_MAX_PARAM = 500


def _one_line(query: str) -> str:
    text = " ".join(str(query or "").split())
    if len(text) > _MAX_QUERY:
        return text[: _MAX_QUERY] + "…"
    return text


def _safe_params(params: dict[str, Any] | None) -> str:
    raw = dict(params or {})
    compact: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and len(value) > _MAX_PARAM:
            compact[key] = value[:_MAX_PARAM] + "…"
        elif isinstance(value, (list, tuple)) and len(value) > 40:
            compact[key] = list(value[:40]) + [f"…(+{len(value) - 40})"]
        else:
            compact[key] = value
    try:
        return json.dumps(compact, ensure_ascii=False, default=str)
    except Exception:
        return str(compact)


def log_cypher_hit(
    query: str,
    params: dict[str, Any] | None = None,
    *,
    source: str = "neo4j",
    rows: int | None = None,
    ms: float | None = None,
) -> None:
    """Print the actual Cypher sent to Neo4j in the serve terminal."""

    extras = []
    if rows is not None:
        extras.append(f"rows={rows}")
    if ms is not None:
        extras.append(f"ms={ms:.1f}")
    suffix = f" | {' '.join(extras)}" if extras else ""
    LOGGER.info(
        "NEO4J CYPHER [%s] %s | params=%s%s",
        source,
        _one_line(query),
        _safe_params(params),
        suffix,
    )
