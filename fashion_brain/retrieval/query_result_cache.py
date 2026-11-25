"""Short-TTL in-memory cache for Neo4j Q&A findings.

Keyed by normalized intent + ontology domain ids + graph ids + metric family.
Fail-open: any error → miss. Never invents data.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger("fashion_brain.query_result_cache")

_LOCK = threading.Lock()
_STORE: dict[str, _CacheEntry] = {}
_DEFAULT_TTL_SEC = 90.0
_MAX_ENTRIES = 256


@dataclass
class _CacheEntry:
    text: str
    evidence: list[Any] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    metric_family: str = ""
    ontology_domains: list[str] = field(default_factory=list)
    graph_ids: list[str] = field(default_factory=list)
    expires_at: float = 0.0


def _normalize_intent(question: str) -> str:
    text = (question or "").casefold().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cache_key(
    question: str,
    *,
    ontology_domains: list[str] | None = None,
    graph_ids: list[str] | None = None,
    metric_family: str | None = None,
    intent: str | None = None,
) -> str:
    """Stable hash for cache lookup (no secrets)."""

    parts = [
        _normalize_intent(question),
        str(intent or "").casefold(),
        str(metric_family or "").casefold(),
        "|".join(sorted(str(x) for x in (ontology_domains or []) if x)),
        "|".join(sorted(str(x) for x in (graph_ids or []) if x)),
    ]
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ttl_seconds(platform=None) -> float:
    try:
        settings = getattr(platform, "settings", None) if platform is not None else None
        value = getattr(settings, "query_cache_ttl_seconds", None)
        if value is not None and float(value) > 0:
            return float(value)
    except Exception:
        pass
    return _DEFAULT_TTL_SEC


def _evict_expired(now: float) -> None:
    dead = [k for k, v in _STORE.items() if v.expires_at <= now]
    for k in dead:
        _STORE.pop(k, None)
    if len(_STORE) > _MAX_ENTRIES:
        # Drop oldest expiry first
        ordered = sorted(_STORE.items(), key=lambda item: item[1].expires_at)
        for k, _ in ordered[: max(0, len(_STORE) - _MAX_ENTRIES)]:
            _STORE.pop(k, None)


def get_cached_result(
    question: str,
    *,
    platform=None,
    ontology_domains: list[str] | None = None,
    graph_ids: list[str] | None = None,
    metric_family: str | None = None,
    intent: str | None = None,
) -> dict[str, Any] | None:
    """Return cached payload or None on miss/expiry."""

    try:
        key = cache_key(
            question,
            ontology_domains=ontology_domains,
            graph_ids=graph_ids,
            metric_family=metric_family,
            intent=intent,
        )
        now = time.monotonic()
        with _LOCK:
            _evict_expired(now)
            entry = _STORE.get(key)
            if entry is None or entry.expires_at <= now:
                return None
            # Hard guard: metric family mismatch never serves
            if metric_family and entry.metric_family and entry.metric_family != metric_family:
                return None
            return {
                "text": entry.text,
                "evidence": list(entry.evidence),
                "caveats": list(entry.caveats),
                "metric_family": entry.metric_family,
                "ontology_domains": list(entry.ontology_domains),
                "graph_ids": list(entry.graph_ids),
                "cache_key": key,
            }
    except Exception as exc:
        LOGGER.debug("query cache get failed: %s", exc)
        return None


def put_cached_result(
    question: str,
    *,
    text: str,
    platform=None,
    evidence: list[Any] | None = None,
    caveats: list[str] | None = None,
    ontology_domains: list[str] | None = None,
    graph_ids: list[str] | None = None,
    metric_family: str | None = None,
    intent: str | None = None,
) -> str | None:
    """Store findings text; return cache key or None on failure."""

    if not (text or "").strip():
        return None
    try:
        key = cache_key(
            question,
            ontology_domains=ontology_domains,
            graph_ids=graph_ids,
            metric_family=metric_family,
            intent=intent,
        )
        ttl = _ttl_seconds(platform)
        now = time.monotonic()
        entry = _CacheEntry(
            text=str(text),
            evidence=list(evidence or []),
            caveats=list(caveats or []),
            metric_family=str(metric_family or ""),
            ontology_domains=[str(x) for x in (ontology_domains or []) if x],
            graph_ids=[str(x) for x in (graph_ids or []) if x],
            expires_at=now + ttl,
        )
        with _LOCK:
            _evict_expired(now)
            _STORE[key] = entry
        return key
    except Exception as exc:
        LOGGER.debug("query cache put failed: %s", exc)
        return None


def clear_query_result_cache() -> None:
    with _LOCK:
        _STORE.clear()


def cache_stats() -> dict[str, Any]:
    now = time.monotonic()
    with _LOCK:
        alive = sum(1 for v in _STORE.values() if v.expires_at > now)
        return {"entries": len(_STORE), "alive": alive}
