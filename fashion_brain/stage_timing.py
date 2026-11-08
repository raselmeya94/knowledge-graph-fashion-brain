"""Per-stage ask-pipeline timing (observation only — no behavior change).

Enable with ``start_timing()`` for one turn; stages are recorded via ``timed()`` /
``record_stage()``. Safe to call when timing is off (no-ops).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageEvent:
    name: str
    ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingBundle:
    started: float = field(default_factory=time.perf_counter)
    events: list[StageEvent] = field(default_factory=list)
    cypher: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ms: float, **meta: Any) -> None:
        self.events.append(StageEvent(name=name, ms=float(ms), meta=dict(meta)))

    def add_cypher(self, query: str, *, params: dict | None = None, rows: int | None = None, ms: float | None = None) -> None:
        self.cypher.append(
            {
                "query": query,
                "params": dict(params or {}),
                "rows": rows,
                "ms": ms,
            }
        )

    def by_stage(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for event in self.events:
            totals[event.name] = totals.get(event.name, 0.0) + event.ms
        return totals

    def total_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total_ms(), 1),
            "by_stage_ms": {key: round(value, 1) for key, value in self.by_stage().items()},
            "events": [
                {"name": event.name, "ms": round(event.ms, 1), **event.meta} for event in self.events
            ],
            "cypher": list(self.cypher),
        }


_CURRENT: ContextVar[TimingBundle | None] = ContextVar("fashion_brain_stage_timing", default=None)


def start_timing() -> TimingBundle:
    bundle = TimingBundle()
    _CURRENT.set(bundle)
    return bundle


def current_timing() -> TimingBundle | None:
    return _CURRENT.get()


def end_timing() -> TimingBundle | None:
    bundle = _CURRENT.get()
    _CURRENT.set(None)
    return bundle


def record_stage(name: str, ms: float, **meta: Any) -> None:
    bundle = _CURRENT.get()
    if bundle is not None:
        bundle.add(name, ms, **meta)


def record_cypher(query: str, *, params: dict | None = None, rows: int | None = None, ms: float | None = None) -> None:
    bundle = _CURRENT.get()
    if bundle is not None:
        bundle.add_cypher(query, params=params, rows=rows, ms=ms)


@contextmanager
def timed(name: str, **meta: Any):
    """Context manager that records wall time for a named stage."""

    bundle = _CURRENT.get()
    started = time.perf_counter()
    try:
        yield
    finally:
        if bundle is not None:
            bundle.add(name, (time.perf_counter() - started) * 1000.0, **meta)


def estimate_tokens(*parts: str) -> int:
    """Rough token estimate (chars/4) for latency reports — not billed usage."""

    total = sum(len(str(part or "")) for part in parts)
    return max(1, total // 4) if total else 0
