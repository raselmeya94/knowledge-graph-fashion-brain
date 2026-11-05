"""Step-by-step ask-pipeline tracing.

Use ``AskTrace`` for one question turn. Logs go to stdout via the standard logger
(JSON when ``log_json`` is on). Never log tokens, DSNs, or API keys.

Traces are stored in a ``ContextVar`` so concurrent HTTP/WebSocket asks do not
interleave STEP numbers into one another.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Any

from knowledge_graph.logging_config import get_logger

LOGGER = get_logger("fashion_brain.trace")


def _short(value: Any, limit: int = 160) -> str:
    text = str(value if value is not None else "")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _fmt_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    parts = []
    for key, value in fields.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        if isinstance(value, (list, tuple)):
            shown = ", ".join(_short(item, 40) for item in value[:8])
            if len(value) > 8:
                shown += f", +{len(value) - 8} more"
            parts.append(f"{key}=[{shown}]")
        elif isinstance(value, dict):
            inner = ", ".join(f"{k}:{_short(v, 32)}" for k, v in list(value.items())[:8])
            parts.append(f"{key}={{{inner}}}")
        else:
            parts.append(f"{key}={_short(value)}")
    return " ".join(parts)


class AskTrace:
    """Numbered step tracer for one Fashion Brain ask turn."""

    def __init__(self, question: str, *, source: str = "ask"):
        self.trace_id = uuid.uuid4().hex[:10]
        self.question = question
        self.source = source
        self.started = time.perf_counter()
        self._step = 0
        self._phase = "start"
        LOGGER.info(
            "TRACE %s START source=%s question=%s",
            self.trace_id,
            source,
            _short(question, 200),
        )

    def step(self, phase: str, message: str, **fields: Any) -> None:
        self._step += 1
        self._phase = phase
        elapsed = time.perf_counter() - self.started
        detail = _fmt_fields(fields)
        suffix = f" | {detail}" if detail else ""
        LOGGER.info(
            "TRACE %s STEP %02d [%s] +%.3fs %s%s",
            self.trace_id,
            self._step,
            phase,
            elapsed,
            message,
            suffix,
        )

    def debug(self, message: str, **fields: Any) -> None:
        detail = _fmt_fields(fields)
        suffix = f" | {detail}" if detail else ""
        LOGGER.debug(
            "TRACE %s DETAIL [%s] %s%s",
            self.trace_id,
            self._phase,
            message,
            suffix,
        )

    def warn(self, message: str, **fields: Any) -> None:
        detail = _fmt_fields(fields)
        suffix = f" | {detail}" if detail else ""
        LOGGER.warning(
            "TRACE %s WARN [%s] %s%s",
            self.trace_id,
            self._phase,
            message,
            suffix,
        )

    def done(self, **fields: Any) -> None:
        elapsed = time.perf_counter() - self.started
        detail = _fmt_fields(fields)
        suffix = f" | {detail}" if detail else ""
        LOGGER.info(
            "TRACE %s DONE steps=%s elapsed=%.3fs%s",
            self.trace_id,
            self._step,
            elapsed,
            suffix,
        )


_CURRENT: ContextVar[AskTrace | None] = ContextVar("fashion_brain_ask_trace", default=None)


def current_trace() -> AskTrace | None:
    return _CURRENT.get()


def start_trace(question: str, *, source: str = "ask") -> AskTrace:
    trace = AskTrace(question, source=source)
    _CURRENT.set(trace)
    return trace


def end_trace(**fields: Any) -> None:
    trace = _CURRENT.get()
    if trace is not None:
        trace.done(**fields)
    _CURRENT.set(None)


def trace_step(phase: str, message: str, **fields: Any) -> None:
    trace = _CURRENT.get()
    if trace is not None:
        trace.step(phase, message, **fields)
    else:
        detail = _fmt_fields(fields)
        suffix = f" | {detail}" if detail else ""
        LOGGER.info("TRACE STEP [%s] %s%s", phase, message, suffix)


def trace_debug(message: str, **fields: Any) -> None:
    trace = _CURRENT.get()
    if trace is not None:
        trace.debug(message, **fields)
    else:
        detail = _fmt_fields(fields)
        suffix = f" | {detail}" if detail else ""
        LOGGER.debug("TRACE DETAIL %s%s", message, suffix)
