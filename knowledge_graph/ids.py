"""Stable namespaced identifiers and the extension registry for source identity."""

from __future__ import annotations

import re

from knowledge_graph.errors import UnknownSourceType

_PREFIX: dict[str, str] = {}
_KIND: dict[str, str] = {}
_LABEL: dict[str, str] = {}


def slug(value: str) -> str:
    """Stable id fragment. Letters from any language are kept. Punctuation becomes hyphens."""

    text = value.strip().casefold().replace("_", "-")
    text = re.sub(r"[^\w]+", "-", text, flags=re.UNICODE)
    text = text.replace("_", "-")
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        raise ValueError("Cannot build an identifier from an empty name.")
    return text


def register_source_identity(source_type: str, prefix: str, kind: str, label: str) -> None:
    """Register how a source type is named in the graph.

    New sources call this from their adapter module. The graph engine reads the
    registry instead of hard-coding every future source.
    """

    _PREFIX[source_type] = prefix
    _KIND[source_type] = kind
    _LABEL[source_type] = label


def require_source_identity(source_type: str) -> tuple[str, str, str]:
    if source_type not in _PREFIX:
        raise UnknownSourceType(
            f"No graph identity is registered for source type '{source_type}'. "
            "Register it with register_source_identity in the source adapter package."
        )
    return _PREFIX[source_type], _KIND[source_type], _LABEL[source_type]


def primary_id(source_type: str, record_id: str) -> str:
    prefix, kind, _label = require_source_identity(source_type)
    return f"{prefix}:{kind}:{record_id}"


def source_entity_id(source_type: str, entity_type: str, name: str) -> str:
    prefix, _kind, _label = require_source_identity(source_type)
    return f"{prefix}:{slug(entity_type)}:{slug(name)}"


def primary_label(source_type: str) -> str:
    _prefix, _kind, label = require_source_identity(source_type)
    return label


def edge_id(rel_type: str, start: str, end: str, record_id: str, field_ref: str | None = None) -> str:
    ref = field_ref or "-"
    return f"{rel_type}:{start}->{end}:{record_id}:{ref}"
