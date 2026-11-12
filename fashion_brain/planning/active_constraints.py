"""Carry prior-turn filters using the **live** RuntimeSchemaContext.

Design rules (multi-KG / multi-source):
- Do **not** maintain ecom-only allowlists (brand/material/country/…).
- Prior ``active_filters`` are an opaque dict produced by Extract / Intention for
  whatever graph was selected. Carry those keys as-is on contextual follow-ups.
- ``group_by`` and “all X / by X” dimensions come from ``schema.filter_keys`` /
  ``facet_types`` via ``detect_group_by_from_schema`` — not a static brand table.
- Question rewrite is generic JSON + relationship hints from the schema allowlist.

Adding a new ontology/source should only require registry + ontology data — not a
Fashion Brain code change for carry logic.
"""

from __future__ import annotations

import json
import re
from typing import Any

from knowledge_graph.ontology import normalize_name


def schema_dimension_keys(schema=None) -> set[str]:
    """Filter / facet dimension names from the live schema (any ontology)."""

    out: set[str] = set()
    if schema is None:
        return out
    for key in getattr(schema, "filter_keys", None) or []:
        k = str(key or "").casefold().strip()
        if k:
            out.add(k)
    for label in getattr(schema, "facet_types", None) or []:
        k = _facet_to_key(str(label))
        if k:
            out.add(k)
    return out


def _facet_to_key(label: str) -> str:
    raw = str(label or "")
    low = raw.casefold()
    if "category" in low:
        return "category"
    if low in {"colour", "color"}:
        return "color"
    # GarmentCategory → already handled; CamelCase → snake-ish
    parts = re.findall(r"[A-Za-z0-9]+", raw)
    if not parts:
        return low
    return "".join(parts).casefold() if len(parts) == 1 else "_".join(p.casefold() for p in parts)


def clean_filters(filters: dict | None, schema=None) -> dict[str, str]:
    """Normalize filter values. Keys are opaque — no ontology allowlist drop.

    When ``schema`` is present, prefer keys that exist on the schema **or** that
    already appear (prior turn). Unknown empty values are dropped.
    """

    dims = schema_dimension_keys(schema)
    out: dict[str, str] = {}
    for key, value in (filters or {}).items():
        if value in (None, "", []):
            continue
        k = str(key).casefold().strip()
        if not k:
            continue
        if isinstance(value, (list, tuple)):
            text = " ".join(str(x).strip() for x in value if x not in (None, ""))
        else:
            text = str(value).strip()
        if not text:
            continue
        # If schema is known and this is a fresh key not in schema and not a
        # structural alias, still keep it — Extract may use graph-specific keys.
        # Only skip pure noise keys.
        if k in {"unsupported", "limit", "suggest", "analysis"}:
            continue
        if dims and k not in dims and k not in {"garment", "buyer"}:
            # Keep anyway if it looks like a facet key (letters); drop garbage.
            if not re.fullmatch(r"[a-z][a-z0-9_]*", k):
                continue
        out[k] = text
    return _dedupe_alias_collisions(out, dims)


def _dedupe_alias_collisions(filters: dict[str, str], dims: set[str]) -> dict[str, str]:
    """If the same value sits on a schema facet and on garment/category, prefer the facet."""

    out = dict(filters)
    # When a schema dimension holds a value, drop identical garment/category copies.
    prefer = [k for k in out if k in dims and k not in {"garment", "category", "garment_type"}]
    for key in prefer:
        nn = normalize_name(out[key])
        for alias in ("garment", "garment_type", "category"):
            if alias in out and normalize_name(out[alias]) == nn and alias != key:
                out.pop(alias, None)
    return out


def merge_filters(prior: dict | None, current: dict | None, *, schema=None) -> dict[str, str]:
    """Prior constraints survive; current overrides same keys. Schema-agnostic merge."""

    base = clean_filters(prior, schema)
    nxt = clean_filters(current, schema)
    base.update(nxt)
    return clean_filters(base, schema)


def constraints_from_history(history: list | None, schema=None) -> dict[str, str]:
    """Best-effort: only used when conversation_state has no active_filters yet.

    Does **not** re-parse ecom-specific grammar. Relies on prior turns having
    stored filters in conversation_state when available. History fallback is empty
    unless the caller already put structured filters on assistant meta (future).
    """

    # Opaque carry requires structured state. Without it, return {} and let
    # ConversationContext + Extract see raw history text in the rewritten prompt.
    _ = history, schema
    return {}


def looks_like_constraint_followup(question: str, history: list | None = None, schema=None) -> bool:
    if not history:
        return False
    q = (question or "").strip()
    if not q:
        return False
    # Schema dimensions: "all brands", "by material", "family wise", …
    for key in schema_dimension_keys(schema):
        if not key:
            continue
        esc = re.escape(key)
        if re.search(rf"\b(?:all|each|every|per|by)\s+{esc}s?\b", q, re.I):
            return True
        if re.search(rf"\b{esc}s?\s*wise\b", q, re.I):
            return True
        if re.search(rf"\bcheck\s+for\s+all\s+{esc}s?\b", q, re.I):
            return True
    # Generic continuation cues (language, not ontology).
    if re.search(
        r"\b(same|those|these|of\s+them|for\s+each|break\s*down|breakdown|"
        r"also\s+show|now\s+show|and\s+for|check\s+for)\b",
        q,
        re.I,
    ):
        return True
    # Do NOT treat every short question as a follow-up — that glued Primark|Zara
    # onto unrelated asks like "which products use cotton".
    return False


def detect_followup_group_by(question: str, schema=None) -> str | None:
    """Group dimension from live schema — never a hard-coded brand/family table."""

    try:
        from fashion_brain.agents.entity_extract import detect_group_by_from_schema

        hit = detect_group_by_from_schema(question, schema)
        if hit:
            return hit
    except Exception:
        pass
    # Fallback: "all|by|per <schema_dimension>"
    q = question or ""
    for key in sorted(schema_dimension_keys(schema), key=len, reverse=True):
        esc = re.escape(key)
        if re.search(rf"\b(?:all|each|every|per|by)\s+{esc}s?\b|{esc}s?\s*wise\b", q, re.I):
            return key
    return None


def relationship_hints_for_filters(schema, filters: dict[str, str]) -> str:
    """Map filter keys → relationship lines from the live allowlist (any ontology)."""

    if not schema or not filters:
        return ""
    allow = getattr(schema, "relationship_allowlist", None) or {}
    facets = {str(t).casefold(): str(t) for t in (getattr(schema, "facet_types", None) or [])}
    lines: list[str] = []
    for key in filters:
        label = None
        if key in facets:
            label = facets[key]
        else:
            for fk, fl in facets.items():
                if key in fk or fk in key or (key == "category" and "category" in fk):
                    label = fl
                    break
        if not label:
            continue
        for rel, pairs in allow.items():
            for pair in pairs or []:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                if str(pair[1]).casefold() == label.casefold() or label.casefold() in str(pair[1]).casefold():
                    lines.append(
                        f"- filter `{key}={filters[key]}` → use relationship {rel} toward :{pair[1]}"
                    )
                    break
    if not lines:
        return ""
    return "Schema relationship hints for REQUIRED filters:\n" + "\n".join(lines)


def rewrite_question_with_constraints(
    question: str,
    filters: dict | None,
    group_by: str | None = None,
    *,
    schema=None,
    prior_user_question: str | None = None,
) -> str:
    """Generic rewrite — works for any graph's filter keys."""

    filters = clean_filters(filters, schema)
    if not filters and not group_by and not prior_user_question:
        return question
    parts = [
        "Continue the prior knowledge-graph question.",
        f"REQUIRED filters (do not drop; keys are from the live schema): {json.dumps(filters, ensure_ascii=False)}",
    ]
    hints = relationship_hints_for_filters(schema, filters)
    if hints:
        parts.append(hints)
    if group_by:
        parts.append(
            f"Group / break down by `{group_by}` (a live schema dimension) while keeping every REQUIRED filter."
        )
    if prior_user_question:
        parts.append(f"Prior user question: {prior_user_question}")
    parts.append(f"User follow-up: {question}")
    return "\n".join(parts)


def apply_turn_constraints(
    question: str,
    history: list | None,
    intention_filters: dict | None,
    intention_group_by: str | None,
    *,
    schema=None,
    conversation_state: dict | None = None,
    is_contextual: bool = False,
    carry_filters: bool = False,
    refined_question: str | None = None,
    context_group_by: str | None = None,
) -> tuple[dict[str, str], str | None, str, dict[str, Any]]:
    """Merge filters when the **context LLM** says the turn is contextual.

    Follow-up detection is not done here — ConversationContextAgent owns that.
    """

    state = dict(conversation_state or {})
    prior = clean_filters(state.get("active_filters"), schema)
    current = clean_filters(intention_filters, schema)

    # Follow-up flag comes only from ConversationContextAgent (LLM), not regex.
    follow = bool(is_contextual and carry_filters)
    if follow and prior:
        filters = merge_filters(prior, current, schema=schema)
    elif follow and not prior:
        # Contextual but no structured prior yet — keep current; rewrite uses history text.
        filters = current
    else:
        filters = current

    # Follow-up group_by detection only when this turn actually continues the prior ask.
    # Otherwise a short standalone question inherits "by brand" from chat state.
    if follow:
        group_by = (
            context_group_by
            or detect_followup_group_by(question, schema)
            or intention_group_by
        )
    else:
        group_by = intention_group_by or None

    prior_q = None
    for turn in reversed(_user_texts(history)):
        prior_q = turn
        break

    # Prefer the context agent's self-contained rewrite when present.
    if refined_question and refined_question.strip() and refined_question.strip() != question.strip():
        effective = refined_question.strip()
        if filters:
            effective = rewrite_question_with_constraints(
                effective,
                filters,
                group_by,
                schema=schema,
                prior_user_question=prior_q if follow else None,
            )
    elif follow and (filters or prior_q):
        effective = rewrite_question_with_constraints(
            question,
            filters,
            group_by,
            schema=schema,
            prior_user_question=prior_q,
        )
    else:
        effective = question

    patch = {
        "active_filters": filters,
        "group_by": group_by,
        "last_question": question,
        "effective_question": effective,
    }
    return filters, group_by, effective, patch


def _user_texts(history: list | None) -> list[str]:
    out: list[str] = []
    for item in history or []:
        if isinstance(item, dict):
            role = str(item.get("role") or "")
            content = str(item.get("content") or item.get("text") or "")
        else:
            role = str(getattr(item, "role", "") or "")
            content = str(getattr(item, "content", "") or getattr(item, "text", "") or "")
        if role in {"user", "human"} and content.strip():
            out.append(content.strip())
    return out
