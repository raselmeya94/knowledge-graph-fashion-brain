"""OpenAI-compatible KG tool schemas + read-only Python wrappers.

Used by the Text2Cypher / catalog path as deterministic helpers (not a replacement
for LangGraph routing). All Neo4j access is read-only.
"""

from __future__ import annotations

from typing import Any

from knowledge_graph.retrieval.cypher_safe import (
    assert_allowlisted_cypher,
    assert_read_only_cypher,
    run_read_cypher,
)

KG_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "resolve_canonical_entity",
            "description": (
                "Resolve a user term (English/Bengali/Banglish) to Canonical / SourceEntity / "
                "Product / Brand / Buyer / Material ids via full-text and aliases. "
                "Call before writing Cypher that filters by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "graph_id": {"type": "string"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional label allowlist",
                    },
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_dynamic_entity",
            "description": (
                "Resolve a free-text term against the live Neo4j graph using full-text "
                "with Lucene fuzzy (~) and CONTAINS fallback. Prefer after ontology fuzzy "
                "misses; never invents ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "graph_id": {"type": "string"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_fashion_kg",
            "description": (
                "Execute one read-only Cypher query against the fashion knowledge graph. "
                "Writes (CREATE/MERGE/DELETE/SET/…) are rejected. Relationship types must "
                "match the ontology relationship_allowlist when available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cypher": {"type": "string"},
                    "params": {"type": "object"},
                    "question": {
                        "type": "string",
                        "description": "Optional question used to select ontology allowlist domains",
                    },
                },
                "required": ["cypher"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_kg_values",
            "description": (
                "Inspect distinct values or property keys for a label/graph to ground filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"},
                    "label": {"type": "string"},
                    "property": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["label"],
            },
        },
    },
]


def tool_resolve_canonical_entity(
    platform,
    term: str,
    *,
    graph_id: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    from knowledge_graph.linking.canonical_resolve import resolve_canonical_entity

    return resolve_canonical_entity(
        platform, term, graph_id=graph_id, labels=labels
    ).to_dict()


def tool_resolve_dynamic_entity(
    platform,
    term: str,
    *,
    graph_id: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    from knowledge_graph.linking.canonical_resolve import resolve_dynamic_entity

    return resolve_dynamic_entity(
        platform, term, graph_id=graph_id, labels=labels
    ).to_dict()


def _tool_allowlist(platform, question: str | None = None) -> dict[str, list[list[str]]] | None:
    try:
        from knowledge_graph.ontology_registry import get_ontology_registry

        reg = get_ontology_registry(platform)
        if question:
            domains = [d.ontology_id for d in reg.domain_for_question(question, limit=3)]
            return reg.allowlist_for_domains(domains or None)
        return reg.allowlist_for_domains(None)
    except Exception:
        return None


def tool_query_fashion_kg(
    platform,
    cypher: str,
    params: dict[str, Any] | None = None,
    *,
    question: str | None = None,
) -> dict[str, Any]:
    try:
        cleaned = assert_read_only_cypher(cypher)
        allow = _tool_allowlist(platform, question)
        if allow:
            cleaned = assert_allowlisted_cypher(cleaned, allow)
        rows = run_read_cypher(platform, cleaned, params or {}, allowlist=allow)
        return {"ok": True, "rows": rows, "count": len(rows)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "rows": [], "count": 0}


def tool_inspect_kg_values(
    platform,
    label: str,
    *,
    graph_id: str | None = None,
    property_name: str = "name",
    limit: int = 25,
) -> dict[str, Any]:
    lab = re_safe_label(label)
    prop = re_safe_prop(property_name)
    if not lab or not prop:
        return {"ok": False, "error": "invalid label/property", "values": []}
    cypher = (
        f"MATCH (n:`{lab}`) "
        f"WHERE ($g IS NULL OR coalesce(n.graph,'') = $g) "
        f"AND coalesce(n.`{prop}`, '') <> '' "
        f"RETURN DISTINCT toString(n.`{prop}`) AS value "
        f"ORDER BY value ASC LIMIT $limit"
    )
    try:
        rows = run_read_cypher(
            platform,
            cypher,
            {"g": graph_id, "limit": max(1, min(int(limit or 25), 100))},
        )
        values = [str(r.get("value")) for r in rows if r.get("value") not in (None, "")]
        return {"ok": True, "values": values, "count": len(values)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "values": []}


def re_safe_label(label: str) -> str:
    import re

    text = str(label or "").strip()
    return text if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text) else ""


def re_safe_prop(name: str) -> str:
    import re

    text = str(name or "").strip()
    return text if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text) else ""


def dispatch_kg_tool(platform, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch a tool call by name (for agent loops / tests)."""

    args = dict(arguments or {})
    if name == "resolve_canonical_entity":
        return tool_resolve_canonical_entity(
            platform,
            str(args.get("term") or ""),
            graph_id=args.get("graph_id"),
            labels=list(args.get("labels") or []) or None,
        )
    if name == "resolve_dynamic_entity":
        return tool_resolve_dynamic_entity(
            platform,
            str(args.get("term") or ""),
            graph_id=args.get("graph_id"),
            labels=list(args.get("labels") or []) or None,
        )
    if name == "query_fashion_kg":
        return tool_query_fashion_kg(
            platform,
            str(args.get("cypher") or ""),
            dict(args.get("params") or {}),
            question=str(args.get("question") or "") or None,
        )
    if name == "inspect_kg_values":
        return tool_inspect_kg_values(
            platform,
            str(args.get("label") or ""),
            graph_id=args.get("graph_id"),
            property_name=str(args.get("property") or "name"),
            limit=int(args.get("limit") or 25),
        )
    return {"ok": False, "error": f"unknown tool {name}"}
