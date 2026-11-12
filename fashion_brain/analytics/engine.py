"""Analytical engine — works on any registered product graph_id(s), not ecom-only."""

from __future__ import annotations

from knowledge_graph.access import Principal
from knowledge_graph.models import EvidenceItem

from fashion_brain.agents.contracts import Retrieved


def pick_analysis_graphs(
    platform,
    *,
    preferred: list[str] | None = None,
    graph_id: str | None = None,
    graph_ids: list[str] | None = None,
    filters: dict | None = None,
) -> list[str]:
    """Resolve which product graphs to analyse — never force ecom.

    Order: explicit graph_ids → graph_id → preferred → capability match from filters
    → all active product graphs only when compare-sized preferred is empty and caller
    already scoped; otherwise first capability-matched product graph.
    """

    from knowledge_graph.retrieval.ecom import registry_product_graphs

    active = registry_product_graphs(platform)
    active_set = set(active)
    explicit = [str(g) for g in (graph_ids or []) if g and str(g) in active_set]
    if explicit:
        return explicit
    if graph_id and str(graph_id) in active_set:
        return [str(graph_id)]
    pref = [str(g) for g in (preferred or []) if g and str(g) in active_set]
    if pref:
        return pref

    filters = dict(filters or {})
    matched = _graphs_for_filters(platform, active, filters)
    if matched:
        return matched[:1]
    return active[:1] if active else []


def _graphs_for_filters(platform, active: list[str], filters: dict) -> list[str]:
    """Prefer PLM-like graphs for buyer; brand-capable for brand — from registry metadata."""

    want_buyer = bool(filters.get("buyer"))
    want_brand = bool(filters.get("brand"))
    if not want_buyer and not want_brand:
        return []
    rows = []
    try:
        for item in platform.kg_registry.list_graphs():
            if getattr(item, "kind", None) != "product":
                continue
            if item.id not in active:
                continue
            blob = " ".join(
                [
                    str(item.id),
                    str(item.name or ""),
                    str(item.description or ""),
                    " ".join(str(x) for x in (getattr(item, "capabilities", None) or [])),
                    " ".join(str(x) for x in (getattr(item, "entities", None) or [])),
                ]
            ).casefold()
            score = 0
            if want_buyer and ("buyer" in blob or "plm" in blob):
                score += 5
            if want_brand and ("brand" in blob or "ecom" in blob or "commerce" in blob):
                score += 5
            if score:
                rows.append((score, item.id))
    except Exception:
        return []
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [gid for _score, gid in rows]


def run_product_analysis(
    platform,
    principal: Principal,
    *,
    graph_ids: list[str],
    filters: dict | None = None,
    limit: int | None = None,
    analysis_fields: list[str] | None = None,
    group_by: str | None = None,
) -> Retrieved:
    """Attribute-wise analysis on one or more product graphs (schema-driven fields)."""

    from knowledge_graph.retrieval.ecom import append_neo4j_analysis_answer, resolve_analysis_fields

    filters = dict(filters or {})
    lines: list[str] = []
    evidence: list[EvidenceItem] = []
    caveats: list[str] = []
    ids = [str(g) for g in (graph_ids or []) if g]
    if not ids:
        return Retrieved(
            text="No product knowledge graph was selected for analysis.",
            evidence=[],
            caveats=["Analysis requires at least one product graph."],
            unavailable=["product_graph"],
            retrieve_mode="analytical_tool",
            response_type="analysis_report",
        )

    field_override = [str(group_by)] if group_by else analysis_fields
    for gid in ids:
        if len(ids) > 1:
            label = _graph_display_name(platform, gid)
            lines.append(f"## {label} (`{gid}`)")
        fields = resolve_analysis_fields(platform, gid, field_override)
        append_neo4j_analysis_answer(
            platform,
            principal,
            gid,
            filters,
            limit,
            lines,
            evidence,
            caveats,
            analysis_fields=fields,
        )
        if len(ids) > 1:
            lines.append("")

    if len(ids) > 1:
        caveats.append(
            f"Analysis ran independently on {len(ids)} product graphs: {', '.join(ids)}."
        )
    else:
        caveats.append(f"Analysis scoped to product graph `{ids[0]}` (not assumed e-commerce).")

    return Retrieved(
        text="\n".join(lines).strip(),
        evidence=evidence,
        caveats=caveats,
        unavailable=[],
        retrieve_mode="analytical_tool",
        response_type="analysis_report" if not group_by else "stats",
    )


def _graph_display_name(platform, graph_id: str) -> str:
    try:
        for item in platform.kg_registry.list_graphs():
            if item.id == graph_id:
                return str(item.name or graph_id)
    except Exception:
        pass
    return graph_id
