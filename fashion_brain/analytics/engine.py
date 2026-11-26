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
    question: str = "",
) -> Retrieved:
    """Attribute-wise analysis on one or more product graphs (schema-driven fields)."""

    from fashion_brain.ask import ensure_compare_party_filters, split_multi_party_filter
    from fashion_brain.planning.semantic_field_map import apply_semantic_field_map
    from knowledge_graph.retrieval.ecom import (
        append_neo4j_analysis_answer,
        clean_product_filters,
        resolve_analysis_fields,
        resolve_focused_analysis_field,
        _align_party_filters_for_graph,
    )

    filters = ensure_compare_party_filters(dict(filters or {}), question)
    # Merchant analysis asks often name a brand in prose; never run unscoped when
    # a live brand is clearly mentioned (stops whole-catalog Primark/Matalan misses).
    if question and not (filters.get("brand") or filters.get("buyer")):
        try:
            from fashion_brain.ask import live_brand_names, live_brands_mentioned

            hits = live_brands_mentioned(question, live_brand_names(platform))
            if len(hits) == 1:
                filters["brand"] = hits[0]
            elif len(hits) >= 2:
                filters["brand"] = " | ".join(hits[:4])
        except Exception:
            pass
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

    parties = split_multi_party_filter(str(filters.get("brand") or filters.get("buyer") or ""))
    # Brand-vs-brand assortment compare belongs on the retail catalog first; avoid
    # unscoped dual-catalog dumps when the ask is clearly a party compare.
    if len(parties) >= 2 and len(ids) > 1 and not group_by:
        ecom_ids = [g for g in ids if "ecom" in g.casefold() or "commerce" in g.casefold()]
        if ecom_ids:
            ids = ecom_ids
            caveats.append(
                "Brand assortment compare scoped to e-commerce (side-by-side brand profiles)."
            )

    field_override = analysis_fields
    if len(parties) >= 2:
        lines.append(
            f"Side-by-side assortment compare: {' vs '.join(parties)}. "
            "Each brand section below is independent — do not merge into one total."
        )
        lines.append("")

    for gid in ids:
        if len(ids) > 1:
            label = _graph_display_name(platform, gid)
            # User-facing heading only — never leak internal graph ids.
            lines.append(f"## {label}")
        # Per-graph aliases: ecom buyer→brand, gender→section, … + hierarchy fallbacks.
        mapped_filters, mapped_gb = apply_semantic_field_map(
            dict(filters),
            gid,
            group_by=group_by,
        )
        mapped_filters = _align_party_filters_for_graph(
            platform, gid, clean_product_filters(mapped_filters)
        )
        if group_by or mapped_gb:
            fields = resolve_focused_analysis_field(
                platform, gid, mapped_gb or group_by, mapped_filters
            )
            if fields and (mapped_gb or group_by) and fields[0] != str(mapped_gb or group_by):
                caveats.append(
                    f"Grouped by {fields[0].replace('_', ' ')} "
                    f"(role-similar to {str(mapped_gb or group_by).replace('_', ' ')} "
                    f"on this catalog)."
                )
        else:
            fields = resolve_analysis_fields(platform, gid, field_override)
        append_neo4j_analysis_answer(
            platform,
            principal,
            gid,
            mapped_filters,
            limit,
            lines,
            evidence,
            caveats,
            analysis_fields=fields,
        )
        if len(ids) > 1:
            lines.append("")

    if len(ids) > 1:
        labels = [_graph_display_name(platform, gid) for gid in ids]
        caveats.append(
            f"Analysis ran independently on {len(ids)} product catalogs: {', '.join(labels)}."
        )
    else:
        caveats.append(
            f"Analysis scoped to {_graph_display_name(platform, ids[0])} (not assumed e-commerce)."
        )

    return Retrieved(
        text="\n".join(lines).strip(),
        evidence=evidence,
        caveats=caveats,
        unavailable=[],
        retrieve_mode="analytical_tool",
        # Full and focused breakdowns share the analysis_report writer path.
        response_type="analysis_report",
    )


def _graph_display_name(platform, graph_id: str) -> str:
    from fashion_brain.agents.answer_formation import source_heading_label, source_display_label

    try:
        for item in platform.kg_registry.list_graphs():
            if item.id == graph_id:
                raw = str(item.name or graph_id)
                # Prefer short stable headings (PLMex / E-commerce) over registry chrome.
                headed = source_heading_label(graph_id)
                if headed in {"PLMex", "PLM", "E-commerce", "Own products"}:
                    return "PLMex" if headed == "PLM" else headed
                from knowledge_graph.retrieval.ecom import _user_facing_source_name

                return _user_facing_source_name(raw) or source_display_label(graph_id)
    except Exception:
        pass
    return source_heading_label(graph_id)
