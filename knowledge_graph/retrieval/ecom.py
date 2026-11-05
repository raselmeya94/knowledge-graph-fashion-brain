"""Query Product nodes in any Neo4j product graph registered in kg_registry.

Graph ids are not hard-coded: callers pass the registry product graph (e.g. ecom_products).
Fashion Brain should only resolve which product graph via GET /graphs (kind=product).
"""

from __future__ import annotations

import json

from knowledge_graph.access import Principal, allows
from knowledge_graph.models import EvidenceItem
from knowledge_graph.ontology import normalize_name

_JUNK_GARMENT = frozenset(
    normalize_name(token)
    for token in (
        "found",
        "matched",
        "match",
        "result",
        "results",
        "available",
        "currently",
        "total",
        "number",
        "count",
        "data",
        "there",
        "here",
        "every",
        "each",
        "all",
        "any",
        "wise",
        "brand",
        "brands",
        "buyer",
        "buyers",
        "material",
        "materials",
    )
)
_JUNK_MATERIAL_LABELS = frozenset(
    normalize_name(token)
    for token in (
        "outer shell",
        "outer",
        "shell",
        "lining",
        "fabric content",
        "fabric",
        "content",
        "body",
        "composition",
        "material",
        "materials",
    )
)


def registry_product_graphs(platform) -> list[str]:
    """Active product-kind graph ids from the KG registry (dynamic)."""

    try:
        graphs = list(platform.kg_registry.list_graphs())
    except Exception:
        return []
    return [
        item.id
        for item in graphs
        if getattr(item, "kind", None) == "product" and getattr(item, "status", "active") == "active"
    ]


def default_product_graph(platform, preferred: str | None = None) -> str | None:
    """Pick preferred graph if it is a product graph; else first registered product graph."""

    ids = registry_product_graphs(platform)
    if preferred and preferred in ids:
        return preferred
    if preferred and preferred in {"shared", "master", "news"}:
        return ids[0] if ids else None
    return ids[0] if ids else None


def neo4j_product_count(platform, graph_id: str) -> int:
    return len(platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False))


_ANALYSIS_FIELDS = (
    "brand",
    "category",
    "family",
    "subfamily",
    "color",
    "material",
    "availability",
    "section",
)


def clean_product_filters(filters: dict) -> dict:
    cleaned = {key: value for key, value in (filters or {}).items() if value not in (None, "")}
    # E-comm graphs use brand, not PLM buyer. "analysis for matalan" arrives as buyer=.
    if cleaned.get("buyer") and not cleaned.get("brand"):
        cleaned["brand"] = cleaned.pop("buyer")
    elif cleaned.get("buyer") and cleaned.get("brand"):
        cleaned.pop("buyer", None)
    garment = cleaned.get("garment") or cleaned.get("garment_type") or cleaned.get("category")
    if garment and normalize_name(str(garment)) in _JUNK_GARMENT:
        cleaned.pop("garment", None)
        cleaned.pop("garment_type", None)
        cleaned.pop("category", None)
    # Brand names mis-tagged as garment (e.g. garment=primark) must not filter categories.
    brand = cleaned.get("brand")
    if brand:
        brand_key = normalize_name(str(brand))
        for key in ("garment", "garment_type", "category"):
            value = cleaned.get(key)
            if value and normalize_name(str(value)) == brand_key:
                cleaned.pop(key, None)
    material = cleaned.get("material")
    if material and normalize_name(str(material)) in _JUNK_MATERIAL_LABELS:
        cleaned.pop("material", None)
    return cleaned


def match_neo4j_products(platform, principal: Principal, graph_id: str, filters: dict) -> list:
    filters = clean_product_filters(filters)
    nodes = platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False)
    matched = []
    for node in nodes:
        if not allows(principal, node.properties.get("classification") or "internal"):
            continue
        if not _product_matches(platform, graph_id, node, filters):
            continue
        matched.append(node)
    matched.sort(key=lambda item: item.id)
    try:
        from fashion_brain.trace import trace_debug

        trace_debug(
            "Neo4j product match",
            graph_id=graph_id,
            scanned=len(nodes),
            matched=len(matched),
            filters=filters,
        )
    except Exception:
        pass
    return matched


def search_neo4j_products(platform, principal: Principal, graph_id: str, filters: dict, *, limit: int = 5) -> tuple[list, int]:
    matched = match_neo4j_products(platform, principal, graph_id, filters)
    return matched[:limit], len(matched)


def append_neo4j_count_answer(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    group_by: str | None,
    limit: int | None,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
) -> None:
    filters = clean_product_filters(filters)
    sample, total = search_neo4j_products(platform, principal, graph_id, filters, limit=5)
    phrase = _filter_phrase(filters)
    label = _graph_label(platform, graph_id)
    try:
        from fashion_brain.trace import trace_step

        trace_step(
            "ecom",
            "Count answer built",
            graph_id=graph_id,
            total=total,
            group_by=group_by,
            filters=filters,
            label=label,
        )
    except Exception:
        pass
    if not total and not filters:
        lines.append(
            f"The {label} catalog contains 0 visible products in Neo4j for this clearance. "
            "Sync the source or raise clearance if products exist."
        )
    else:
        lines.append(f"The {label} catalog contains {total} products" + (f" {phrase}." if phrase else "."))
        if group_by:
            groups = [
                (name, count)
                for name, count in _count_by(platform, principal, graph_id, group_by, filters)
                if _usable_value(name)
            ]
            lines.extend(_dimension_breakdown_lines(group_by, groups, total, limit))
    for node in sample:
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id=str(node.properties.get("prov_record_id") or node.id),
                title=str(node.properties.get("name") or node.id),
                excerpt=_excerpt(node),
                source_path=str(node.properties.get("source_id") or graph_id),
                verification_status="observed",
                graph=graph_id,
                node_id=node.id,
                synthetic=False,
            )
        )


def append_neo4j_analysis_answer(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    limit: int | None,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
    analysis_fields: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Type-wise top rankings for a brand/selection on a Neo4j product graph.

    ``analysis_fields`` should be the unique columns of the selected KG; falls back
    to the e-comm defaults when omitted.
    """

    filters = clean_product_filters(filters)
    matched = match_neo4j_products(platform, principal, graph_id, filters)
    total = len(matched)
    top_n = max(int(limit or 5), 1)
    label = _graph_label(platform, graph_id)
    who = filters.get("brand") or filters.get("buyer") or "this selection"
    fields = tuple(analysis_fields) if analysis_fields else _ANALYSIS_FIELDS
    try:
        from fashion_brain.trace import trace_step

        trace_step(
            "ecom",
            "Analysis answer built",
            graph_id=graph_id,
            total=total,
            who=who,
            fields=list(fields),
            filters=filters,
        )
    except Exception:
        pass
    if not total:
        lines.append(f"There are no products for {who} in {label} to analyse.")
        return
    lines.append(f"Product analysis for {who}: {total} products in {label}.")
    lines.append(f"Top {top_n} values by type (highest count first).")
    ranked = _profile_nodes(platform, graph_id, matched, fields=fields)
    # Chart-only fields vs table-only fields — never both for the same values.
    chart_fields = {"material", "color", "availability", "gender"}
    for field in fields:
        choices = [item for item in (ranked.get(field) or []) if _usable_value(item[0])][:top_n]
        if not choices:
            continue
        field_label = field.replace("_", " ")
        lines.append(f"### {field_label}")
        if field in chart_fields:
            chart_type = "pie" if field in {"material", "color", "gender"} else "bar"
            lines.extend(
                _chart_lines(
                    chart_type,
                    f"{who} — {field_label}",
                    [name for name, _count in choices],
                    [count for _name, count in choices],
                )
            )
            # One-line share summary only — no duplicate table of the same numbers.
            top_name, top_count = choices[0]
            share = round(100.0 * top_count / total, 1) if total else 0.0
            lines.append(f"Leading value: {top_name} ({top_count}, {share}%).")
        else:
            lines.append("| Name | Count | Share % |")
            lines.append("| --- | ---: | ---: |")
            for name, count in choices:
                share = round(100.0 * count / total, 1) if total else 0.0
                lines.append(f"| {name} | {count} | {share} |")
    for node in matched[:5]:
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id=str(node.properties.get("prov_record_id") or node.id),
                title=str(node.properties.get("name") or node.id),
                excerpt=_excerpt(node),
                source_path=str(node.properties.get("source_id") or graph_id),
                verification_status="observed",
                graph=graph_id,
                node_id=node.id,
                synthetic=False,
            )
        )


# Back-compat aliases used by earlier wiring
ECOM_GRAPH = "ecom_products"


def ecom_product_count(platform) -> int:
    graph = default_product_graph(platform, ECOM_GRAPH) or ECOM_GRAPH
    return neo4j_product_count(platform, graph)


def clean_ecom_filters(filters: dict) -> dict:
    return clean_product_filters(filters)


def append_ecom_count_answer(platform, principal, filters, group_by, limit, lines, evidence, caveats) -> None:
    graph = default_product_graph(platform) or ECOM_GRAPH
    append_neo4j_count_answer(platform, principal, graph, filters, group_by, limit, lines, evidence, caveats)


def _graph_label(platform, graph_id: str) -> str:
    try:
        for item in platform.kg_registry.list_graphs():
            if item.id == graph_id:
                return str(item.name or graph_id)
    except Exception:
        pass
    return graph_id


def _product_matches(platform, graph_id: str, node, filters: dict) -> bool:
    props = node.properties or {}
    brand = filters.get("brand")
    if brand:
        prop_brand = normalize_name(str(props.get("brand") or ""))
        want = normalize_name(str(brand))
        if prop_brand != want and not _has_related(platform, graph_id, node.id, "HAS_BRAND", str(brand)):
            return False
    color = filters.get("color")
    if color:
        if normalize_name(str(props.get("color") or "")) != normalize_name(str(color)):
            if not _has_related(platform, graph_id, node.id, "HAS_COLOR", str(color)):
                return False
    category = filters.get("category") or filters.get("garment") or filters.get("garment_type")
    if category:
        cat = normalize_name(str(category))
        hay = " ".join(
            normalize_name(str(props.get(key) or ""))
            for key in ("category", "family", "subfamily", "category_path", "name")
        )
        if cat not in hay:
            if not (
                _has_related(platform, graph_id, node.id, "BELONGS_TO_CATEGORY", str(category))
                or _has_related(platform, graph_id, node.id, "HAS_FAMILY", str(category))
                or _has_related(platform, graph_id, node.id, "HAS_SUB_CATEGORY", str(category))
            ):
                return False
    material = filters.get("material")
    if material and not _material_matches(platform, graph_id, node, str(material)):
        return False
    availability = filters.get("availability")
    if availability and normalize_name(str(props.get("availability") or "")) != normalize_name(str(availability)):
        if not _has_related(platform, graph_id, node.id, "HAS_AVAILABILITY", str(availability)):
            return False
    return True


def _has_related(platform, graph_id: str, product_id: str, rel_type: str, name: str) -> bool:
    target = normalize_name(name)
    try:
        edges = platform.graph.find_edges(graph=graph_id, rel_type=rel_type, start_id=product_id, include_deleted=False)
    except Exception:
        return False
    for edge in edges:
        end = platform.graph.get_node(edge.end_id, include_deleted=False)
        if end and normalize_name(str(end.properties.get("name") or "")) == target:
            return True
    return False


def _material_matches(platform, graph_id: str, node, material: str) -> bool:
    """True when composition text or USES_MATERIAL mentions the requested fibre."""

    want = normalize_name(material)
    if not want or want in _JUNK_MATERIAL_LABELS:
        return True
    props = node.properties or {}
    haystacks = [
        normalize_name(str(props.get("composition") or "")),
        normalize_name(str(props.get("material") or "")),
        normalize_name(str(props.get("composition_detail") or "")),
    ]
    if any(want == text or (len(want) >= 4 and want in text) for text in haystacks if text):
        return True
    for value in _field_values(platform, graph_id, node, "material"):
        name = normalize_name(value)
        if not name or name in _JUNK_MATERIAL_LABELS:
            continue
        if name == want or (len(want) >= 4 and want in name) or (len(name) >= 4 and name in want):
            return True
    return False


def _count_by(platform, principal: Principal, graph_id: str, field: str, filters: dict) -> list[tuple[str, int]]:
    nodes = match_neo4j_products(platform, principal, graph_id, filters)
    tallies: dict[str, int] = {}
    for node in nodes:
        for key in _field_values(platform, graph_id, node, field):
            tallies[key] = tallies.get(key, 0) + 1
    return sorted(tallies.items(), key=lambda item: (-item[1], item[0].casefold()))


def _profile_nodes(
    platform,
    graph_id: str,
    nodes: list,
    *,
    fields: tuple[str, ...] | list[str] | None = None,
) -> dict[str, list[tuple[str, int]]]:
    ranked: dict[str, list[tuple[str, int]]] = {}
    for field in fields or _ANALYSIS_FIELDS:
        tallies: dict[str, int] = {}
        for node in nodes:
            for key in _field_values(platform, graph_id, node, field):
                tallies[key] = tallies.get(key, 0) + 1
        ranked[field] = sorted(tallies.items(), key=lambda item: (-item[1], item[0].casefold()))
    return ranked


_REL_FIELDS = {
    "category": ("BELONGS_TO_CATEGORY",),
    "family": ("HAS_FAMILY",),
    "subfamily": ("HAS_SUB_CATEGORY",),
    "color": ("HAS_COLOR",),
    "material": ("USES_MATERIAL",),
    "availability": ("HAS_AVAILABILITY",),
    "brand": ("HAS_BRAND",),
}


def _field_values(platform, graph_id: str, node, field: str) -> list[str]:
    props = node.properties or {}
    direct = props.get(field)
    values: list[str] = []
    if direct not in (None, "", []):
        if isinstance(direct, list):
            values.extend(str(item).strip() for item in direct if str(item).strip())
        else:
            values.append(str(direct).strip())
    for rel_type in _REL_FIELDS.get(field, ()):
        try:
            edges = platform.graph.find_edges(graph=graph_id, rel_type=rel_type, start_id=node.id, include_deleted=False)
        except Exception:
            continue
        for edge in edges:
            end = platform.graph.get_node(edge.end_id, include_deleted=False)
            name = str((end.properties or {}).get("name") or "").strip() if end else ""
            if name:
                values.append(name)
    # Deduplicate while keeping order.
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = normalize_name(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _usable_value(value: str) -> bool:
    folded = str(value or "").strip().casefold()
    if folded in {"", "unspecified", "null", "none", "n/a", "na", "-"}:
        return False
    return normalize_name(folded) not in _JUNK_MATERIAL_LABELS


def _group_text(field: str, groups: list[tuple[str, int]], limit: int | None) -> str:
    shown = groups[: max(int(limit or 10), 1)]
    body = ", ".join(f"{name} {count}" for name, count in shown)
    return f"{field} counts: {body}." if body else f"No {field} breakdown."


def _chart_lines(chart_type: str, title: str, labels: list[str], values: list[int]) -> list[str]:
    payload = {
        "type": chart_type,
        "title": title,
        "labels": labels,
        "values": values,
    }
    return ["```chart", json.dumps(payload, ensure_ascii=False), "```"]


def _dimension_breakdown_lines(
    field: str,
    groups: list[tuple[str, int]],
    total: int,
    limit: int | None,
) -> list[str]:
    """One-dimension breakdown: chart OR table, never both with the same numbers."""

    top_n = max(int(limit or 10), 1)
    shown = groups[:top_n]
    if not shown:
        return [f"No {field.replace('_', ' ')} breakdown."]
    field_label = field.replace("_", " ")
    lines = [f"{field_label} breakdown (highest count first):"]
    # Share-style dimensions → chart only; ranked dimensions → table only.
    if field in {"material", "color", "gender", "availability"}:
        chart_type = "pie" if field in {"material", "color", "gender"} else "bar"
        lines.extend(
            _chart_lines(
                chart_type,
                field_label,
                [name for name, _count in shown],
                [count for _name, count in shown],
            )
        )
        top_name, top_count = shown[0]
        share = round(100.0 * top_count / total, 1) if total else 0.0
        lines.append(f"Leading {field_label}: {top_name} ({top_count} of {total}, {share}%).")
        return lines
    lines.append("| Name | Count | Share % |")
    lines.append("| --- | ---: | ---: |")
    for name, count in shown:
        share = round(100.0 * count / total, 1) if total else 0.0
        lines.append(f"| {name} | {count} | {share} |")
    return lines


def _filter_phrase(filters: dict) -> str:
    if not filters:
        return ""
    return "with " + " and ".join(f"{key} {value}" for key, value in filters.items())


def _excerpt(node) -> str:
    props = node.properties or {}
    bits = [props.get("brand"), props.get("name"), props.get("category"), props.get("price"), props.get("currency")]
    return " | ".join(str(bit) for bit in bits if bit)
