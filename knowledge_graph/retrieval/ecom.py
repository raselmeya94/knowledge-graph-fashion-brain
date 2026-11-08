"""Query Product nodes in any Neo4j product graph registered in kg_registry.

Graph ids are not hard-coded: callers pass the registry product graph (e.g. ecom_products).
Fashion Brain should only resolve which product graph via GET /graphs (kind=product).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowledge_graph.access import Principal, allows
from knowledge_graph.models import EvidenceItem
from knowledge_graph.ontology import normalize_name


def _cypher_token(value: str) -> str:
    """Lowercase token for Cypher equality — keep hyphens (c-and-a), drop other punctuation."""

    text = str(value or "").strip().casefold()
    text = re.sub(r"[^a-z0-9\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _avail_token(value: str) -> str:
    """Canonical availability key: InStock / IN_STOCK / in_stock → instock."""

    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _price_num_cypher(prop: str = "p.price") -> str:
    """Prefer stored price_num; else parse display prices like €219.00 / $14.00."""

    parsed = (
        "toFloat(replace(replace(replace(replace(replace("
        f"coalesce({prop}, ''), '€', ''), '$', ''), '£', ''), ',', ''), ' ', ''))"
    )
    if prop == "p.price":
        return f"coalesce(p.price_num, {parsed})"
    if prop == "p.original_price":
        return f"coalesce(p.original_price_num, {parsed})"
    return parsed


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
        "main fabric",
        "secondary fabric",
        "unspecified",
        "fill",
        "padding",
        "interlining",
    )
)
_JUNK_MATERIAL_PARTS = (
    "outer shell",
    "main fabric",
    "secondary fabric",
    "lining",
    "interlining",
    "padding",
    "fill ",
    "insole",
    "sole",
    "upper",
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
    """Prefer sync-time kg_stats; fall back to a count query, never a full node load."""

    try:
        from knowledge_graph.retrieval.kg_stats import product_total

        generated = getattr(getattr(platform, "settings", None), "generated", None)
        cached = product_total(graph_id, generated)
        if cached is not None and cached > 0:
            return int(cached)
    except Exception:
        pass
    session_factory = getattr(platform.graph, "_session", None)
    if callable(session_factory):
        with session_factory() as session:
            row = session.run(
                "MATCH (n:Entity:Product {graph: $g}) "
                "WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                "RETURN count(n) AS c",
                {"g": graph_id},
            ).single()
            return int(row["c"]) if row else 0
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
    cleaned.pop("parent_of", None)  # structural lookup — handled in brain, not product WHERE
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


def _session(platform):
    factory = getattr(platform.graph, "_session", None)
    return factory if callable(factory) else None


def _brand_where_cypher() -> str:
    """Brand match on product.brand or HAS_BRAND name (case-insensitive; hyphens preserved)."""

    return (
        "("
        "toLower(coalesce(p.brand, '')) = $brand "
        "OR EXISTS { MATCH (p)-[:HAS_BRAND]->(b:Entity) "
        "WHERE toLower(coalesce(b.name, '')) = $brand }"
        ")"
    )


def _availability_where_cypher() -> str:
    """Match InStock / IN_STOCK / in_stock after stripping non-alphanumerics."""

    return (
        "("
        "replace(replace(replace(toLower(coalesce(p.availability,'')),'_',''),'-',''),' ','') = $availability "
        "OR EXISTS { MATCH (p)-[:HAS_AVAILABILITY]->(a:Entity) "
        "WHERE replace(replace(replace(toLower(coalesce(a.name,'')),'_',''),'-',''),' ','') = $availability }"
        ")"
    )


_KNOWN_FILTERS = frozenset(
    {
        "brand",
        "buyer",
        "category",
        "garment",
        "garment_type",
        "material",
        "availability",
        "color",
        "price_max",
        "price_min",
        "currency",
        "section",
    }
)


def _filter_where(filters: dict, params: dict) -> list[str] | None:
    """Build WHERE clauses + params for product filters. None = unsupported combo."""

    filters = clean_product_filters(filters)
    if set(filters) - _KNOWN_FILTERS:
        return None
    where = ["coalesce(p.lifecycle,'active') <> 'deleted'", "p.graph = $g"]
    brand = filters.get("brand")
    if brand:
        params["brand"] = _cypher_token(str(brand))
        where.append(_brand_where_cypher())
    category = filters.get("category") or filters.get("garment") or filters.get("garment_type")
    if category:
        # Prefer category_path + category relationship names (Product often has no family/category props).
        params["category"] = _cypher_token(str(category)).replace("-", " ")
        where.append(
            "("
            "toLower(coalesce(p.category_path,'')) CONTAINS $category "
            "OR toLower(coalesce(p.name,'')) CONTAINS $category "
            "OR EXISTS { MATCH (p)-[:BELONGS_TO_CATEGORY|HAS_FAMILY|HAS_SUB_CATEGORY]->(c:Entity) "
            "WHERE toLower(coalesce(c.name,'')) CONTAINS $category }"
            ")"
        )
    section = filters.get("section")
    if section:
        params["section"] = _cypher_token(str(section)).replace("-", " ")
        where.append(
            "("
            "toLower(coalesce(p.category_path,'')) CONTAINS $section "
            "OR EXISTS { MATCH (p)-[:HAS_SECTION]->(s:Entity) "
            "WHERE toLower(coalesce(s.name,'')) CONTAINS $section }"
            ")"
        )
    material = filters.get("material")
    if material:
        params["material"] = _cypher_token(str(material)).replace("-", " ")
        where.append(
            "("
            "toLower(coalesce(p.composition,'')) CONTAINS $material "
            "OR EXISTS { MATCH (p)-[:USES_MATERIAL]->(m:Entity) "
            "WHERE toLower(coalesce(m.name,'')) CONTAINS $material }"
            ")"
        )
    availability = filters.get("availability")
    if availability:
        params["availability"] = _avail_token(str(availability))
        where.append(_availability_where_cypher())
    color = filters.get("color")
    if color:
        params["color"] = _cypher_token(str(color)).replace("-", " ")
        where.append(
            "("
            "EXISTS { MATCH (p)-[:HAS_COLOR]->(c:Entity) "
            "WHERE toLower(coalesce(c.name,'')) = $color }"
            ")"
        )
    currency = filters.get("currency")
    if currency:
        params["currency"] = _cypher_token(str(currency)).replace("-", "")
        where.append("replace(toLower(coalesce(p.currency,'')),' ','') CONTAINS $currency")
    price_max = filters.get("price_max")
    if price_max is not None and str(price_max).strip() != "":
        try:
            params["price_max"] = float(price_max)
        except (TypeError, ValueError):
            return None
        where.append(f"{_price_num_cypher()} IS NOT NULL AND {_price_num_cypher()} < $price_max")
    price_min = filters.get("price_min")
    if price_min is not None and str(price_min).strip() != "":
        try:
            params["price_min"] = float(price_min)
        except (TypeError, ValueError):
            return None
        where.append(f"{_price_num_cypher()} IS NOT NULL AND {_price_num_cypher()} >= $price_min")
    return where


def cypher_count_products(platform, graph_id: str, filters: dict) -> int | None:
    """Server-side product count with optional brand/category/material filters."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    query = "MATCH (p:Entity:Product) WHERE " + " AND ".join(where) + " RETURN count(p) AS c"
    try:
        import time

        from fashion_brain.stage_timing import record_cypher, record_stage

        started = time.perf_counter()
        with session_factory() as session:
            row = session.run(query, params).single()
            value = int(row["c"]) if row else 0
        ms = (time.perf_counter() - started) * 1000.0
        record_stage("cypher_exec", ms, kind="count", rows=1)
        record_cypher(query, params=params, rows=1, ms=ms)
        return value
    except Exception:
        return None


def cypher_group_by(
    platform, graph_id: str, field: str, filters: dict
) -> list[tuple[str, int]] | None:
    """Aggregate products by a facet using Neo4j (supports material/color/… filters)."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    field = str(field or "").strip()
    # garment_type → category relationship facet
    if field in {"garment", "garment_type"}:
        field = "category"
    rel = _REL_FIELDS.get(field)
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    if field == "brand":
        query = (
            "MATCH (p:Entity:Product) WHERE "
            + " AND ".join(where)
            + " OPTIONAL MATCH (p)-[:HAS_BRAND]->(x:Entity) "
            "WHERE coalesce(x.lifecycle,'active') <> 'deleted' "
            "WITH p, coalesce(x.name, p.brand, '(unspecified)') AS value "
            "RETURN value, count(DISTINCT p) AS n "
            "ORDER BY n DESC, value ASC"
        )
    elif field == "availability":
        query = (
            "MATCH (p:Entity:Product) WHERE "
            + " AND ".join(where)
            + " OPTIONAL MATCH (p)-[:HAS_AVAILABILITY]->(x:Entity) "
            "WHERE coalesce(x.lifecycle,'active') <> 'deleted' "
            "WITH p, coalesce(x.name, p.availability, '(unspecified)') AS value "
            "RETURN value, count(DISTINCT p) AS n "
            "ORDER BY n DESC, value ASC"
        )
    elif rel:
        query = (
            "MATCH (p:Entity:Product) WHERE "
            + " AND ".join(where)
            + f" OPTIONAL MATCH (p)-[:{rel[0]}]->(x:Entity) "
            "WHERE coalesce(x.lifecycle,'active') <> 'deleted' "
            "WITH p, coalesce(x.name, '(unspecified)') AS value "
            "RETURN value, count(DISTINCT p) AS n "
            "ORDER BY n DESC, value ASC"
        )
    else:
        # Only allow safe product properties that exist on ecom Product nodes.
        if field not in {"brand", "availability", "name", "currency", "price", "category_path"}:
            return None
        query = (
            "MATCH (p:Entity:Product) WHERE "
            + " AND ".join(where)
            + f" RETURN coalesce(toString(p.{field}), '(unspecified)') AS value, count(p) AS n "
            "ORDER BY n DESC, value ASC"
        )
    try:
        import time

        from fashion_brain.stage_timing import record_cypher, record_stage

        started = time.perf_counter()
        with session_factory() as session:
            rows = list(session.run(query, params))
        result = [(str(row["value"]), int(row["n"])) for row in rows if row["value"] is not None]
        ms = (time.perf_counter() - started) * 1000.0
        record_stage("cypher_exec", ms, kind="group_by", rows=len(result), field=field)
        record_cypher(query, params=params, rows=len(result), ms=ms)
        return result
    except Exception:
        return None


def cypher_sample_products(platform, graph_id: str, filters: dict, *, limit: int = 5) -> list:
    """Fetch a few matching product nodes via Cypher (full filter support)."""

    session_factory = _session(platform)
    if not session_factory:
        return []
    params: dict = {"g": graph_id, "limit": max(int(limit), 1)}
    where = _filter_where(filters, params)
    if where is None:
        return []
    order = "ORDER BY p.id ASC"
    if filters.get("price_max") is not None or filters.get("price_min") is not None:
        order = f"ORDER BY {_price_num_cypher()} ASC, p.id ASC"
    query = (
        "MATCH (p:Entity:Product) WHERE "
        + " AND ".join(where)
        + f" RETURN p, labels(p) AS labels {order} LIMIT $limit"
    )
    try:
        import time

        from fashion_brain.stage_timing import record_cypher, record_stage
        from knowledge_graph.models import GraphNode

        nodes: list = []
        started = time.perf_counter()
        with session_factory() as session:
            for row in session.run(query, params):
                raw = row["p"]
                labels = list(row["labels"] or [])
                props = dict(raw)
                node_id = str(props.pop("id", raw.id if hasattr(raw, "id") else ""))
                nodes.append(
                    GraphNode(
                        id=node_id,
                        labels=labels,
                        graph=str(props.get("graph") or graph_id),
                        properties=props,
                    )
                )
        ms = (time.perf_counter() - started) * 1000.0
        record_stage("cypher_exec", ms, kind="sample", rows=len(nodes), limit=params["limit"])
        record_cypher(query, params=params, rows=len(nodes), ms=ms)
        return nodes
    except Exception:
        return []


def cypher_parent_category(platform, graph_id: str, name: str, *, limit: int = 10) -> list[dict]:
    """Resolve PARENT_CATEGORY: child -[:PARENT_CATEGORY]-> parent."""

    session_factory = _session(platform)
    if not session_factory:
        return []
    token = _cypher_token(name).replace("-", " ")
    query = (
        "MATCH (child:Entity {graph:$g})-[:PARENT_CATEGORY]->(parent:Entity) "
        "WHERE toLower(coalesce(child.name,'')) = $name "
        "OR toLower(coalesce(child.name,'')) CONTAINS $name "
        "RETURN DISTINCT child.name AS child, parent.name AS parent "
        "LIMIT $limit"
    )
    params = {"g": graph_id, "name": token, "limit": max(int(limit), 1)}
    try:
        import time

        from fashion_brain.stage_timing import record_cypher, record_stage

        started = time.perf_counter()
        with session_factory() as session:
            rows = [dict(record) for record in session.run(query, params)]
        ms = (time.perf_counter() - started) * 1000.0
        record_stage("cypher_exec", ms, kind="parent_category", rows=len(rows))
        record_cypher(query, params=params, rows=len(rows), ms=ms)
        return rows
    except Exception:
        return []


def match_neo4j_products(platform, principal: Principal, graph_id: str, filters: dict) -> list:
    """Match products via Cypher only — never load the full Product label set."""

    filters = clean_product_filters(filters)
    sample = cypher_sample_products(platform, graph_id, filters, limit=5000)
    matched = [
        node
        for node in (sample or [])
        if allows(principal, node.properties.get("classification") or "internal")
    ]
    matched.sort(key=lambda item: item.id)
    try:
        from fashion_brain.trace import trace_debug

        trace_debug(
            "Neo4j product match",
            graph_id=graph_id,
            scanned=len(sample or []),
            matched=len(matched),
            filters=filters,
        )
    except Exception:
        pass
    return matched


def search_neo4j_products(platform, principal: Principal, graph_id: str, filters: dict, *, limit: int = 5) -> tuple[list, int]:
    filters = clean_product_filters(filters)
    total = cypher_count_products(platform, graph_id, filters)
    sample = cypher_sample_products(platform, graph_id, filters, limit=limit)
    if total is not None and sample is not None:
        sample = [
            node
            for node in sample
            if allows(principal, node.properties.get("classification") or "internal")
        ]
        return sample[:limit], int(total)
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
    phrase = _filter_phrase(filters)
    label = _graph_label(platform, graph_id)
    generated = getattr(getattr(platform, "settings", None), "generated", None)
    stats_file = (
        str(Path(generated) / "kg_stats" / f"{graph_id}.json")
        if generated
        else f"kg_stats/{graph_id}.json"
    )

    def _stats_breakdown(field: str) -> tuple[int, list[tuple[str, int]]] | None:
        try:
            from knowledge_graph.retrieval.kg_stats import load_graph_stats

            stats = load_graph_stats(graph_id, generated) or {}
            bucket = (stats.get("value_counts") or {}).get(field) or {}
            if not bucket:
                return None
            total = int((stats.get("totals") or {}).get("products") or sum(int(v) for v in bucket.values()))
            groups = sorted(
                ((str(name), int(count)) for name, count in bucket.items() if _usable_value(str(name))),
                key=lambda item: (-item[1], item[0].casefold()),
            )
            return total, groups
        except Exception:
            return None

    # Unfiltered totals + brand/category/… wise: answer from sync-time kg_stats.
    if not filters and not group_by:
        total = neo4j_product_count(platform, graph_id)
        cached = _stats_breakdown("brand")
        if cached:
            total, groups = cached
            top = groups[:5]
            lines.append(f"There are {total} products in {label}.")
            lines.append(
                "By brand: "
                + ", ".join(f"{name} ({count})" for name, count in top)
                + ("…" if len(groups) > 5 else "")
                + "."
            )
            evidence.append(
                EvidenceItem(
                    source_id=graph_id,
                    record_id="kg_stats",
                    title=f"{label} product total",
                    excerpt=f"products={total}; brands={len(groups)} (from generated/kg_stats)",
                    source_path=stats_file,
                    verification_status="observed",
                    graph=graph_id,
                    node_id=None,
                    synthetic=False,
                )
            )
            return
        lines.append(f"There are {total} products in {label}.")
        return
    if not filters and group_by:
        cached = _stats_breakdown(str(group_by))
        if cached:
            total, groups = cached
            lines.append(f"The {label} catalog contains {total} products.")
            lines.extend(_dimension_breakdown_lines(str(group_by), groups, total, limit))
            evidence.append(
                EvidenceItem(
                    source_id=graph_id,
                    record_id="kg_stats",
                    title=f"{label} {group_by} breakdown",
                    excerpt=f"products={total}; {group_by} values={len(groups)} (from generated/kg_stats)",
                    source_path=stats_file,
                    verification_status="observed",
                    graph=graph_id,
                    node_id=None,
                    synthetic=False,
                )
            )
            return
    if len(filters) == 1 and not group_by:
        attr, value = next(iter(filters.items()))
        try:
            from knowledge_graph.retrieval.kg_stats import value_count

            generated = getattr(getattr(platform, "settings", None), "generated", None)
            cached = value_count(graph_id, str(attr), str(value), generated)
            if cached is not None:
                lines.append(f"There are {cached} products{phrase} in {label}.")
                evidence.append(
                    EvidenceItem(
                        source_id=graph_id,
                        record_id="kg_stats",
                        title=f"{label} {attr}={value}",
                        excerpt=f"{attr}={value} → {cached} (from generated/kg_stats)",
                        source_path=f"kg_stats/{graph_id}.json",
                        verification_status="observed",
                        graph=graph_id,
                        node_id=None,
                        synthetic=False,
                    )
                )
                return
        except Exception:
            pass
    # Filtered (+ optional group_by): Cypher aggregates — never scan all Product nodes.
    if group_by:
        groups = cypher_group_by(platform, graph_id, str(group_by), filters)
        total = cypher_count_products(platform, graph_id, filters)
        if groups is not None and total is not None:
            usable = [(name, count) for name, count in groups if _usable_value(name)]
            lines.append(f"The {label} catalog contains {total} products" + (f" {phrase}." if phrase else "."))
            lines.extend(_dimension_breakdown_lines(str(group_by), usable, total, limit))
            sample = cypher_sample_products(platform, graph_id, filters, limit=5)
            for node in sample:
                if not allows(principal, node.properties.get("classification") or "internal"):
                    continue
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
            return

    show_n = max(int(limit or 5), 1)
    sample, total = search_neo4j_products(platform, principal, graph_id, filters, limit=show_n)
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
            grouped = cypher_group_by(platform, graph_id, str(group_by), filters)
            if grouped is None:
                grouped = _count_by(platform, principal, graph_id, group_by, filters)
            groups = [(name, count) for name, count in (grouped or []) if _usable_value(name)]
            lines.extend(_dimension_breakdown_lines(group_by, groups, total, limit))
        elif filters and sample:
            # Listing-style asks ("show 5 InStock …") — include name + price rows.
            lines.append(f"Showing up to {min(len(sample), show_n)} matching products:")
            lines.append("| Name | Price | Availability |")
            lines.append("| --- | --- | --- |")
            for node in sample[:show_n]:
                props = node.properties or {}
                lines.append(
                    f"| {props.get('name') or node.id} | {props.get('price') or '—'} | "
                    f"{props.get('availability') or '—'} |"
                )
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
    total = cypher_count_products(platform, graph_id, filters)
    sample = cypher_sample_products(platform, graph_id, filters, limit=5)
    if total is None:
        matched = match_neo4j_products(platform, principal, graph_id, filters)
        total = len(matched)
        sample = matched[:5]
        ranked = _profile_nodes(platform, graph_id, matched, fields=analysis_fields)
    else:
        ranked = {}
        fields_pre = tuple(analysis_fields) if analysis_fields else _ANALYSIS_FIELDS
        for field in fields_pre:
            grouped = cypher_group_by(platform, graph_id, str(field), filters)
            if grouped is not None:
                ranked[field] = grouped
        sample = [
            node
            for node in (sample or [])
            if allows(principal, node.properties.get("classification") or "internal")
        ]
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
    # Chart-only fields vs table-only fields — never both for the same values.
    chart_fields = {"material", "color", "availability", "gender"}
    for field in fields:
        choices = [item for item in (ranked.get(field) or []) if _usable_value(item[0])][:top_n]
        if not choices:
            continue
        field_label = field.replace("_", " ")
        lines.append(f"### {field_label}")
        if field in chart_fields:
            # Multi-label fields → bar; exclusive partitions → pie.
            chart_type = "pie" if field in {"gender", "availability"} else "bar"
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
    for node in (sample or [])[:5]:
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
    if availability:
        want = _avail_token(str(availability))
        have = _avail_token(str(props.get("availability") or ""))
        if have != want and not _has_related(platform, graph_id, node.id, "HAS_AVAILABILITY", str(availability)):
            return False
    price_max = filters.get("price_max")
    if price_max is not None and str(price_max).strip() != "":
        raw = str(props.get("price") or "")
        digits = re.sub(r"[^\d.]", "", raw.replace(",", ""))
        try:
            if not digits or float(digits) >= float(price_max):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _has_related(platform, graph_id: str, product_id: str, rel_type: str, name: str) -> bool:
    target = _avail_token(name) if rel_type == "HAS_AVAILABILITY" else normalize_name(name)
    try:
        edges = platform.graph.find_edges(graph=graph_id, rel_type=rel_type, start_id=product_id, include_deleted=False)
    except Exception:
        return False
    for edge in edges:
        end = platform.graph.get_node(edge.end_id, include_deleted=False)
        if not end:
            continue
        got = str(end.properties.get("name") or "")
        if rel_type == "HAS_AVAILABILITY":
            if _avail_token(got) == target:
                return True
        elif normalize_name(got) == target:
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
    key = normalize_name(folded)
    if key in _JUNK_MATERIAL_LABELS:
        return False
    return not any(part in key for part in _JUNK_MATERIAL_PARTS)


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
    """Breakdown with a chart plus a share table for analysis-style asks."""

    # Material/color mixes need enough slices for a useful chart.
    default_n = 12 if field in {"material", "color", "brand"} else 10
    top_n = max(int(limit or default_n), 1)
    shown = [(name, count) for name, count in groups if _usable_value(name)][:top_n]
    if not shown:
        return [f"No {field.replace('_', ' ')} breakdown."]
    field_label = field.replace("_", " ")
    lines = [f"{field_label} breakdown (highest count first):"]
    if field in {"material", "color", "gender", "availability", "brand"}:
        # Material/color are multi-label (shares can sum past 100%) — bar, not pie.
        chart_type = "pie" if field in {"gender", "availability"} else "bar"
        lines.extend(
            _chart_lines(
                chart_type,
                field_label.title() + " mix",
                [name for name, _count in shown],
                [count for _name, count in shown],
            )
        )
        lines.append("")
        lines.append("| Name | Count | Share % |")
        lines.append("| --- | ---: | ---: |")
        for name, count in shown:
            share = round(100.0 * count / total, 1) if total else 0.0
            lines.append(f"| {name} | {count} | {share} |")
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
