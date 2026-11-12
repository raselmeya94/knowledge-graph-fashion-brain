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


def _party_match_token(value: str) -> str:
    """Compact brand/buyer key so C&A / c-and-a / C and A all become ``canda``."""

    text = str(value or "").strip().casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


def _party_norm_cypher(expr: str) -> str:
    """Cypher form of ``_party_match_token`` for stored brand/buyer strings."""

    return (
        "replace(replace(replace(replace("
        f"toLower(coalesce({expr}, '')), '&', 'and'), '-', ''), ' ', ''), '_', '')"
    )


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


# Exact keys that are never merchandising facets (identity, free text, price scalars).
_NON_FACET_ATTRS = frozenset(
    {
        "id",
        "product_id",
        "name",
        "title",
        "description",
        "url",
        "image",
        "image_url",
        "images",
        "content_hash",
        "prov_record_id",
        "source_id",
        "lifecycle",
        "classification",
        "composition",
        "composition_detail",
        "category_path",
        "price",
        "original_price",
        "price_num",
        "original_price_num",
        "price_min",
        "price_max",
        "currency",
        "product_name",
        "normalized_name",
        "reference",
        "display_reference",
        "graph",
        "source",
        "domain",
    }
)
# Token fragments that mark scrape / warehouse / infra columns (public answers must not use these).
_META_ATTR_TOKENS = frozenset(
    {
        "url",
        "uri",
        "href",
        "scrape",
        "scraped",
        "crawl",
        "crawled",
        "domain",
        "graph",
        "source",
        "hash",
        "prov",
        "reference",
        "snapshot",
        "enrich",
        "enriched",
        "image",
        "images",
        "media",
        "thumbnail",
        "payload",
        "timestamp",
        "html",
        "json",
        "raw",
        "meta",
        "catalog",
    }
)


def is_analysis_facet_attr(key: str) -> bool:
    """True for product-mix attributes safe to show publicly — schema-driven, not a fashion allow-list.

    Drops identity, scrape/ops metadata, and warehouse plumbing by name shape.
    """

    folded = str(key or "").strip().casefold().replace(" ", "_").replace("-", "_")
    if not folded or folded in _NON_FACET_ATTRS:
        return False
    if folded.endswith(("_id", "_ids", "_url", "_uri", "_hash", "_path", "_html", "_json", "_at")):
        return False
    if folded.startswith(("scraped", "crawl", "pdp_", "raw_", "meta_", "prov_")):
        return False
    # Inventory counters / ops tallies — not assortment mix.
    if folded.endswith("_count") or folded.endswith("_counts"):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", folded))
    if tokens & _META_ATTR_TOKENS:
        return False
    return True


def _facet_worth_showing(
    choices: list[tuple[str, int]],
    *,
    total: int,
) -> bool:
    """Drop empty, all-unspecified, near-unique ID, or URL/timestamp-like slices."""

    if not choices or total <= 0:
        return False
    top_name, top_count = choices[0]
    top_key = normalize_name(str(top_name))
    share = top_count / float(total)
    if top_key in {"", "unspecified", "null", "none", "n/a", "na"} and share >= 0.85:
        return False
    sample = [str(name or "") for name, _ in choices[:5]]
    if any(text.casefold().startswith("http") for text in sample):
        return False
    if any(re.search(r"\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}", text) for text in sample):
        return False
    # Per-row identifiers: top bucket is tiny and almost every row is distinct.
    if top_count <= 2 and share < 0.02:
        return False
    return True


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
    """Pick preferred graph if it is a product graph; else first registered product graph.

    Do not treat the first registry id as "the ecom graph" — order is registry order.
    For analysis/asks with filters, prefer ``pick_product_graph_for_ask``.
    """

    ids = registry_product_graphs(platform)
    if preferred and preferred in ids:
        return preferred
    if preferred and preferred in {"shared", "master", "news"}:
        return ids[0] if ids else None
    return ids[0] if ids else None


def pick_product_graph_for_ask(
    platform,
    *,
    preferred: str | None = None,
    graph_ids: list[str] | None = None,
    filters: dict | None = None,
) -> str | None:
    """Choose one product graph for a catalog ask without assuming e-commerce.

    Prefer explicit ids, then filter/capability match (buyer→PLM-like, brand→brand-capable).
    """

    try:
        from fashion_brain.analytics.engine import pick_analysis_graphs

        picked = pick_analysis_graphs(
            platform,
            preferred=[preferred] if preferred else None,
            graph_id=preferred,
            graph_ids=graph_ids,
            filters=filters,
        )
        return picked[0] if picked else None
    except Exception:
        return default_product_graph(platform, preferred)


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


def resolve_analysis_fields(
    platform,
    graph_id: str | None,
    override: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Product-mix facets from live schema (or planner override), minus scrape/meta columns.

    Single-field overrides (user said \"material wise\") are kept even if unusual.
    """

    if override is not None and len(override) == 1:
        only = str(override[0] or "").strip()
        return [only] if only else []
    if not graph_id and not override:
        return []
    keys: list[str] = []
    if override:
        keys = [str(item) for item in override if str(item or "").strip()]
    else:
        try:
            from knowledge_graph.retrieval.metadata import metadata_overview

            overview = metadata_overview(platform, graph_id) or {}
        except Exception:
            overview = {}
        schema = overview.get("schema") if isinstance(overview, dict) else None
        product = schema.get("Product") if isinstance(schema, dict) else None
        raw_attrs = (product or {}).get("attributes") if isinstance(product, dict) else None
        if isinstance(raw_attrs, dict):
            keys = [str(k) for k in raw_attrs.keys()]
        elif isinstance(raw_attrs, (list, tuple)):
            keys = [str(k) for k in raw_attrs]
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        folded = key.strip().casefold().replace(" ", "_").replace("-", "_")
        if not folded or folded in seen or not is_analysis_facet_attr(folded):
            continue
        seen.add(folded)
        out.append(folded)
    # Relationship-backed facets first (full _REL_FIELDS map), then remaining schema keys.
    rel_order = [name for name in out if name in _REL_FIELDS]
    rest = [name for name in out if name not in set(rel_order)]
    return rel_order + rest


def clean_product_filters(filters: dict) -> dict:
    cleaned = {key: value for key, value in (filters or {}).items() if value not in (None, "")}
    cleaned.pop("parent_of", None)  # structural lookup — handled in brain, not product WHERE
    # Intention already remaps buyer→brand for brand-only graphs. Keep buyer for PLM/HAS_BUYER.
    cleaned.pop("season_code", None)
    cleaned.pop("upcoming", None)
    # season=2028 → season_year; season=winter → season_name
    season = cleaned.pop("season", None)
    if season not in (None, ""):
        token = str(season).strip()
        if re.fullmatch(r"(19|20)\d{2}", token):
            cleaned.setdefault("season_year", token)
        else:
            named = normalize_name(token)
            if named in {"winter", "spring", "summer", "autumn", "fall"}:
                cleaned.setdefault("season_name", "autumn" if named == "fall" else named)
    # Drop filters whose value is just the field name itself (garment=garment, material=materials).
    for key, value in list(cleaned.items()):
        want = normalize_name(str(value))
        field = normalize_name(str(key))
        if not want:
            cleaned.pop(key, None)
            continue
        if want == field or want == field + "s" or want + "s" == field:
            cleaned.pop(key, None)
    # Brand/buyer names mis-tagged as garment must not filter categories.
    for party_key in ("brand", "buyer"):
        party = cleaned.get(party_key)
        if not party:
            continue
        party_key_norm = normalize_name(str(party))
        for key in ("garment", "garment_type", "category"):
            value = cleaned.get(key)
            if value and normalize_name(str(value)) == party_key_norm:
                cleaned.pop(key, None)
    # Drop ranking/metric question phrasing pasted into facet filters.
    try:
        from fashion_brain.ask import _reliable_filter

        for key in ("garment", "garment_type", "category", "family", "subfamily", "material", "color", "section"):
            value = cleaned.get(key)
            if value is not None and not _reliable_filter(str(value)):
                cleaned.pop(key, None)
    except Exception:
        pass
    return cleaned


def _session(platform):
    factory = getattr(platform.graph, "_session", None)
    return factory if callable(factory) else None


def _brand_where_cypher() -> str:
    """Brand match on product.brand or HAS_BRAND (C&A ≡ c-and-a ≡ C and A)."""

    return (
        "("
        f"{_party_norm_cypher('p.brand')} = $brand "
        "OR EXISTS { MATCH (p)-[:HAS_BRAND]->(b:Entity) "
        f"WHERE {_party_norm_cypher('b.name')} = $brand }}"
        ")"
    )


def _buyer_where_cypher() -> str:
    """Buyer match on product.buyer or HAS_BUYER (same compact party matching)."""

    return (
        "("
        f"{_party_norm_cypher('p.buyer')} = $buyer "
        "OR EXISTS { MATCH (p)-[:HAS_BUYER]->(b:Entity) "
        f"WHERE {_party_norm_cypher('b.name')} = $buyer }}"
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
        "season_year",
        "season_name",
    }
)

# Product properties that are not always relationship facets (PLM composition/gsm/…).
_PROP_FILTER_FIELDS = frozenset({"composition", "gsm", "weight", "unit", "cuttable_width"})


def _rel_filter_clause(field: str, param: str) -> str:
    """Match a facet via product property and/or its ontology relationships."""

    rels = _REL_FIELDS.get(field) or ()
    prop = (
        f"toLower(coalesce(toString(p.`{field}`),'')) = ${param} "
        f"OR toLower(coalesce(toString(p.`{field}`),'')) CONTAINS ${param}"
    )
    if not rels:
        return f"({prop})"
    rel_alt = "|".join(rels)
    return (
        f"("
        f"{prop} "
        f"OR EXISTS {{ MATCH (p)-[:{rel_alt}]->(x:Entity) "
        f"WHERE toLower(coalesce(x.name,'')) = ${param} "
        f"OR toLower(coalesce(x.name,'')) CONTAINS ${param} }}"
        f")"
    )


def _filter_where(filters: dict, params: dict) -> list[str] | None:
    """Build WHERE clauses + params for product filters. None = unsupported combo.

    Known retail filters stay special-cased. PLM/spec facet keys (plm_*, gender, …)
    are accepted via `_REL_FIELDS` so specification anchoring can narrow the set.
    """

    filters = clean_product_filters(filters)
    # product_id is resolved by dedicated timeline helpers, not Product WHERE facets.
    filters.pop("product_id", None)
    allowed = _KNOWN_FILTERS | frozenset(_REL_FIELDS) | _PROP_FILTER_FIELDS
    if set(filters) - allowed:
        return None
    where = ["coalesce(p.lifecycle,'active') <> 'deleted'", "p.graph = $g"]
    handled: set[str] = set()
    brand = filters.get("brand")
    buyer = filters.get("buyer")
    # Same party in both fields (e.g. brand=C&A and buyer=C&A) must not AND to empty.
    if brand and buyer and _party_match_token(str(brand)) == _party_match_token(str(buyer)):
        handled.add("buyer")
        buyer = None
    if brand:
        params["brand"] = _party_match_token(str(brand))
        where.append(_brand_where_cypher())
        handled.add("brand")
    if buyer:
        params["buyer"] = _party_match_token(str(buyer))
        where.append(_buyer_where_cypher())
        handled.add("buyer")
    # Soft category/garment match for retail asks. garment_type is handled precisely below
    # so specification anchoring can apply category AND garment_type together.
    category = filters.get("category") or filters.get("garment")
    if category:
        params["category"] = _cypher_token(str(category)).replace("-", " ")
        where.append(
            "("
            "toLower(coalesce(p.category_path,'')) CONTAINS $category "
            "OR toLower(coalesce(p.name,'')) CONTAINS $category "
            "OR EXISTS { MATCH (p)-[:BELONGS_TO_CATEGORY|HAS_FAMILY|HAS_SUB_CATEGORY|HAS_GARMENT_TYPE|"
            "HAS_PLM_SUB_SUB_CATEGORY]->(c:Entity) "
            "WHERE toLower(coalesce(c.name,'')) CONTAINS $category }"
            ")"
        )
        handled.update({"category", "garment"})
    season_year = filters.get("season_year")
    if season_year:
        params["season_year"] = str(season_year).strip()
        where.append(
            "EXISTS { MATCH (p)-[:IN_SEASON_YEAR]->(y:Entity) "
            "WHERE toString(coalesce(y.name, '')) CONTAINS $season_year "
            "OR toString(coalesce(p.season_year, '')) CONTAINS $season_year }"
        )
        handled.add("season_year")
    season_name = filters.get("season_name")
    if season_name:
        params["season_name"] = _cypher_token(str(season_name)).replace("-", " ")
        where.append(
            "("
            "EXISTS { MATCH (p)-[:HAS_EVENT_TAG]->(e:Entity) "
            "WHERE toLower(coalesce(e.name,'')) = $season_name }"
            "OR toLower(coalesce(p.style_reference,'')) CONTAINS $season_name "
            ")"
        )
        handled.add("season_name")
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
        handled.add("section")
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
        handled.add("material")
    availability = filters.get("availability")
    if availability:
        params["availability"] = _avail_token(str(availability))
        where.append(_availability_where_cypher())
        handled.add("availability")
    color = filters.get("color")
    if color:
        params["color"] = _cypher_token(str(color)).replace("-", " ")
        where.append(
            "("
            "EXISTS { MATCH (p)-[:HAS_COLOR]->(c:Entity) "
            "WHERE toLower(coalesce(c.name,'')) = $color }"
            ")"
        )
        handled.add("color")
    currency = filters.get("currency")
    if currency:
        params["currency"] = _cypher_token(str(currency)).replace("-", "")
        where.append("replace(toLower(coalesce(p.currency,'')),' ','') CONTAINS $currency")
        handled.add("currency")
    price_max = filters.get("price_max")
    if price_max is not None and str(price_max).strip() != "":
        try:
            params["price_max"] = float(price_max)
        except (TypeError, ValueError):
            return None
        where.append(f"{_price_num_cypher()} IS NOT NULL AND {_price_num_cypher()} < $price_max")
        handled.add("price_max")
    price_min = filters.get("price_min")
    if price_min is not None and str(price_min).strip() != "":
        try:
            params["price_min"] = float(price_min)
        except (TypeError, ValueError):
            return None
        where.append(f"{_price_num_cypher()} IS NOT NULL AND {_price_num_cypher()} >= $price_min")
        handled.add("price_min")

    # Remaining facet keys (PLM hierarchy, garment_type, gender, AI tags, composition, …).
    for key, value in filters.items():
        if key in handled or value in (None, ""):
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            return None
        param = f"f_{key}"
        params[param] = _cypher_token(str(value)).replace("-", " ")
        if key in _REL_FIELDS:
            where.append(_rel_filter_clause(str(key), param))
            handled.add(key)
            continue
        if key in _PROP_FILTER_FIELDS:
            where.append(
                f"toLower(coalesce(toString(p.`{key}`),'')) CONTAINS ${param}"
            )
            handled.add(key)
            continue
        return None
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


def cypher_price_overview(platform, graph_id: str, filters: dict) -> dict[str, float | int] | None:
    """Overview metrics for analysis: count + price avg/min/max when price_num exists."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    price = _price_num_cypher()
    query = (
        "MATCH (p:Entity:Product) WHERE "
        + " AND ".join(where)
        + f" RETURN count(DISTINCT p) AS total_products, "
        f"avg({price}) AS average_price, min({price}) AS min_price, max({price}) AS max_price"
    )
    try:
        import time

        from fashion_brain.stage_timing import record_cypher, record_stage

        started = time.perf_counter()
        with session_factory() as session:
            row = session.run(query, params).single()
        ms = (time.perf_counter() - started) * 1000.0
        record_stage("cypher_exec", ms, kind="price_overview", rows=1)
        record_cypher(query, params=params, rows=1, ms=ms)
        if not row:
            return None
        total = int(row["total_products"] or 0)

        def _num(key: str) -> float | None:
            value = row.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return {
            "total_products": total,
            "average_price": _num("average_price"),
            "min_price": _num("min_price"),
            "max_price": _num("max_price"),
        }
    except Exception:
        return None


def cypher_resolve_product_id(platform, graph_id: str, ref: str) -> str | None:
    """Resolve a full id, trailing sku, or numeric token to a Product.id."""

    session_factory = _session(platform)
    if not session_factory or not ref:
        return None
    query = (
        "MATCH (p:Entity:Product) WHERE p.graph = $g "
        "AND coalesce(p.lifecycle,'active') <> 'deleted' "
        "AND ("
        "toLower(p.id) = toLower($ref) OR p.id ENDS WITH $ref OR "
        "toLower(coalesce(p.source_id,'')) = toLower($ref) OR "
        "toLower(coalesce(p.sku,'')) = toLower($ref)"
        ") "
        "RETURN p.id AS id LIMIT 1"
    )
    try:
        with session_factory() as session:
            row = session.run(query, {"g": graph_id, "ref": str(ref)}).single()
        return str(row["id"]) if row and row.get("id") else None
    except Exception:
        return None


def cypher_count_nodes(platform, graph_id: str, label: str) -> int | None:
    """Count temporal nodes for one product graph."""

    session_factory = _session(platform)
    if not session_factory or label not in {"PriceChangeEvent", "AvailabilitySnapshot"}:
        return None
    query = f"MATCH (n:{label}) WHERE n.graph = $g RETURN count(n) AS c"
    try:
        with session_factory() as session:
            row = session.run(query, {"g": graph_id}).single()
        return int(row["c"] or 0) if row else 0
    except Exception:
        # Some syncs omit n.graph on event nodes — fall back to unscoped count.
        try:
            with session_factory() as session:
                row = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
            return int(row["c"] or 0) if row else 0
        except Exception:
            return None


def cypher_price_history(
    platform,
    graph_id: str,
    product_id: str,
    *,
    limit: int = 100,
) -> list[dict] | None:
    """Return PriceChangeEvent timeline for one product (newest last)."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    limit = max(1, min(int(limit or 100), 500))
    resolved = cypher_resolve_product_id(platform, graph_id, product_id) or product_id
    query = (
        "MATCH (p:Entity:Product {id: $pid})-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE p.graph = $g AND e.detected_at IS NOT NULL "
        "RETURN e.detected_at AS at, e.old_price_num AS old_price, e.new_price_num AS new_price, "
        "e.delta_pct AS delta_pct, e.direction AS direction, e.currency AS currency, "
        "e.currency_changed AS currency_changed, p.name AS name, p.id AS product_id "
        "ORDER BY e.detected_at ASC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            rows = list(session.run(query, {"pid": resolved, "g": graph_id}))
        return [dict(row) for row in rows]
    except Exception:
        return None


def cypher_price_events_by_brand(platform, graph_id: str, *, limit: int = 20) -> list[dict] | None:
    session_factory = _session(platform)
    if not session_factory:
        return None
    limit = max(1, min(int(limit or 20), 50))
    query = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE p.graph = $g AND coalesce(p.lifecycle,'active') <> 'deleted' "
        "OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Entity) "
        "WITH coalesce(b.name, p.brand, '(unspecified)') AS brand, count(e) AS events, "
        "count(DISTINCT p) AS products "
        "RETURN brand, events, products ORDER BY events DESC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            return [dict(row) for row in session.run(query, {"g": graph_id})]
    except Exception:
        return None


def cypher_availability_events_by_brand(platform, graph_id: str, *, limit: int = 20) -> list[dict] | None:
    """Rank brands by AvailabilitySnapshot volume (availability change events)."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    limit = max(1, min(int(limit or 20), 50))
    query = (
        "MATCH (p:Entity:Product)-[:HAS_AVAILABILITY_SNAPSHOT]->(s:AvailabilitySnapshot) "
        "WHERE p.graph = $g AND coalesce(p.lifecycle,'active') <> 'deleted' "
        "OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Entity) "
        "WITH coalesce(b.name, p.brand, s.brand, '(unspecified)') AS brand, "
        "count(s) AS events, count(DISTINCT p) AS products "
        "RETURN brand, events, products ORDER BY events DESC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            return [dict(row) for row in session.run(query, {"g": graph_id})]
    except Exception:
        return None


def cypher_recent_availability_changes(
    platform,
    graph_id: str,
    filters: dict | None = None,
    *,
    limit: int = 20,
) -> list[dict] | None:
    """Most recent AvailabilitySnapshot rows with product + brand context."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    limit = max(1, min(int(limit or 20), 50))
    params: dict = {"g": graph_id}
    where = _filter_where(clean_product_filters(dict(filters or {})), params)
    if where is None:
        return None
    query = (
        "MATCH (p:Entity:Product)-[:HAS_AVAILABILITY_SNAPSHOT]->(s:AvailabilitySnapshot) "
        "WHERE " + " AND ".join(where) + " "
        "OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Entity) "
        "RETURN p.id AS product_id, "
        "coalesce(p.name, p.title, p.id) AS name, "
        "coalesce(b.name, p.brand, s.brand, '(unspecified)') AS brand, "
        "coalesce(s.status, s.name, '—') AS status, "
        "s.observed_at AS observed_at, "
        "s.in_stock_count AS in_stock_count, "
        "s.out_stock_count AS out_stock_count "
        "ORDER BY s.observed_at DESC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            return [dict(row) for row in session.run(query, params)]
    except Exception:
        return None


def append_neo4j_availability_history_answer(
    platform,
    principal: Principal,
    graph_id: str,
    question: str,
    limit: int | None,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
    filters: dict | None = None,
) -> bool:
    """Templated AvailabilitySnapshot answers (counts / brand rankings / recent)."""

    try:
        from fashion_brain.ask import availability_history_mode, temporal_count_entity
    except Exception:
        return False
    mode = availability_history_mode(question)
    if mode is None and temporal_count_entity(question) != "AvailabilitySnapshot":
        return False
    if mode == "count_snapshots" or temporal_count_entity(question) == "AvailabilitySnapshot":
        n = cypher_count_nodes(platform, graph_id, "AvailabilitySnapshot")
        if n is None:
            return False
        lines.append(f"There are {n} AvailabilitySnapshot nodes stored in the knowledge graph.")
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="availability_snapshot_count",
                title="AvailabilitySnapshot count",
                excerpt=f"count={n}",
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        return True
    if mode == "recent":
        rows = cypher_recent_availability_changes(
            platform,
            graph_id,
            filters,
            limit=max(int(limit or 15), 1),
        )
        if not rows:
            lines.append("No recent availability change snapshots were found.")
            caveats.append(
                "Recent availability uses AvailabilitySnapshot.observed_at from product_change_events."
            )
            return True
        lines.append("Products with recent availability changes:")
        lines.append("| When | Brand | Product | Status | In stock |")
        lines.append("| --- | --- | --- | --- | ---: |")
        for row in rows:
            when = str(row.get("observed_at") or "—")
            if len(when) > 19:
                when = when[:19]
            lines.append(
                f"| {when} | {_display_label(str(row.get('brand') or '—'))} | "
                f"{row.get('name') or row.get('product_id') or '—'} | "
                f"{row.get('status') or '—'} | "
                f"{row.get('in_stock_count') if row.get('in_stock_count') is not None else '—'} |"
            )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="recent_availability_changes",
                title="Recent availability changes",
                excerpt="; ".join(
                    f"{r.get('product_id')}={r.get('status')}" for r in rows[:5]
                ),
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        caveats.append(
            "Ordered by AvailabilitySnapshot.observed_at descending; one row per snapshot event."
        )
        return True
    # Default ranking shape for "which brands have the most availability changes"
    rows = cypher_availability_events_by_brand(
        platform, graph_id, limit=max(int(limit or 10), 1)
    )
    if not rows:
        lines.append("No availability change snapshots were found to rank by brand.")
        caveats.append("Availability changes use AvailabilitySnapshot nodes from product_change_events.")
        return True
    total = sum(int(r.get("events") or 0) for r in rows)
    lines.append("Availability changes by brand (AvailabilitySnapshot counts):")
    lines.append("| Brand | Availability changes | Products | Share % |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in rows:
        events = int(row.get("events") or 0)
        share = (100.0 * events / total) if total else 0.0
        lines.append(
            f"| {_display_label(str(row.get('brand') or '—'))} | {events} | "
            f"{int(row.get('products') or 0)} | {share:.1f} |"
        )
    evidence.append(
        EvidenceItem(
            source_id=graph_id,
            record_id="availability_events_by_brand",
            title="Availability changes by brand",
            excerpt="; ".join(f"{r.get('brand')}={r.get('events')}" for r in rows[:5]),
            source_path="product_change_events",
            verification_status="observed",
            graph=graph_id,
            node_id=None,
            synthetic=False,
        )
    )
    caveats.append(
        "Ranked by AvailabilitySnapshot count per brand; not current InStock product totals."
    )
    return True


def cypher_price_trend_by_category(platform, graph_id: str, *, limit: int = 20) -> list[dict] | None:
    session_factory = _session(platform)
    if not session_factory:
        return None
    limit = max(1, min(int(limit or 20), 50))
    query = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE p.graph = $g AND coalesce(p.lifecycle,'active') <> 'deleted' "
        "AND coalesce(e.currency_changed, false) = false AND e.delta_pct IS NOT NULL "
        "OPTIONAL MATCH (p)-[:BELONGS_TO_CATEGORY|HAS_FAMILY|HAS_SUB_CATEGORY]->(c:Entity) "
        "WITH coalesce(c.name, p.category_path, '(unspecified)') AS category, "
        "avg(e.delta_pct) AS avg_delta_pct, count(e) AS events, count(DISTINCT p) AS products "
        "RETURN category, avg_delta_pct, events, products "
        "ORDER BY avg_delta_pct ASC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            rows = [dict(row) for row in session.run(query, {"g": graph_id})]
        for row in rows:
            if row.get("avg_delta_pct") is not None:
                try:
                    row["avg_delta_pct"] = float(row["avg_delta_pct"])
                except (TypeError, ValueError):
                    row["avg_delta_pct"] = None
        return rows
    except Exception:
        return None


def cypher_avg_price_by_brand(platform, graph_id: str, filters: dict, *, limit: int = 20) -> list[dict] | None:
    """Current-state average price_num by brand (not historical deltas)."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    # Drop multi-party brand so each side can be computed by caller if needed.
    where = _filter_where({k: v for k, v in (filters or {}).items() if k != "brand" or "|" not in str(v)}, params)
    if where is None:
        return None
    price = _price_num_cypher()
    limit = max(1, min(int(limit or 20), 50))
    query = (
        "MATCH (p:Entity:Product) WHERE "
        + " AND ".join(where)
        + " OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Entity) "
        + f"WITH coalesce(b.name, p.brand, '(unspecified)') AS brand, avg({price}) AS avg_price, "
        "count(p) AS products "
        "WHERE avg_price IS NOT NULL "
        "RETURN brand, avg_price, products ORDER BY avg_price DESC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            rows = [dict(row) for row in session.run(query, params)]
        for row in rows:
            if row.get("avg_price") is not None:
                row["avg_price"] = float(row["avg_price"])
        return rows
    except Exception:
        return None


def cypher_price_movers(
    platform,
    graph_id: str,
    filters: dict,
    *,
    direction: str = "up",
    limit: int = 25,
) -> list[dict] | None:
    """Largest same-currency price moves."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    direction = str(direction or "up").strip().lower()
    if direction not in {"up", "down"}:
        direction = "up"
    params["dir"] = direction
    limit = max(1, min(int(limit or 25), 100))
    query = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND e.direction = $dir AND coalesce(e.currency_changed, false) = false "
        "AND e.detected_at IS NOT NULL "
        "OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Brand) "
        "RETURN p.id AS product_id, p.name AS name, b.name AS brand, "
        "e.old_price_num AS old_price, e.new_price_num AS new_price, "
        "e.delta_pct AS delta_pct, e.detected_at AS detected_at, e.currency AS currency "
        "ORDER BY abs(coalesce(e.delta_pct, 0)) DESC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            rows = list(session.run(query, params))
        return [dict(row) for row in rows]
    except Exception:
        return None


def price_history_mode(question: str) -> str | None:
    """Classify catalog price-history asks.

    Returns:
      - ``count_events``: how many PriceChangeEvent nodes
      - ``product``: timeline / deltas for one product id
      - ``events_by_brand``: brands ranked by change-event volume
      - ``trend_by_category``: category-wise average delta trends
      - ``trend``: average / overall change summary
      - ``movers_up`` / ``movers_down``: ranked individual movers
      - ``None``: not a price-history ask
    """

    text = normalize_name(question or "")
    try:
        from fashion_brain.ask import product_ref_from_question, temporal_count_entity

        if temporal_count_entity(question) == "PriceChangeEvent":
            return "count_events"
        if product_ref_from_question(question) and re.search(
            r"\b(price|change|changed|old|new|percent|pct|history|historical|over time)\b",
            text,
        ):
            return "product"
    except Exception:
        pass
    cheaper_or_currency = bool(
        re.search(r"\b(cheaper|currency\s+switch|currency\s+switches|same[- ]currency)\b", text)
    )
    if not re.search(r"\bprices?\b", text) and not cheaper_or_currency and not re.search(
        r"\bpricechangeevent\b", text
    ):
        return None
    has_change = bool(
        re.search(
            r"\b(change|changes|changed|trend|trends|trending|history|historical|"
            r"fluctuation|fluctuations|move|moves|moved|mover|movers|"
            r"decrease|decreases|decreased|"
            r"increase|increases|increased|drop|drops|dropped|rise|rises|rose|rising|"
            r"fallen|falling|cheaper|expensive|over time)\b",
            text,
        )
    )
    if not has_change and not re.search(r"\b(largest|biggest|highest|lowest)\s+price\b", text):
        return None
    # Unsupported geo/time scopes stay trend-mode; answer path emits an honest caveat.
    if re.search(r"\b(most|largest|biggest|highest)\b", text) and re.search(
        r"\b(brands?|buyers?)\b", text
    ) and re.search(r"\b(events?|changes?|moves?)\b", text):
        return "events_by_brand"
    if re.search(r"\b(categor(?:y|ies))\b", text) and re.search(
        r"\b(trend|trends|decreasing|decrease|drop|dropping)\b", text
    ):
        return "trend_by_category"
    # Aggregate / average trend asks take priority over mover ranking.
    if re.search(r"\b(average|avg|mean|median|overall|typical|aggregate|summary)\b", text):
        return "trend"
    if re.search(r"\b(trend|trends|trending)\b", text) and not re.search(
        r"\b(largest|biggest|highest|lowest|top|most)\b", text
    ):
        return "trend"
    if re.search(
        r"\b(decrease|decreases|decreased|drop|drops|dropped|cheaper|fallen|falling|down)\b",
        text,
    ):
        return "movers_down"
    if re.search(
        r"\b(increase|increases|increased|rise|rises|rose|rising|expensive|up)\b",
        text,
    ):
        return "movers_up"
    if re.search(r"\b(largest|biggest|highest|lowest|top|most)\b", text):
        return "movers_down"
    return "trend"


def price_mover_direction(question: str) -> str | None:
    """Back-compat: 'up'/'down' for mover asks only (not average-trend asks)."""

    mode = price_history_mode(question)
    if mode == "movers_up":
        return "up"
    if mode == "movers_down":
        return "down"
    return None


def cypher_price_trend_timeseries(
    platform, graph_id: str, filters: dict, *, limit: int = 24
) -> list[dict] | None:
    """Point-wise same-currency delta_pct series for an analytics trend line.

    Prefers individual events when there are enough points; otherwise daily, then
    weekly buckets. Avoids a flat two-endpoint average line when denser data exists.
    """

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    limit = max(8, min(int(limit or 24), 40))
    base = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND coalesce(e.currency_changed, false) = false "
        "AND e.delta_pct IS NOT NULL AND e.detected_at IS NOT NULL "
    )
    week_query = (
        base
        + "WITH date(datetime(e.detected_at)) AS d, e.delta_pct AS pct "
        "WITH d.year AS y, d.week AS w, avg(pct) AS avg_delta_pct, "
        "min(pct) AS min_delta_pct, max(pct) AS max_delta_pct, count(*) AS events "
        "RETURN y + '-W' + (CASE WHEN w < 10 THEN '0' ELSE '' END) + toString(w) AS period, "
        "avg_delta_pct, min_delta_pct, max_delta_pct, events "
        "ORDER BY y ASC, w ASC "
        f"LIMIT {limit}"
    )
    day_query = (
        base
        + "WITH substring(toString(e.detected_at), 0, 10) AS period, "
        "avg(e.delta_pct) AS avg_delta_pct, "
        "min(e.delta_pct) AS min_delta_pct, max(e.delta_pct) AS max_delta_pct, "
        "count(e) AS events "
        "RETURN period, avg_delta_pct, min_delta_pct, max_delta_pct, events "
        "ORDER BY period ASC "
        f"LIMIT {limit}"
    )
    event_query = (
        base
        + "RETURN substring(toString(e.detected_at), 5, 11) AS period, "
        "e.delta_pct AS avg_delta_pct, e.delta_pct AS min_delta_pct, "
        "e.delta_pct AS max_delta_pct, 1 AS events "
        "ORDER BY e.detected_at ASC "
        f"LIMIT {limit}"
    )

    def _rows(session, query: str) -> list[dict]:
        out: list[dict] = []
        for row in session.run(query, params):
            avg = row.get("avg_delta_pct")
            try:
                avg_f = float(avg) if avg is not None else None
            except (TypeError, ValueError):
                avg_f = None
            if avg_f is None:
                continue
            def _f(key: str) -> float | None:
                raw = row.get(key)
                try:
                    return float(raw) if raw is not None else None
                except (TypeError, ValueError):
                    return None

            out.append(
                {
                    "period": str(row.get("period") or ""),
                    "avg_delta_pct": round(avg_f, 3),
                    "min_delta_pct": round(_f("min_delta_pct"), 3) if _f("min_delta_pct") is not None else round(avg_f, 3),
                    "max_delta_pct": round(_f("max_delta_pct"), 3) if _f("max_delta_pct") is not None else round(avg_f, 3),
                    "events": int(row.get("events") or 0),
                }
            )
        return out

    try:
        with session_factory() as session:
            events: list[dict] = []
            days: list[dict] = []
            weeks: list[dict] = []
            try:
                events = _rows(session, event_query)
            except Exception:
                events = []
            try:
                days = _rows(session, day_query)
            except Exception:
                days = []
            try:
                weeks = _rows(session, week_query)
            except Exception:
                weeks = []
            # Dense event trail when we have real point-to-point motion.
            if len(events) >= 3:
                return events
            if len(days) >= 3:
                return days
            if len(weeks) >= 2:
                return weeks
            if len(days) >= 2:
                return days
            if len(events) >= 2:
                return events
        return events or days or weeks
    except Exception:
        return None


def cypher_price_trend_summary(platform, graph_id: str, filters: dict) -> dict | None:
    """Aggregate same-currency PriceChangeEvent stats for the filtered product set."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    query = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND coalesce(e.currency_changed, false) = false "
        "AND e.delta_pct IS NOT NULL "
        "RETURN count(e) AS change_events, "
        "count(DISTINCT p) AS products_changed, "
        "avg(e.delta_pct) AS avg_delta_pct, "
        "min(e.delta_pct) AS min_delta_pct, "
        "max(e.delta_pct) AS max_delta_pct, "
        "sum(CASE WHEN e.direction = 'up' THEN 1 ELSE 0 END) AS ups, "
        "sum(CASE WHEN e.direction = 'down' THEN 1 ELSE 0 END) AS downs, "
        "sum(CASE WHEN e.direction = 'unchanged' THEN 1 ELSE 0 END) AS unchanged"
    )
    try:
        with session_factory() as session:
            row = session.run(query, params).single()
        if not row:
            return None
        out = dict(row)
        for key in ("avg_delta_pct", "min_delta_pct", "max_delta_pct"):
            value = out.get(key)
            if value is not None:
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = None
        for key in ("change_events", "products_changed", "ups", "downs", "unchanged"):
            try:
                out[key] = int(out.get(key) or 0)
            except (TypeError, ValueError):
                out[key] = 0
        return out
    except Exception:
        return None


def cypher_price_currency_switch_summary(platform, graph_id: str, filters: dict) -> dict | None:
    """Count currency-switch PriceChangeEvent rows when same-currency deltas are absent."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    totals_q = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND coalesce(e.currency_changed, false) = true "
        "RETURN count(e) AS events, count(DISTINCT p) AS products"
    )
    pairs_q = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND coalesce(e.currency_changed, false) = true "
        "WITH coalesce(e.old_currency, '?') + '→' + coalesce(e.new_currency, e.currency, '?') AS pair, "
        "count(e) AS events, count(DISTINCT p) AS products "
        "RETURN pair, events, products "
        "ORDER BY events DESC "
        "LIMIT 5"
    )
    try:
        with session_factory() as session:
            total = session.run(totals_q, params).single()
            rows = [dict(row) for row in session.run(pairs_q, params)]
        events = int((total or {}).get("events") or 0)
        products = int((total or {}).get("products") or 0)
        return {
            "events": events,
            "products": products,
            "pairs": [
                {
                    "pair": str(r.get("pair") or "—"),
                    "events": int(r.get("events") or 0),
                    "products": int(r.get("products") or 0),
                }
                for r in rows
            ],
        }
    except Exception:
        return None


def _append_currency_switch_only_summary(
    who: str,
    switches: dict,
    graph_id: str,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
    *,
    heading: str = "Recorded currency changes",
) -> bool:
    """Explain why no percentage trend can be reported and show the observed switch mix."""

    switch_n = int(switches.get("events") or 0)
    if switch_n <= 0:
        return False
    products = int(switches.get("products") or 0)
    pairs = switches.get("pairs") or []
    lines.append(
        f"No comparable percentage price trend is available for {_display_label(who)}: "
        f"the {switch_n} recorded events are currency-label changes across {products} products, "
        "not same-currency price moves."
    )
    if pairs:
        lines.append(f"{heading}:")
        lines.append("| Currency change | Events | Products |")
        lines.append("| --- | ---: | ---: |")
        for pair in pairs[:5]:
            lines.append(
                f"| {pair.get('pair') or 'Unspecified'} | "
                f"{int(pair.get('events') or 0)} | "
                f"{int(pair.get('products') or 0)} |"
            )
    evidence.append(
        EvidenceItem(
            source_id=graph_id,
            record_id="price_currency_switch_summary",
            title=f"Currency-switch price events · {_display_label(who)}",
            excerpt=(
                f"currency_switch_events={switch_n}; products={products}; "
                + "; ".join(f"{p.get('pair')}={p.get('events')}" for p in pairs[:5])
            ),
            source_path="product_change_events",
            verification_status="observed",
            graph=graph_id,
            node_id=None,
            synthetic=False,
        )
    )
    caveats.append(
        "Currency-label changes cannot be interpreted as percentage price moves without comparable "
        "same-currency prices."
    )
    return True


def append_neo4j_price_history_answer(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    question: str,
    limit: int | None,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
) -> bool:
    """Handle average trend summaries and ranked movers from PriceChangeEvent."""

    mode = price_history_mode(question)
    if mode is None:
        return False
    filters = clean_product_filters(filters)
    # Metric wording must never become a garment/category filter (e.g. price→"special prices").
    _noise = {
        "special prices",
        "special price",
        "trend",
        "trends",
        "price",
        "prices",
        "change",
        "changes",
        "average",
        "avg",
    }
    for key in ("garment", "garment_type", "category", "family", "subfamily", "subcategory"):
        val = normalize_name(str(filters.get(key) or ""))
        if val in _noise or any(tok in val for tok in ("special price", "trend")):
            filters.pop(key, None)
    # Geo/time scopes not modeled yet — fail closed instead of inventing a global trend.
    qn = normalize_name(question or "")
    if re.search(r"\b(region|regions|country|countries)\b", qn) and re.search(
        r"\b(uk|eu|us|usa|europe|asia)\b", qn
    ):
        lines.append(
            "Regional / country-scoped price trends are not available yet "
            "(Domain/Country nodes are not wired into PriceChangeEvent)."
        )
        caveats.append("No Region dimension on price-history events; refusing global substitute.")
        return True

    if mode == "count_events":
        n = cypher_count_nodes(platform, graph_id, "PriceChangeEvent")
        if n is None:
            return False
        lines.append(f"There are {n} PriceChangeEvent nodes in the knowledge graph.")
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="price_change_event_count",
                title="PriceChangeEvent count",
                excerpt=f"count={n}",
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        return True

    if mode == "product":
        try:
            from fashion_brain.ask import product_ref_from_question

            pref = str(filters.get("product_id") or product_ref_from_question(question) or "").strip()
        except Exception:
            pref = str(filters.get("product_id") or "").strip()
        if not pref:
            return False
        rows = cypher_price_history(platform, graph_id, pref, limit=max(int(limit or 50), 1))
        if not rows:
            lines.append(f"No price change events were found for product {pref}.")
            caveats.append("Product timeline uses HAS_PRICE_CHANGE → PriceChangeEvent.")
            return True
        name = rows[0].get("name") or pref
        pid = rows[0].get("product_id") or pref
        lines.append(f"Price history for {name} (`{pid}`):")
        lines.append("| When | Old | New | Change % | Direction | Currency |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in rows:
            pct = row.get("delta_pct")
            pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "—"
            cur = "switch" if row.get("currency_changed") else (row.get("currency") or "—")
            lines.append(
                f"| {str(row.get('at') or '—')[:19]} | {row.get('old_price') if row.get('old_price') is not None else '—'} | "
                f"{row.get('new_price') if row.get('new_price') is not None else '—'} | {pct_s} | "
                f"{row.get('direction') or '—'} | {cur} |"
            )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id=str(pid),
                title=f"Price history · {name}",
                excerpt=f"events={len(rows)}",
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=str(pid),
                synthetic=False,
            )
        )
        return True

    if mode == "events_by_brand":
        rows = cypher_price_events_by_brand(platform, graph_id, limit=max(int(limit or 10), 1))
        if not rows:
            lines.append("No price change events were found to rank by brand.")
            return True
        lines.append("Brands with the most price change events:")
        lines.append("| Brand | Events | Products |")
        lines.append("| --- | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {_display_label(str(row.get('brand') or '—'))} | "
                f"{int(row.get('events') or 0)} | {int(row.get('products') or 0)} |"
            )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="price_events_by_brand",
                title="Price change events by brand",
                excerpt="; ".join(f"{r.get('brand')}={r.get('events')}" for r in rows[:5]),
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        return True

    if mode == "trend_by_category":
        rows = cypher_price_trend_by_category(platform, graph_id, limit=max(int(limit or 10), 1))
        if not rows:
            switches = cypher_price_currency_switch_summary(platform, graph_id, {}) or {}
            if not _append_currency_switch_only_summary(
                "the catalog",
                switches,
                graph_id,
                lines,
                evidence,
                caveats,
                heading="Currency changes in the catalog (no category breakdown available)",
            ):
                lines.append("No category-level same-currency price trends were found.")
            return True
        lines.append("Categories by average same-currency price change (most decreasing first):")
        lines.append("| Category | Avg change | Events | Products |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in rows:
            avg = row.get("avg_delta_pct")
            avg_s = f"{avg:.2f}%" if isinstance(avg, float) else "—"
            lines.append(
                f"| {row.get('category') or '—'} | {avg_s} | "
                f"{int(row.get('events') or 0)} | {int(row.get('products') or 0)} |"
            )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="price_trend_by_category",
                title="Price trend by category",
                excerpt="; ".join(
                    f"{r.get('category')}={r.get('avg_delta_pct')}" for r in rows[:5]
                ),
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        return True

    # "Zara | Primark" must never be treated as a single brand name.
    parties: list[str] = []
    try:
        from fashion_brain.ask import split_multi_party_filter

        parties = split_multi_party_filter(str(filters.get("brand") or filters.get("buyer") or ""))
    except Exception:
        parties = []
    if len(parties) >= 2:
        lines.append("Comparing historical price moves by brand:")
        lines.append(
            "| Brand | Avg change | Events | Products | Up | Down | Unchanged |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        excerpts: list[str] = []
        any_events = False
        switch_summaries: list[tuple[str, dict]] = []
        for party in parties:
            party_filters = {**filters, "brand": party}
            party_filters.pop("buyer", None)
            summary = cypher_price_trend_summary(platform, graph_id, party_filters) or {}
            events = int(summary.get("change_events") or 0)
            any_events = any_events or events > 0
            switches = cypher_price_currency_switch_summary(platform, graph_id, party_filters) or {}
            if events == 0 and int(switches.get("events") or 0) > 0:
                switch_summaries.append((party, switches))
            avg = summary.get("avg_delta_pct")
            avg_s = f"{avg:.2f}%" if isinstance(avg, float) else "—"
            lines.append(
                f"| {_display_label(party)} | {avg_s} | {events} | "
                f"{int(summary.get('products_changed') or 0)} | "
                f"{int(summary.get('ups') or 0)} | {int(summary.get('downs') or 0)} | "
                f"{int(summary.get('unchanged') or 0)} |"
            )
            excerpts.append(f"{party} avg={avg_s} events={events}")
            sample_n = min(max(int(limit or 3), 1), 3)
            if mode in {"movers_up", "movers_down"}:
                direction = "up" if mode == "movers_up" else "down"
            else:
                direction = "down" if isinstance(avg, float) and avg < 0 else "up"
            sample = cypher_price_movers(
                platform, graph_id, party_filters, direction=direction, limit=sample_n
            )
            if sample:
                lines.append(f"Example {_display_label(party)} moves:")
                for row in sample:
                    pct = row.get("delta_pct")
                    pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "—"
                    lines.append(
                        f"- {row.get('name') or row.get('product_id')}: "
                        f"{row.get('old_price')} → {row.get('new_price')} ({pct_s})"
                    )
        for party, switches in switch_summaries:
            _append_currency_switch_only_summary(
                party,
                switches,
                graph_id,
                lines,
                evidence,
                caveats,
            )
        if not any_events and not switch_summaries:
            lines.append(
                "No same-currency price change events were found for these brands "
                "(currency switches are excluded)."
            )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="price_trend_brand_compare",
                title="Price trend brand comparison",
                excerpt="; ".join(excerpts),
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        caveats.append(
            "Brand comparison uses same-currency PriceChangeEvent deltas; "
            "paired similar-product matching across brands is not applied yet."
        )
        return True
    if mode == "trend":
        summary = cypher_price_trend_summary(platform, graph_id, filters)
        who = filters.get("brand") or filters.get("buyer") or "this selection"
        if not summary or not summary.get("change_events"):
            switches = cypher_price_currency_switch_summary(platform, graph_id, filters) or {}
            if _append_currency_switch_only_summary(
                str(who),
                switches,
                graph_id,
                lines,
                evidence,
                caveats,
            ):
                return True
            lines.append(
                f"No same-currency price change events were found for {who} "
                "(currency switches are excluded)."
            )
            caveats.append("Price trends use PriceChangeEvent nodes from product_change_events.")
            return True
        avg = summary.get("avg_delta_pct")
        avg_s = f"{avg:.2f}%" if isinstance(avg, float) else "—"
        direction = "decreasing" if isinstance(avg, float) and avg < 0 else "increasing" if isinstance(avg, float) and avg > 0 else "flat"
        ups = int(summary.get("ups") or 0)
        downs = int(summary.get("downs") or 0)
        lines.append(f"Average same-currency price change trend for {who}:")
        lines.append(f"- Average change: {avg_s} ({direction})")
        lines.append(f"- Change events: {summary.get('change_events')}")
        lines.append(f"- Products with changes: {summary.get('products_changed')}")
        lines.append(
            f"- Direction mix: {ups} up, {downs} down, "
            f"{summary.get('unchanged')} unchanged"
        )
        min_pct = summary.get("min_delta_pct")
        max_pct = summary.get("max_delta_pct")
        if isinstance(min_pct, float) and isinstance(max_pct, float):
            lines.append(f"- Range: {min_pct:.2f}% to {max_pct:.2f}%")

        series = cypher_price_trend_timeseries(
            platform, graph_id, filters, limit=max(int(limit or 24), 16)
        ) or []
        if len(series) >= 2:
            sample_period = str(series[0].get("period") or "")
            if "-W" in sample_period:
                grain = "Weekly"
            elif len(sample_period) == 10 and sample_period[4:5] == "-":
                grain = "Daily"
            else:
                grain = "Event"
            values = [float(r.get("avg_delta_pct") or 0.0) for r in series]
            series_min = min(values)
            series_max = max(values)
            series_avg = sum(values) / len(values)
            # Prefer overall event summary avg when present (same-currency mean).
            overall_avg = avg if isinstance(avg, float) else series_avg
            overall_min = min_pct if isinstance(min_pct, float) else series_min
            overall_max = max_pct if isinstance(max_pct, float) else series_max
            lines.append(
                f"{grain} % change trail (same-currency) — "
                f"min {overall_min:.2f}% · avg {overall_avg:.2f}% · max {overall_max:.2f}%:"
            )
            lines.append("| Period | Change % | Events |")
            lines.append("| --- | ---: | ---: |")
            # Keep the table readable: show endpoints + extremes when many points.
            show_idx = set(range(len(series))) if len(series) <= 12 else {
                0,
                len(series) - 1,
                values.index(series_min),
                values.index(series_max),
            }
            for i, row in enumerate(series):
                if i not in show_idx and len(series) > 12:
                    continue
                lines.append(
                    f"| {row.get('period') or '—'} | {row.get('avg_delta_pct'):.2f} | "
                    f"{int(row.get('events') or 0)} |"
                )
            if len(series) > 12:
                lines.append(f"| … | ({len(series)} points plotted) | … |")
            chart = {
                "type": "line",
                "smooth": True,
                "analytics": True,
                "title": f"Price trend · {who}",
                "labels": [str(r.get("period") or "") for r in series],
                "values": values,
                "avg": round(float(overall_avg), 3),
                "min": round(float(overall_min), 3),
                "max": round(float(overall_max), 3),
            }
            lines.append("```chart")
            lines.append(json.dumps(chart, ensure_ascii=False))
            lines.append("```")

        # Grounded drivers only — never invent market narratives.
        lines.append("Probable drivers (from recorded events only):")
        total_dir = ups + downs
        if total_dir > 0 and downs > ups * 1.25:
            lines.append(
                f"- Markdown pressure: {downs} down vs {ups} up moves "
                f"({100.0 * downs / total_dir:.0f}% of directional events are decreases)."
            )
        elif total_dir > 0 and ups > downs * 1.25:
            lines.append(
                f"- Upward repricing: {ups} up vs {downs} down moves "
                f"({100.0 * ups / total_dir:.0f}% of directional events are increases)."
            )
        elif total_dir > 0:
            lines.append(
                f"- Mixed moves: {ups} up and {downs} down — no single direction dominates."
            )
        switches = cypher_price_currency_switch_summary(platform, graph_id, filters) or {}
        switch_n = int(switches.get("events") or 0)
        if switch_n > 0:
            pairs = switches.get("pairs") or []
            pair_bits = ", ".join(f"{p.get('pair')} ({p.get('events')})" for p in pairs[:2])
            lines.append(
                f"- Currency-label switches also appear ({switch_n} events"
                + (f": {pair_bits}" if pair_bits else "")
                + ") and are excluded from the % trend line."
            )
        # Category hotspots are graph-wide — only cite them when the ask is unscoped.
        if not any(filters.get(k) for k in ("category", "brand", "material", "garment", "buyer")):
            cat_rows = cypher_price_trend_by_category(platform, graph_id, limit=3) or []
            bits = []
            for row in cat_rows[:3]:
                cavg = row.get("avg_delta_pct")
                if isinstance(cavg, float):
                    bits.append(f"{row.get('category') or '—'} ({cavg:.1f}%)")
            if bits:
                lines.append(
                    "- Category hotspots by average change: " + "; ".join(bits) + "."
                )
        lines.append(
            "- These are observed event patterns in the catalog, not external market causes "
            "(inflation, FX policy, promotions) unless those appear as data fields."
        )

        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="price_trend_summary",
                title=f"Price trend · {who}",
                excerpt=(
                    f"avg_delta_pct={avg_s} events={summary.get('change_events')} "
                    f"ups={ups} downs={downs} series={len(series)}"
                ),
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        # Small sample of movers for context (not the primary answer).
        sample = cypher_price_movers(
            platform,
            graph_id,
            filters,
            direction="down" if isinstance(avg, float) and avg < 0 else "up",
            limit=min(max(int(limit or 5), 1), 5),
        )
        if sample:
            lines.append("Example moves:")
            for row in sample:
                pct = row.get("delta_pct")
                pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "—"
                lines.append(
                    f"- {row.get('name') or row.get('product_id')}: "
                    f"{row.get('old_price')} → {row.get('new_price')} ({pct_s})"
                )
        caveats.append(
            "Trend line uses same-currency delta_pct over time (week/day/event buckets); "
            "currency switches excluded."
        )
        return True

    direction = "up" if mode == "movers_up" else "down"
    show_n = max(int(limit or 10), 1)
    rows = cypher_price_movers(
        platform,
        graph_id,
        filters,
        direction=direction,
        limit=show_n,
    )
    label = "decreases" if direction == "down" else "increases"
    if not rows:
        switches = cypher_price_currency_switch_summary(platform, graph_id, filters) or {}
        if not _append_currency_switch_only_summary(
            str(filters.get("brand") or filters.get("buyer") or "this selection"),
            switches,
            graph_id,
            lines,
            evidence,
            caveats,
        ):
            lines.append(
                f"No same-currency price {label} were found in the product knowledge graph "
                "(currency switches are excluded)."
            )
            caveats.append("Price movers use PriceChangeEvent nodes from product_change_events.")
        return True
    lines.append(f"Largest same-currency price {label} (from catalog change events):")
    lines.append("| Product | Brand | Old | New | Change % | When |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        name = row.get("name") or row.get("product_id") or "—"
        brand = row.get("brand") or "—"
        old = row.get("old_price")
        new = row.get("new_price")
        pct = row.get("delta_pct")
        when = str(row.get("detected_at") or "—")[:19]
        pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "—"
        lines.append(f"| {name} | {brand} | {old if old is not None else '—'} | {new if new is not None else '—'} | {pct_s} | {when} |")
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id=str(row.get("product_id") or ""),
                title=str(name),
                excerpt=f"price {old} → {new} ({pct_s}) brand={brand} at={when}",
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=str(row.get("product_id") or "") or None,
                synthetic=False,
            )
        )
    caveats.append("Only same-currency moves are ranked; direction=currency_change events are excluded.")
    return True


def append_neo4j_price_movers_answer(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    question: str,
    limit: int | None,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
) -> bool:
    """Back-compat wrapper around append_neo4j_price_history_answer."""

    return append_neo4j_price_history_answer(
        platform,
        principal,
        graph_id,
        filters,
        question,
        limit,
        lines,
        evidence,
        caveats,
    )


def overlap_facet_from_question(question: str, platform, graph_ids: list[str] | None = None) -> str | None:
    """Pick the attribute named in the question from live product schema — not a fixed question map."""

    text = normalize_name(question or "")
    if not text:
        return None
    try:
        from fashion_brain.planning.runtime_schema import get_runtime_schema, resolve_product_graph_id
    except Exception:
        return None
    ids = [str(item) for item in (graph_ids or []) if item]
    if not ids:
        gid = resolve_product_graph_id(platform) if platform is not None else None
        if gid:
            ids = [str(gid)]
    seen: set[str] = set()
    hits: list[str] = []
    for gid in ids[:4]:
        try:
            schema = get_runtime_schema(platform, gid)
        except Exception:
            continue
        attrs = []
        for raw in list(getattr(schema, "facet_types", None) or []) + list(getattr(schema, "attributes", None) or []):
            key = str(raw or "").strip().casefold().replace(" ", "_")
            if not key or key in _NON_FACET_ATTRS or key in seen:
                continue
            if not is_analysis_facet_attr(key):
                continue
            seen.add(key)
            attrs.append(key)
        for key in attrs:
            stem = key[:-1] if key.endswith("s") and len(key) > 3 else key
            if re.search(rf"\b{re.escape(key)}s?\b", text) or re.search(rf"\b{re.escape(stem)}s?\b", text):
                hits.append(key)
    if not hits:
        return None
    # Prefer the longest concrete attribute token mentioned in the question.
    return sorted(hits, key=lambda item: (-len(item), item))[0]


def _facet_overlap_key(field: str, label: str) -> str:
    """Case/spacing merge key; keep fibre percentages distinct from bare fibre names."""

    raw = str(label or "").strip()
    if not raw:
        return ""
    field_key = normalize_name(field)
    if field_key in {"material", "materials", "composition"}:
        # Align with display normalization so "POLYESTER" and "polyester" merge,
        # while "polyester 22" stays separate from bare "polyester".
        return normalize_name(_material_display_label(raw))
    return normalize_name(raw)


def _facet_overlap_display(field: str, label: str) -> str:
    field_key = normalize_name(field)
    if field_key in {"material", "materials", "composition"}:
        return _material_display_label(label)
    return _display_label(label)


def analysis_value_limit(question: str, request_limit: int | None = None) -> int | None:
    """Row cap for attribute analysis only.

    - Explicit top-N from the planner → that N
    - full / complete / all / every → None (return every aggregated value)
    - otherwise None for analysis (aggregates are cheap; product listings use their own limit)
    """

    if request_limit is not None:
        try:
            return max(1, int(request_limit))
        except (TypeError, ValueError):
            pass
    text = str(question or "").casefold()
    if re.search(r"\b(full|complete|entire|exhaustive|all|every)\b", text):
        return None
    # Attribute aggregates are small (distinct facet values), not product rows.
    return None


def facet_overlap_across_graphs(
    platform,
    graph_ids: list[str],
    field: str,
    *,
    filters: dict | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Attribute-wise sum per product graph, then keep values present in more than one graph.

    No product-id scrape and no question-specific Cypher — reuses cypher_group_by.
    Values are denormalized (case/spacing), BOM/component roles dropped for materials,
    and labels polished for display. ``limit`` is optional top-N only; None returns all
    aggregated values (analysis path). Product-detail listings use a separate sample limit.
    """

    field = str(field or "").strip()
    ids = [str(item) for item in (graph_ids or []) if item]
    if not field or len(ids) < 2:
        return []
    filters = dict(filters or {})
    per_graph: dict[str, dict[str, tuple[str, int]]] = {}
    for gid in ids:
        rows = cypher_group_by(platform, gid, field, filters) or []
        bucket: dict[str, tuple[str, int]] = {}
        for value, count in rows:
            label = str(value or "").strip()
            if not _usable_value(label, field):
                continue
            key = _facet_overlap_key(field, label)
            if not key:
                continue
            display = _facet_overlap_display(field, label)
            n = int(count)
            prev = bucket.get(key)
            if prev is None:
                bucket[key] = (display, n)
            else:
                # Prefer the longer polished label when merging case/spacing variants.
                best = display if len(display) >= len(prev[0]) else prev[0]
                bucket[key] = (best, prev[1] + n)
        per_graph[gid] = bucket
    shared_keys = None
    for bucket in per_graph.values():
        keys = set(bucket)
        shared_keys = keys if shared_keys is None else (shared_keys & keys)
    if not shared_keys:
        return []
    groups: list[dict] = []
    for key in shared_keys:
        graph_counts: dict[str, int] = {}
        label = key
        total = 0
        for gid in ids:
            hit = per_graph.get(gid, {}).get(key)
            if not hit:
                continue
            label = hit[0]
            graph_counts[gid] = hit[1]
            total += hit[1]
        if len(graph_counts) < 2:
            continue
        groups.append(
            {
                "value": label,
                "field": field,
                "product_count": total,
                "graph_ids": sorted(graph_counts),
                "graph_counts": graph_counts,
                "verified": False,
            }
        )
    # Bare fibre/attribute names first, then percentage mixes; both ranked by volume.
    groups.sort(
        key=lambda item: (
            1 if re.search(r"\d", str(item.get("value") or "")) else 0,
            -int(item["product_count"]),
            str(item["value"]).casefold(),
        )
    )
    if limit is None:
        return groups
    return groups[: max(1, int(limit))]


def cypher_group_by(
    platform, graph_id: str, field: str, filters: dict
) -> list[tuple[str, int]] | None:
    """Aggregate products by a facet using Neo4j (supports material/color/… filters)."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    field = str(field or "").strip()
    # Bare "garment" → category facet; keep garment_type on HAS_GARMENT_TYPE when present.
    if field == "garment":
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
        # Prefer related entity name, then the product property, then category_path
        # for category-like facets (ecom often stores path text, not a bare category prop).
        prop_fallback = (
            f"p.`{field}`, p.category_path"
            if field in {"category", "family", "subfamily"}
            else f"p.`{field}`"
        )
        query = (
            "MATCH (p:Entity:Product) WHERE "
            + " AND ".join(where)
            + f" OPTIONAL MATCH (p)-[:{rel[0]}]->(x:Entity) "
            "WHERE coalesce(x.lifecycle,'active') <> 'deleted' "
            f"WITH p, coalesce(x.name, {prop_fallback}, '(unspecified)') AS value "
            "RETURN value, count(DISTINCT p) AS n "
            "ORDER BY n DESC, value ASC"
        )
    else:
        # Any schema property with a safe identifier — no fixed allow-list of facet names.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
            return None
        query = (
            "MATCH (p:Entity:Product) WHERE "
            + " AND ".join(where)
            + f" RETURN coalesce(toString(p.`{field}`), '(unspecified)') AS value, count(p) AS n "
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


def cypher_sample_products(
    platform, graph_id: str, filters: dict, *, limit: int = 5, offset: int = 0
) -> list:
    """Fetch a few matching product nodes via Cypher (full filter support)."""

    session_factory = _session(platform)
    if not session_factory:
        return []
    params: dict = {
        "g": graph_id,
        "limit": max(int(limit), 1),
        "offset": max(int(offset), 0),
    }
    where = _filter_where(filters, params)
    if where is None:
        return []
    order = "ORDER BY p.id ASC"
    if filters.get("price_max") is not None or filters.get("price_min") is not None:
        order = f"ORDER BY {_price_num_cypher()} ASC, p.id ASC"
    query = (
        "MATCH (p:Entity:Product) WHERE "
        + " AND ".join(where)
        + f" RETURN p, labels(p) AS labels {order} SKIP $offset LIMIT $limit"
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
    # Internal match helper only — never a user-facing dump of thousands of rows.
    sample = cypher_sample_products(platform, graph_id, filters, limit=200)
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


# Sample-ready sheet order (PLMEX-first). Source of truth: fashion_brain.agents.product_spec.
try:
    from fashion_brain.agents.product_spec import (
        SPEC_ANCHOR_FIELDS as _SPEC_ANCHOR_FIELDS,
        SPEC_FIELD_ORDER as _SPEC_FIELD_ORDER,
    )
except Exception:  # pragma: no cover - keep retrieval usable without agents package
    _SPEC_ANCHOR_FIELDS = (
        "plm_sub_sub_category",
        "category",
        "garment_type",
    )
    _SPEC_FIELD_ORDER = (
        "plm_sub_sub_category",
        "category",
        "garment_type",
        "plm_main_category",
        "plm_sub_category",
        "family",
        "subfamily",
        "gender",
        "design_for",
        "material",
        "color",
        "gauge",
        "knit_machine_type",
        "gsm",
        "weight",
        "design_feature",
        "style_feature",
        "ai_construction_tags",
        "ai_material_tags",
        "ai_finish_tags",
        "ai_special_tags",
        "fit",
        "construction",
        "availability",
        "section",
    )

# Ontology facet / entity labels → specification field keys.
_SPEC_FACET_ALIASES = {
    "garmenttype": "garment_type",
    "garmentcategory": "category",
    "plmmaincategory": "plm_main_category",
    "plmsubcategory": "plm_sub_category",
    "plmsubsubcategory": "plm_sub_sub_category",
    "sampletype": "sample_collection_type",
    "designpurpose": "design_for",
    "machinetype": "knit_machine_type",
    "designfeature": "design_feature",
    "stylefeature": "style_feature",
    "eventtag": "ai_event_tags",
    "constructiontag": "ai_construction_tags",
    "aimaterialtag": "ai_material_tags",
    "finishtag": "ai_finish_tags",
    "specialtag": "ai_special_tags",
    "materialgroup": "material",
    "subfamily": "subfamily",
    "family": "family",
    "section": "section",
    "fit": "fit",
    "construction": "construction",
    "availability": "availability",
}


def _specification_fields_for_graph(platform, graph_id: str) -> list[str]:
    """Required product-spec attributes for this catalog (from product_spec.SPEC_FIELD_ORDER).

    PLM / plmex graphs: analyze the full predefined sheet (every SPEC_FIELD_ORDER row).
    Thin retail graphs: intersect with facets the catalog actually exposes.
    """

    # Ops / scope keys never become mode-of-spec rows (season is a requirement, not a mode).
    exclude = {
        "brand",
        "buyer",
        "season_year",
        "department",
        "sample_collection_type",
    }
    predefined = [field for field in _SPEC_FIELD_ORDER if field not in exclude]
    plm_like = "plmex" in str(graph_id or "").casefold() or "plm" in str(graph_id or "").casefold()
    if plm_like:
        # User contract: for a product spec, analyze every predefined field.
        return predefined

    allow = (set(predefined) | set(_REL_FIELDS) | set(_PROP_FILTER_FIELDS)) - exclude
    available: set[str] = set()
    try:
        from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

        ctx = build_runtime_schema(platform, graph_id)
        rels = {str(name).upper() for name in (ctx.relationship_allowlist or {})}
        for field, rel_types in _REL_FIELDS.items():
            if field in allow and any(rel in rels for rel in rel_types):
                available.add(field)
        for facet in ctx.facet_types or []:
            key = normalize_name(str(facet)).replace(" ", "").replace("_", "")
            mapped = _SPEC_FACET_ALIASES.get(key)
            if not mapped:
                snake = normalize_name(str(facet)).replace(" ", "_")
                mapped = snake if snake in allow else None
            if mapped and mapped in allow:
                available.add(mapped)
        for key in ctx.filter_keys or []:
            folded = normalize_name(str(key)).replace(" ", "").replace("_", "")
            mapped = _SPEC_FACET_ALIASES.get(folded)
            snake = normalize_name(str(key)).replace(" ", "_")
            if mapped and mapped in allow:
                available.add(mapped)
            elif snake in allow:
                available.add(snake)
    except Exception:
        pass
    if not available:
        available = {"category", "material", "color", "availability", "family", "subfamily"}
    return [field for field in predefined if field in available]


def _align_party_filters_for_graph(platform, graph_id: str, filters: dict) -> dict:
    """Map Primark-style party to buyer and/or brand for this catalog's schema."""

    out = dict(filters or {})
    party = out.get("buyer") or out.get("brand")
    if not party:
        return out
    has_buyer = False
    has_brand = False
    brand_first = False
    try:
        from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

        ctx = build_runtime_schema(platform, graph_id)
        keys = {str(k).casefold() for k in (ctx.filter_keys or [])}
        rels = {str(r).upper() for r in (ctx.relationship_allowlist or {})}
        has_buyer = "buyer" in keys or "HAS_BUYER" in rels
        has_brand = "brand" in keys or "HAS_BRAND" in rels or bool(ctx.brand_label)
        # Retail/ecom graphs are brand-primary even when a buyer key also exists.
        brand_first = bool(ctx.brand_label) or "HAS_BRAND" in rels or "ecom" in str(graph_id or "").casefold()
    except Exception:
        has_brand = True
        brand_first = "ecom" in str(graph_id or "").casefold()
    if has_buyer and not has_brand:
        out["buyer"] = party
        out.pop("brand", None)
    elif has_brand and not has_buyer:
        out["brand"] = party
        out.pop("buyer", None)
    elif has_brand and brand_first:
        out["brand"] = party
        out.pop("buyer", None)
    elif has_buyer:
        # PLM-like: prefer buyer (development customer), keep brand out of AND-filter.
        out["buyer"] = party
        out.pop("brand", None)
    else:
        out["brand"] = party
        out.pop("buyer", None)
    return out


def append_neo4j_specification_answer(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
    *,
    season_name: str | None = None,
    graph_ids: list[str] | None = None,
) -> None:
    """Build product specification sheet(s) from Neo4j facet modes.

    One graph → that catalog's fields only (PLMEX rich / ecom retail).
    Several graphs (e.g. study ecom brand + build PLM spec) → one section per catalog,
    with party remapped (brand on ecom, buyer on PLMEX).
    """

    targets = [str(g) for g in (graph_ids or []) if g] or ([graph_id] if graph_id else [])
    targets = list(dict.fromkeys(targets))
    if not targets:
        lines.append("No product catalog was selected for the specification.")
        return
    if len(targets) > 1:
        lines.append(
            "Combined specification: retail mix from e-commerce plus development fields from PLM "
            "(each catalog only contributes fields it actually stores)."
        )
        caveats.append(
            "Multi-catalog specification: e-comm informs market/retail modes; PLM informs factory/development modes."
        )
    for gid in targets:
        if len(targets) > 1:
            lines.append(f"### {_graph_label(platform, gid)}")
        _append_neo4j_specification_for_graph(
            platform,
            principal,
            gid,
            _align_party_filters_for_graph(platform, gid, filters),
            lines,
            evidence,
            season_name=season_name,
        )


def _append_neo4j_specification_for_graph(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    lines: list[str],
    evidence: list[EvidenceItem],
    *,
    season_name: str | None = None,
) -> None:
    """User requirements → required attributes → analyze values → pick best → final sheet."""

    base = clean_product_filters(dict(filters or {}))
    if season_name and not base.get("season_name"):
        base["season_name"] = season_name
    party_key = "buyer" if base.get("buyer") else "brand"
    party = base.get("buyer") or base.get("brand") or "catalog"
    party_only = {
        key: value
        for key, value in base.items()
        if key not in {"season_name", "season_year", "season"}
    }
    buyer_total = cypher_count_products(platform, graph_id, party_only)
    if buyer_total is None:
        buyer_total = 0
    lines.append(f"Scope for {party}: {buyer_total} products in {_graph_label(platform, graph_id)}.")
    if not buyer_total:
        lines.append(f"There are no matching styles for {party} in {_graph_label(platform, graph_id)}.")
        return

    active = dict(base)
    total = cypher_count_products(platform, graph_id, active)
    if total is None:
        total = 0
    season_bits = []
    if active.get("season_name"):
        season_bits.append(str(active["season_name"]).capitalize())
    if active.get("season_year"):
        season_bits.append(str(active["season_year"]))
    if season_bits:
        lines.append(f"{' '.join(season_bits)} styles in that scope: {total} products.")
    if total == 0 and active.get("season_year"):
        relaxed = dict(active)
        dropped_year = relaxed.pop("season_year")
        total = cypher_count_products(platform, graph_id, relaxed) or 0
        if total:
            active = relaxed
            lines.append(
                f"No products for season year {dropped_year}; using "
                f"{active.get('season_name') or 'all seasons'} styles instead ({total} products)."
            )
    if total == 0 and active.get("season_name"):
        relaxed = dict(active)
        dropped_season = relaxed.pop("season_name")
        total = cypher_count_products(platform, graph_id, relaxed) or 0
        if total:
            active = relaxed
            lines.append(
                f"No {dropped_season} styles matched; using all {total} products in the {party_key}/garment scope."
            )
    if not total:
        lines.append(f"There are no styles left after filters for {party} in {_graph_label(platform, graph_id)}.")
        return

    spec_fields = _specification_fields_for_graph(platform, graph_id)
    requirement_rows = [(party_key.replace("_", " ").capitalize(), str(party))]
    if active.get("season_name"):
        requirement_rows.append(("Season", str(active["season_name"])))
    if active.get("season_year"):
        requirement_rows.append(("Season year", str(active["season_year"])))
    garment = active.get("garment") or active.get("garment_type") or active.get("category")
    if garment:
        requirement_rows.append(("Garment", str(garment)))
    for key, value in active.items():
        if key in {party_key, "season_name", "season_year", "season", "garment", "garment_type", "category", "brand", "buyer"}:
            continue
        if value:
            requirement_rows.append((key.replace("_", " ").capitalize(), str(value)))

    lines.append("User requirements:")
    for label, value in requirement_rows:
        lines.append(f"- {label}: {value}")

    field_labels = [field.replace("_", " ") for field in spec_fields]
    lines.append(
        f"Required specification attributes ({len(spec_fields)} — analyze all): "
        + (", ".join(field_labels) if field_labels else "none")
        + "."
    )

    # 1) anchors narrow (when supported)  2) every required field is analyzed  3) full sheet
    scope = dict(active)
    scope_total = int(total)
    # field -> (best_label, best_count, alternatives_text, raw_value|None)
    analysis_rows: dict[str, tuple[str, int, str, str | None]] = {}
    decided_fields: set[str] = set()
    lines.append(
        f"Analysis on {scope_total} matching products "
        f"(predefined attributes; anchor hierarchy first when possible):"
    )

    def _probe_field(field: str, filters: dict, population: int) -> tuple[str, int, str, str] | None:
        groups = cypher_group_by(platform, graph_id, field, filters) or []
        usable = _usable_ranked(groups, field, limit=5)
        if not usable:
            return None
        options = "; ".join(
            f"{_display_label(str(name))} ({count} of {population})" for name, count in usable
        )
        best_name, best_count = usable[0]
        return _display_label(str(best_name)), int(best_count), options, str(best_name)

    for field in _SPEC_ANCHOR_FIELDS:
        if field not in spec_fields or field in decided_fields:
            continue
        picked = _probe_field(field, scope, scope_total)
        label = field.replace("_", " ")
        if not picked:
            analysis_rows[field] = ("(no usable value)", 0, "—", None)
            continue
        best, best_count, options, raw = picked
        analysis_rows[field] = (best, best_count, options, raw)
        decided_fields.add(field)
        candidate = dict(scope)
        candidate[field] = raw
        narrowed = cypher_count_products(platform, graph_id, candidate)
        if narrowed:
            scope = candidate
            scope_total = int(narrowed)
            lines.append(f"  Anchored on {label}={best}; remaining products: {scope_total}.")
        else:
            lines.append(
                f"  Kept {label}={best}; could not narrow further "
                f"(continuing on {scope_total} products)."
            )

    for field in spec_fields:
        if field in analysis_rows:
            continue
        picked = _probe_field(field, scope, scope_total)
        if not picked:
            analysis_rows[field] = ("(no usable value)", 0, "—", None)
            continue
        best, best_count, options, raw = picked
        analysis_rows[field] = (best, best_count, options, raw)
        decided_fields.add(field)

    # Full breakdown table — one row per required attribute (never stop after hierarchy).
    lines.append("| Attribute | Best value | Count | Top alternatives |")
    lines.append("| --- | --- | ---: | --- |")
    for field in spec_fields:
        best, best_count, options, _raw = analysis_rows.get(
            field, ("(no usable value)", 0, "—", None)
        )
        label = field.replace("_", " ")
        count_cell = str(best_count) if best_count else "—"
        # Alternatives column: drop the leading "Best (n of N); " duplication when possible.
        alt = options if options and options != "—" else "—"
        if alt != "—" and "; " in alt:
            # Keep 2nd.. values as alternatives when present.
            parts = [p.strip() for p in alt.split(";") if p.strip()]
            alt = "; ".join(parts[1:4]) if len(parts) > 1 else "—"
        lines.append(f"| {label} | {best} | {count_cell} | {alt} |")

    filled = sum(1 for field in spec_fields if analysis_rows.get(field, ("", 0, "", None))[3])
    missing_fields = [field for field in spec_fields if not analysis_rows.get(field, ("", 0, "", None))[3]]
    lines.append(
        "Decision rule: user requirements set scope; every required attribute above is analyzed; "
        "anchors narrow when supported; each best value is the most common usable mode on the "
        f"active set ({filled}/{len(spec_fields)} filled)."
    )

    lines.append("Product specification:")
    for label, value in requirement_rows:
        lines.append(f"{label}: {value}")
    for field in spec_fields:
        best, best_count, _options, raw = analysis_rows.get(
            field, ("(no usable value)", 0, "—", None)
        )
        label = field.replace("_", " ")
        if raw and best_count:
            lines.append(f"{label}: {best} ({best_count} of {scope_total})")
        elif raw:
            lines.append(f"{label}: {best}")
        else:
            lines.append(f"{label}: (no usable value in matching products)")

    evidence.append(
        EvidenceItem(
            source_id=graph_id,
            record_id="specification",
            title=f"{_graph_label(platform, graph_id)} specification",
            excerpt=(
                f"{party_key}={party}; products={scope_total}; "
                f"fields={filled}/{len(spec_fields)}; "
                f"missing={len(missing_fields)}; filters={scope}"
            ),
            source_path=graph_id,
            verification_status="observed",
            graph=graph_id,
            node_id=None,
            synthetic=False,
        )
    )


def append_neo4j_product_details_answer(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    lines: list[str],
    evidence: list[EvidenceItem],
    *,
    limit: int = 5,
) -> None:
    """Return user-facing properties for a small sample of matching products."""

    products, _total = search_neo4j_products(
        platform, principal, graph_id, filters, limit=max(1, min(int(limit), 10))
    )
    if not products:
        lines.append("No matching product specifications were found in the selected product catalog.")
        return

    lines.append(f"Matching product specifications (showing up to {len(products)} results):")
    hidden = {
        "classification",
        "content_hash",
        "graph",
        "lifecycle",
        "prov_record_id",
        "source_id",
        "source_path",
    }
    for index, node in enumerate(products, start=1):
        props = node.properties or {}
        title = str(props.get("name") or props.get("title") or node.id)
        lines.append(f"{index}. {title}")
        lines.append("| Specification | Value |")
        lines.append("| --- | --- |")
        product_id = str(props.get("product_id") or node.id or "").strip()
        if product_id:
            lines.append(f"| Product ID | {product_id} |")
        fields = sorted(
            (
                (str(key), value)
                for key, value in props.items()
                if str(key).casefold() not in hidden
                and not str(key).casefold().startswith(
                    ("crawl_", "meta_", "prov_", "raw_", "scrape_")
                )
                and not str(key).casefold().endswith(
                    ("_at", "_hash", "_id", "_timestamp", "_url")
                )
                and value not in (None, "", [], {})
                and str(key).casefold() not in {"id", "product_id", "name", "title"}
                and str(key).casefold() not in {"image", "image_url", "images", "url"}
            ),
            key=lambda item: item[0].casefold(),
        )
        for key, value in fields:
            label = re.sub(r"[_\s]+", " ", key).strip().capitalize()
            display = (
                json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (dict, list))
                else str(value)
            )
            lines.append(f"| {label} | {display.replace('|', '/').replace(chr(10), ' ')} |")

        excerpt_fields = "; ".join(
            f"{key}={value}" for key, value in fields[:8]
        )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id=str(props.get("prov_record_id") or product_id or node.id),
                title=title,
                excerpt=excerpt_fields or title,
                source_path=graph_id,
                verification_status="observed",
                graph=graph_id,
                node_id=node.id,
                synthetic=False,
            )
        )


def append_neo4j_composition_compare_answer(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    question: str,
    limit: int | None,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
) -> bool:
    """Side-by-side material/composition mix for a garment across named brands."""

    try:
        from fashion_brain.ask import (
            garment_type_from_question,
            is_composition_compare_question,
            split_multi_party_filter,
        )
    except Exception:
        return False
    if not is_composition_compare_question(question):
        return False
    filters = clean_product_filters(filters)
    parties = split_multi_party_filter(str(filters.get("brand") or filters.get("buyer") or ""))
    if len(parties) < 2:
        return False
    garment = (
        filters.get("garment")
        or filters.get("category")
        or filters.get("garment_type")
        or garment_type_from_question(question)
    )
    show_n = max(int(limit or 8), 3)
    who = garment or "products"
    party_blocks: list[tuple[str, int, list[tuple[str, int]]]] = []
    chart_labels: list[str] = []
    chart_values: list[float] = []
    for party in parties:
        party_filters = {"brand": party}
        if garment:
            party_filters["garment"] = garment
        total = cypher_count_products(platform, graph_id, party_filters)
        if total is None:
            total = 0
        materials = cypher_group_by(platform, graph_id, "material", party_filters) or []
        usable = _usable_ranked(materials, "material", limit=show_n)
        # Fallback: raw composition text buckets when USES_MATERIAL is sparse.
        if not usable:
            compositions = cypher_group_by(platform, graph_id, "composition", party_filters) or []
            usable = _usable_ranked(compositions, "composition", limit=show_n)
        label = _display_label(party)
        party_blocks.append((label, int(total), usable))
        for name, count in usable[:6]:
            chart_labels.append(f"{label}: {_material_display_label(str(name))}")
            chart_values.append(float(count))
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id=f"composition_compare_{normalize_name(party)}",
                title=f"Composition · {label} · {who}",
                excerpt="; ".join(
                    f"{_material_display_label(str(n))}={c}" for n, c in usable[:5]
                )
                or f"total={total}",
                source_path=graph_id,
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )

    names = [block[0] for block in party_blocks]
    totals = [block[1] for block in party_blocks]
    lines.append(
        f"Composition compare — similar {who} across {' vs '.join(names)} "
        f"({', '.join(f'{n} {t}' for n, t in zip(names, totals))})."
    )
    lines.append(
        "Response must include EVERY brand section below with its full markdown table; "
        "do not drop or merge brands. Material labels are already reader-normalized "
        "(e.g. Cotton 100% means 100% cotton from scraped 'cotton 100')."
    )
    for label, total, usable in party_blocks:
        lines.append("")
        lines.append(f"**{label}** — {total} matching {who}")
        if not usable:
            lines.append(f"No material/composition breakdown was found for {label} {who}.")
            continue
        lines.append("| Material / composition | Products | Share |")
        lines.append("| --- | ---: | ---: |")
        for name, count in usable:
            share = (100.0 * count / total) if total else 0.0
            lines.append(
                f"| {_material_display_label(str(name))} | {count} | {share:.1f}% |"
            )
    if len(chart_labels) >= 2:
        lines.append("")
        chart = {
            "type": "bar",
            "title": f"Material mix · {who} · {' vs '.join(names)}",
            "labels": chart_labels[:12],
            "values": chart_values[:12],
        }
        lines.append("```chart")
        lines.append(json.dumps(chart, ensure_ascii=False))
        lines.append("```")
    # One-line contrast from top materials when both sides have data.
    leads = []
    for label, _total, usable in party_blocks:
        if usable:
            leads.append(f"{label} leads with {_material_display_label(str(usable[0][0]))}")
    if len(leads) >= 2:
        lines.append("")
        lines.append("Compare takeaway: " + "; ".join(leads) + ".")
    caveats.append(
        "Composition compare uses material relationships (and composition text as fallback) "
        "for the named garment across each brand."
    )
    return True


def _fold_facet_token(raw: str | None) -> str | None:
    """Normalize a user/schema label to snake_case; singularize simple plurals."""

    text = str(raw or "").strip().casefold().replace(" ", "_").replace("-", "_")
    if not text:
        return None
    if text in {"product", "products"}:
        return "product"
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith("s") and len(text) > 3 and not text.endswith(("ss", "us", "is")):
        return text[:-1]
    return text


def resolve_count_facet(
    platform,
    graph_id: str,
    raw: str | None,
    *,
    question: str = "",
) -> str | None:
    """Map 'how many <X>' onto a facet key that exists on this graph.

    Keys come from live schema / mapping / kg_stats / relationship facets — not a fixed list.
    """

    cand = _fold_facet_token(raw)
    if cand is None and question:
        try:
            from fashion_brain.ask import _facet_entity_count
            from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

            schema = build_runtime_schema(platform, graph_id)
            cand = _fold_facet_token(_facet_entity_count(question, schema))
        except Exception:
            try:
                from fashion_brain.ask import _facet_entity_count

                cand = _fold_facet_token(_facet_entity_count(question))
            except Exception:
                cand = None
    if not cand or cand == "product":
        return cand

    keys: set[str] = set()
    try:
        keys.update(resolve_analysis_fields(platform, graph_id))
    except Exception:
        pass
    keys.update(_REL_FIELDS)
    try:
        from knowledge_graph.retrieval.kg_stats import facet_inventory

        generated = getattr(getattr(platform, "settings", None), "generated", None)
        keys.update(k for k in facet_inventory(graph_id, generated) if k != "products")
    except Exception:
        pass
    try:
        from knowledge_graph.retrieval.runtime_schema import _facet_filter_key, build_runtime_schema

        ctx = build_runtime_schema(platform, graph_id)
        for item in list(ctx.filter_keys or []) + list(ctx.facet_types or []) + list(
            ctx.product_properties or []
        ):
            keys.add(_facet_filter_key(str(item)))
    except Exception:
        pass
    keys = {k for k in keys if k and is_analysis_facet_attr(k)}

    if cand in keys:
        return cand
    compact = cand.replace("_", "")
    for key in keys:
        variants = {
            key,
            key + "s",
            key[:-1] if key.endswith("s") and len(key) > 3 else key,
            key.replace("_", ""),
            key.replace("_", "") + "s",
        }
        if key.endswith("y") and len(key) > 2:
            variants.add(key[:-1] + "ies")
        if cand in variants or compact in variants:
            return key
    # Unknown to inventory but still try as a property/rel key (cypher may resolve).
    return cand


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
    *,
    question: str = "",
    count_entity: str | None = None,
) -> None:
    filters = clean_product_filters(filters)
    # "how many <facet>s" → distinct values of that graph facet, never the product total.
    entity = resolve_count_facet(platform, graph_id, count_entity, question=question)
    if entity and entity != "product" and not filters and not group_by:
        label = _graph_label(platform, graph_id)
        noun = entity.replace("_", " ")
        generated = getattr(getattr(platform, "settings", None), "generated", None)
        n: int | None = None
        source = "cypher"
        try:
            from knowledge_graph.retrieval.kg_stats import unique_count

            n = unique_count(graph_id, entity, generated)
            if n is not None:
                source = "kg_stats"
        except Exception:
            n = None
        if n is None:
            groups = cypher_group_by(platform, graph_id, entity, {})
            if groups is not None:
                n = sum(
                    1
                    for name, _count in groups
                    if _usable_value(str(name))
                    and str(name).strip().casefold() not in {"(empty)", "(unspecified)"}
                )
                source = "cypher"
        if n is not None:
            lines.append(f"There are {n:,} distinct {noun}s in {label}.")
            evidence.append(
                EvidenceItem(
                    source_id=graph_id,
                    record_id="kg_stats" if source == "kg_stats" else "facet_count",
                    title=f"{label} distinct {noun} count",
                    excerpt=f"{entity}s={n}",
                    source_path=f"kg_stats/{graph_id}.json" if source == "kg_stats" else graph_id,
                    verification_status="observed",
                    graph=graph_id,
                    node_id=None,
                    synthetic=False,
                )
            )
            return
    # Brand/buyer compare: "Primark | Zara" → count each side, do not treat as one name.
    # Drop multi-party only when it looks like prior-chat contamination (facet ask without
    # an explicit compare in the latest question).
    compare_raw = str(filters.get("brand") or filters.get("buyer") or "")
    compare_parties = [p.strip() for p in compare_raw.split("|") if p.strip()] if "|" in compare_raw else []
    other_facets = {
        key: value
        for key, value in filters.items()
        if key not in {"brand", "buyer"} and value not in (None, "")
    }
    named_compare = False
    composition_compare = False
    try:
        from fashion_brain.ask import _compare_party_names, is_composition_compare_question

        named_compare = len(_compare_party_names(question)) >= 2
        composition_compare = is_composition_compare_question(question)
    except Exception:
        named_compare = False
        composition_compare = False
    if composition_compare and append_neo4j_composition_compare_answer(
        platform,
        principal,
        graph_id,
        filters,
        question,
        limit,
        lines,
        evidence,
        caveats,
    ):
        return
    if len(compare_parties) >= 2 and other_facets and not named_compare:
        # Drop stale multi-party brand; answer the facet ask (e.g. material=cotton).
        filters = dict(other_facets)
        compare_parties = []
    if len(compare_parties) >= 2 and other_facets and named_compare:
        # Intentional compare + garment/material scope: keep parties; apply shared facets per side.
        rows: list[tuple[str, int]] = []
        for party in compare_parties:
            side = {**other_facets, "brand": party}
            n = cypher_count_products(platform, graph_id, side)
            if n is None:
                n = cypher_count_products(platform, graph_id, {**other_facets, "buyer": party})
            rows.append((_display_label(party), int(n or 0)))
        scope = _filter_phrase(other_facets) or "matching products"
        lines.append(f"Comparing product counts ({scope}):")
        lines.append("| Brand | Products |")
        lines.append("| --- | ---: |")
        for name, count in rows:
            lines.append(f"| {name} | {count} |")
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="brand_compare_scoped",
                title="Scoped brand product count comparison",
                excerpt="; ".join(f"{name}={count}" for name, count in rows),
                source_path=graph_id,
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        return
    if len(compare_parties) >= 2:
        rows: list[tuple[str, int]] = []
        for party in compare_parties:
            n = cypher_count_products(platform, graph_id, {"brand": party})
            if n is None:
                n = cypher_count_products(platform, graph_id, {"buyer": party})
            rows.append((_display_label(party), int(n or 0)))
        lines.append("Comparing product counts:")
        lines.append("| Brand | Products |")
        lines.append("| --- | ---: |")
        for name, count in rows:
            lines.append(f"| {name} | {count} |")
        ranked = sorted(rows, key=lambda item: (-item[1], item[0].casefold()))
        if ranked[0][1] > ranked[1][1]:
            lines.append(
                f"{ranked[0][0]} has more products ({ranked[0][1]} vs {ranked[1][1]})."
            )
        elif ranked[0][1] == ranked[1][1]:
            lines.append(f"Both have the same number of products ({ranked[0][1]}).")
        else:
            lines.append(
                f"{ranked[1][0]} has more products ({ranked[1][1]} vs {ranked[0][1]})."
            )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="brand_compare",
                title="Brand product count comparison",
                excerpt="; ".join(f"{name}={count}" for name, count in rows),
                source_path=graph_id,
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        return

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

    # Plain unfiltered total — no automatic brand mix (ask "by brand" for that).
    if not filters and not group_by:
        total = neo4j_product_count(platform, graph_id)
        cached = _stats_breakdown("brand")
        if cached:
            total, _groups = cached
        lines.append(f"There are {total:,} products.")
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="kg_stats" if cached else "count",
                title=f"{label} product total",
                excerpt=f"products={total}",
                source_path=stats_file if cached else graph_id,
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        return
    if not filters and group_by:
        cached = _stats_breakdown(str(group_by))
        if cached:
            total, groups = cached
            lines.append(f"There are {total:,} products.")
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
                lines.append(f"There are {cached:,} products{phrase}.")
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
            usable = _usable_ranked(groups, str(group_by))
            lines.append(f"There are {total:,} products" + (f" {phrase}." if phrase else "."))
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

    # Open listings (no user top-N): short sample. Explicit limit: honour it, still capped.
    open_listing = limit is None
    show_n = max(int(limit or 5), 1)
    if open_listing:
        show_n = min(show_n, 5)
    else:
        show_n = min(show_n, 25)
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
            open_listing=open_listing,
            show_n=show_n,
        )
    except Exception:
        pass
    if not total and not filters:
        lines.append("There are 0 visible products for this ask.")
    else:
        lines.append(f"There are {total:,} products" + (f" {phrase}." if phrase else "."))
        if group_by:
            grouped = cypher_group_by(platform, graph_id, str(group_by), filters)
            if grouped is None:
                grouped = _count_by(platform, principal, graph_id, group_by, filters)
            groups = _usable_ranked(grouped or [], str(group_by))
            lines.extend(_dimension_breakdown_lines(group_by, groups, total, limit))
        elif filters:
            # Prefer count-wise description over dumping every matching name.
            brand_groups = cypher_group_by(platform, graph_id, "brand", filters)
            if brand_groups:
                usable_brands = _usable_ranked(brand_groups, "brand")
                if usable_brands:
                    lines.append("By brand:")
                    lines.extend(
                        _dimension_breakdown_lines("brand", usable_brands, total, 8)
                    )
            if "material" in filters or "color" in filters:
                cat_groups = cypher_group_by(platform, graph_id, "category", filters)
                if cat_groups:
                    usable_cats = _usable_ranked(cat_groups, "category")
                    if usable_cats:
                        lines.append("By category:")
                        lines.extend(
                            _dimension_breakdown_lines("category", usable_cats, total, 8)
                        )
            if sample:
                lines.append(
                    f"Example products ({min(len(sample), show_n)} of {total}):"
                )
                lines.append("| Brand | Name | Price | Availability |")
                lines.append("| --- | --- | --- | --- |")
                for node in sample[:show_n]:
                    props = node.properties or {}
                    lines.append(
                        f"| {_display_label(str(props.get('brand') or '—'))} | "
                        f"{props.get('name') or node.id} | {props.get('price') or '—'} | "
                        f"{props.get('availability') or '—'} |"
                    )
            if open_listing and int(total or 0) > show_n:
                caveats.append(
                    "Open product listing: total + brand/category mix + a few examples. "
                    "Ask for a top-N or a brand filter to see more named products."
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
    """Type-wise top rankings for a brand/selection on any Neo4j product graph.

    ``analysis_fields`` come from that graph's live schema (via resolve_analysis_fields).
    Works for ecom, plmex, or any registered product KG — not e-commerce-only.
    """

    filters = _align_party_filters_for_graph(platform, graph_id, clean_product_filters(filters))
    overview = cypher_price_overview(platform, graph_id, filters)
    total = int((overview or {}).get("total_products") or 0) if overview else None
    if total is None:
        total = cypher_count_products(platform, graph_id, filters)
    sample = cypher_sample_products(platform, graph_id, filters, limit=5)
    if total is None:
        matched = match_neo4j_products(platform, principal, graph_id, filters)
        total = len(matched)
        sample = matched[:5]
        fields_resolved = resolve_analysis_fields(platform, graph_id, analysis_fields)
        ranked = _profile_nodes(platform, graph_id, matched, fields=fields_resolved)
    else:
        ranked = {}
        fields_pre = resolve_analysis_fields(platform, graph_id, analysis_fields)
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
    who_raw = filters.get("brand") or filters.get("buyer") or "this selection"
    who = _display_label(str(who_raw))
    fields = resolve_analysis_fields(platform, graph_id, analysis_fields)
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
            facet_queries=len(ranked),
        )
    except Exception:
        pass
    if not total:
        lines.append(f"There are no products for {who} to analyse.")
        return

    # Metrics as plain lines (no ###). Only real attribute slices use ### so merge can
    # pick visualizations by structure — no heading-name allow/deny lists.
    lines.append(f"Product analysis for {who}: **{total}** products.")
    if overview:
        avg_p = overview.get("average_price")
        min_p = overview.get("min_price")
        max_p = overview.get("max_price")
        if avg_p is not None or min_p is not None or max_p is not None:
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")
            lines.append(f"| Products | {total} |")
            if avg_p is not None:
                lines.append(f"| Average price | {round(float(avg_p), 2)} |")
            if min_p is not None:
                lines.append(f"| Min price | {round(float(min_p), 2)} |")
            if max_p is not None:
                lines.append(f"| Max price | {round(float(max_p), 2)} |")
    active_party = {
        normalize_name(str(filters.get(k) or ""))
        for k in ("brand", "buyer")
        if filters.get(k)
    }
    reserved_values = _schema_reserved_value_keys(platform, graph_id)
    # Compact uniqueness summary (structural + schema-reserved only; LLM curates the rest).
    uniqueness: list[str] = []
    for field in fields:
        if not is_analysis_facet_attr(str(field)):
            continue
        usable = _usable_ranked(
            ranked.get(field),
            str(field),
            active_party=active_party,
            reserved_values=reserved_values,
        )
        if usable and _facet_worth_showing(usable, total=total):
            label = field.replace("_", " ")
            uniqueness.append(f"{len(usable)} {label}{'s' if not label.endswith('s') else ''}")
    if uniqueness:
        lines.append("Distinct values: " + ", ".join(uniqueness[:8]) + ".")

    lines.append(f"Attribute rankings (top {top_n} each). One visualization per attribute.")
    # Skip facets the ask already fixed to one value (self-echo breakdowns).
    fixed_filters = {
        normalize_name(str(k)): normalize_name(str(v))
        for k, v in (filters or {}).items()
        if v not in (None, "")
    }
    for field in fields:
        if not is_analysis_facet_attr(str(field)):
            continue
        choices = _usable_ranked(
            ranked.get(field),
            str(field),
            active_party=active_party,
            reserved_values=reserved_values,
            limit=top_n,
        )
        if not choices:
            continue
        field_key = normalize_name(str(field))
        names = {normalize_name(str(name)) for name, _ in choices}
        # Already filtered to one party/value → 100% slice is not insight.
        if field_key in fixed_filters and (
            names <= {fixed_filters[field_key]} or (len(choices) == 1 and next(iter(names)) == fixed_filters[field_key])
        ):
            continue
        if active_party and (names <= active_party or (len(choices) == 1 and next(iter(names)) in active_party)):
            continue
        # Exact retail prices already covered by overview avg/min/max — skip noisy €xx.xx tables/charts.
        if _looks_like_price_labels([name for name, _ in choices]):
            continue
        if not _facet_worth_showing(choices, total=total):
            continue
        field_label = field.replace("_", " ").title()
        display_choices = [(_display_label(name), count) for name, count in choices]
        # One chart per attribute — UI renders ```chart; markers are for merge only.
        lines.append(f"<!--fb:viz:{field}-->")
        lines.append(f"### {field_label}")
        top_name, top_count = display_choices[0]
        share = round(100.0 * top_count / total, 1) if total else 0.0
        lines.append(f"Leading value: {top_name} ({top_count}, {share}%).")
        lines.extend(
            _chart_lines(
                "bar",
                field_label,
                [name for name, _ in display_choices],
                [int(count) for _, count in display_choices],
            )
        )
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
                return _user_facing_source_name(str(item.name or graph_id))
    except Exception:
        pass
    return _user_facing_source_name(graph_id)


def _user_facing_source_name(name: str) -> str:
    """Strip internal infra words from registry titles shown to end users."""

    text = str(name or "").strip()
    text = re.sub(r"\bknowledge\s*graphs?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:neo4j|postgres|plmex|kg)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip(" -–—|/\\")
    return text or "our products"


def _display_label(value: str) -> str:
    """User-facing label from stored/normalized facet values."""

    raw = str(value or "").strip()
    if not raw:
        return raw
    folded = re.sub(r"\s+", " ", raw.casefold().replace("&", " and "))
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    fixes = {
        "instock": "In stock",
        "outofstock": "Out of stock",
        "comingsoon": "Coming soon",
        "lowonstock": "Low on stock",
        "mixed": "Mixed",
        "multicoloured": "Multicoloured",
        "multicolored": "Multicolored",
        "greymarl": "Grey marl",
        "navyblue": "Navy blue",
        "cowleather": "Cow leather",
        "fauxsuede": "Faux suede",
        "fauxleather": "Faux leather",
        "coordsets": "Co-ord sets",
        "coord": "Co-ord",
        "canda": "C&A",
        "handm": "H&M",
    }
    if compact in fixes:
        return fixes[compact]
    # "women s shoes" / "men s tops" → Women's / Men's
    text = re.sub(r"\b(women|men)\s+s\b", r"\1's", folded, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    # Keep small words lower inside a title-cased phrase.
    small = {"and", "or", "of", "the", "a", "an", "for", "to", "in", "on"}
    words = []
    for i, part in enumerate(text.split()):
        if "'" in part:
            # women's → Women's
            head, _, tail = part.partition("'")
            part = head.capitalize() + "'" + tail
        elif i > 0 and part in small:
            pass
        else:
            part = part.capitalize()
        words.append(part)
    return " ".join(words)


def _material_display_label(value: str) -> str:
    """Normalize scraped composition tokens for readers (Cotton 100 → Cotton 100%)."""

    raw = str(value or "").strip()
    if not raw:
        return raw
    # Already a clear percent composition.
    if re.search(r"\d\s*%", raw):
        base = _display_label(re.sub(r"\s*%", "%", raw))
        return re.sub(r"(\d)\s*%", r"\1%", base)
    # "Cotton 100" / "Elastane 5" / "Cotton100"
    match = re.match(r"^([A-Za-z][A-Za-z\s\-_/]*?)\s*(\d{1,3})$", raw.strip())
    if match:
        fibre = _display_label(match.group(1))
        pct = int(match.group(2))
        if 1 <= pct <= 100 and fibre:
            return f"{fibre} {pct}%"
    # "100 Cotton" / "95 cotton"
    match = re.match(r"^(\d{1,3})\s+([A-Za-z][A-Za-z\s\-_/]*)$", raw.strip())
    if match:
        pct = int(match.group(1))
        fibre = _display_label(match.group(2))
        if 1 <= pct <= 100 and fibre:
            return f"{fibre} {pct}%"
    return _display_label(raw)


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
    # Value is just the field name → not a real material filter.
    if not want or want in {"material", "materials", "composition", "fabric"}:
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
        if not name or name in {"material", "materials", "composition", "fabric"}:
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
    for field in fields or ():
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
    "color": ("HAS_COLOR", "PRIMARY_COLOR"),
    "material": ("USES_MATERIAL", "PRIMARY_MATERIAL"),
    "availability": ("HAS_AVAILABILITY",),
    "brand": ("HAS_BRAND",),
    "buyer": ("HAS_BUYER",),
    "garment_type": ("HAS_GARMENT_TYPE",),
    "gender": ("HAS_GENDER",),
    "gauge": ("HAS_GAUGE",),
    "sample_collection_type": ("HAS_SAMPLE_TYPE",),
    "plm_main_category": ("HAS_PLM_MAIN_CATEGORY",),
    "plm_sub_category": ("HAS_PLM_SUB_CATEGORY",),
    "plm_sub_sub_category": ("HAS_PLM_SUB_SUB_CATEGORY",),
    "design_for": ("DESIGNED_FOR",),
    "knit_machine_type": ("USES_KNIT_MACHINE",),
    "design_feature": ("HAS_DESIGN_FEATURE",),
    "style_feature": ("HAS_STYLE_FEATURE",),
    "department": ("BELONGS_TO_DEPARTMENT",),
    "fit": ("HAS_FIT",),
    "construction": ("HAS_CONSTRUCTION",),
    "section": ("HAS_SECTION",),
    "family": ("HAS_FAMILY",),
    "subfamily": ("HAS_SUBFAMILY", "HAS_SUB_CATEGORY"),
    "ai_event_tags": ("HAS_EVENT_TAG",),
    "ai_construction_tags": ("HAS_CONSTRUCTION_TAG",),
    "ai_material_tags": ("HAS_AI_MATERIAL_TAG",),
    "ai_finish_tags": ("HAS_FINISH_TAG",),
    "ai_special_tags": ("HAS_SPECIAL_TAG",),
    "season_year": ("IN_SEASON_YEAR",),
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


def _schema_reserved_value_keys(platform, graph_id: str | None) -> set[str]:
    """Live schema dimension/type names — never valid as facet VALUES (API-derived)."""

    out: set[str] = set()
    if not graph_id:
        return out
    try:
        from knowledge_graph.retrieval.metadata import metadata_overview

        overview = metadata_overview(platform, graph_id) or {}
    except Exception:
        return out
    schema = overview.get("schema") if isinstance(overview, dict) else None
    if not isinstance(schema, dict):
        return out

    def _add(token: str) -> None:
        key = normalize_name(str(token or ""))
        if not key or len(key) < 2:
            return
        out.add(key)
        out.add(key + "s")
        for part in re.findall(r"[a-z0-9]+", key):
            if len(part) >= 3:
                out.add(part)
                out.add(part + "s")

    for label, body in schema.items():
        _add(str(label))
        if not isinstance(body, dict):
            continue
        for attr in body.get("attributes") or []:
            _add(str(attr))
        for rel in body.get("relationships") or []:
            if isinstance(rel, dict):
                _add(str(rel.get("type") or rel.get("name") or ""))
            else:
                _add(str(rel))
    return out


# BOM / composition *roles* (not fibres) — appear as material values from scraped PDP parts.
_COMPOSITION_ROLE_KEYS = frozenset(
    normalize_name(token)
    for token in (
        "outer shell",
        "outer",
        "shell",
        "lining",
        "main fabric",
        "secondary fabric",
        "fabric content",
        "fabric",
        "content",
        "composition",
        "body",
        "fill",
        "filling",
        "padding",
        "interlining",
        "insole",
        "sole",
        "upper",
        "top",
        "bottom",
        "hood",
        "sleeve",
        "sleeves",
        "collar",
        "pocket",
        "pockets",
        "main",
        "secondary",
        "trim",
        "trims",
        "warning",
        "fire safety",
        "care",
        "keep away from fire",
    )
)


def _usable_value(
    value: str,
    field: str | None = None,
    *,
    active_party: set[str] | None = None,
    reserved_values: set[str] | None = None,
) -> bool:
    """Structural usability — empties, schema-as-value, party self-echo, BOM roles as material."""

    folded = str(value or "").strip().casefold()
    if folded in {"", "unspecified", "(unspecified)", "null", "none", "n/a", "na", "-", "other", "(other)"}:
        return False
    key = normalize_name(folded)
    if not key or key in {"unspecified", "null", "none", "n a", "na", "other"}:
        return False
    # Value that is only the field name itself is never a real facet value.
    if field:
        field_key = normalize_name(str(field))
        if key == field_key or key == field_key + "s" or key + "s" == field_key:
            return False
        # Material mix should be fibres, not BOM roles or yarn-count labels (nm 1/50).
        if field_key in {"material", "materials", "composition"} and (
            key in _COMPOSITION_ROLE_KEYS
            or any(part in key for part in ("outer shell", "main fabric", "lining", "keep away from fire", "fire safety"))
            or len(key.split()) >= 8
            or key.startswith("nm ")
            or re.fullmatch(r"nm\s*\d+(?:\s+\d+)?", key)
            or re.fullmatch(r"\d+(?:\.\d+)?", key)
        ):
            return False
        # Empty numeric gsm/weight placeholders are not sample-ready values.
        if field_key in {"gsm", "weight"} and re.fullmatch(r"0+(?:[.\s]0+)*", key):
            return False
    if active_party and key in active_party:
        return False
    if reserved_values and key in reserved_values:
        return False
    return True


def _usable_ranked(
    ranked_items: list[tuple[str, int]] | None,
    field: str,
    *,
    active_party: set[str] | None = None,
    reserved_values: set[str] | None = None,
    limit: int | None = None,
) -> list[tuple[str, int]]:
    usable = [
        (name, count)
        for name, count in (ranked_items or [])
        if _usable_value(
            name,
            field,
            active_party=active_party,
            reserved_values=reserved_values,
        )
    ]
    if limit is None:
        return usable
    return usable[: max(int(limit), 1)]


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


def _looks_like_price_labels(names: list[str]) -> bool:
    """True when most values look like currency price tags (€45.95), not categories."""

    if not names:
        return False
    priced = 0
    for name in names:
        text = str(name or "").strip()
        if re.search(r"(?:€|\$|£)\s*\d", text) or re.fullmatch(r"\d+[.,]\d{2}", text):
            priced += 1
    return priced >= max(1, (len(names) + 1) // 2)


def _analysis_viz_for_field(
    choices: list[tuple[str, int]],
    *,
    total: int | None = None,
) -> str:
    """Pick one viz from the data shape: pie | bar | table — never by field-name lists."""

    n = len(choices)
    if n < 2:
        return "table"
    if _looks_like_price_labels([name for name, _ in choices]):
        return "table"
    counts = [int(count) for _name, count in choices]
    covered = sum(counts)
    # Multi-label facets (e.g. materials) often sum far above product total → bar, not pie.
    multi_label = bool(total and covered > int(total) * 1.15)
    if not multi_label and 2 <= n <= 8:
        return "pie"
    if 2 <= n <= 12:
        return "bar"
    return "table"


def _dimension_breakdown_lines(
    field: str,
    groups: list[tuple[str, int]],
    total: int,
    limit: int | None,
) -> list[str]:
    """Breakdown with one best-fit visualization (chart XOR table)."""

    top_n = max(int(limit or 5), 1)
    shown = _usable_ranked(groups, str(field), limit=top_n)
    if not shown:
        return [f"No {field.replace('_', ' ')} breakdown."]
    field_label = field.replace("_", " ")
    labeler = (
        _material_display_label
        if normalize_name(str(field)) in {"material", "materials", "composition"}
        else _display_label
    )
    display_shown = [(labeler(name), count) for name, count in shown]
    top_name, top_count = display_shown[0]
    share = round(100.0 * top_count / total, 1) if total else 0.0
    lines = [
        f"{field_label} breakdown (highest count first):",
        f"Leading {field_label}: {top_name} ({top_count} of {total}, {share}%).",
    ]
    # Ranked numbers only — response LLM decides chart vs table vs prose.
    lines.append("| Name | Count | Share % |")
    lines.append("| --- | ---: | ---: |")
    for name, count in display_shown:
        row_share = round(100.0 * count / total, 1) if total else 0.0
        lines.append(f"| {name} | {count} | {row_share} |")
    return lines


def _filter_phrase(filters: dict) -> str:
    if not filters:
        return ""
    return "with " + " and ".join(f"{key} {value}" for key, value in filters.items())


def _excerpt(node) -> str:
    props = node.properties or {}
    bits = [props.get("brand"), props.get("name"), props.get("category"), props.get("price"), props.get("currency")]
    return " | ".join(str(bit) for bit in bits if bit)
