"""Query Product nodes in any Neo4j product graph registered in kg_registry.

Graph ids are not hard-coded: callers pass the registry product graph (e.g. ecom_products).
Fashion Brain should only resolve which product graph via GET /graphs (kind=product).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from knowledge_graph.access import Principal, allows
from knowledge_graph.models import EvidenceItem
from knowledge_graph.ontology import normalize_name


def _cypher_token(value: str) -> str:
    """Lowercase token for Cypher equality — keep hyphens (c-and-a), drop other punctuation."""

    text = str(value or "").strip().casefold()
    text = re.sub(r"[^a-z0-9\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _soft_match_needles(value: str) -> list[str]:
    """Universal case/plural/spacing needles for ANY facet value in Neo4j.

    Applies to every column (category, color, gender, brand labels, material,
    garment_type, AI tags, …) — not garment-only. Example: Sweaters / sweater /
    SWEATER all become searchable needles against stored ``sweater``.
    """

    raw = str(value or "").strip()
    if not raw:
        return []
    token = _cypher_token(raw).replace("-", " ")
    compact = re.sub(r"\s+", "", token)
    needles: list[str] = []

    def _add(item: str) -> None:
        t = str(item or "").strip().casefold()
        t = re.sub(r"\s+", " ", t)
        if t and t not in needles:
            needles.append(t)

    _add(token)
    _add(normalize_name(raw))
    if compact and compact != token:
        _add(compact)

    try:
        from knowledge_graph.retrieval.kg_stats import _plural_forms

        seeds = list(needles)
        for seed in seeds:
            for form in _plural_forms(seed):
                _add(form)
            # Also pluralize each whitespace token (navy blue → navy blues).
            parts = seed.split()
            if len(parts) > 1:
                for i, part in enumerate(parts):
                    for form in _plural_forms(part):
                        alt = list(parts)
                        alt[i] = form
                        _add(" ".join(alt))
    except Exception:
        if token.endswith("s") and len(token) > 2:
            _add(token[:-1])
        else:
            _add(token + "s")

    # Optional known garment surface forms (still general needle list).
    try:
        from fashion_brain.ask import _GARMENT_MATCH_ALIASES

        for alias in _GARMENT_MATCH_ALIASES.get(normalize_name(token), ()):
            _add(_cypher_token(alias).replace("-", " "))
    except Exception:
        pass
    return needles


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
        "ref_norm",
        "search_text",
        "searchtext",
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
        "search",
        "ref",
        "norm",
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


def _pluralize_label(label: str) -> str:
    if label.endswith("y") and len(label) > 1 and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    if label.endswith(("s", "x", "z", "ch", "sh")):
        return label + "es"
    return label + "s"


_ANALYSIS_FIELD_LABELS = {
    "plm_main_category": "Main category",
    "plm_sub_category": "Product group",
    "plm_sub_sub_category": "Product type",
    "garment_type": "Garment type",
    "color_family": "Colour family",
    "care_instruction": "Care",
    "season_year": "Season year",
    "color": "Colour",
}


def _human_analysis_field_label(field: str) -> str:
    """Buyer-friendly facet title (hide raw PLM column chrome)."""

    key = str(field or "").strip().casefold().replace(" ", "_").replace("-", "_")
    if key in _ANALYSIS_FIELD_LABELS:
        return _ANALYSIS_FIELD_LABELS[key]
    label = key.replace("_", " ").strip()
    label = re.sub(r"(?i)^plm\s+", "", label)
    return label[:1].upper() + label[1:] if label else "Attribute"


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


def resolve_focused_analysis_field(
    platform,
    graph_id: str,
    group_by: str | None,
    filters: dict | None = None,
) -> list[str]:
    """Pick one analysis facet for focused group_by, with hierarchy role fallbacks.

    LLM/user often say ``by category``; ecom/plmex store that role under different
    columns. Try remapped primary then category-like siblings until a usable mix
    appears (not mostly unspecified).
    """

    from fashion_brain.planning.semantic_field_map import resolve_group_by_candidates

    candidates = resolve_group_by_candidates(group_by, graph_id)
    if not candidates:
        return []
    clean = _align_party_filters_for_graph(
        platform, graph_id, clean_product_filters(dict(filters or {}))
    )
    total = cypher_count_products(platform, graph_id, clean)
    if total is None:
        total = 0
    for field in candidates:
        grouped = cypher_group_by(platform, graph_id, str(field), clean)
        if grouped is None:
            continue
        usable = _usable_ranked(grouped, str(field))
        facet_total = int(total) if total else sum(int(n) for _, n in grouped) or 0
        if not usable or facet_total <= 0:
            continue
        if not _facet_worth_showing(usable, total=max(facet_total, 1)):
            continue
        # Thin tails after dropping unspecified do not count as a real mix —
        # fall through to role-similar columns (family/section/…).
        covered = sum(int(n) for _, n in usable)
        if covered / float(facet_total) < 0.15:
            continue
        return [str(field)]
    # Last resort: keep the remapped primary so callers still run one field.
    return [candidates[0]]


def resolve_group_by_with_fallback(
    platform,
    graph_id: str,
    group_by: str | None,
    filters: dict | None = None,
) -> str | None:
    """Count/breakdown path: remap + hierarchy fallback to a facet that has data."""

    picked = resolve_focused_analysis_field(platform, graph_id, group_by, filters)
    return picked[0] if picked else (str(group_by).strip() if group_by else None)


_GRAPH_SCOPE_FILTER_NOISE = frozenset(
    {
        "plm",
        "plmex",
        "plm-ex",
        "ecom",
        "ecomm",
        "ecommerce",
        "commerce",
        "own",
        "own_products",
        "ecom_products",
        "plmex_products",
        "catalog",
        "catalogue",
        "database",
        "graph",
        "graphs",
        "knowledge",
        "kg",
        "source",
        "sources",
        "system",
        "data",
        "are in plm",
        "in plm",
        "in plmex",
    }
)


def clean_product_filters(filters: dict) -> dict:
    cleaned = {key: value for key, value in (filters or {}).items() if value not in (None, "")}
    cleaned.pop("parent_of", None)  # structural lookup — handled in brain, not product WHERE
    # Intention already remaps buyer→brand for brand-only graphs. Keep buyer for PLM/HAS_BUYER.
    cleaned.pop("season_code", None)
    cleaned.pop("upcoming", None)
    # "PLM data" / graph names are scope, never brand/buyer filters.
    _currency_noise = {
        "usd", "gbp", "eur", "sek", "dkk", "nok", "pln", "czk", "chf", "jpy", "cny",
        "aud", "cad", "dollar", "dollars", "pound", "pounds", "sterling", "euro",
        "euros", "currency", "currencies", "both",
    }
    for key in ("brand", "buyer"):
        raw = cleaned.get(key)
        if raw in (None, ""):
            continue
        want = normalize_name(str(raw))
        if want in _currency_noise or want in _GRAPH_SCOPE_FILTER_NOISE or any(
            want == noise or want.startswith(noise + " ") for noise in ("plm", "plmex", "ecom", "own")
        ):
            cleaned.pop(key, None)
    brand = cleaned.get("brand")
    buyer = cleaned.get("buyer")
    if brand and buyer and _party_match_token(str(brand)) == _party_match_token(str(buyer)):
        cleaned.pop("buyer", None)
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
    """Return a session factory that logs every Cypher hit to the serve terminal."""

    factory = getattr(platform.graph, "_session", None)
    if not callable(factory):
        return None

    class _LoggingSession:
        def __init__(self, session):
            self._session = session

        def run(self, query, parameters=None, **kwargs):
            try:
                from knowledge_graph.retrieval.cypher_log import log_cypher_hit

                params = parameters if parameters is not None else kwargs.get("parameters")
                log_cypher_hit(str(query), params if isinstance(params, dict) else None, source="template")
            except Exception:
                pass
            if parameters is None:
                return self._session.run(query, **kwargs)
            return self._session.run(query, parameters, **kwargs)

        def __getattr__(self, name):
            return getattr(self._session, name)

    class _LoggingSessionFactory:
        def __call__(self):
            return self

        def __enter__(self):
            self._cm = factory()
            self._raw = self._cm.__enter__()
            return _LoggingSession(self._raw)

        def __exit__(self, exc_type, exc, tb):
            return self._cm.__exit__(exc_type, exc, tb)

    return _LoggingSessionFactory


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


def _rel_filter_clause(field: str, param: str, *, needles_param: str | None = None) -> str:
    """Match a facet via product property and/or its ontology relationships.

    When ``needles_param`` is set, match any soft needle (sweater/sweaters) via
    equality or CONTAINS against property and related entity names.
    """

    rels = _REL_FIELDS.get(field) or ()
    if needles_param:
        prop = (
            f"ANY(needle IN ${needles_param} WHERE toLower(coalesce(toString(p.`{field}`),'')) = needle "
            f"OR toLower(coalesce(toString(p.`{field}`),'')) CONTAINS needle)"
        )
        if not rels:
            return f"({prop})"
        rel_alt = "|".join(rels)
        return (
            f"("
            f"{prop} "
            f"OR EXISTS {{ MATCH (p)-[:{rel_alt}]->(x:Entity) "
            f"WHERE ANY(needle IN ${needles_param} WHERE toLower(coalesce(x.name,'')) = needle "
            f"OR toLower(coalesce(x.name,'')) CONTAINS needle) }}"
            f")"
        )
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

    # Graph-aware aliases when caller already set params["g"] (count / retrieve paths).
    graph_id = params.get("g")
    if graph_id:
        try:
            from fashion_brain.planning.semantic_field_map import apply_semantic_field_map

            filters, _ = apply_semantic_field_map(filters, str(graph_id))
        except Exception:
            pass
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
        needles = _soft_match_needles(str(category))
        params["category_needles"] = needles
        # Colour words as category needles must NOT match product titles
        # ("Black Pull On…") — that path invented 743 "category Black" hits.
        try:
            from fashion_brain.planning.metadata_resolver import _looks_like_basic_color

            colorish_category = _looks_like_basic_color(str(category)) or any(
                _looks_like_basic_color(n) for n in needles
            )
        except Exception:
            colorish_category = False
        if colorish_category:
            where.append(
                "("
                "EXISTS { MATCH (p)-[:BELONGS_TO_CATEGORY|HAS_FAMILY|HAS_SUBFAMILY|HAS_SUB_CATEGORY|"
                "HAS_GARMENT_TYPE|HAS_PLM_SUB_SUB_CATEGORY|HAS_PLM_MAIN_CATEGORY|HAS_PLM_SUB_CATEGORY]->(c:Entity) "
                "WHERE ANY(needle IN $category_needles WHERE toLower(coalesce(c.name,'')) = needle) }"
                ")"
            )
        else:
            where.append(
                "("
                "ANY(needle IN $category_needles WHERE toLower(coalesce(p.category_path,'')) CONTAINS needle) "
                "OR ANY(needle IN $category_needles WHERE toLower(coalesce(p.name,'')) CONTAINS needle) "
                "OR EXISTS { MATCH (p)-[:BELONGS_TO_CATEGORY|HAS_FAMILY|HAS_SUBFAMILY|HAS_SUB_CATEGORY|"
                "HAS_GARMENT_TYPE|HAS_PLM_SUB_SUB_CATEGORY|HAS_PLM_MAIN_CATEGORY|HAS_PLM_SUB_CATEGORY]->(c:Entity) "
                "WHERE ANY(needle IN $category_needles WHERE toLower(coalesce(c.name,'')) = needle "
                "OR toLower(coalesce(c.name,'')) CONTAINS needle) }"
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
            "EXISTS { MATCH (p)-[:HAS_EVENT_TAG|SUITED_FOR_OCCASION]->(e:Entity) "
            "WHERE toLower(coalesce(e.name,'')) = $season_name }"
            "OR toLower(coalesce(p.style_reference,'')) CONTAINS $season_name "
            ")"
        )
        handled.add("season_name")
    section = filters.get("section")
    if section:
        needles = _soft_match_needles(str(section))
        params["section_needles"] = needles
        where.append(
            "("
            "ANY(needle IN $section_needles WHERE toLower(coalesce(p.category_path,'')) CONTAINS needle) "
            "OR EXISTS { MATCH (p)-[:HAS_SECTION|FOR_TARGET_GROUP]->(s:Entity) "
            "WHERE ANY(needle IN $section_needles WHERE toLower(coalesce(s.name,'')) = needle "
            "OR toLower(coalesce(s.name,'')) CONTAINS needle) }"
            ")"
        )
        handled.add("section")
    material = filters.get("material")
    material_min_pct = filters.get("material_min_pct")
    if material_min_pct in (None, "") and filters.get("min_percentage") not in (None, ""):
        material_min_pct = filters.get("min_percentage")
    min_pct_value: float | None = None
    if material_min_pct not in (None, ""):
        try:
            min_pct_value = float(material_min_pct)
        except (TypeError, ValueError):
            min_pct_value = None
    if material:
        needles = _soft_match_needles(str(material))
        params["material_needles"] = needles
        mat_rels = "|".join(_REL_FIELDS.get("material") or ("USES_MATERIAL", "PRIMARY_MATERIAL"))
        if min_pct_value is not None:
            params["material_min_pct"] = min_pct_value
            # Percentage threshold requires real USES_MATERIAL/PRIMARY_MATERIAL edges.
            where.append(
                "("
                f"EXISTS {{ MATCH (p)-[rm:{mat_rels}]->(m:Entity) "
                "WHERE ANY(needle IN $material_needles WHERE toLower(coalesce(m.name,'')) = needle "
                "OR toLower(coalesce(m.name,'')) CONTAINS needle) "
                "AND rm.percentage IS NOT NULL AND rm.percentage >= $material_min_pct }"
                ")"
            )
        else:
            where.append(
                "("
                "ANY(needle IN $material_needles WHERE toLower(coalesce(p.composition,'')) CONTAINS needle) "
                f"OR EXISTS {{ MATCH (p)-[:{mat_rels}]->(m:Entity) "
                "WHERE ANY(needle IN $material_needles WHERE toLower(coalesce(m.name,'')) = needle "
                "OR toLower(coalesce(m.name,'')) CONTAINS needle) }"
                ")"
            )
        handled.add("material")
    if "material_min_pct" in filters:
        handled.add("material_min_pct")
    if "min_percentage" in filters:
        handled.add("min_percentage")
    availability = filters.get("availability")
    if availability:
        params["availability"] = _avail_token(str(availability))
        where.append(_availability_where_cypher())
        handled.add("availability")
    color = filters.get("color")
    if color:
        needles = _soft_match_needles(str(color))
        params["color_needles"] = needles
        where.append(
            "("
            "EXISTS { MATCH (p)-[:HAS_COLOR|PRIMARY_COLOR]->(c:Entity) "
            "WHERE ANY(needle IN $color_needles WHERE toLower(coalesce(c.name,'')) = needle "
            "OR toLower(coalesce(c.name,'')) CONTAINS needle) }"
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
        needles_param = f"f_{key}_needles"
        needles = _soft_match_needles(str(value))
        params[param] = needles[0] if needles else _cypher_token(str(value)).replace("-", " ")
        params[needles_param] = needles or [params[param]]
        if key in _REL_FIELDS:
            where.append(_rel_filter_clause(str(key), param, needles_param=needles_param))
            handled.add(key)
            continue
        if key in _PROP_FILTER_FIELDS:
            where.append(
                f"ANY(needle IN ${needles_param} WHERE "
                f"toLower(coalesce(toString(p.`{key}`),'')) CONTAINS needle)"
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


def cypher_availability_transitions(
    platform,
    graph_id: str,
    filters: dict | None = None,
    *,
    from_status: str | None = None,
    to_status: str | None = None,
    limit: int = 10,
) -> list[dict] | None:
    """Self-join ordered AvailabilitySnapshot pairs for from→to status transitions."""

    session_factory = _session(platform)
    if not session_factory:
        return None
    limit = max(1, min(int(limit or 10), 50))
    params: dict = {"g": graph_id}
    where = _filter_where(clean_product_filters(dict(filters or {})), params)
    if where is None:
        return None
    from_clause = ""
    to_clause = ""
    if from_status:
        params["from_status"] = str(from_status).casefold()
        from_clause = (
            "AND toLower(coalesce(s1.status, s1.name, '')) CONTAINS $from_status "
        )
    if to_status:
        params["to_status"] = str(to_status).casefold()
        to_clause = "AND toLower(coalesce(s2.status, s2.name, '')) CONTAINS $to_status "
    query = (
        "MATCH (p:Entity:Product)-[:HAS_AVAILABILITY_SNAPSHOT]->(s1:AvailabilitySnapshot) "
        "WHERE " + " AND ".join(where) + " "
        "MATCH (p)-[:HAS_AVAILABILITY_SNAPSHOT]->(s2:AvailabilitySnapshot) "
        "WHERE s2.observed_at > s1.observed_at "
        + from_clause
        + to_clause
        + "OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Entity) "
        "WITH p, b, s1, s2 "
        "ORDER BY s2.observed_at DESC "
        "WITH p, b, s1, s2, "
        "coalesce(s1.status, s1.name) AS from_status, "
        "coalesce(s2.status, s2.name) AS to_status "
        "WHERE from_status IS NOT NULL AND to_status IS NOT NULL "
        "AND toLower(toString(from_status)) <> toLower(toString(to_status)) "
        "RETURN p.id AS product_id, "
        "coalesce(p.name, p.title, p.id) AS name, "
        "coalesce(b.name, p.brand, '(unspecified)') AS brand, "
        "from_status AS from_status, to_status AS to_status, "
        "s1.observed_at AS from_at, s2.observed_at AS to_at "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            return [dict(row) for row in session.run(query, params)]
    except Exception:
        return None


def cypher_entity_frequency(
    platform,
    graph_id: str,
    field: str,
    filters: dict | None = None,
    *,
    min_products: int = 2,
    limit: int = 25,
) -> list[tuple[str, int]] | None:
    """Attribute values appearing on more than ``min_products`` products."""

    rows = cypher_group_by(platform, graph_id, field, dict(filters or {}))
    if rows is None:
        return None
    min_n = max(1, int(min_products or 2))
    limit = max(1, min(int(limit or 25), 100))
    out = [(value, count) for value, count in rows if int(count) >= min_n]
    return out[:limit]


def _canonical_product_url(url: Any) -> str:
    """Collapse browse-path PDPs to ``/{locale}/p/{slug}``.

    Scraped Primark links often look like
    ``…/en-us/r/baby/explore-by-product/en-us/p/<slug>``; the stable storefront
    URL is ``https://www.primark.com/en-us/p/<slug>``.
    """

    text = str(url or "").strip()
    if not text or not re.match(r"^https?://", text, flags=re.I):
        return text
    # Prefer locale immediately before /p/, else first locale in the path.
    near_p = re.search(
        r"/(?P<locale>[a-z]{2}-[a-z]{2})/p/(?P<slug>[^/?#]+)",
        text,
        flags=re.I,
    )
    if not near_p:
        return text
    slug = near_p.group("slug")
    first_loc = re.match(
        r"^https?://[^/]+/(?P<locale>[a-z]{2}-[a-z]{2})(?:/|$)",
        text,
        flags=re.I,
    )
    locale = (first_loc.group("locale") if first_loc else near_p.group("locale")).casefold()
    host_m = re.match(r"^(https?://[^/]+)", text, flags=re.I)
    if not host_m:
        return text
    return f"{host_m.group(1)}/{locale}/p/{slug}"


def _title_from_product_url(url: Any) -> str:
    """Derive a readable title from a PDP slug (common when name is stored as 'null')."""

    text = _canonical_product_url(url) or str(url or "").strip()
    if not text:
        return ""
    # …/p/<slug> or last path segment
    m = re.search(r"/p/([^/?#]+)", text, flags=re.I)
    slug = m.group(1) if m else text.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"\.[a-z0-9]+$", "", slug, flags=re.I)
    # Drop trailing numeric sku / style id.
    slug = re.sub(r"[-_]?\d{8,}$", "", slug)
    # Keep age/size hyphens (0-18mths, 7-15yrs) while spacing other slug segments.
    slug = re.sub(
        r"(?<![a-z0-9])(\d+)-(\d+)(mths?|yrs?|years?|months?|pk)(?![a-z0-9])",
        r"\1_\2\3",
        slug,
        flags=re.I,
    )
    slug = slug.replace("-", " ").replace("_", "-")
    slug = re.sub(r"\s+", " ", slug).strip(" -_")
    if not slug or slug.casefold() in {"null", "none", "p"}:
        return ""
    # Title-case lightly without wrecking size tokens like 0-18mths / 2pk.
    parts = []
    for tok in slug.split():
        if re.fullmatch(r"\d+[a-z]*|\d+[-/]\d+[a-z]*|\d+pk", tok, flags=re.I):
            parts.append(tok)
        else:
            parts.append(tok[:1].upper() + tok[1:] if tok else tok)
    return " ".join(parts)


def _human_product_label(
    name: Any,
    product_id: Any = None,
    *,
    product_url: Any = None,
) -> str:
    """Prefer a human-readable product title; never lead with colon-prefixed ids."""

    def _clean(raw: Any) -> str:
        text = str(raw or "").strip()
        if not text or text.casefold() in {"null", "none", "undefined", "(unspecified)"}:
            return ""
        return text

    def _is_bare_code(text: str) -> bool:
        if not text:
            return True
        if re.match(r"^[a-z0-9_]+:", text, flags=re.I):
            return True
        # Pure numeric / sku-like codes are not useful as the only label.
        return bool(re.fullmatch(r"\d{6,}", text))

    label = _clean(name)
    if label and not _is_bare_code(label):
        return label.replace("|", "\\|")

    from_url = _title_from_product_url(product_url)
    if from_url:
        return from_url.replace("|", "\\|")

    pid = _clean(product_id)
    if pid and not _is_bare_code(pid):
        return pid.replace("|", "\\|")
    # Strip leading source:brand: prefixes for display when that is all we have.
    for raw in (label, pid):
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) >= 2 and parts[-1]:
            tail = parts[-1].strip()
            if tail and tail.casefold() not in {"null", "none"} and not _is_bare_code(tail):
                return tail.replace("|", "\\|")
            if tail:
                return tail.replace("|", "\\|")
    return "—"


def _format_money(price: Any, currency: Any = None, price_text: Any = None) -> str:
    """Display current price with currency when available."""

    text = str(price_text or "").strip()
    if text and text.casefold() not in {"null", "none"}:
        return text.replace("|", "\\|")
    if isinstance(price, (int, float)):
        cur = str(currency or "").strip().upper()
        if cur in {"USD", "$"}:
            return f"${price:.2f}"
        if cur in {"GBP", "£"}:
            return f"£{price:.2f}"
        if cur in {"EUR", "€"}:
            return f"€{price:.2f}"
        if cur:
            return f"{price:.2f} {cur}"
        return f"{price:.2f}"
    return "—"


def _parse_status_transition(question: str) -> tuple[str | None, str | None]:
    """Extract from/to availability phrases from ``from X to Y`` wording."""

    text = normalize_name(question or "")
    m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+and\b|\s+with\b|$)", text)
    if not m:
        return None, None
    left = _avail_token(m.group(1))
    right = _avail_token(m.group(2))
    return (left or None, right or None)


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
    if mode == "transition":
        from_status, to_status = _parse_status_transition(question)
        rows = cypher_availability_transitions(
            platform,
            graph_id,
            filters,
            from_status=from_status,
            to_status=to_status,
            limit=max(int(limit or 10), 1),
        )
        if not rows:
            lines.append(
                "No products were found with that availability status transition in the snapshot history."
            )
            caveats.append(
                "Transitions use ordered AvailabilitySnapshot pairs (earlier → later observed_at)."
            )
            return True
        lines.append("Products with availability status transitions:")
        lines.append("| Product | Brand | From | To | From when | To when |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in rows:
            name = _human_product_label(row.get("name"), row.get("product_id"))
            from_at = str(row.get("from_at") or "—")[:19]
            to_at = str(row.get("to_at") or "—")[:19]
            lines.append(
                f"| {name} | {_display_label(str(row.get('brand') or '—'))} | "
                f"{_display_label(str(row.get('from_status') or '—'))} | "
                f"{_display_label(str(row.get('to_status') or '—'))} | "
                f"{from_at} | {to_at} |"
            )
        first_id = str(rows[0].get("product_id") or "").strip()
        if first_id:
            hist = cypher_recent_availability_changes(
                platform,
                graph_id,
                {**(filters or {}), "product_id": first_id},
                limit=8,
            ) or []
            if hist:
                lines.append("")
                lines.append(
                    f"Availability history for {_human_product_label(rows[0].get('name'), first_id)}:"
                )
                lines.append("| When | Status | In stock |")
                lines.append("| --- | --- | ---: |")
                for row in hist:
                    when = str(row.get("observed_at") or "—")[:19]
                    lines.append(
                        f"| {when} | {_display_label(str(row.get('status') or '—'))} | "
                        f"{row.get('in_stock_count') if row.get('in_stock_count') is not None else '—'} |"
                    )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="availability_transition",
                title="Availability status transition",
                excerpt="; ".join(
                    f"{r.get('name') or r.get('product_id')}:{r.get('from_status')}→{r.get('to_status')}"
                    for r in rows[:5]
                ),
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
                f"{_human_product_label(row.get('name'), row.get('product_id'))} | "
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


def cypher_availability_price_ops_brief(
    platform,
    graph_id: str,
    filters: dict | None = None,
) -> list[dict] | None:
    """Current stock-status mix joined to PriceChangeEvent volume per availability."""

    if platform is None:
        return None
    try:
        session_factory = _session(platform)
    except Exception:
        return None
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    query = (
        "MATCH (p:Entity:Product) "
        "WHERE " + " AND ".join(where) + " "
        "WITH p, coalesce("
        "head([(p)-[:HAS_AVAILABILITY]->(av) | av.name]), "
        "nullif(p.availability, 'null'), '(unspecified)'"
        ") AS availability "
        "OPTIONAL MATCH (p)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WITH availability, "
        "count(DISTINCT p) AS products, "
        "count(e) AS price_events, "
        "count(DISTINCT CASE WHEN e IS NOT NULL THEN p END) AS products_with_events "
        "RETURN availability, products, price_events, products_with_events "
        "ORDER BY products DESC"
    )
    try:
        with session_factory() as session:
            return [dict(row) for row in session.run(query, params)]
    except Exception:
        return None


def answer_availability_price_ops_brief(
    platform,
    graph_id: str,
    question: str,
    filters: dict | None = None,
) -> tuple[str, list[EvidenceItem], list[str]] | None:
    """Ops brief: stock-status mix + whether price events concentrate on in-stock vs mixed."""

    try:
        from fashion_brain.ask import is_availability_price_ops_brief_ask
    except Exception:
        return None
    if not is_availability_price_ops_brief_ask(question):
        return None
    rows = cypher_availability_price_ops_brief(platform, graph_id, filters)
    if rows is None:
        return None
    party = (
        str((filters or {}).get("brand") or (filters or {}).get("buyer") or "").strip()
        or "this selection"
    )
    evidence: list[EvidenceItem] = []
    caveats: list[str] = []
    lines: list[str] = [
        f"Ops brief for {party} — availability stock-status mix joined to price-change events.",
        "",
    ]
    if not rows:
        lines.append("No matching products were found for this scope.")
        caveats.append("Scoped with current product filters on e-commerce listings.")
        return "\n".join(lines), evidence, caveats

    total_products = sum(int(r.get("products") or 0) for r in rows)
    total_events = sum(int(r.get("price_events") or 0) for r in rows)
    lines.append("## Stock-status mix")
    lines.append("| Availability | Products | Share % |")
    lines.append("| --- | ---: | ---: |")
    for row in rows:
        products = int(row.get("products") or 0)
        share = (100.0 * products / total_products) if total_products else 0.0
        lines.append(
            f"| {_display_label(str(row.get('availability') or '—'))} | {products} | {share:.1f} |"
        )

    lines.append("")
    lines.append("## Price events by stock status")
    if total_events <= 0:
        lines.append(
            "No PriceChangeEvent rows were found on products in this scope, "
            "so concentration by stock status cannot be measured yet."
        )
        caveats.append(
            "Join uses current listing availability with HAS_PRICE_CHANGE events; "
            "empty event volume means the price half of the brief is unavailable."
        )
    else:
        lines.append(
            "| Availability | Products | Products with events | Price events | "
            "Events / product | Share of events % |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        focus = []
        for row in rows:
            products = int(row.get("products") or 0)
            events = int(row.get("price_events") or 0)
            with_ev = int(row.get("products_with_events") or 0)
            per = (events / products) if products else 0.0
            share = (100.0 * events / total_events) if total_events else 0.0
            label = _display_label(str(row.get("availability") or "—"))
            lines.append(
                f"| {label} | {products} | {with_ev} | {events} | {per:.2f} | {share:.1f} |"
            )
            token = _avail_token(str(row.get("availability") or ""))
            if token in {"instock", "mixed"} or "stock" in token:
                focus.append((label, events, share, per, products))
        if focus:
            focus.sort(key=lambda item: (-item[1], -item[2]))
            top = focus[0]
            lines.append("")
            lines.append(
                f"Price events are most concentrated on **{top[0]}** "
                f"({top[1]} events, {top[2]:.1f}% of events"
                + (f", {top[3]:.2f} per product" if top[4] else "")
                + ")."
            )
        caveats.append(
            "Join is current availability on the listing × PriceChangeEvent volume for those products "
            "(not a historical availability-at-event-time reconstruction)."
        )

    evidence.append(
        EvidenceItem(
            source_id=graph_id,
            record_id="availability_price_ops_brief",
            title="Availability × price-event ops brief",
            excerpt=(
                f"party={party}; products={total_products}; price_events={total_events}; "
                + "; ".join(
                    f"{r.get('availability')}:{r.get('products')}/{r.get('price_events')}"
                    for r in rows[:6]
                )
            ),
            source_path="ecom_products",
            verification_status="observed",
            graph=graph_id,
            node_id=None,
            synthetic=False,
        )
    )
    return "\n".join(lines), evidence, caveats


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
    min_abs_pct: float | None = None,
) -> list[dict] | None:
    """Largest same-currency price moves.

    ``direction``: ``up`` / ``down`` filter by event direction; ``any`` ranks by
    absolute percentage change (volatility).
    ``min_abs_pct``: optional floor on ``abs(delta_pct)`` (e.g. drops of more than 20%).
    """

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    direction = str(direction or "up").strip().lower()
    if direction not in {"up", "down", "any"}:
        direction = "up"
    params["dir"] = direction
    limit = max(1, min(int(limit or 25), 100))
    dir_clause = (
        ""
        if direction == "any"
        else "AND e.direction = $dir "
    )
    pct_clause = ""
    if min_abs_pct is not None and float(min_abs_pct) > 0:
        params["min_abs_pct"] = float(min_abs_pct)
        pct_clause = "AND abs(coalesce(e.delta_pct, 0)) >= $min_abs_pct "
    query = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        + dir_clause
        + pct_clause
        + "AND coalesce(e.currency_changed, false) = false "
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


def cypher_trending_products(
    platform,
    graph_id: str,
    filters: dict,
    *,
    limit: int = 15,
) -> list[dict] | None:
    """Rank products by recent catalog activity (price ups + stock snaps + events).

    Score favors same-currency price increases, then availability snapshot volume,
    then any price-change activity (including currency-label switches).
    """

    session_factory = _session(platform)
    if not session_factory:
        return None
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return None
    limit = max(1, min(int(limit or 15), 50))
    # Aggregate activity first; enrichment via pattern comprehensions (one row / product).
    query = (
        "MATCH (p:Entity:Product) "
        "WHERE " + " AND ".join(where) + " "
        "OPTIONAL MATCH (p)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "OPTIONAL MATCH (p)-[:HAS_AVAILABILITY_SNAPSHOT]->(a:AvailabilitySnapshot) "
        "WITH p, "
        "count(DISTINCT e) AS price_events, "
        "sum(CASE WHEN e.direction = 'up' AND coalesce(e.currency_changed,false) = false "
        "THEN 1 ELSE 0 END) AS price_ups, "
        "sum(CASE WHEN e.direction = 'down' AND coalesce(e.currency_changed,false) = false "
        "THEN 1 ELSE 0 END) AS price_downs, "
        "sum(CASE WHEN coalesce(e.currency_changed,false) = true THEN 1 ELSE 0 END) AS currency_switches, "
        "count(DISTINCT a) AS avail_snaps, "
        "max(e.detected_at) AS last_price_at, "
        "max(a.observed_at) AS last_avail_at "
        "WITH p, price_events, price_ups, price_downs, currency_switches, avail_snaps, "
        "last_price_at, last_avail_at, "
        "(price_ups * 5 + avail_snaps * 2 + price_events + currency_switches) AS score "
        "WHERE score > 0 "
        "RETURN p.id AS product_id, "
        "coalesce(nullif(p.name, 'null'), nullif(p.title, 'null'), p.display_reference, "
        "p.reference, p.source_id, p.id) AS name, "
        "p.product_url AS product_url, "
        "p.price AS price_text, p.price_num AS price_num, p.currency AS currency, "
        "coalesce(head([(p)-[:HAS_BRAND]->(b:Brand) | b.name]), p.brand) AS brand, "
        "head([(p)-[:HAS_SECTION]->(sec) | sec.name]) AS section, "
        "head([(p)-[:HAS_FAMILY]->(fam) | fam.name]) AS family, "
        "coalesce("
        "head([(p)-[:HAS_SUBFAMILY]->(sub) | sub.name]), "
        "head([(p)-[:BELONGS_TO_CATEGORY]->(cat) | cat.name])"
        ") AS subcategory, "
        "coalesce(p.category_path, '(unspecified)') AS category_path, "
        "coalesce("
        "head([(p)-[:PRIMARY_COLOR]->(c) | c.name]), "
        "head([(p)-[:HAS_COLOR]->(c) | c.name])"
        ") AS color, "
        "coalesce("
        "head([(p)-[:HAS_AVAILABILITY]->(av) | av.name]), "
        "p.availability"
        ") AS availability, "
        "price_events, price_ups, price_downs, currency_switches, avail_snaps, "
        "last_price_at, last_avail_at, score "
        "ORDER BY score DESC, last_price_at DESC, last_avail_at DESC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            return [dict(row) for row in session.run(query, params)]
    except Exception:
        return None


def _price_pct_threshold(question: str) -> float | None:
    """Parse 'more than 20%' / '>20%' style floors from the ask."""

    # Keep raw % before normalize_name strips punctuation.
    raw = str(question or "")
    match = re.search(
        r"(?:more than|over|above|at least|greater than|>)\s*(\d+(?:\.\d+)?)\s*%",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", raw)
    if match and re.search(
        r"\b(drop|drops|decrease|decreases|increase|increases)\b",
        raw,
        flags=re.IGNORECASE,
    ):
        return float(match.group(1))
    # normalize_name drops '%'; still catch "more than 20 percent/pct".
    text = normalize_name(raw)
    match = re.search(
        r"(?:more than|over|above|at least|greater than)\s*(\d+(?:\.\d+)?)\s*(?:percent|pct|pc)?\b",
        text,
    )
    if match and re.search(r"\b(drop|drops|decrease|decreases|increase|increases)\b", text):
        return float(match.group(1))
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
    # Current-state average / list price asks are NOT historical (avoid matching
    # the word "history" inside "not history" / "current-state").
    if re.search(
        r"\b(right now|current[- ]state|current average|avg price now|average price of|"
        r"average prices?(?:\s+by|\s+for)?|current price)\b",
        text,
    ) and not re.search(
        r"\b(change|changes|changed|trend|trends|history|historical|over time|"
        r"fluctuation|mover|movers|decrease|increase|drop|rise)\b",
        re.sub(r"\bnot\s+history\b", " ", text),
    ):
        return None
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
    trending_product_ask = False
    try:
        from fashion_brain.ask import is_price_trend_analysis_ask, is_trending_product_ask

        bare_trend_ask = is_price_trend_analysis_ask(question)
        trending_product_ask = is_trending_product_ask(question)
    except Exception:
        bare_trend_ask = bool(
            re.search(r"\b(trending\s+analysis|trend\s+analysis|make\s+trending)\b", text)
        )
        trending_product_ask = bool(
            re.search(r"\b(products?|items?)\b", text)
            and re.search(r"\b(trending|hot|popular)\b", text)
        )
    if (
        not re.search(r"\bprices?\b", text)
        and not cheaper_or_currency
        and not re.search(r"\bpricechangeevent\b", text)
        and not bare_trend_ask
        and not trending_product_ask
    ):
        return None
    if trending_product_ask:
        return "trending_products"
    # Explicit "price-trend analysis" / "trend analysis" phrasing → aggregate trend.
    # Do not use bare price+change (that would steal events_by_brand / movers asks).
    if re.search(
        r"\b("
        r"(?:price\s+)?(?:trend|trends|trending)(?:\s+analysis|\s+analyse|\s+analyze)"
        r"|trending\s+analysis|trend\s+analysis|price\s*[- ]?\s*trend"
        r")\b",
        text,
    ):
        return "trend"
    # Strip negated "not history" so it does not trigger history mode.
    change_text = re.sub(r"\bnot\s+history\b", " ", text)
    has_change = bool(
        re.search(
            r"\b(change|changes|changed|trend|trends|trending|history|historical|"
            r"fluctuation|fluctuations|swing|swings|swinging|volatil|volatile|"
            r"move|moves|moved|moving|mover|movers|"
            r"decrease|decreases|decreased|"
            r"increase|increases|increased|drop|drops|dropped|rise|rises|rose|rising|"
            r"fallen|falling|cheaper|expensive|over time)\b",
            change_text,
        )
    )
    # Current catalog "highest/lowest price products" is a listing — not PriceChangeEvent.
    try:
        from fashion_brain.ask import is_ranked_product_listing_ask

        if is_ranked_product_listing_ask(question):
            return None
    except Exception:
        pass
    if not has_change:
        return None
    # Unsupported geo/time scopes stay trend-mode; answer path emits an honest caveat.
    if re.search(r"\b(most|largest|biggest|highest)\b", text) and re.search(
        r"\b(brands?|buyers?)\b", text
    ) and re.search(r"\b(events?|changes?|moves?)\b", text):
        return "events_by_brand"
    if re.search(r"\b(categor(?:y|ies)|family|families|section|department)\b", text) and re.search(
        r"\b(trend|trends|decreasing|decrease|drop|dropping|cheaper|cheapening|"
        r"getting cheaper|price drop|price drops)\b",
        text,
    ):
        return "trend_by_category"
    # Multi-brand historical compare ("Zara and Primark … price moves compare").
    if re.search(r"\b(compare|comparison|versus|vs\.?|similar|side by side)\b", text) and (
        re.search(r"\b\w+\s+and\s+\w+\b", text) or " | " in (question or "")
    ):
        return "brand_compare"
    # Volatility / swinging: rank by absolute delta magnitude (both directions).
    if re.search(
        r"\b(swing|swings|swinging|volatil|fluctuat|up and down|most (volatile|unstable)|"
        r"price (moves?|movement|movements|fluctuations?))\b",
        text,
    ):
        return "volatility"
    # "which product is trending" → per-product activity (stock + price), not brand avg.
    try:
        from fashion_brain.ask import is_trending_product_ask

        if is_trending_product_ask(question):
            return "trending_products"
    except Exception:
        if re.search(r"\b(which|what|more|most)\b", text) and re.search(
            r"\b(products?|items?)\b", text
        ) and re.search(r"\b(trending|hot|popular)\b", text):
            return "trending_products"
    # Aggregate / average trend asks take priority over mover ranking.
    if re.search(r"\b(average|avg|mean|median|overall|typical|aggregate|summary)\b", text):
        return "trend"
    if re.search(r"\b(trend|trends|trending)\b", text) and not re.search(
        r"\b(largest|biggest|highest|lowest|top|most|which|what|products?|items?)\b", text
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


def _price_trend_scope_options(platform, graph_id: str, *, limit: int = 6) -> list[dict]:
    """Concrete brand / category choices for an unscoped price-trend clarify ask."""

    options: list[dict] = []
    seen: set[str] = set()

    def add(field: str, value: str, *, kind: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = f"{field}:{normalize_name(text)}"
        if key in seen:
            return
        seen.add(key)
        options.append(
            {
                "field": field,
                "value": text,
                "label": f"{kind}: {text}",
                "filters": {field: text},
            }
        )

    # Prefer parties that actually have PriceChangeEvent volume.
    try:
        brands = cypher_price_events_by_brand(platform, graph_id, limit=max(int(limit or 6), 3)) or []
        for row in brands[:4]:
            add("brand", str(row.get("brand") or ""), kind="Brand")
    except Exception:
        pass
    try:
        cats = cypher_price_trend_by_category(platform, graph_id, limit=4) or []
        for row in cats[:3]:
            add("category", str(row.get("category") or ""), kind="Category")
    except Exception:
        pass
    if len(options) >= 2:
        return options[: max(int(limit or 6), 2)]

    # Fallback: live metadata values.
    try:
        from knowledge_graph.retrieval.metadata import discover_values

        for field, kind in (("brand", "Brand"), ("buyer", "Buyer"), ("category", "Category")):
            payload = discover_values(platform, graph_id=graph_id, attribute=field, q="", limit=4)
            for value in payload.get("values") or []:
                add(field, str(value), kind=kind)
                if len(options) >= limit:
                    return options
    except Exception:
        pass
    return options[: max(int(limit or 6), 0)]


def cypher_price_trend_facet_series(
    platform,
    graph_id: str,
    filters: dict,
    *,
    facet: str = "category",
    top_n: int = 4,
    limit: int = 24,
) -> list[dict]:
    """Per-facet time series under the current filters (for multi-line charts).

    Returns ``[{"name": str, "points": [{"period", "avg_delta_pct", "events"}, …]}, …]``.
    """

    session_factory = _session(platform)
    if not session_factory:
        return []
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return []
    facet_key = str(facet or "category").casefold()
    if facet_key in {"brand", "buyer"} and (filters.get("brand") or filters.get("buyer")):
        # Already scoped to one party — break down by category instead.
        facet_key = "category"
    top_n = max(1, min(int(top_n or 4), 6))
    limit = max(8, min(int(limit or 24), 40))

    if facet_key == "section":
        facet_expr = "coalesce(p.section, head([(p)-[:HAS_SECTION]->(s:Entity) | s.name]), '(unspecified)')"
    elif facet_key == "family":
        facet_expr = (
            "coalesce(head([(p)-[:HAS_FAMILY]->(f:Entity) | f.name]), "
            "p.family, '(unspecified)')"
        )
    elif facet_key == "brand":
        facet_expr = (
            "coalesce(head([(p)-[:HAS_BRAND]->(b:Entity) | b.name]), "
            "p.brand, '(unspecified)')"
        )
    else:
        facet_expr = (
            "coalesce(head([(p)-[:BELONGS_TO_CATEGORY|HAS_FAMILY|HAS_SUB_CATEGORY|"
            "HAS_PLM_SUB_SUB_CATEGORY]->(c:Entity) | c.name]), "
            "p.category_path, '(unspecified)')"
        )

    # Rank facets by event volume under the current filter, then pull weekly averages.
    rank_q = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND coalesce(e.currency_changed, false) = false AND e.delta_pct IS NOT NULL "
        f"WITH {facet_expr} AS facet, count(e) AS events "
        "WHERE facet IS NOT NULL AND facet <> '' "
        "RETURN facet, events ORDER BY events DESC "
        f"LIMIT {top_n}"
    )
    series_q = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND coalesce(e.currency_changed, false) = false "
        "AND e.delta_pct IS NOT NULL AND e.detected_at IS NOT NULL "
        f"AND {facet_expr} = $facet "
        "WITH date(datetime(e.detected_at)) AS d, e.delta_pct AS pct "
        "WITH d.year AS y, d.week AS w, avg(pct) AS avg_delta_pct, count(*) AS events "
        "RETURN y + '-W' + (CASE WHEN w < 10 THEN '0' ELSE '' END) + toString(w) AS period, "
        "avg_delta_pct, events "
        "ORDER BY y ASC, w ASC "
        f"LIMIT {limit}"
    )
    out: list[dict] = []
    try:
        with session_factory() as session:
            facets = [str(row.get("facet") or "").strip() for row in session.run(rank_q, params)]
            facets = [f for f in facets if f and f.casefold() != "(unspecified)"]
            for name in facets[:top_n]:
                sp = dict(params)
                sp["facet"] = name
                points: list[dict] = []
                try:
                    for row in session.run(series_q, sp):
                        avg = row.get("avg_delta_pct")
                        try:
                            avg_f = float(avg) if avg is not None else None
                        except (TypeError, ValueError):
                            avg_f = None
                        if avg_f is None:
                            continue
                        points.append(
                            {
                                "period": str(row.get("period") or ""),
                                "avg_delta_pct": round(avg_f, 3),
                                "events": int(row.get("events") or 0),
                            }
                        )
                except Exception:
                    points = []
                if len(points) >= 2:
                    out.append({"name": name, "points": points})
    except Exception:
        return []
    return out


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


def _currencies_from_switch_pairs(switches: dict) -> list[str]:
    """Unique currency codes from pairs like USD→GBP."""
    found: list[str] = []
    seen: set[str] = set()
    for pair in switches.get("pairs") or []:
        text = str(pair.get("pair") or "")
        for part in re.split(r"[→><\-]+", text):
            code = re.sub(r"[^A-Za-z]", "", part).upper()
            if len(code) == 3 and code not in seen and code not in {"NAN", "NULL"}:
                seen.add(code)
                found.append(code)
    return found


def _currency_preference_from_question(question: str) -> str | None:
    """Return USD / GBP / BOTH when the user named a currency preference."""
    text = normalize_name(question or "")
    if not text:
        return None
    if re.search(r"\b(both|all currencies|each currency|compare currencies)\b", text):
        return "BOTH"
    # Prefer explicit ISO codes; also accept "us dollar" / "pound".
    if re.search(r"\b(usd|us\s*dollars?|dollar)\b", text):
        return "USD"
    if re.search(r"\b(gbp|pounds?|sterling)\b", text):
        return "GBP"
    if re.search(r"\b(eur|euros?)\b", text):
        return "EUR"
    m = re.search(r"\b([a-z]{3})\b", text)
    if m and m.group(1).upper() in {"USD", "GBP", "EUR", "SEK", "DKK", "NOK", "PLN", "CZK"}:
        return m.group(1).upper()
    return None


def cypher_price_level_timeseries(
    platform,
    graph_id: str,
    filters: dict,
    *,
    currency: str,
    limit: int = 24,
) -> list[dict]:
    """Avg new_price_num over time for events landing in one currency (incl. after switches)."""

    session_factory = _session(platform)
    if not session_factory:
        return []
    params: dict = {"g": graph_id, "currency": str(currency or "").upper()}
    where = _filter_where(filters, params)
    if where is None or not params["currency"]:
        return []
    limit = max(8, min(int(limit or 24), 40))
    # Prefer numeric props; fall back to toFloat(new_price) when new_price_num is sparse.
    query = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND e.detected_at IS NOT NULL "
        "AND toUpper(coalesce(e.new_currency, e.currency, '')) = $currency "
        "WITH date(datetime(e.detected_at)) AS d, "
        "coalesce(e.new_price_num, toFloat(e.new_price)) AS price "
        "WHERE price IS NOT NULL "
        "WITH d.year AS y, d.week AS w, avg(price) AS avg_price, "
        "min(price) AS min_price, max(price) AS max_price, count(*) AS events "
        "RETURN y + '-W' + (CASE WHEN w < 10 THEN '0' ELSE '' END) + toString(w) AS period, "
        "avg_price, min_price, max_price, events "
        "ORDER BY y ASC, w ASC "
        f"LIMIT {limit}"
    )
    out: list[dict] = []
    try:
        with session_factory() as session:
            for row in session.run(query, params):
                try:
                    avg = float(row.get("avg_price"))
                except (TypeError, ValueError):
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
                        "avg_price": round(avg, 3),
                        "min_price": round(_f("min_price"), 3) if _f("min_price") is not None else round(avg, 3),
                        "max_price": round(_f("max_price"), 3) if _f("max_price") is not None else round(avg, 3),
                        "events": int(row.get("events") or 0),
                    }
                )
    except Exception:
        return []
    return out


def cypher_currency_switch_timeseries(
    platform,
    graph_id: str,
    filters: dict,
    *,
    currency: str | None = None,
    limit: int = 24,
) -> list[dict]:
    """Weekly currency-switch event counts (optionally involving one currency)."""

    session_factory = _session(platform)
    if not session_factory:
        return []
    params: dict = {"g": graph_id}
    where = _filter_where(filters, params)
    if where is None:
        return []
    limit = max(8, min(int(limit or 24), 40))
    cur_clause = ""
    if currency:
        params["currency"] = str(currency).upper()
        cur_clause = (
            "AND (toUpper(coalesce(e.old_currency, '')) = $currency "
            "OR toUpper(coalesce(e.new_currency, e.currency, '')) = $currency) "
        )
    query = (
        "MATCH (p:Entity:Product)-[:HAS_PRICE_CHANGE]->(e:PriceChangeEvent) "
        "WHERE " + " AND ".join(where) + " "
        "AND coalesce(e.currency_changed, false) = true AND e.detected_at IS NOT NULL "
        + cur_clause
        + "WITH date(datetime(e.detected_at)) AS d "
        "WITH d.year AS y, d.week AS w, count(*) AS events "
        "RETURN y + '-W' + (CASE WHEN w < 10 THEN '0' ELSE '' END) + toString(w) AS period, "
        "events ORDER BY y ASC, w ASC "
        f"LIMIT {limit}"
    )
    try:
        with session_factory() as session:
            return [
                {"period": str(row.get("period") or ""), "events": int(row.get("events") or 0)}
                for row in session.run(query, params)
            ]
    except Exception:
        return []


def _append_currency_preference_clarify(
    who: str,
    switches: dict,
    graph_id: str,
    filters: dict,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
) -> bool:
    """Ask USD / GBP / both when only currency-switch events exist."""

    switch_n = int(switches.get("events") or 0)
    codes = _currencies_from_switch_pairs(switches)
    if switch_n <= 0 or len(codes) < 2:
        return False
    products = int(switches.get("products") or 0)
    pairs = switches.get("pairs") or []
    lines.append(
        f"For {_display_label(who)}, the {switch_n} recorded price events across {products} products "
        "are currency-label changes (not same-currency % moves). "
        "Which currency should I trend?"
    )
    options = []
    for code in codes[:4]:
        options.append(
            {
                "label": f"{code} only",
                "restated_question": (
                    f"Summarize average price change trends for {who} in {code}"
                ),
                "filters": dict(filters or {}),
                "currency": code,
            }
        )
    options.append(
        {
            "label": "Both currencies",
            "restated_question": (
                f"Summarize average price change trends for {who} in both currencies"
            ),
            "filters": dict(filters or {}),
            "currency": "BOTH",
        }
    )
    for i, opt in enumerate(options, 1):
        lines.append(f"{i}. {opt['label']}")
    lines.append("Reply with the number, or say USD / GBP / both.")
    if pairs:
        lines.append("Observed currency changes:")
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
            record_id="price_trend_needs_currency",
            title=f"Price trend needs currency · {_display_label(who)}",
            excerpt=json.dumps(
                {
                    "options": [o["label"] for o in options],
                    "interpretations": [
                        {
                            "restated_question": o["restated_question"],
                            "filters": o["filters"],
                            "currency": o["currency"],
                        }
                        for o in options
                    ],
                    "who": who,
                    "currencies": codes,
                },
                ensure_ascii=False,
            ),
            source_path="product_change_events",
            verification_status="observed",
            graph=graph_id,
            node_id=None,
            synthetic=False,
        )
    )
    caveats.append(
        "Awaiting currency preference (USD / GBP / both) before building the price trend chart."
    )
    try:
        from fashion_brain.agents.query_understanding import QueryUnderstandingAgent

        caveats.append(QueryUnderstandingAgent.AWAITING_CAVEAT)
    except Exception:
        caveats.append("Awaiting clarification before retrieval.")
    return True


def _append_currency_scoped_trend(
    platform,
    graph_id: str,
    filters: dict,
    who: str,
    currency_pref: str,
    switches: dict,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
    *,
    limit: int | None = None,
) -> bool:
    """Build absolute-price (and switch-volume) charts for one currency or both."""

    codes = (
        _currencies_from_switch_pairs(switches)
        if currency_pref == "BOTH"
        else [currency_pref]
    )
    codes = [c for c in codes if c][:4]
    if not codes:
        return False

    series_payload: list[dict] = []
    labels: list[str] = []
    label_index: dict[str, int] = {}

    def _ensure_labels(periods: list[str]) -> None:
        nonlocal labels, label_index
        for p in periods:
            if p not in label_index:
                label_index[p] = len(labels)
                labels.append(p)

    level_notes: list[str] = []
    for code in codes:
        pts = cypher_price_level_timeseries(
            platform, graph_id, filters, currency=code, limit=max(int(limit or 24), 16)
        )
        if len(pts) < 2:
            continue
        _ensure_labels([str(p.get("period") or "") for p in pts])
        aligned = [None] * len(labels)
        for pt in pts:
            idx = label_index.get(str(pt.get("period") or ""))
            if idx is None:
                continue
            aligned[idx] = float(pt.get("avg_price") or 0.0)
        last = None
        filled: list[float] = []
        for v in aligned:
            if v is not None:
                last = v
            filled.append(float(last if last is not None else 0.0))
        series_payload.append({"name": f"Avg price · {code}", "values": filled})
        vals = [float(p.get("avg_price") or 0.0) for p in pts]
        level_notes.append(
            f"- {code}: avg {sum(vals)/len(vals):.2f}, "
            f"min {min(vals):.2f}, max {max(vals):.2f} "
            f"({sum(int(p.get('events') or 0) for p in pts)} events)"
        )

    sw_chart = None
    # Always try switch-volume series — useful when absolute price levels are sparse.
    if currency_pref == "BOTH":
        multi_sw: list[dict] = []
        shared_labels: list[str] = []
        shared_index: dict[str, int] = {}
        for code in codes:
            sw_pts = cypher_currency_switch_timeseries(
                platform, graph_id, filters, currency=code, limit=max(int(limit or 24), 16)
            )
            if len(sw_pts) < 2:
                continue
            for p in sw_pts:
                per = str(p.get("period") or "")
                if per and per not in shared_index:
                    shared_index[per] = len(shared_labels)
                    shared_labels.append(per)
            aligned = [0.0] * len(shared_labels)
            for pt in sw_pts:
                idx = shared_index.get(str(pt.get("period") or ""))
                if idx is not None:
                    aligned[idx] = float(pt.get("events") or 0.0)
            # Pad earlier series if label axis grew.
            for series in multi_sw:
                if len(series["values"]) < len(shared_labels):
                    series["values"] = series["values"] + [0.0] * (
                        len(shared_labels) - len(series["values"])
                    )
            if len(aligned) < len(shared_labels):
                aligned = aligned + [0.0] * (len(shared_labels) - len(aligned))
            multi_sw.append({"name": f"Switch events · {code}", "values": aligned})
        if multi_sw and shared_labels:
            sw_chart = {
                "type": "line",
                "smooth": True,
                "analytics": True,
                "title": f"Currency-switch volume · {_display_label(who)}",
                "labels": shared_labels,
                "values": multi_sw[0]["values"],
                "series": multi_sw,
                "yLabel": "Events",
            }
    elif codes:
        sw_pts = cypher_currency_switch_timeseries(
            platform, graph_id, filters, currency=codes[0], limit=max(int(limit or 24), 16)
        )
        if len(sw_pts) >= 2:
            sw_chart = {
                "type": "line",
                "smooth": True,
                "analytics": True,
                "title": f"Currency-switch volume · {codes[0]} · {_display_label(who)}",
                "labels": [str(p.get("period") or "") for p in sw_pts],
                "values": [float(p.get("events") or 0.0) for p in sw_pts],
                "series": [
                    {
                        "name": f"Switch events · {codes[0]}",
                        "values": [float(p.get("events") or 0.0) for p in sw_pts],
                    }
                ],
                "yLabel": "Events",
            }

    # No absolute prices AND no switch volume → cannot chart.
    if not series_payload and not sw_chart:
        return False

    # Price-level chart when available; otherwise lead with switch-volume analysis.
    if series_payload:
        all_vals = [v for s in series_payload for v in s["values"]]
        lines.append(
            f"Price level trend for {_display_label(who)} "
            f"({'both currencies' if currency_pref == 'BOTH' else currency_pref}):"
        )
        lines.extend(level_notes)
        lines.append(
            "These are average listed prices in the chosen currency over time "
            "(includes post-switch listings). Same-currency % deltas were not present."
        )
        chart = {
            "type": "line",
            "smooth": True,
            "analytics": True,
            "title": f"Price levels · {_display_label(who)} · {currency_pref}",
            "labels": labels,
            "values": series_payload[0]["values"],
            "series": series_payload,
            "avg": round(sum(all_vals) / len(all_vals), 3) if all_vals else 0.0,
            "min": round(min(all_vals), 3) if all_vals else 0.0,
            "max": round(max(all_vals), 3) if all_vals else 0.0,
            "yLabel": "Price",
        }
        lines.append("```chart")
        lines.append(json.dumps(chart, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append(
            f"No same-currency % price moves for {_display_label(who)} in "
            f"{'both currencies' if currency_pref == 'BOTH' else currency_pref}. "
            "Showing currency-switch activity over time as the recorded change pattern."
        )

    if sw_chart:
        lines.append("Currency-switch event volume over time:")
        lines.append("```chart")
        lines.append(json.dumps(sw_chart, ensure_ascii=False))
        lines.append("```")

    # Keep the switch mix table for context.
    pairs = switches.get("pairs") or []
    if pairs:
        lines.append("Currency-label changes in this scope:")
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
            record_id="price_trend_currency_scoped",
            title=f"Price change trend · {_display_label(who)} · {currency_pref}",
            excerpt=(
                f"currency={currency_pref}; price_series={len(series_payload)}; "
                f"switch_chart={bool(sw_chart)}; "
                f"switch_events={int(switches.get('events') or 0)}"
            ),
            source_path="product_change_events",
            verification_status="observed",
            graph=graph_id,
            node_id=None,
            synthetic=False,
        )
    )
    caveats.append(
        "Same-currency % deltas were absent; chart uses absolute price levels and/or "
        "currency-switch volume for the chosen currency."
    )
    return True


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


def _price_history_source_label(graph_id: str) -> str:
    low = str(graph_id or "").casefold()
    if "plm" in low:
        return "PLMex"
    if "ecom" in low or low in {"master_products", "master"}:
        return "e-commerce listings"
    return str(graph_id or "this catalog")


def append_price_history_unavailable(
    graph_id: str,
    lines: list[str],
    evidence: list[EvidenceItem],
    caveats: list[str],
    *,
    question: str | None = None,
) -> None:
    """Polite closed answer when PriceChangeEvent data is missing on this graph."""

    label = _price_history_source_label(graph_id)
    lines.append(
        f"Price change history is not available in {label} "
        f"(no PriceChangeEvent data on this catalog)."
    )
    lines.append(
        "Assortment, materials, colors, and other product attributes are still available here — "
        "ask about those, or ask the same price question against e-commerce listings."
    )
    evidence.append(
        EvidenceItem(
            source_id=graph_id,
            record_id="price_history_unavailable",
            title="Price history unavailable",
            excerpt=f"graph={graph_id}; events=0; ask={str(question or '')[:120]}",
            source_path="product_change_events",
            verification_status="observed",
            graph=graph_id,
            node_id=None,
            synthetic=False,
        )
    )
    caveats.append(
        f"Price trends use PriceChangeEvent nodes; {label} has none. "
        "Other catalog facets on this source remain queryable."
    )


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
    # PLM (and any graph without change events) must not fall through to assortment.
    # Trending-product ranking can still use availability snapshots when price events=0.
    try:
        event_n = cypher_count_nodes(platform, graph_id, "PriceChangeEvent")
    except Exception:
        event_n = None
    if (
        event_n is not None
        and int(event_n) <= 0
        and mode not in {"count_events", "trending_products"}
    ):
        append_price_history_unavailable(
            graph_id, lines, evidence, caveats, question=question
        )
        return True
    if mode == "count_events" and event_n is not None and int(event_n) <= 0:
        lines.append(
            f"There are 0 PriceChangeEvent nodes in {_price_history_source_label(graph_id)}. "
            "Price-drop / trend rankings need e-commerce change events."
        )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="price_change_event_count",
                title="PriceChangeEvent count",
                excerpt="count=0",
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        return True
    filters = clean_product_filters(filters)
    try:
        from fashion_brain.ask import sanitize_party_filters

        filters = sanitize_party_filters(filters)
    except Exception:
        pass
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
    # Drop brand/buyer ghosts from fuzzy measure words (price→Prince) only.
    # Keep sticky/deixis brands (e.g. prior Zara) that are not near-matches of metric tokens.
    qn = normalize_name(question or "")
    _measure_toks = {
        "price",
        "prices",
        "cost",
        "costs",
        "change",
        "changes",
        "trend",
        "trends",
        "drop",
        "drops",
        "increase",
        "increases",
        "decrease",
        "decreases",
        "more",
        "getting",
        "recorded",
        "than",
        "ignoring",
        "actually",
        "similar",
        "satin",
    }
    for key in ("brand", "buyer"):
        raw = str(filters.get(key) or "").strip()
        if not raw:
            continue
        val = normalize_name(raw)
        if not val:
            filters.pop(key, None)
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(val)}(?![a-z0-9])", qn):
            continue
        from difflib import SequenceMatcher

        for tok in qn.split():
            if tok not in _measure_toks and tok not in _noise:
                continue
            if SequenceMatcher(None, tok, val).ratio() >= 0.88:
                filters.pop(key, None)
                break
    # Geo/time scopes not modeled yet — fail closed instead of inventing a global trend.
    qn = normalize_name(question or "")
    geo_hit = bool(
        re.search(r"\b(uk|eu|us|usa|europe|asia)\b", qn)
        or (
            re.search(r"\b(region|regions|country|countries)\b", qn)
            and re.search(r"\b(uk|eu|us|usa|europe|asia)\b", qn)
        )
    )
    time_window = bool(
        re.search(
            r"\b(last\s+\d+\s+(days?|weeks?|months?|years?)|six\s+months|"
            r"over the last|past\s+\d+\s+(days?|weeks?|months?|years?))\b",
            qn,
        )
    )
    if geo_hit or time_window:
        if geo_hit:
            lines.append(
                "Regional / country-scoped price trends are not available yet "
                "(Domain/Country nodes are not wired into PriceChangeEvent)."
            )
            caveats.append("No Region dimension on price-history events; refusing global substitute.")
        if time_window:
            lines.append(
                "Fixed calendar windows (e.g. last six months) are not filtered on "
                "PriceChangeEvent yet — refusing an unscoped global substitute."
            )
            caveats.append("No event-time window filter for 'last N months'; refusing global substitute.")
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
        # Line plot of price level over time (start at first old_price, then each new_price).
        chart_labels: list[str] = []
        chart_values: list[float] = []
        first_old = rows[0].get("old_price")
        first_at = str(rows[0].get("at") or "")[:10]
        if isinstance(first_old, (int, float)):
            chart_labels.append(first_at or "start")
            chart_values.append(float(first_old))
        for row in rows:
            new_p = row.get("new_price")
            if not isinstance(new_p, (int, float)):
                continue
            chart_labels.append(str(row.get("at") or "")[:19] or "—")
            chart_values.append(float(new_p))
        if len(chart_values) >= 2:
            lines.append("```chart")
            lines.append(
                json.dumps(
                    {
                        "type": "line",
                        "smooth": True,
                        "title": f"Price over time · {_human_product_label(name, pid)}",
                        "labels": chart_labels,
                        "values": chart_values,
                        "series": [{"name": "Price", "values": chart_values}],
                        "yLabel": "Price",
                    },
                    ensure_ascii=False,
                )
            )
            lines.append("```")
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

    if mode == "trending_products":
        who = (
            filters.get("brand")
            or filters.get("buyer")
            or filters.get("category")
            or "this selection"
        )
        show_n = max(int(limit or 10), 1)
        rows = cypher_trending_products(platform, graph_id, filters, limit=show_n) or []
        if not rows:
            lines.append(
                f"No recent stock/price activity was found to rank trending products for {who}."
            )
            caveats.append(
                "Trending products use PriceChangeEvent + AvailabilitySnapshot activity scores."
            )
            return True
        ups_total = sum(int(r.get("price_ups") or 0) for r in rows)
        who_label = _display_label(str(who))
        lines.append(
            f"Most active / trending products for {who_label} right now "
            "(higher score = more recent stock updates and price-change activity — "
            "not sales/traffic):"
        )
        lines.append(
            "| Product | Brand | Price | Category | Color | Availability | Score | "
            "Stock snaps | Price ups | Last activity | Product link |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |"
        )
        for row in rows:
            name = _human_product_label(
                row.get("name"),
                row.get("product_id"),
                product_url=row.get("product_url"),
            )
            price = _format_money(
                row.get("price_num"), row.get("currency"), row.get("price_text")
            )
            category = (
                str(row.get("category_path") or "").strip()
                or " > ".join(
                    x
                    for x in (
                        str(row.get("section") or "").strip(),
                        str(row.get("family") or "").strip(),
                        str(row.get("subcategory") or "").strip(),
                    )
                    if x and x.casefold() not in {"null", "none"}
                )
                or "—"
            ).replace("|", "\\|")
            color = str(row.get("color") or "—").strip() or "—"
            if color.casefold() in {"null", "none"}:
                color = "—"
            color = color.replace("|", "\\|")
            avail = str(row.get("availability") or "—").strip() or "—"
            if avail.casefold() in {"null", "none"}:
                avail = "—"
            avail = avail.replace("|", "\\|")
            last_at = str(row.get("last_price_at") or row.get("last_avail_at") or "—")[:19]
            raw_url = str(row.get("product_url") or "").strip()
            canon_url = _canonical_product_url(raw_url)
            if (
                canon_url
                and canon_url.casefold() not in {"null", "none"}
                and canon_url.startswith(("http://", "https://"))
            ):
                link = f"[Open on site]({canon_url})"
            else:
                link = "—"
            lines.append(
                f"| {name} | {_display_label(str(row.get('brand') or '—'))} | {price} | "
                f"{category} | {color} | {avail} | "
                f"{int(row.get('score') or 0)} | {int(row.get('avail_snaps') or 0)} | "
                f"{int(row.get('price_ups') or 0)} | {last_at} | {link} |"
            )
            evidence.append(
                EvidenceItem(
                    source_id=graph_id,
                    record_id=str(row.get("product_id") or ""),
                    title=str(name),
                    excerpt=(
                        f"price={price}; category={category}; color={color}; "
                        f"url={canon_url or '—'}; score={row.get('score')} "
                        f"ups={row.get('price_ups')} avail_snaps={row.get('avail_snaps')} "
                        f"events={row.get('price_events')}"
                    ),
                    source_path="product_change_events",
                    verification_status="observed",
                    graph=graph_id,
                    node_id=str(row.get("product_id") or "") or None,
                    synthetic=False,
                )
            )
        if ups_total <= 0:
            lines.append(
                "Note: no same-currency price *increases* were recorded in this scope — "
                "ranking leans on stock/availability updates and other price-change activity "
                "(including currency-label switches)."
            )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="trending_products",
                title=f"Trending products · {_display_label(str(who))}",
                excerpt=f"products={len(rows)}; same_currency_ups={ups_total}",
                source_path="product_change_events",
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
            )
        )
        caveats.append(
            "Trending score = (same-currency price ups × 5) + (availability snapshots × 2) "
            "+ price events + currency switches. Not a sales/traffic rank."
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

    # Multi-brand compare before volatility/movers so "Zara and Primark … compare"
    # is not swallowed by the swing/moves classifier.
    parties: list[str] = []
    try:
        from fashion_brain.ask import live_brands_mentioned, live_brand_names, split_multi_party_filter

        parties = split_multi_party_filter(str(filters.get("brand") or filters.get("buyer") or ""))
        if len(parties) < 2 and mode == "brand_compare":
            parties = live_brands_mentioned(question, live_brand_names(platform, graph_id))[:4]
    except Exception:
        parties = []
    if len(parties) >= 2 or mode == "brand_compare":
        if len(parties) < 2:
            lines.append(
                "I can compare historical price moves once two brands are in scope "
                "(e.g. Zara and Primark)."
            )
            caveats.append("Brand comparison needs at least two named brands.")
            return True
        lines.append("Comparing historical price moves by brand:")
        lines.append(
            "| Brand | Avg change | Events | Products | Up | Down | Unchanged |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        excerpts: list[str] = []
        any_events = False
        switch_summaries: list[tuple[str, dict]] = []
        chart_labels: list[str] = []
        chart_series: list[dict] = []
        for party in parties:
            party_filters = {**{k: v for k, v in filters.items() if k not in {"brand", "buyer"}}, "brand": party}
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
            series = cypher_price_trend_timeseries(
                platform, graph_id, party_filters, limit=max(int(limit or 24), 16)
            ) or []
            if len(series) >= 2:
                if not chart_labels:
                    chart_labels = [str(r.get("period") or "") for r in series]
                label_index = {lab: i for i, lab in enumerate(chart_labels)}
                aligned = [None] * len(chart_labels)
                for pt in series:
                    idx = label_index.get(str(pt.get("period") or ""))
                    if idx is None:
                        continue
                    aligned[idx] = float(pt.get("avg_delta_pct") or 0.0)
                last = None
                filled: list[float] = []
                for v in aligned:
                    if v is not None:
                        last = v
                    filled.append(float(last if last is not None else 0.0))
                if sum(1 for v in aligned if v is not None) >= 2:
                    chart_series.append(
                        {"name": _display_label(party), "values": filled}
                    )
            sample_n = min(max(int(limit or 3), 1), 3)
            sample = cypher_price_movers(
                platform, graph_id, party_filters, direction="down", limit=sample_n
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
        if len(chart_series) >= 1 and len(chart_labels) >= 2:
            lines.append("```chart")
            lines.append(
                json.dumps(
                    {
                        "type": "line",
                        "smooth": True,
                        "analytics": True,
                        "title": "Historical price moves · brand compare",
                        "labels": chart_labels,
                        "values": chart_series[0]["values"],
                        "series": chart_series,
                        "yLabel": "Change %",
                    },
                    ensure_ascii=False,
                )
            )
            lines.append("```")
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

    if mode == "volatility":
        rows = cypher_price_movers(
            platform,
            graph_id,
            filters,
            direction="any",
            limit=max(int(limit or 15), 1),
        )
        if not rows:
            lines.append("No same-currency price swings were found to rank by volatility.")
            return True
        lines.append("Products with the largest absolute price swings (same currency):")
        lines.append("| Product | Brand | Old | New | Change % | Direction |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in rows:
            pct = row.get("delta_pct")
            pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "—"
            direction = "down" if isinstance(pct, (int, float)) and pct < 0 else (
                "up" if isinstance(pct, (int, float)) and pct > 0 else (row.get("direction") or "—")
            )
            name = _human_product_label(row.get("name"), row.get("product_id"))
            lines.append(
                f"| {name} | {_display_label(str(row.get('brand') or '—'))} | "
                f"{row.get('old_price') if row.get('old_price') is not None else '—'} | "
                f"{row.get('new_price') if row.get('new_price') is not None else '—'} | "
                f"{pct_s} | {direction} |"
            )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="price_volatility",
                title="Price volatility ranking",
                excerpt="; ".join(
                    f"{r.get('name') or r.get('product_id')}={r.get('delta_pct')}" for r in rows[:5]
                ),
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
        # Prefer categories that are getting cheaper when the ask is about cheapening.
        if rows and re.search(r"\b(cheaper|cheapening|getting cheaper|decrease|drop|dropping)\b", qn):
            cheaper = [
                row
                for row in rows
                if isinstance(row.get("avg_delta_pct"), (int, float)) and float(row["avg_delta_pct"]) < 0
            ]
            rows = cheaper or rows
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

    if mode == "trend":
        scope_keys = (
            "brand",
            "buyer",
            "category",
            "garment",
            "garment_type",
            "family",
            "subfamily",
            "section",
            "plm_main_category",
            "subcategory",
            "plm_sub_sub_category",
        )
        if not any(filters.get(k) for k in scope_keys):
            # Refuse unscoped global trend — ask which brand / buyer / category.
            options = _price_trend_scope_options(platform, graph_id)
            lines.append(
                "I can run a price trend analysis once I know the scope. "
                "Which brand, buyer, or category should I track?"
            )
            for i, opt in enumerate(options, 1):
                lines.append(f"{i}. {opt['label']}")
            if options:
                lines.append("Reply with the number, or name a brand / buyer / category.")
            evidence.append(
                EvidenceItem(
                    source_id=graph_id,
                    record_id="price_trend_needs_scope",
                    title="Price trend needs scope",
                    excerpt=json.dumps(
                        {
                            "options": [o.get("label") for o in options],
                            "interpretations": [
                                {
                                    "restated_question": (
                                        f"Summarize average price change trends for {o.get('value')}"
                                    ),
                                    "filters": dict(o.get("filters") or {}),
                                }
                                for o in options
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    source_path="product_change_events",
                    verification_status="observed",
                    graph=graph_id,
                    node_id=None,
                    synthetic=False,
                )
            )
            caveats.append("Price trend requires a brand, buyer, or category filter.")
            return True

        summary = cypher_price_trend_summary(platform, graph_id, filters)
        who = (
            filters.get("brand")
            or filters.get("buyer")
            or filters.get("category")
            or filters.get("garment")
            or filters.get("section")
            or filters.get("family")
            or "this selection"
        )
        if not summary or not summary.get("change_events"):
            # Drop currency codes wrongly stored as buyer/brand before trending.
            scoped_filters = dict(filters or {})
            for key in ("brand", "buyer"):
                if str(scoped_filters.get(key) or "").casefold() in {
                    "usd", "gbp", "eur", "both", "dollar", "dollars", "pound", "pounds",
                }:
                    scoped_filters.pop(key, None)
            switches = cypher_price_currency_switch_summary(
                platform, graph_id, scoped_filters
            ) or {}
            currency_pref = _currency_preference_from_question(question)
            codes = _currencies_from_switch_pairs(switches)
            who_scoped = (
                scoped_filters.get("brand")
                or scoped_filters.get("buyer")
                or who
            )
            # Multi-currency switch mix and no preference yet → ask USD / GBP / both.
            if not currency_pref and len(codes) >= 2 and _append_currency_preference_clarify(
                str(who_scoped),
                switches,
                graph_id,
                scoped_filters,
                lines,
                evidence,
                caveats,
            ):
                return True
            # User already named a currency (or only one exists) → absolute price-level trend.
            if not currency_pref and len(codes) == 1:
                currency_pref = codes[0]
            if currency_pref and _append_currency_scoped_trend(
                platform,
                graph_id,
                scoped_filters,
                str(who_scoped),
                currency_pref,
                switches,
                lines,
                evidence,
                caveats,
                limit=limit,
            ):
                return True
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
            # Align multi-facet lines onto the overall period axis.
            labels = [str(r.get("period") or "") for r in series]
            chart_series: list[dict] = [
                {"name": f"Overall · {_display_label(str(who))}", "values": values}
            ]
            # Min / max envelope as soft guide lines when present on the buckets.
            min_line = [
                float(r["min_delta_pct"])
                if isinstance(r.get("min_delta_pct"), (int, float))
                else float(r.get("avg_delta_pct") or 0.0)
                for r in series
            ]
            max_line = [
                float(r["max_delta_pct"])
                if isinstance(r.get("max_delta_pct"), (int, float))
                else float(r.get("avg_delta_pct") or 0.0)
                for r in series
            ]
            if any(a != b for a, b in zip(min_line, max_line)):
                chart_series.append({"name": "Period min %", "values": min_line, "style": "dashed"})
                chart_series.append({"name": "Period max %", "values": max_line, "style": "dashed"})

            facet_key = "category"
            if filters.get("category") or filters.get("garment") or filters.get("family"):
                facet_key = "section" if not filters.get("section") else "brand"
            facet_rows = cypher_price_trend_facet_series(
                platform,
                graph_id,
                filters,
                facet=facet_key,
                top_n=4,
                limit=max(len(series), 16),
            )
            label_index = {lab: i for i, lab in enumerate(labels)}
            for facet in facet_rows[:4]:
                aligned = [None] * len(labels)
                for pt in facet.get("points") or []:
                    idx = label_index.get(str(pt.get("period") or ""))
                    if idx is None:
                        continue
                    aligned[idx] = float(pt.get("avg_delta_pct") or 0.0)
                # Forward-fill gaps so the line stays continuous.
                last = None
                filled: list[float] = []
                for v in aligned:
                    if v is not None:
                        last = v
                    filled.append(float(last if last is not None else 0.0))
                if sum(1 for v in aligned if v is not None) >= 2:
                    chart_series.append(
                        {
                            "name": _display_label(str(facet.get("name") or "facet")),
                            "values": filled,
                        }
                    )

            chart = {
                "type": "line",
                "smooth": True,
                "analytics": True,
                "title": f"Price change trends · {_display_label(str(who))}",
                "labels": labels,
                "values": values,
                "series": chart_series,
                "avg": round(float(overall_avg), 3),
                "min": round(float(overall_min), 3),
                "max": round(float(overall_max), 3),
                "yLabel": "Change %",
            }
            lines.append("```chart")
            lines.append(json.dumps(chart, ensure_ascii=False))
            lines.append("```")
            if len(chart_series) > 1:
                lines.append(
                    "Chart lines: overall average under your filters, period min/max envelope, "
                    "and the busiest categories/sections inside that scope."
                )

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
    min_abs = _price_pct_threshold(question)
    rows = cypher_price_movers(
        platform,
        graph_id,
        filters,
        direction=direction,
        limit=show_n,
        min_abs_pct=min_abs,
    )
    label = "decreases" if direction == "down" else "increases"
    thresh_note = f" of {min_abs:g}% or more" if min_abs else ""
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
                f"No same-currency price {label}{thresh_note} were found in the product knowledge graph "
                "(currency switches are excluded)."
            )
            caveats.append("Price movers use PriceChangeEvent nodes from product_change_events.")
        return True
    heading = f"Largest same-currency price {label}"
    if min_abs:
        heading = f"Same-currency price {label} of {min_abs:g}% or more"
    lines.append(f"{heading} (from catalog change events):")
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
        rel_alt = "|".join(rel)
        query = (
            "MATCH (p:Entity:Product) WHERE "
            + " AND ".join(where)
            + f" OPTIONAL MATCH (p)-[:{rel_alt}]->(x:Entity) "
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
    platform,
    graph_id: str,
    filters: dict,
    *,
    limit: int = 5,
    offset: int = 0,
    order_by_price: str | None = None,
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
    price_dir = str(order_by_price or "").casefold().strip()
    if price_dir in {"desc", "asc"}:
        direction = "DESC" if price_dir == "desc" else "ASC"
        order = (
            f"ORDER BY {_price_num_cypher()} IS NULL ASC, "
            f"{_price_num_cypher()} {direction}, p.id ASC"
        )
    elif filters.get("price_max") is not None or filters.get("price_min") is not None:
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


def search_neo4j_products(
    platform,
    principal: Principal,
    graph_id: str,
    filters: dict,
    *,
    limit: int = 5,
    order_by_price: str | None = None,
) -> tuple[list, int]:
    filters = clean_product_filters(filters)
    # Sample first for the row dump; always count separately so "There are N products"
    # never collapses to the LIMIT size (was: total=len(sample) when limit<=25).
    sample = cypher_sample_products(
        platform, graph_id, filters, limit=limit, order_by_price=order_by_price
    )
    if sample is not None:
        sample = [
            node
            for node in sample
            if allows(principal, node.properties.get("classification") or "internal")
        ][:limit]
        total = cypher_count_products(platform, graph_id, filters)
        if total is None:
            total = len(sample)
        return sample, int(total)
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
            "Combined specification: retail mix from e-commerce plus development fields from PLMex "
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
    """User requirements → historical attribute modes → proposed NEW sheet (not an existing SKU)."""

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
    if not buyer_total:
        lines.append(f"There are no matching styles for {party} in {_graph_label(platform, graph_id)}.")
        return

    active = dict(base)
    # Season is a hard filter for new-spec asks — never silently analyze the whole brand first.
    total = cypher_count_products(platform, graph_id, active)
    if total is None:
        total = 0
    season_bits = []
    if active.get("season_name"):
        season_bits.append(str(active["season_name"]).capitalize())
    if active.get("season_year"):
        season_bits.append(str(active["season_year"]))
    if season_bits:
        lines.append(
            f"Scope for {party} · {' '.join(season_bits)}: {total} products in "
            f"{_graph_label(platform, graph_id)} "
            f"(of {buyer_total} total {party} products)."
        )
    else:
        lines.append(f"Scope for {party}: {total} products in {_graph_label(platform, graph_id)}.")
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

    # 1) anchors narrow (when supported)  2) every required field is analyzed  3) proposed sheet
    scope = dict(active)
    scope_total = int(total)
    # field -> (best_label, best_count, alternatives_text, raw_value|None)
    analysis_rows: dict[str, tuple[str, int, str, str | None]] = {}
    decided_fields: set[str] = set()

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

    # Progressive compatible locks: user requirements → hierarchy anchors → core
    # attributes decided on the narrowed historical set (not independent modes).
    decision_rows: list[dict[str, str]] = []
    decision_scope = dict(scope)
    decision_total = int(scope_total)
    season_requested = str(active.get("season_name") or season_name or "").strip().casefold()
    locked_raw: dict[str, str] = {
        key: str(value)
        for key, value in decision_scope.items()
        if key not in {party_key, "brand", "buyer", "season_name", "season_year", "season"}
        and value not in (None, "")
    }

    def _confidence(count: int, population: int) -> str:
        if population <= 0 or count <= 0:
            return "insufficient"
        share = count / population
        if share >= 0.50 and count >= 5:
            return "high"
        if share >= 0.25 or (share >= 0.15 and count >= 3):
            return "medium"
        return "low"

    def _evidence_cell(label: str, count: int, population: int) -> str:
        if population <= 0 or count <= 0:
            return "insufficient historical evidence"
        # Never report count > population (stale probe vs narrowed scope).
        use_count = min(int(count), int(population))
        share = round(100.0 * use_count / population, 1)
        return f"{label} on {use_count} of {population} ({share}%) in scoped historical set"

    def _pick_season_aware(
        field: str, population: int
    ) -> tuple[str, int, str, str] | None:
        groups = cypher_group_by(platform, graph_id, field, decision_scope) or []
        usable = _usable_ranked(groups, field, limit=8)
        if not usable:
            return None
        # Prefer event/season tags that match the user-requested season (Winter > Fall).
        if field in {"ai_event_tags", "ai_event_tag"} and season_requested:
            scored: list[tuple[int, str, int]] = []
            for name, count in usable:
                key = normalize_name(str(name))
                score = 0
                if key == season_requested or key.startswith(season_requested + " "):
                    score = 100
                elif season_requested in key:
                    score = 80
                elif season_requested == "winter" and any(
                    tok in key for tok in ("winter", "aw", "autumn winter", "fall winter")
                ):
                    score = 90 if "winter" in key else 40
                elif season_requested == "winter" and key in {"fall", "autumn"}:
                    score = 0  # never propose Fall/Autumn for a Winter ask
                scored.append((score, str(name), int(count)))
            scored.sort(key=lambda row: (-row[0], -row[2], row[1]))
            # Require a season-matching tag; otherwise leave pending for buyer.
            if scored and scored[0][0] >= 40:
                name, count = scored[0][1], scored[0][2]
                options = "; ".join(
                    f"{_display_label(str(n))} ({c} of {population})" for n, c in usable
                )
                return _display_label(str(name)), int(count), options, str(name)
            return None
        best_name, best_count = usable[0]
        # Never propose yarn-count units as material (Nm / tex / denier).
        if field in {"material", "materials"}:
            if not _usable_value(str(best_name), field):
                nxt = next(
                    ((n, c) for n, c in usable if _usable_value(str(n), field)),
                    None,
                )
                if not nxt:
                    return None
                best_name, best_count = nxt
        options = "; ".join(
            f"{_display_label(str(name))} ({count} of {population})" for name, count in usable
        )
        return _display_label(str(best_name)), int(best_count), options, str(best_name)

    for label, value in requirement_rows:
        decision_rows.append(
            {
                "field": label,
                "value": value,
                "evidence": "stated in user question",
                "reason": "User-provided requirement — locks analysis scope",
                "confidence": "user-provided",
                "confirm": "No",
                "basis": "user-provided",
                "raw_key": None,
                "raw_value": None,
            }
        )

    decide_order = [
        *[f for f in _SPEC_ANCHOR_FIELDS if f in spec_fields],
        *[f for f in spec_fields if f not in set(_SPEC_ANCHOR_FIELDS)],
    ]
    relationship_notes: list[str] = []
    for field in decide_order:
        label = field.replace("_", " ")
        # Always re-probe on the current decision scope so evidence counts match population.
        picked = _pick_season_aware(field, decision_total)
        if field in locked_raw and not picked:
            decision_rows.append(
                {
                    "field": label,
                    "value": _display_label(locked_raw[field]),
                    "evidence": "locked by hierarchy / prior filter",
                    "reason": "Kept to preserve a coherent attribute combination",
                    "confidence": "medium",
                    "confirm": "No",
                    "basis": "historically observed",
                    "raw_key": field,
                    "raw_value": locked_raw[field],
                }
            )
            continue
        if field in locked_raw and picked:
            best, best_count, _opts, raw = picked
            # Prefer the locked raw when still present; otherwise take fresh probe.
            if normalize_name(str(raw)) != normalize_name(locked_raw[field]):
                # Re-count the locked value on current scope.
                locked_groups = cypher_group_by(platform, graph_id, field, decision_scope) or []
                match = next(
                    (
                        (n, c)
                        for n, c in locked_groups
                        if normalize_name(str(n)) == normalize_name(locked_raw[field])
                    ),
                    None,
                )
                if match:
                    best = _display_label(str(match[0]))
                    best_count = int(match[1])
                    raw = str(match[0])
            conf = _confidence(best_count, decision_total)
            decision_rows.append(
                {
                    "field": label,
                    "value": best,
                    "evidence": _evidence_cell(best, best_count, decision_total),
                    "reason": "Hierarchy/compatible lock from historical narrowing",
                    "confidence": conf if conf != "insufficient" else "medium",
                    "confirm": "No",
                    "basis": "historically observed",
                    "raw_key": field,
                    "raw_value": locked_raw[field],
                }
            )
            continue

        if not picked:
            decision_rows.append(
                {
                    "field": label,
                    "value": "Pending — insufficient evidence",
                    "evidence": "no usable historical values in the season/party scope",
                    "reason": "Do not invent a value; buyer confirmation required",
                    "confidence": "insufficient",
                    "confirm": "Yes",
                    "basis": "not available / buyer confirmation",
                    "raw_key": field,
                    "raw_value": None,
                }
            )
            continue

        best, best_count, options, raw = picked
        conf = _confidence(best_count, decision_total)
        analysis_rows[field] = (best, best_count, options, raw)
        alt_bits = []
        if options and "; " in options:
            alt_bits = [p.strip() for p in options.split(";") if p.strip()][1:3]
        reason = "Leading historically observed value on the current compatible scope"
        if alt_bits:
            reason += f"; alternatives: {'; '.join(alt_bits)}"
        narrowed_ok = False
        if conf in {"high", "medium"} and raw:
            candidate = dict(decision_scope)
            candidate[field] = raw
            narrowed = cypher_count_products(platform, graph_id, candidate)
            if narrowed and int(narrowed) > 0:
                if locked_raw:
                    relationship_notes.append(
                        f"{label}={best} co-occurs with locked attributes on "
                        f"{int(narrowed)} of {decision_total} scoped products."
                    )
                decision_scope = candidate
                decision_total = int(narrowed)
                locked_raw[field] = raw
                narrowed_ok = True
                reason += f"; combination retained ({decision_total} products remain)"
        if not narrowed_ok and conf == "low":
            decision_rows.append(
                {
                    "field": label,
                    "value": best,
                    "evidence": _evidence_cell(best, best_count, decision_total),
                    "reason": reason + "; weak share — optional recommendation only",
                    "confidence": conf,
                    "confirm": "Yes",
                    "basis": "AI-recommended",
                    "raw_key": field,
                    "raw_value": raw,
                }
            )
            continue
        if not narrowed_ok and conf in {"high", "medium"}:
            locked_raw[field] = raw
        decision_rows.append(
            {
                "field": label,
                "value": best,
                "evidence": _evidence_cell(best, best_count, decision_total),
                "reason": reason,
                "confidence": conf,
                "confirm": "Yes" if conf in {"low", "insufficient"} else "No",
                "basis": "historically observed" if conf in {"high", "medium"} else "AI-recommended",
                "raw_key": field,
                "raw_value": raw if conf in {"high", "medium", "low"} else None,
            }
        )

    filled = sum(
        1
        for row in decision_rows
        if row.get("raw_value") or row.get("basis") == "user-provided"
    )
    missing_fields = [
        row["field"]
        for row in decision_rows
        if row.get("basis") == "not available / buyer confirmation"
    ]
    confirm_fields = [
        row["field"] for row in decision_rows if str(row.get("confirm") or "").casefold() == "yes"
    ]

    catalog_label = _graph_label(platform, graph_id)
    season_title = (season_requested or "catalog").strip().title()
    party_title = str(party).strip().title()

    def _alt_cell(options: str, best: str) -> str:
        parts = [p.strip() for p in (options or "").split(";") if p.strip()]
        alts: list[str] = []
        for part in parts:
            label_bit = part.split("(", 1)[0].strip()
            if normalize_name(label_bit) == normalize_name(best):
                continue
            # Prefer "Name (count)" without the "of N" population suffix.
            if "(" in part and " of " in part:
                name, rest = part.split("(", 1)
                count_bit = rest.split(" of ", 1)[0].strip()
                alts.append(f"{name.strip()} ({count_bit})")
            else:
                alts.append(part)
            if len(alts) >= 3:
                break
        return ", ".join(alts) if alts else "—"

    def _sheet_value(row: dict[str, str]) -> str:
        basis = row.get("basis") or ""
        if basis in {"not available / buyer confirmation", "AI-recommended"}:
            return "Pending buyer confirmation"
        if str(row.get("confirm") or "").casefold() == "yes" and basis != "user-provided":
            return "Pending buyer confirmation"
        return str(row.get("value") or "Pending buyer confirmation")

    by_field = {
        str(row.get("raw_key") or "").casefold(): row
        for row in decision_rows
        if row.get("raw_key")
    }

    def _val(field: str, default: str = "Pending buyer confirmation") -> str:
        row = by_field.get(field.casefold())
        if not row:
            return default
        return _sheet_value(row)

    # --- Buyer intelligence brief (requirements → analysis → summary → JSON) ---
    lines.append(f"# {season_title} Product Specification — {party_title}")
    lines.append("")
    lines.append("## 1. User Requirements")
    lines.append("")
    lines.append(
        f"The user requests a new {season_title}-season product specification for "
        f"{party_title} buyers, developed using historical {catalog_label} product data."
    )
    lines.append("")
    for label, value in requirement_rows:
        lines.append(f"- **{label}:** {value}")
    lines.append(f"- **Data Source:** {catalog_label}")
    lines.append(
        "- **Objective:** Develop an evidence-based product specification using "
        "historical product attributes and patterns."
    )
    lines.append(
        "- **Deliverable:** A new proposed product specification for buyer review."
    )
    lines.append("")
    lines.append("## 2. Historical Data Analysis")
    lines.append("")
    lines.append(
        f"Historical {party_title} products were examined using the user-provided "
        f"requirements and relevant available product attributes "
        f"({scope_total} in season/party scope; {decision_total} after compatible narrowing)."
    )
    lines.append("")
    lines.append(
        "The analysis focuses on identifying commonly used attribute values, "
        "alternative choices, and compatible combinations within the selected product scope."
    )
    lines.append("")
    lines.append("### Attribute Analysis")
    lines.append("")
    lines.append("| Attribute | Leading Value | Count | Alternative Values |")
    lines.append("|---|---|---:|---|")
    attr_shown = 0
    for field in decide_order:
        best, best_count, options, raw = analysis_rows.get(field, ("", 0, "", None))
        if not raw or not best_count:
            # Prefer decision-row display when probe was season-aware / locked.
            row = by_field.get(field)
            if not row or not row.get("raw_value"):
                continue
            best = str(row.get("value") or "")
            best_count = 0
            options = ""
            m = re.search(r" on (\d+) of ", str(row.get("evidence") or ""))
            if m:
                best_count = int(m.group(1))
        if not best or not best_count:
            continue
        lines.append(
            f"| {field.replace('_', ' ').title()} | {best} | {best_count} | "
            f"{_alt_cell(options, best)} |"
        )
        attr_shown += 1
    if not attr_shown:
        lines.append("| — | insufficient historical evidence | 0 | — |")
    lines.append("")
    lines.append(
        "*Counts are historical observations from the supplied analysis, "
        "not proposed production quantities.*"
    )
    lines.append("")
    lines.append("## 3. Analysis Summary")
    lines.append("")
    cat = _val("category", "")
    subsub = _val("plm_sub_sub_category", "")
    gtype = _val("garment_type", "")
    gender = _val("gender", "")
    material = _val("material", "")
    color = _val("color", "")
    gauge = _val("gauge", "")
    event = _val("ai_event_tags", "")
    if subsub != "Pending buyer confirmation" or gtype != "Pending buyer confirmation":
        direction = " / ".join(
            v
            for v in (subsub if subsub != "Pending buyer confirmation" else "", gtype if gtype != "Pending buyer confirmation" else "")
            if v
        ) or (cat if cat != "Pending buyer confirmation" else "historical knitwear")
        lines.append(
            f"- **Product direction:** {direction} is a prominent historical product type "
            "in the analyzed records."
        )
    if gender != "Pending buyer confirmation":
        lines.append(
            f"- **Target customer:** {gender}'s products represent the leading gender "
            "selection in the narrowed historical comparison."
        )
    if relationship_notes:
        for note in relationship_notes[:3]:
            lines.append(f"- {note}")
    if material != "Pending buyer confirmation":
        lines.append(
            f"- **Material:** {material} leads historically; final material should consider "
            f"the {season_title} requirement and available evidence."
        )
    elif "material" in confirm_fields or "material" in missing_fields:
        lines.append(
            f"- **Material:** insufficient/weak evidence — buyer confirmation required "
            f"for the {season_title} sheet."
        )
    if color != "Pending buyer confirmation":
        lines.append(
            f"- **Colour:** {color} is the leading individual colour in the reported comparison."
        )
    if gauge != "Pending buyer confirmation":
        lines.append(f"- **Gauge:** {gauge} is the leading historical gauge.")
    if event != "Pending buyer confirmation":
        lines.append(f"- **Season tags:** {event} aligns with the requested season scope.")
    if confirm_fields:
        lines.append(
            "- **Evidence limitations:** "
            + ", ".join(confirm_fields[:8])
            + ("…" if len(confirm_fields) > 8 else "")
            + " should remain subject to buyer confirmation."
        )
    lines.append("")
    conclusion_bits = [
        v
        for v in (
            gender if gender != "Pending buyer confirmation" else "",
            season_title,
            subsub if subsub != "Pending buyer confirmation" else (
                cat if cat != "Pending buyer confirmation" else "product"
            ),
        )
        if v
    ]
    lines.append(
        f"**Conclusion:** Based on the user's {season_title} requirement and the historical "
        f"{catalog_label} patterns, a {' '.join(conclusion_bits)} is a reasonable proposed "
        "product direction. The final values below are recommendations for buyer review, "
        "not automatically approved production specifications."
    )
    lines.append("")
    lines.append("## 4. Final Product Specification")
    lines.append("")
    # Detector / grounding markers (kept for product_spec + answer_card).
    lines.append("Product specification:")
    lines.append("Proposed new product specification (NOT an existing PLM/catalog record):")
    lines.append("Product identity: Pending PLMEX creation/assignment")
    lines.append("")

    product_name_parts = [
        p
        for p in (
            f"{gender}'s" if gender not in {"Pending buyer confirmation", ""} else "",
            season_title if season_requested else "",
            subsub if subsub != "Pending buyer confirmation" else (
                cat if cat != "Pending buyer confirmation" else "Product"
            ),
        )
        if p
    ]
    product_name = " ".join(product_name_parts) or f"{party_title} Proposed Product"

    # Nested construction from AI construction tags when present.
    construction_tag = _val("ai_construction_tags")
    finish_tag = _val("ai_finish_tags")
    design_for = _val("design_for")
    composition = _val("composition")
    knit_machine = _val("knit_machine_type")
    gsm = _val("gsm")
    weight = _val("weight")
    plm_sub = _val("plm_sub_category")
    plm_main = _val("plm_main_category")

    sheet = {
        "product_name": product_name,
        "brand": party_title,
        "season": season_title if season_requested else "Pending buyer confirmation",
        "product_code": "Pending PLMEX creation",
        "style_reference": "Pending PLMEX creation",
        "category": cat if cat != "Pending buyer confirmation" else _val("category"),
        "sub_category": plm_sub,
        "main_category": plm_main,
        "sub_sub_category": subsub,
        "garment_type": gtype,
        "gender": gender,
        "design_for": design_for,
        "composition": composition,
        "material": material,
        "colour": color,
        "gauge": gauge,
        "knit_machine_type": knit_machine,
        "gsm": gsm,
        "weight": weight,
        "construction": construction_tag,
        "finish": finish_tag,
        "ai_event_tags": event,
        "ai_material_tags": _val("ai_material_tags"),
        "approval_status": "Draft - Pending Buyer Review",
    }
    lines.append("```json")
    lines.append(json.dumps(sheet, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    if confirm_fields:
        lines.append("### Buyer confirmation requirements")
        for field in confirm_fields:
            lines.append(f"- {field}")
        lines.append("")
    lines.append(
        "Historical benchmark note: do not reuse historical product codes or customer style "
        "references as identifiers for this proposed product."
    )

    evidence.append(
        EvidenceItem(
            source_id=graph_id,
            record_id="specification",
            title=f"{catalog_label} proposed specification",
            excerpt=(
                f"{party_key}={party}; products={decision_total}; "
                f"fields={filled}/{len(decision_rows)}; "
                f"missing={len(missing_fields)}; filters={decision_scope}; "
                f"season={season_requested or 'unset'}; "
                "proposed_new=true; identity=Pending PLMEX creation; "
                "decision_engine=true; report=buyer_brief"
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

    filters = dict(filters or {})
    # Style / SKU tokens must resolve via id ENDS WITH — not be dropped as a facet.
    pref = str(
        filters.get("product_id")
        or filters.get("style_reference")
        or filters.get("sku")
        or ""
    ).strip()
    products: list = []
    if pref:
        resolved = cypher_resolve_product_id(platform, graph_id, pref)
        if resolved:
            session_factory = _session(platform)
            if session_factory:
                try:
                    with session_factory() as session:
                        row = session.run(
                            "MATCH (p:Entity:Product) WHERE p.graph = $g AND p.id = $id "
                            "AND coalesce(p.lifecycle,'active') <> 'deleted' "
                            "RETURN p",
                            {"g": graph_id, "id": resolved},
                        ).single()
                    if row and row.get("p"):
                        node = row["p"]
                        # Adapt neo4j node to the shape search_neo4j_products returns.
                        from types import SimpleNamespace

                        try:
                            props = dict(node)
                        except Exception:
                            props = {k: node.get(k) for k in node.keys()} if hasattr(node, "keys") else {}
                        if not props.get("id"):
                            props["id"] = resolved
                        products = [
                            SimpleNamespace(
                                id=str(props.get("id") or resolved),
                                properties=props,
                            )
                        ]
                except Exception:
                    products = []
        filters.pop("product_id", None)
        filters.pop("style_reference", None)
        filters.pop("sku", None)
    if not products:
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
    meta_result: dict | None = None,
    include_optional_metadata: bool = True,
) -> None:
    filters = clean_product_filters(filters)
    try:
        from fashion_brain.planning.semantic_field_map import apply_semantic_field_map

        filters, group_by = apply_semantic_field_map(
            filters, graph_id, group_by=group_by
        )
        filters = clean_product_filters(filters)
    except Exception:
        pass
    filters = _align_party_filters_for_graph(platform, graph_id, filters)
    if group_by:
        resolved_gb = resolve_group_by_with_fallback(
            platform, graph_id, group_by, filters
        )
        if resolved_gb and resolved_gb != str(group_by):
            caveats.append(
                f"Grouped by {resolved_gb.replace('_', ' ')} "
                f"(role-similar to {str(group_by).replace('_', ' ')} on this catalog)."
            )
            group_by = resolved_gb
        elif resolved_gb:
            group_by = resolved_gb
    generated = getattr(getattr(platform, "settings", None), "generated", None)
    try:
        from knowledge_graph.retrieval.kg_stats import answer_simple_count, stats_available

        has_stats = stats_available(graph_id, generated)
    except Exception:
        has_stats = False
        answer_simple_count = None  # type: ignore[assignment]

    # Ranked product lists (top-N / highest price) must never collapse to a meta count.
    ranked_listing = False
    price_sort = None
    try:
        from fashion_brain.ask import (
            is_ranked_product_listing_ask,
            product_price_sort_direction,
        )

        ranked_listing = bool(is_ranked_product_listing_ask(question)) or (
            limit is not None and bool(product_price_sort_direction(question))
        )
        price_sort = product_price_sort_direction(question)
    except Exception:
        ranked_listing = False
        price_sort = None

    # Metadata-first: 0–1 filter simple counts never hit Cypher when kg_stats exists.
    # Multi-filter / uncached dimension → answer_simple_count returns None (cache_miss).
    if (
        has_stats
        and answer_simple_count is not None
        and not group_by
        and len(filters) <= 1
        and not ranked_listing
        and limit is None
    ):
        # Skip meta short-circuit for intentional multi-party compare ("Primark | Zara").
        compare_raw = str(filters.get("brand") or filters.get("buyer") or "")
        if "|" not in compare_raw:
            meta = meta_result
            if meta is None:
                meta = answer_simple_count(
                    graph_id,
                    filters=filters,
                    count_entity=count_entity,
                    generated=generated,
                )
            if meta is not None:
                from knowledge_graph.retrieval.kg_stats import filters_match, normalize_filter_dict

                requested = normalize_filter_dict(
                    meta.get("requested_filters") if meta.get("requested_filters") is not None else filters
                )
                applied = normalize_filter_dict(meta.get("applied_filters") or {})
                # Guard: never accept an unfiltered global hit for a filtered ask.
                if requested and not filters_match(requested, applied):
                    meta = None
                else:
                    label = _graph_label(platform, graph_id)
                    n = int(meta["n"])
                    kind = str(meta.get("kind") or "")
                    if kind == "distinct_facet":
                        noun = str(meta.get("attribute") or "value").replace("_", " ")
                        attr = str(meta.get("attribute") or noun)
                        example_bit = _facet_example_phrase(
                            graph_id, attr, generated=generated, limit=5
                        )
                        line = f"There are {n:,} distinct {noun}s in {label}"
                        if example_bit:
                            line = f"{line} ({example_bit})"
                        lines.append(line + ".")
                        title = f"{label} distinct {noun} count"
                        excerpt = f"{meta.get('attribute')}s={n}"
                        if example_bit:
                            excerpt = f"{excerpt}; {example_bit}"
                    elif kind == "value_count":
                        phrase = _filter_phrase(filters)
                        # Enrich with top values of the counted facet itself
                        # (e.g. brand=primark count → n/a; category filter → other
                        # categories are competing — only distinct-facet gets names).
                        # For single-filter product counts, name the filter value clearly;
                        # optional_metadata may still carry related context for the writer.
                        lines.append(f"There are {n:,} products{phrase}.")
                        title = f"{label} {meta.get('attribute')}={meta.get('value')}"
                        filter_label = str(
                            meta.get("filter")
                            or f"filter: {meta.get('attribute')}={meta.get('value')}"
                        )
                        excerpt = f"{filter_label} → {n} (from generated/kg_stats)"
                    else:
                        line = f"There are {n:,} products"
                        brand_bit = _facet_example_phrase(
                            graph_id, "brand", generated=generated, limit=4
                        )
                        if brand_bit:
                            line = f"{line} ({brand_bit})"
                        lines.append(line + ".")
                        title = f"{label} product total"
                        excerpt = f"products={n}"
                    evidence.append(
                        EvidenceItem(
                            source_id=graph_id,
                            record_id="kg_stats",
                            title=title,
                            excerpt=excerpt,
                            source_path=f"kg_stats/{graph_id}.json",
                            verification_status="observed",
                            graph=graph_id,
                            node_id=None,
                            synthetic=False,
                            requested_filters=requested,
                            applied_filters=applied,
                        )
                    )
                    # Optional facet facts for ResponseAgent — not forced into the answer.
                    if (
                        include_optional_metadata
                        and kind in {"value_count", "product_total", "distinct_facet", ""}
                        and n > 0
                    ):
                        _attach_optional_facet_metadata(
                            platform,
                            graph_id,
                            filters,
                            total=n,
                            evidence=evidence,
                            question=question,
                        )
                    return
            # Cache-miss with filters present: do NOT fall through to a global total.
            if filters:
                # Continue to Cypher / multi-filter path below — never invent products=total.
                pass

    # "how many <facet>s" → distinct values of that graph facet, never the product total.
    entity = resolve_count_facet(platform, graph_id, count_entity, question=question)
    if entity and entity != "product" and not filters and not group_by:
        label = _graph_label(platform, graph_id)
        noun = entity.replace("_", " ")
        n: int | None = None
        source = "cypher"
        try:
            from knowledge_graph.retrieval.kg_stats import unique_count

            n = unique_count(graph_id, entity, generated)
            if n is not None:
                source = "kg_stats"
        except Exception:
            n = None
        # When stats exist, never fall back to Cypher for bare facet counts.
        if n is None and not has_stats:
            groups = cypher_group_by(platform, graph_id, entity, {})
            if groups is not None:
                n = sum(
                    1
                    for name, _count in groups
                    if _usable_value(str(name))
                    and str(name).strip().casefold() not in {"(empty)", "(unspecified)"}
                )
                source = "cypher"
        elif n is None and has_stats:
            n = 0
            source = "kg_stats"
        if n is not None:
            example_bit = _facet_example_phrase(
                graph_id, entity, generated=generated, limit=5
            )
            line = f"There are {n:,} distinct {noun}s in {label}"
            if example_bit:
                line = f"{line} ({example_bit})"
            lines.append(line + ".")
            excerpt = f"{entity}s={n}"
            if example_bit:
                excerpt = f"{excerpt}; {example_bit}"
            evidence.append(
                EvidenceItem(
                    source_id=graph_id,
                    record_id="kg_stats" if source == "kg_stats" else "facet_count",
                    title=f"{label} distinct {noun} count",
                    excerpt=excerpt,
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
        scope = (_filter_phrase(other_facets) or " matching products").strip()
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

    # Plain unfiltered total — metadata first; Cypher only when stats file is missing.
    if not filters and not group_by:
        total: int | None = None
        from_stats = False
        try:
            from knowledge_graph.retrieval.kg_stats import product_total

            total = product_total(graph_id, generated)
            if total is not None:
                from_stats = True
        except Exception:
            total = None
        if total is None and not has_stats:
            total = neo4j_product_count(platform, graph_id)
        if total is None:
            total = 0
            from_stats = has_stats
        lines.append(f"There are {total:,} products.")
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="kg_stats" if from_stats else "count",
                title=f"{label} product total",
                excerpt=f"products={total}",
                source_path=stats_file if from_stats else graph_id,
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
                requested_filters={},
                applied_filters={},
            )
        )
        return
    if not filters and group_by:
        cached = _stats_breakdown(str(group_by))
        if cached:
            total, groups = cached
            lines.append(f"There are {total:,} products.")
            breakdown_lines = _dimension_breakdown_lines(
                str(group_by),
                groups,
                total,
                _explicit_facet_limit(question, limit),
                graph_id=graph_id,
                platform=platform,
                generated=generated,
            )
            lines.extend(breakdown_lines)
            evidence.append(
                EvidenceItem(
                    source_id=graph_id,
                    record_id="kg_stats",
                    title=f"{label} {group_by} breakdown",
                    excerpt=(
                        f"products={total}; {group_by} values={len(groups)} "
                        f"(from generated/kg_stats)\n" + "\n".join(breakdown_lines)
                    ),
                    source_path=stats_file,
                    verification_status="observed",
                    graph=graph_id,
                    node_id=None,
                    synthetic=False,
                )
            )
            return
        # Stats exist but this facet shard is missing (e.g. ecom color/material) —
        # fall through to live Cypher instead of inventing an empty catalog.
    if len(filters) == 1 and not group_by and limit is None:
        # Single-filter: use kg_stats only when that dimension is pre-computed.
        # Uncached dimension → fall through to live Cypher (never fake a global/zero meta hit).
        attr, value = next(iter(filters.items()))
        try:
            from knowledge_graph.retrieval.kg_stats import has_dimension, normalize_filter_dict, value_count

            if has_dimension(graph_id, str(attr), generated):
                cached = value_count(graph_id, str(attr), str(value), generated)
                n = int(cached if cached is not None else 0)
                applied = normalize_filter_dict({str(attr): str(value)})
                lines.append(f"There are {n:,} products{phrase}.")
                evidence.append(
                    EvidenceItem(
                        source_id=graph_id,
                        record_id="kg_stats",
                        title=f"{label} {attr}={value}",
                        excerpt=f"filter: {attr}={value} → {n} (from generated/kg_stats)",
                        source_path=f"kg_stats/{graph_id}.json",
                        verification_status="observed",
                        graph=graph_id,
                        node_id=None,
                        synthetic=False,
                        requested_filters=applied,
                        applied_filters=applied,
                    )
                )
                if n > 0:
                    _attach_optional_facet_metadata(
                        platform,
                        graph_id,
                        filters,
                        total=n,
                        evidence=evidence,
                        question=question,
                    )
                return
        except Exception:
            pass
    # Multi-filter / filtered+group_by: Cypher aggregates — never scan all Product nodes.
    if group_by:
        groups = cypher_group_by(platform, graph_id, str(group_by), filters)
        if groups is None:
            groups = _count_by(platform, principal, graph_id, str(group_by), filters)
        total = cypher_count_products(platform, graph_id, filters)
        if groups is not None and total is not None:
            usable = _usable_ranked(groups, str(group_by))
            lines.append(f"There are {total:,} products" + (f"{phrase}." if phrase else "."))
            breakdown_lines = _dimension_breakdown_lines(
                str(group_by),
                usable,
                total,
                _explicit_facet_limit(question, limit),
                graph_id=graph_id,
                platform=platform,
                generated=generated,
            )
            lines.extend(breakdown_lines)
            # Structured evidence so answer cards do not depend on prose parsing alone.
            evidence.append(
                EvidenceItem(
                    source_id=graph_id,
                    record_id="group_by",
                    title=f"{label} {group_by} breakdown",
                    excerpt=f"products={total}\n" + "\n".join(breakdown_lines),
                    source_path=graph_id,
                    verification_status="observed",
                    graph=graph_id,
                    node_id=None,
                    synthetic=False,
                    requested_filters=dict(filters or {}),
                    applied_filters=dict(filters or {}),
                )
            )
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

    # Pure "how many …" asks: one Cypher COUNT + one sentence. No samples,
    # brand/category mix, or "open product listing" caveats — those belong to
    # listing / show-me asks, not quantity asks.
    try:
        from fashion_brain.ask import is_product_count_ask

        pure_count = (
            is_product_count_ask(question or "")
            and not ranked_listing
            and not group_by
            and limit is None
        )
    except Exception:
        pure_count = False
    if pure_count and filters:
        total = cypher_count_products(platform, graph_id, filters)
        if total is None:
            total = 0
        lines.append(
            f"There are {int(total):,} products" + (f"{phrase}." if phrase else ".")
        )
        evidence.append(
            EvidenceItem(
                source_id=graph_id,
                record_id="cypher_count",
                title=f"{label} product count",
                excerpt=f"products={int(total)}; filters={filters}",
                source_path=graph_id,
                verification_status="observed",
                graph=graph_id,
                node_id=None,
                synthetic=False,
                requested_filters=dict(filters or {}),
                applied_filters=dict(filters or {}),
            )
        )
        return

    # Open listings (no user top-N): short sample. Explicit limit: honour it, still capped.
    open_listing = limit is None and not ranked_listing
    show_n = max(int(limit or 5), 1)
    if open_listing:
        show_n = min(show_n, 5)
    else:
        show_n = min(show_n, 25)
    sample, total = search_neo4j_products(
        platform,
        principal,
        graph_id,
        filters,
        limit=show_n,
        order_by_price=price_sort,
    )
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
            ranked_listing=ranked_listing,
            price_sort=price_sort,
        )
    except Exception:
        pass
    if not total and not filters:
        lines.append("There are 0 visible products for this ask.")
    elif ranked_listing:
        order_label = (
            "highest price first"
            if price_sort == "desc"
            else "lowest price first"
            if price_sort == "asc"
            else "ranked"
        )
        lines.append(
            f"Top {min(len(sample), show_n)} products{phrase} by price ({order_label}) "
            f"from {total:,} matching."
        )
        table = _product_table_lines(
            sample, filters, question, limit=show_n, force_price=True
        )
        if table:
            lines.extend(table)
        else:
            lines.append("No priced products matched this scope.")
    else:
        lines.append(f"There are {total:,} products" + (f"{phrase}." if phrase else "."))
        if group_by:
            grouped = cypher_group_by(platform, graph_id, str(group_by), filters)
            if grouped is None:
                grouped = _count_by(platform, principal, graph_id, group_by, filters)
            groups = _usable_ranked(grouped or [], str(group_by))
            lines.extend(
                _dimension_breakdown_lines(
                    group_by,
                    groups,
                    total,
                    _explicit_facet_limit(question, limit),
                    graph_id=graph_id,
                    platform=platform,
                    generated=generated,
                )
            )
        elif filters:
            # Prefer count-wise description over dumping every matching name.
            brand_groups = cypher_group_by(platform, graph_id, "brand", filters)
            if brand_groups:
                usable_brands = _usable_ranked(brand_groups, "brand")
                brand_lines = _dimension_breakdown_lines(
                    "brand",
                    usable_brands,
                    total,
                    8,
                    graph_id=graph_id,
                    platform=platform,
                    generated=generated,
                    user_set_limit=True,
                )
                if brand_lines:
                    lines.append("By brand:")
                    lines.extend(brand_lines)
            if "material" in filters or "color" in filters:
                cat_groups = cypher_group_by(platform, graph_id, "category", filters)
                if cat_groups:
                    usable_cats = _usable_ranked(cat_groups, "category")
                    cat_lines = _dimension_breakdown_lines(
                        "category",
                        usable_cats,
                        total,
                        8,
                        graph_id=graph_id,
                        platform=platform,
                        generated=generated,
                        user_set_limit=True,
                    )
                    if cat_lines:
                        lines.append("By category:")
                        lines.extend(cat_lines)
            table = _product_table_lines(sample, filters, question, limit=show_n)
            if table:
                lines.append(
                    f"Example products ({min(len(table) - 2, show_n)} of {total}):"
                )
                lines.extend(table)
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


# Soft facet tables: show every value when the set is small; otherwise preview + note.
_FACET_AUTO_FULL_MAX = 25
_FACET_AUTO_PREVIEW = 20
_EXPLICIT_FACET_LIMIT_RE = re.compile(
    r"\b(?:top|first|show|list|give|about|around)\s+(\d{1,3})\b",
    re.IGNORECASE,
)


def _explicit_facet_limit(question: str | None, limit: int | None) -> int | None:
    """Return limit only when the question named an explicit top/show N."""

    if limit is None:
        return None
    if _EXPLICIT_FACET_LIMIT_RE.search(str(question or "")):
        return max(1, int(limit))
    return None


def _facet_row_budget(
    available: int,
    *,
    limit: int | None = None,
    user_set_limit: bool = False,
) -> tuple[int, str | None]:
    """How many ranked facet rows to show + optional truncation note.

    - User named N → honor that N (note when more exist).
    - No N and set is small → show all.
    - No N and set is huge → preview 20 and say "Showing 20 of N unique values."
    """

    total = max(0, int(available or 0))
    if total <= 0:
        return 0, None
    if user_set_limit and limit is not None:
        n = max(1, int(limit))
        if total > n:
            return n, f"Showing {n} of {total} unique values (your limit)."
        return total, None
    if total <= _FACET_AUTO_FULL_MAX:
        return total, None
    n = min(_FACET_AUTO_PREVIEW, total)
    return n, f"Showing {n} of {total} unique values."


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
    *,
    brief: bool = False,
    user_set_limit: bool = False,
) -> None:
    """Type-wise top rankings for a brand/selection on any Neo4j product graph.

    ``analysis_fields`` come from that graph's live schema (via resolve_analysis_fields).
    Works for ecom, plmex, or any registered product KG — not e-commerce-only.
    ``brief=True`` titles the answer as a product brief and keeps only requested fields.
    ``user_set_limit=True`` means ``limit`` came from the question (top/show N).
    """

    filters = _align_party_filters_for_graph(platform, graph_id, clean_product_filters(filters))
    # Multi-party "Primark | Zara" cannot be one brand token — analyse each side.
    try:
        from fashion_brain.ask import split_multi_party_filter

        parties = split_multi_party_filter(str(filters.get("brand") or filters.get("buyer") or ""))
    except Exception:
        parties = []
    if len(parties) >= 2:
        party_key = "brand" if filters.get("brand") else "buyer"
        for party in parties:
            side = {k: v for k, v in filters.items() if k not in {"brand", "buyer"}}
            side[party_key] = party
            lines.append(f"### {_display_label(party)}")
            append_neo4j_analysis_answer(
                platform,
                principal,
                graph_id,
                side,
                limit,
                lines,
                evidence,
                caveats,
                analysis_fields=analysis_fields,
                brief=brief,
                user_set_limit=user_set_limit,
            )
            lines.append("")
        return
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
    # Per-field row budget is resolved after counting unique values (see loop below).
    focused = bool(analysis_fields) and len(list(analysis_fields)) == 1
    who_raw = filters.get("brand") or filters.get("buyer") or "this selection"
    who = _display_label(str(who_raw), graph_id=graph_id, platform=platform)
    garment_bit = (
        filters.get("garment")
        or filters.get("garment_type")
        or filters.get("category")
        or ""
    )
    garment_label = _display_label(str(garment_bit), graph_id=graph_id, platform=platform) if garment_bit else ""
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
            brief=brief,
        )
    except Exception:
        pass
    if not total:
        lines.append(f"There are no products for {who} to analyse.")
        return

    # Metrics as plain lines (no ###). Only real attribute slices use ### so merge can
    # pick visualizations by structure — no heading-name allow/deny lists.
    if brief and garment_label:
        lines.append(f"Product brief for {who} {garment_label}: **{total}** products.")
    elif brief:
        lines.append(f"Product brief for {who}: **{total}** products.")
    else:
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
    # Compact uniqueness summary — skip on focused briefs (requested fields only).
    if not brief:
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
                uniqueness.append(f"{len(usable)} {_pluralize_label(label)}")
        if uniqueness:
            # Full analysis should surface every facet that has distinct values.
            lines.append("Distinct values: " + ", ".join(uniqueness) + ".")

    if brief:
        scope = "requested attributes"
    elif focused:
        scope = "focused attribute"
    else:
        scope = "all attributes"
    if user_set_limit and limit is not None:
        lines.append(
            f"Field-wise analysis ({scope}; up to {int(limit)} unique values per field with counts and shares)."
        )
    else:
        lines.append(
            f"Field-wise analysis ({scope}; all values when ≤{_FACET_AUTO_FULL_MAX}, "
            f"otherwise top {_FACET_AUTO_PREVIEW} with a count note)."
        )
    # Skip facets the ask already fixed to one value (self-echo breakdowns).
    fixed_filters = {
        normalize_name(str(k)): normalize_name(str(v))
        for k, v in (filters or {}).items()
        if v not in (None, "")
    }
    generated = getattr(getattr(platform, "settings", None), "generated", None)
    for field in fields:
        if not is_analysis_facet_attr(str(field)):
            continue
        all_usable = _usable_ranked(
            ranked.get(field),
            str(field),
            active_party=active_party,
            reserved_values=reserved_values,
        )
        if not all_usable:
            continue
        show_n, trunc_note = _facet_row_budget(
            len(all_usable),
            limit=limit,
            user_set_limit=user_set_limit,
        )
        choices = all_usable[:show_n] if show_n else all_usable
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
        field_label = _human_analysis_field_label(str(field))
        # Country codes + names are usually disjoint product sets → sum after aliasing.
        combine = (
            "sum"
            if field_key in {"country", "country_of_origin", "origin", "origin_country", "made_in"}
            else "max"
        )
        display_choices = _merge_display_facet_rows(
            [
                (
                    _display_label(
                        name,
                        graph_id=graph_id,
                        attribute=str(field),
                        generated=generated,
                        platform=platform,
                    ),
                    count,
                )
                for name, count in choices
            ],
            combine=combine,
        )
        if not display_choices:
            continue
        # Field-wise block: heading + leader + full value table (+ optional chart).
        lines.append(f"<!--fb:viz:{field}-->")
        lines.append(f"### {field_label}")
        top_name, top_count = display_choices[0]
        share = round(100.0 * top_count / total, 1) if total else 0.0
        lines.append(f"Leading value: {top_name} ({top_count}, {share}%).")
        if trunc_note:
            lines.append(trunc_note)
        else:
            lines.append(f"{len(display_choices)} unique values shown (of this field).")
        lines.append("| Value | Count | Share % |")
        lines.append("| --- | ---: | ---: |")
        for name, count in display_choices:
            row_share = round(100.0 * int(count) / total, 1) if total else 0.0
            lines.append(f"| {name} | {int(count)} | {row_share} |")
        # Chart for denser mixes (skip 1-row noise). One chart only — no second table dump.
        if len(display_choices) >= 2:
            lines.extend(
                _chart_lines(
                    "bar",
                    field_label,
                    [name for name, _ in display_choices[:12]],
                    [int(count) for _, count in display_choices[:12]],
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


def _facet_example_phrase(
    graph_id: str,
    attribute: str,
    *,
    generated=None,
    limit: int = 5,
) -> str:
    """Short 'including X, Y, Z' clause from kg_stats top values."""

    try:
        from knowledge_graph.retrieval.kg_stats import top_facet_examples

        examples = top_facet_examples(
            graph_id, attribute, generated=generated, limit=limit
        )
    except Exception:
        return ""
    if not examples:
        return ""
    names = [name for name, _n in examples if name]
    if not names:
        return ""
    if len(names) == 1:
        return f"including {names[0]}"
    if len(names) == 2:
        return f"including {names[0]} and {names[1]}"
    return "including " + ", ".join(names[:-1]) + f", and {names[-1]}"


def _graph_label(platform, graph_id: str) -> str:
    try:
        from fashion_brain.agents.answer_card import SOURCE_LABELS

        if graph_id in SOURCE_LABELS:
            return str(SOURCE_LABELS[graph_id])
    except Exception:
        pass
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
    low = text.casefold()
    # Always surface the catalog as PLMex — never bare "PLM".
    if "plmex" in low or re.search(r"\bplm\b", low) or low in {"plm records", "plm data"}:
        if "ecom" not in low and "commerce" not in low:
            return "PLMex"
    text = re.sub(r"\bknowledge\s*graphs?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:neo4j|postgres|kg)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip(" -–—|/\\")
    return text or "our products"


# Tiny universal fixes only — catalog surfaces live in kg_stats display_values / Neo4j name.
_FACET_DISPLAY_CANON: dict[str, str] = {
    "instock": "In stock",
    "outofstock": "Out of stock",
    "comingsoon": "Coming soon",
    "lowonstock": "Low on stock",
    "mixed": "Mixed",
    "coordsets": "Co-ord sets",
    "coord": "Co-ord",
    "canda": "C&A",
    "handm": "H&M",
}

_PRICE_BAND_TIERS: dict[str, str] = {
    "budget": "Budget",
    "value": "Value",
    "mid": "Mid",
    "premium": "Premium",
}

_CURRENCY_DISPLAY: dict[str, str] = {
    "eur": "EUR",
    "euro": "EUR",
    "gbp": "GBP",
    "usd": "USD",
    "sek": "SEK",
    "dkk": "DKK",
    "nok": "NOK",
    "pln": "PLN",
}


def _price_band_display_label(value: str) -> str | None:
    """Turn internal ``eur:value`` codes into end-user ``EUR · Value`` labels."""

    raw = str(value or "").strip()
    if not raw:
        return None
    match = re.fullmatch(
        r"([A-Za-z]{2,4})\s*[:_\-]\s*(budget|value|mid|premium)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    cur = _CURRENCY_DISPLAY.get(match.group(1).casefold(), match.group(1).upper())
    tier = _PRICE_BAND_TIERS.get(match.group(2).casefold(), match.group(2).capitalize())
    return f"{cur} · {tier}"


def _merge_display_facet_rows(
    rows: list[tuple[str, int]],
    *,
    combine: str = "max",
) -> list[tuple[str, int]]:
    """Collapse rows that only differ by raw spelling once displayed (Women×2, In stock×2).

    ``combine``:
      - ``max`` — overlapping near-duplicate facet nodes (Women×2); default
      - ``sum`` — disjoint aliases of the same concept (CN + China product sets)
    """

    use_sum = str(combine or "max").casefold() == "sum"
    merged: dict[str, list] = {}
    order: list[str] = []
    for name, count in rows:
        label = str(name or "").strip() or "—"
        key = re.sub(r"[^a-z0-9]+", "", label.casefold())
        if not key:
            key = label.casefold()
        n = int(count or 0)
        if key not in merged:
            merged[key] = [label, n]
            order.append(key)
            continue
        if use_sum:
            merged[key][1] = int(merged[key][1]) + n
        elif n > int(merged[key][1]):
            merged[key][1] = n
        # Prefer the longer / better-cased surface when merging.
        if len(label) > len(str(merged[key][0])) or (
            any(ch.isupper() for ch in label)
            and not any(ch.isupper() for ch in str(merged[key][0]))
        ):
            merged[key][0] = label
    out = [(merged[k][0], int(merged[k][1])) for k in order]
    out.sort(key=lambda item: (-item[1], item[0].casefold()))
    return out


def _title_case_facet(text: str) -> str:
    """Title-case a facet phrase; keep small words lower mid-phrase; preserve & / - segments."""

    small = {"and", "or", "of", "the", "a", "an", "for", "to", "in", "on"}

    def _one_word(part: str, *, first: bool) -> str:
        if not part:
            return part
        low = part.casefold()
        if low in {"t-shirt", "tshirt"}:
            return "T-Shirt"
        if "'" in part:
            head, _, tail = part.partition("'")
            return head.capitalize() + "'" + tail
        if not first and low in small:
            return low
        if len(part) <= 3 and part.isupper():
            return part
        return part.capitalize()

    out_words: list[str] = []
    for i, word in enumerate(text.split()):
        if "/" in word:
            segs = [_one_word(seg, first=(i == 0 and j == 0)) for j, seg in enumerate(word.split("/"))]
            out_words.append("/".join(segs))
        elif "-" in word and not word.startswith("-"):
            segs = [_one_word(seg, first=(i == 0 and j == 0)) for j, seg in enumerate(word.split("-"))]
            out_words.append("-".join(segs))
        else:
            out_words.append(_one_word(word, first=i == 0))
    return " ".join(out_words)


def _neo4j_facet_display(
    platform,
    graph_id: str,
    value: str,
) -> str | None:
    """Restore real ``name`` from Neo4j via ``normalized_name`` match key."""

    session_factory = _session(platform) if platform is not None else None
    if not session_factory or not graph_id:
        return None
    from knowledge_graph.ontology import normalize_name

    nn = normalize_name(value)
    if not nn:
        return None
    try:
        with session_factory() as session:
            row = session.run(
                "MATCH (e:Entity {graph: $g}) "
                "WHERE coalesce(e.lifecycle,'active') <> 'deleted' "
                "AND coalesce(e.normalized_name, '') = $nn "
                "AND coalesce(e.name, '') <> '' "
                "RETURN e.name AS name "
                "LIMIT 1",
                {"g": str(graph_id), "nn": nn},
            ).single()
        if row and row.get("name"):
            return str(row["name"]).strip() or None
    except Exception:
        return None
    return None


def _display_label(
    value: str,
    *,
    graph_id: str | None = None,
    attribute: str | None = None,
    generated=None,
    platform=None,
) -> str:
    """User-facing label: kg_stats display_values → Neo4j name → light title-case.

    Matching stays on normalized keys; answers never invent catalog-specific maps.
    """

    raw = str(value or "").strip()
    if not raw:
        return raw
    attr = str(attribute or "").strip().casefold().replace(" ", "_")
    # Country of origin: CN / China / china → China (and multi-origin order-normalize).
    if attr in {
        "country",
        "country_of_origin",
        "origin",
        "origin_country",
        "made_in",
    } or (
        len(raw) <= 32
        and (
            re.fullmatch(r"[A-Za-z]{2}", raw)
            or re.search(r"(?i)\b(china|bangladesh|turkey|cambodia|morocco|pakistan|vietnam|india)\b", raw)
        )
    ):
        try:
            from knowledge_graph.geo_country import country_display_label

            country = country_display_label(raw)
            if country:
                return country
        except Exception:
            pass
    # Price-band ETL codes (eur:value) — never show raw colon form to end users.
    if attr in {"price_band", "priceband"} or re.search(
        r"(?i)^[a-z]{2,4}[:_\-](budget|value|mid|premium)$", raw
    ):
        band = _price_band_display_label(raw)
        if band:
            return band
    compact = re.sub(r"[^a-z0-9]+", "", raw.casefold())
    if compact in _FACET_DISPLAY_CANON:
        return _FACET_DISPLAY_CANON[compact]

    # 1) Meta shard mapping (normalized → real) written at stats build time.
    try:
        from knowledge_graph.retrieval.kg_stats import resolve_facet_display

        mapped = resolve_facet_display(
            raw,
            graph_id=graph_id,
            attribute=attribute,
            generated=generated,
        )
        if mapped:
            return mapped
    except Exception:
        pass

    # 2) Neo4j entity.name restored via normalized_name.
    if platform is not None and graph_id:
        neo = _neo4j_facet_display(platform, str(graph_id), raw)
        if neo and (any(ch.isupper() for ch in neo) or any(ch in neo for ch in "&/-")):
            return neo
        if neo and neo != raw:
            # Even if Neo also stored scrubbed text, prefer its surface then title-case.
            raw = neo

    # Already well-formed — keep punctuation, light title-case.
    has_mark = any(ch in raw for ch in "&/-")
    has_upper = any(ch.isupper() for ch in raw)
    if has_mark and has_upper and raw != raw.casefold():
        return _title_case_facet(re.sub(r"\s+", " ", raw))

    folded = re.sub(r"\s+", " ", raw.casefold().replace("_", " "))
    folded = re.sub(r"\b(women|men)\s+s\b", r"\1's", folded, flags=re.IGNORECASE)
    return _title_case_facet(folded)


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
    # Ecom garment category. PLM production craft is MADE_WITH_METHOD (false friend for "category").
    "category": ("BELONGS_TO_CATEGORY", "MADE_WITH_METHOD"),
    "production_method": ("MADE_WITH_METHOD",),
    "family": ("HAS_FAMILY",),
    "subfamily": ("HAS_SUBFAMILY", "HAS_SUB_CATEGORY"),
    "color": ("HAS_COLOR", "PRIMARY_COLOR"),
    "material": ("USES_MATERIAL", "PRIMARY_MATERIAL"),
    "availability": ("HAS_AVAILABILITY",),
    "brand": ("HAS_BRAND",),
    "buyer": ("HAS_BUYER",),
    "garment_type": ("HAS_GARMENT_TYPE",),
    # PLM HAS_GENDER + shared TargetGroup hub (ecom + plmex).
    "gender": ("HAS_GENDER", "FOR_TARGET_GROUP"),
    "target_group": ("FOR_TARGET_GROUP", "HAS_GENDER", "HAS_SECTION"),
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
    "construction": ("HAS_CONSTRUCTION", "HAS_CONSTRUCTION_TAG"),
    "section": ("HAS_SECTION", "FOR_TARGET_GROUP"),
    "occasion": ("SUITED_FOR_OCCASION", "HAS_EVENT_TAG"),
    "pattern": ("HAS_PATTERN",),
    "price_band": ("IN_PRICE_BAND",),
    "reference": ("HAS_REFERENCE", "HAS_STYLE_REFERENCE", "HAS_CUSTOMER_STYLE_REFERENCE"),
    "style_reference": ("HAS_REFERENCE", "HAS_STYLE_REFERENCE"),
    "customer_style_reference": ("HAS_REFERENCE", "HAS_CUSTOMER_STYLE_REFERENCE"),
    # New unified Occasion/Construction + legacy AI tag edges (backward compatible).
    "ai_event_tags": ("SUITED_FOR_OCCASION", "HAS_EVENT_TAG"),
    "ai_construction_tags": ("HAS_CONSTRUCTION", "HAS_CONSTRUCTION_TAG"),
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
            or key in {"nm", "tex", "denier"}
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
    *,
    graph_id: str | None = None,
    platform=None,
    generated=None,
    user_set_limit: bool = False,
) -> list[str]:
    """Breakdown with one best-fit visualization (chart XOR table)."""

    all_usable = _usable_ranked(groups, str(field))
    if not all_usable:
        return [f"No {field.replace('_', ' ')} breakdown."]
    # Hard cap when an N is provided (user top-N or caller preview cap); else soft.
    show_n, trunc_note = _facet_row_budget(
        len(all_usable),
        limit=limit,
        user_set_limit=bool(user_set_limit or limit is not None),
    )
    shown = all_usable[:show_n]
    field_label = field.replace("_", " ")

    def _label(name: str) -> str:
        if normalize_name(str(field)) in {"material", "materials", "composition"}:
            return _material_display_label(name)
        return _display_label(
            name,
            graph_id=graph_id,
            attribute=str(field),
            generated=generated,
            platform=platform,
        )

    display_shown = [(_label(name), count) for name, count in shown]
    # Skip trivial single-value 100% slices (e.g. By brand after filtering to Primark).
    if len(display_shown) == 1 and total and int(display_shown[0][1]) == int(total):
        return []
    top_name, top_count = display_shown[0]
    share = round(100.0 * top_count / total, 1) if total else 0.0
    lines = [
        f"{field_label} breakdown (highest count first):",
        f"Leading {field_label}: {top_name} ({top_count} of {total}, {share}%).",
    ]
    if trunc_note:
        lines.append(trunc_note)
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
    # Leading space so callers can write ``products{phrase}`` without "productswith".
    return " with " + " and ".join(
        f"{key} {_display_label(str(value))}" for key, value in filters.items()
    )


_TABLE_COLUMNS: dict[str, tuple[str, tuple[str, ...]]] = {
    # key: (header, property keys to try in order)
    "brand": ("Brand", ("brand", "buyer")),
    "name": ("Name", ("name", "title")),
    "price": ("Price", ("price", "retail_price")),
    "availability": ("Availability", ("availability", "stock_status")),
    "category": ("Category", ("category", "product_type", "category_path", "garment_type")),
    "color": ("Color", ("color", "colour")),
    "material": ("Material", ("material", "fabric", "composition")),
    "season": ("Season", ("season", "season_name", "season_year")),
    "size": ("Size", ("size", "sizes")),
}
_DEFAULT_COLUMNS = ["brand", "name", "price", "availability"]
_QUESTION_HINTS: dict[str, tuple[str, ...]] = {
    "category": ("category", "categories", "type", "garment", "sweater", "shirt", "dress"),
    "color": ("color", "colour", "colors", "colours"),
    "material": ("material", "fabric", "cotton", "linen", "wool", "composition"),
    "season": ("season", "ss", "aw", "summer", "winter"),
    "size": ("size", "sizes"),
}
# Filter keys that pin the same display column (drop redundant column).
_PINNED_COLUMN_ALIASES: dict[str, str] = {
    "buyer": "brand",
    "colour": "color",
    "colors": "color",
    "colours": "color",
    "composition": "material",
    "fabric": "material",
    "garment": "category",
    "garment_type": "category",
    "subfamily": "category",
    "product_type": "category",
}


def _table_cell(node, keys: tuple[str, ...], *, column: str = "") -> str:
    """Read the first usable property; polish brand/availability labels."""

    props = getattr(node, "properties", None) or {}
    for key in keys:
        raw = props.get(key)
        if raw in (None, "", [], "(empty)", "(unspecified)"):
            continue
        text = str(raw).replace("|", "/").strip()
        if not text or text.casefold() in {"null", "none", "—", "-", "n/a"}:
            continue
        if column == "price" and re.search(r"(?i)(?:£|€|\$)\s*(?:19|20)\d{2}\b", text):
            continue
        if column in {"brand", "availability"}:
            return _display_label(text)
        if column == "name" and (re.fullmatch(r"\d{6,}", text) or re.fullmatch(r"[A-Z]{0,3}\d{5,}", text)):
            continue
        return text
    return ""


def _product_table_lines(
    sample,
    filters: dict,
    question: str,
    *,
    limit: int,
    force_price: bool = False,
) -> list[str]:
    """Header + rows with columns chosen per answer. Empty list → caller falls back."""

    if not sample:
        return []
    q = (question or "").casefold()
    pinned: set[str] = set()
    for key, value in (filters or {}).items():
        if value in (None, ""):
            continue
        # Multi-party compare ("Primark | Zara") is not a single pinned brand.
        if "|" in str(value):
            continue
        folded = str(key).casefold().replace(" ", "_").replace("-", "_")
        col = _PINNED_COLUMN_ALIASES.get(folded, folded)
        if col in _TABLE_COLUMNS:
            pinned.add(col)

    cols = [c for c in _DEFAULT_COLUMNS if c not in pinned]
    for key, hints in _QUESTION_HINTS.items():
        if key not in pinned and key not in cols and any(h in q for h in hints):
            cols.append(key)

    cap = max(1, min(int(limit or 25), 25))
    usable_nodes = []
    for node in sample or []:
        name = _table_cell(node, _TABLE_COLUMNS["name"][1], column="name")
        if not name:
            continue
        usable_nodes.append(node)
        if len(usable_nodes) >= cap:
            break
    if not usable_nodes:
        return []

    data = {
        c: [_table_cell(n, _TABLE_COLUMNS[c][1], column=c) for n in usable_nodes]
        for c in cols
        if c in _TABLE_COLUMNS
    }
    # Drop columns with no data at all (keep name; keep price when ranking by price).
    cols = [
        c
        for c in cols
        if c in data and (c == "name" or any(data[c]) or (c == "price" and force_price))
    ]
    if "name" not in cols:
        cols.insert(0, "name")
        data["name"] = [_table_cell(n, _TABLE_COLUMNS["name"][1], column="name") for n in usable_nodes]
    if len(cols) < 2:
        return []

    out = [
        "| " + " | ".join(_TABLE_COLUMNS[c][0] for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for i in range(len(usable_nodes)):
        out.append("| " + " | ".join((data[c][i] or "—") for c in cols) + " |")
    return out


def _example_product_table_rows(nodes: list, *, limit: int = 25) -> list[str]:
    """Back-compat body rows for Brand|Name|Price|Availability (tests / older callers)."""

    table = _product_table_lines(nodes, {}, "", limit=limit)
    # Drop header + separator; keep body only when defaults survived.
    return table[2:] if len(table) > 2 else []


def _attach_optional_facet_metadata(
    platform,
    graph_id: str,
    filters: dict,
    *,
    total: int,
    evidence: list,
    question: str = "",
) -> None:
    """Attach optional facet facts as evidence — never as forced answer text.

    Prefer filter-wise ``kg_stats/<kg>/<filter>.json`` shards (selected dims only).
    Fall back to live Cypher group_by only when shards are missing.
    Findings stay the count; ResponseAgent uses findings + these facts together.
    """

    if total <= 0:
        return
    if any(str(getattr(item, "record_id", "") or "") == "optional_metadata" for item in evidence):
        return
    # Party / empty / single catalog facet — attach same-dimension + totals context.
    # Multi-filter asks already carry their own focus; skip to avoid noise.
    keys = {str(k).casefold() for k in (filters or {})}
    if len(keys) >= 2:
        return

    generated = getattr(getattr(platform, "settings", None), "generated", None)
    facts: list[str] = []
    source_path = f"kg_stats/{graph_id}"
    try:
        from knowledge_graph.retrieval.kg_stats import build_meta_reference_lines

        facts = build_meta_reference_lines(
            graph_id,
            filters=dict(filters or {}),
            question=question or "",
            generated=generated,
            max_filters=4,
            top_n=3,
        )
        # Drop the global products_total line when we already stated ``total``.
        facts = [line for line in facts if not str(line).startswith("products_total=")]
        if facts:
            source_path = f"kg_stats/{graph_id}/filters"
    except Exception:
        facts = []

    if not facts:
        dims = [dim for dim in ("section", "category", "family", "garment_type") if dim not in keys]
        for dim in dims:
            try:
                groups = cypher_group_by(platform, graph_id, dim, dict(filters or {}))
            except Exception:
                groups = None
            if not groups:
                continue
            usable = _usable_ranked(groups, dim, limit=3)
            if not usable:
                continue
            top_name, top_count = usable[0]
            top_label = _display_label(str(top_name))
            share = round(100.0 * int(top_count) / int(total), 1) if total else 0.0
            line = f"{dim.replace('_', ' ')}: {top_label} ({int(top_count):,}; {share}%)"
            if dim == "category" and len(usable) > 1:
                nxt = ", ".join(
                    f"{_display_label(str(n))} ({int(c):,})" for n, c in usable[1:3]
                )
                line = f"{line}; next: {nxt}"
            facts.append(line)
            if len(facts) >= 3:
                break
        source_path = f"neo4j/{graph_id}/facet_context"

    if not facts:
        return

    evidence.append(
        EvidenceItem(
            source_id=graph_id,
            record_id="optional_metadata",
            title="optional facet context",
            excerpt="; ".join(facts),
            source_path=source_path,
            verification_status="observed",
            graph=graph_id,
            node_id=None,
            synthetic=False,
            requested_filters=dict(filters or {}),
            applied_filters=dict(filters or {}),
        )
    )


def _excerpt(node) -> str:
    props = node.properties or {}
    bits = [props.get("brand"), props.get("name"), props.get("category"), props.get("price"), props.get("currency")]
    return " | ".join(str(bit) for bit in bits if bit)
