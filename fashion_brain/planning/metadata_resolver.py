"""fashion_brain/planning/metadata_resolver.py

Validate a plan against LIVE METADATA across EVERY product KG (ecom, plmex, …),
repair wrong field/value picks, and when understanding is weak ask a clarification
that restates the closest interpretations as full questions.
No fixed vocabulary: everything is scored against the multi-graph metadata index.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from knowledge_graph.ontology import normalize_name

LOGGER = logging.getLogger(__name__)

MATCH_MIN = 0.78          # below this a phrase is "unknown"
CONFIDENT = 0.90          # below this (but >= MATCH_MIN) we confirm: "did you mean X?"
AMBIGUOUS_MARGIN = 0.06   # same phrase hits two fields this close -> ask
MAX_NGRAM = 3
_INDEX_TTL = 600.0
_INDEX_CACHE: dict[tuple, tuple[float, "MetaIndex"]] = {}

# Always resolve these categorical facets against live metadata when present on a plan.
# Includes PLM hierarchy (main → sub → sub-sub) + family/subfamily and other PLM facets.
CORE_METADATA_FIELDS = (
    "brand",
    "buyer",
    "garment",
    "garment_type",
    "category",
    "subcategory",           # PLM Sub Category
    "plm_main_category",     # PLM Main Category
    "plm_sub_category",      # alias used on some ecom/runtime paths
    "plm_sub_sub_category",  # PLM Sub Sub Category
    "family",
    "subfamily",
    "section",
    "color",
    "colour",
    "material",
    "composition",
    "gender",
    "season",
    "availability",
    "country",
    "department",
    "designer",
    "merchandiser",
    "gauge",
    "sample_collection_type",
    "knit_machine_type",
    "shopping_department",
)

# Never accept an LLM/keyword value for these unless it exists in the live index.
STRICT_METADATA_FIELDS = frozenset(
    {
        "brand",
        "buyer",
        "garment",
        "garment_type",
        "category",
        "subcategory",
        "plm_main_category",
        "plm_sub_category",
        "plm_sub_sub_category",
        "family",
        "subfamily",
        "section",
        "color",
        "colour",
        "material",
        "gender",
        "season",
        "availability",
        "country",
        "department",
        "designer",
        "merchandiser",
        "gauge",
        "sample_collection_type",
        "knit_machine_type",
        "shopping_department",
    }
)

# Free / numeric plan keys — not catalog enum values.
FREEFORM_FILTER_KEYS = frozenset(
    {
        "price_min",
        "price_max",
        "limit",
        "parent_of",
        "product_id",
        "sku",
        "time_range",
    }
)

# Important columns every product KG is cross-checked for (presence, not values).
# Graphs may legitimately lack some (e.g. ecom often has brand, plmex often has buyer).
IMPORTANT_COLUMNS = CORE_METADATA_FIELDS


@dataclass
class MetaIndex:
    """Multi-KG metadata index: merged values + per-graph column/value coverage."""

    values: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    by_graph: dict[str, dict[str, list[tuple[str, str]]]] = field(default_factory=dict)
    columns: dict[str, list[str]] = field(default_factory=dict)
    # field -> graph_ids that expose the column (schema/mapping/live attrs)
    field_graphs: dict[str, list[str]] = field(default_factory=dict)
    # graph_id -> important columns missing from that graph's attribute set
    missing_important: dict[str, list[str]] = field(default_factory=dict)
    product_graphs: list[str] = field(default_factory=list)


def _product_graph_ids(platform) -> list[str]:
    """Active product KGs only (ecom, plmex, own_products, …) — not system/shared."""
    try:
        from knowledge_graph.retrieval.ecom import registry_product_graphs

        ids = [str(g) for g in (registry_product_graphs(platform) or []) if str(g).strip()]
        if ids:
            return list(dict.fromkeys(ids))
    except Exception:
        pass
    try:
        return [
            str(item.id)
            for item in platform.kg_registry.list_graphs()
            if str(getattr(item, "kind", "") or "").casefold() == "product"
        ]
    except Exception:
        return []


def _graph_attribute_keys(platform, graph_id: str) -> list[str]:
    """Distinct filterable columns for one KG (mapping + ontology + sampled Product props)."""
    keys: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        key = str(name or "").strip().lower().replace(" ", "_")
        if not key or key in seen or key in {"id", "name", "labels", "graph"}:
            return
        seen.add(key)
        keys.append(key)

    try:
        from knowledge_graph.retrieval.metadata import metadata_overview

        row = metadata_overview(platform, graph_id=graph_id)
        if isinstance(row, dict):
            schema = row.get("schema") or {}
            product = schema.get("Product") if isinstance(schema, dict) else None
            if isinstance(product, dict):
                for attr in (product.get("attributes") or {}):
                    add(attr)
            for entity in row.get("entities") or []:
                mapped = {
                    "Brand": "brand",
                    "Material": "material",
                    "GarmentCategory": "category",
                    "Color": "color",
                    "Colour": "colour",
                    "Season": "season",
                    "Supplier": "supplier",
                    "Company": "buyer",
                    "Buyer": "buyer",
                    "Availability": "availability",
                    "Section": "section",
                    "Currency": "currency",
                    "Gender": "gender",
                    "Department": "department",
                    "Family": "family",
                    "Subfamily": "subfamily",
                    "PlmMainCategory": "plm_main_category",
                    "PlmSubCategory": "plm_sub_category",
                    "PlmSubSubCategory": "plm_sub_sub_category",
                    "Designer": "designer",
                    "Merchandiser": "merchandiser",
                }.get(str(entity))
                if mapped:
                    add(mapped)
    except Exception:
        LOGGER.debug("metadata_overview columns failed for %s", graph_id, exc_info=True)

    # Live Product property keys as a safety net when overview is thin.
    try:
        nodes = platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False)
        for node in (nodes or [])[:80]:
            for key in (getattr(node, "properties", None) or {}):
                add(key)
    except Exception:
        pass
    return keys


def cross_check_important_columns(
    platform,
    *,
    graphs: list[str] | None = None,
    important: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Cross-check important columns across every product KG (ecom + plmex + …).

    Returns::
        {
          "product_graphs": [...],
          "columns": {graph_id: [attrs…]},
          "field_graphs": {field: [graph_ids that have it]},
          "missing_important": {graph_id: [core fields absent on that KG]},
          "present_on_any": [fields found on ≥1 KG],
          "missing_everywhere": [important fields on no product KG],
        }
    """
    gids = list(graphs or _product_graph_ids(platform))
    want = tuple(important or IMPORTANT_COLUMNS)
    columns: dict[str, list[str]] = {}
    field_graphs: dict[str, list[str]] = {f: [] for f in want}
    missing_important: dict[str, list[str]] = {}

    for gid in gids:
        attrs = _graph_attribute_keys(platform, gid)
        columns[gid] = attrs
        attr_set = set(attrs)
        # colour/color alias
        if "color" in attr_set:
            attr_set.add("colour")
        if "colour" in attr_set:
            attr_set.add("color")
        miss = [f for f in want if f not in attr_set]
        if miss:
            missing_important[gid] = miss
        for f in want:
            if f in attr_set:
                field_graphs.setdefault(f, []).append(gid)

    present_on_any = [f for f in want if field_graphs.get(f)]
    missing_everywhere = [f for f in want if not field_graphs.get(f)]
    return {
        "product_graphs": gids,
        "columns": columns,
        "field_graphs": {k: v for k, v in field_graphs.items() if v},
        "missing_important": missing_important,
        "present_on_any": present_on_any,
        "missing_everywhere": missing_everywhere,
    }


# ---------------------------------------------------------------- value index
def build_meta_index(platform, fields: list[str], per_field: int = 200) -> MetaIndex:
    """Build a multi-KG value + column index (ecom, plmex, every product graph).

    Values are discovered per graph then merged. Important columns are cross-checked
    so the resolver knows which facets exist on which KG.
    """
    merged = list(dict.fromkeys([*(fields or []), *CORE_METADATA_FIELDS]))
    key = (id(platform), "meta", tuple(sorted(merged)))
    hit = _INDEX_CACHE.get(key)
    if hit and time.time() - hit[0] < _INDEX_TTL:
        return hit[1]

    from knowledge_graph.retrieval.metadata import discover_values

    limit = max(1, min(int(per_field or 200), 200))
    gids = _product_graph_ids(platform)
    coverage = cross_check_important_columns(platform, graphs=gids)

    by_graph: dict[str, dict[str, list[tuple[str, str]]]] = {}
    flat: dict[str, list[tuple[str, str]]] = {}
    flat_seen: dict[str, set[str]] = {}

    # Probe fields that exist on at least one graph, plus any caller-requested fields.
    probe_fields = list(
        dict.fromkeys(
            [
                *merged,
                *[f for f, gs in (coverage.get("field_graphs") or {}).items() if gs],
            ]
        )
    )

    for gid in gids or [None]:
        graph_attrs = set((coverage.get("columns") or {}).get(gid or "", []) or [])
        if "color" in graph_attrs:
            graph_attrs.add("colour")
        if "colour" in graph_attrs:
            graph_attrs.add("color")
        # If we don't know columns yet, still try CORE + requested fields.
        fields_for_graph = (
            [f for f in probe_fields if f in graph_attrs or f in FREEFORM_FILTER_KEYS]
            if graph_attrs
            else list(probe_fields)
        )
        if not fields_for_graph:
            fields_for_graph = list(probe_fields)
        gkey = str(gid) if gid else "_all"
        by_graph.setdefault(gkey, {})
        for f in fields_for_graph:
            try:
                payload = discover_values(
                    platform,
                    graph_id=gid,
                    attribute=f,
                    q="",
                    limit=limit,
                )
            except Exception:
                continue
            vals = [str(v).strip() for v in (payload.get("values") or []) if str(v).strip()]
            if not vals:
                continue
            pairs = [(v, normalize_name(v)) for v in vals]
            by_graph[gkey][f] = pairs
            bucket = flat.setdefault(f, [])
            seen = flat_seen.setdefault(f, set())
            for raw, norm in pairs:
                if norm in seen:
                    continue
                seen.add(norm)
                bucket.append((raw, norm))

    # Fallback: if no product graphs registered, keep legacy platform-wide discover.
    if not gids and not flat:
        for f in merged:
            try:
                payload = discover_values(platform, attribute=f, q="", limit=limit)
            except Exception:
                continue
            vals = [str(v).strip() for v in (payload.get("values") or []) if str(v).strip()]
            if vals:
                flat[f] = [(v, normalize_name(v)) for v in vals]

    meta = MetaIndex(
        values=flat,
        by_graph=by_graph,
        columns=dict(coverage.get("columns") or {}),
        field_graphs=dict(coverage.get("field_graphs") or {}),
        missing_important=dict(coverage.get("missing_important") or {}),
        product_graphs=list(coverage.get("product_graphs") or gids),
    )
    _INDEX_CACHE[key] = (time.time(), meta)
    if meta.missing_important:
        LOGGER.info(
            "metadata column cross-check missing_important=%s field_graphs=%s",
            meta.missing_important,
            {k: v for k, v in meta.field_graphs.items()},
        )
    return meta


def build_value_index(platform, fields: list[str], per_field: int = 200) -> dict[str, list[tuple[str, str]]]:
    """{field: [(raw_value, normalized)]} merged from EVERY product KG.

    Prefer ``build_meta_index`` when column coverage / per-graph provenance is needed.
    """
    return build_meta_index(platform, fields, per_field=per_field).values


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 4 and a in b.split():     # whole-word containment: weaker than equality
        return 0.85
    return SequenceMatcher(None, a, b).ratio()


# When one token hits several fields, prefer party/color over nav category.
# Stops "black Zara" becoming category=Black (name CONTAINS trap).
_FIELD_MOVE_PRIORITY = {
    "brand": 0,
    "buyer": 1,
    "color": 2,
    "colour": 2,
    "material": 3,
    "availability": 4,
    "section": 5,
    "gender": 6,
    "garment": 7,
    "garment_type": 7,
    "family": 8,
    "subfamily": 8,
    "category": 9,
    "plm_main_category": 9,
    "plm_sub_category": 9,
    "plm_sub_sub_category": 9,
}

# Basic colour tokens that must never be treated as a product category.
_BASIC_COLOR_TOKENS = frozenset(
    {
        "red",
        "blue",
        "green",
        "black",
        "white",
        "yellow",
        "pink",
        "purple",
        "grey",
        "gray",
        "brown",
        "orange",
        "beige",
        "navy",
        "burgundy",
        "cream",
        "ivory",
        "khaki",
        "olive",
        "maroon",
        "teal",
        "gold",
        "silver",
        "multicolor",
        "multicolour",
    }
)

_TYPE_LIKE_KEYS = frozenset(
    {
        "category",
        "garment",
        "garment_type",
        "family",
        "subfamily",
        "plm_main_category",
        "plm_sub_category",
        "plm_sub_sub_category",
    }
)


def _looks_like_basic_color(value: str) -> bool:
    norm = normalize_name(str(value or ""))
    if not norm or " " in norm:
        return False
    return norm in _BASIC_COLOR_TOKENS


def _pick_move_target(moved: list[tuple[str, str]]) -> tuple[str, str]:
    return min(
        moved,
        key=lambda item: (_FIELD_MOVE_PRIORITY.get(item[0], 50), item[0], item[1]),
    )


def demote_color_words_from_category(filters: dict, index: dict | None = None) -> dict:
    """Rewrite category/garment=Black → color=black when the token is a colour."""

    out = {
        str(k): str(v)
        for k, v in dict(filters or {}).items()
        if k not in (None, "") and v not in (None, "")
    }
    color_index = []
    if index:
        color_index = list(index.get("color") or []) + list(index.get("colour") or [])
    for key in list(out):
        if key not in _TYPE_LIKE_KEYS:
            continue
        raw = out.get(key)
        if not raw:
            continue
        norm = normalize_name(raw)
        color_hit = next((r for r, n in color_index if n == norm), None)
        if color_hit is None and not _looks_like_basic_color(raw):
            continue
        if color_hit is None and _looks_like_basic_color(raw):
            color_hit = str(raw).strip()
        # Single-token colour masquerading as a type/nav facet.
        if " " in norm and not _looks_like_basic_color(raw):
            continue
        out.setdefault("color", color_hit)
        out.pop(key, None)
    # Same token on both color and category → keep color only.
    color_norm = normalize_name(str(out.get("color") or ""))
    if color_norm:
        for key in list(out):
            if key in _TYPE_LIKE_KEYS and normalize_name(str(out.get(key) or "")) == color_norm:
                out.pop(key, None)
    return out


# ---------------------------------------------------------------- repair
def repair_filters(filters: dict, index: dict) -> tuple[dict, list[str]]:
    """Fix wrong-field picks using metadata. Returns (repaired, unresolved_values).

    Strict categorical fields (brand/buyer/garment/…) never keep a value that is
    absent from the live index. Freeform keys (price_max, limit, …) pass through.
    """
    out: dict[str, str] = {}
    unresolved: list[str] = []
    for key, raw in (filters or {}).items():
        key_s = str(key or "").strip()
        if not key_s or raw in (None, ""):
            continue
        if key_s in FREEFORM_FILTER_KEYS:
            out[key_s] = str(raw).strip()
            continue
        parts = [p.strip() for p in str(raw).split("|") if p.strip()]
        fixed: list[str] = []
        for part in parts:
            norm = normalize_name(part)
            # Known colour tokens on type-like keys → color (never keep category=Black).
            if key_s in _TYPE_LIKE_KEYS and _looks_like_basic_color(part):
                color_vals = list(index.get("color") or []) + list(index.get("colour") or [])
                color_hit = next((r for r, n in color_vals if n == norm), None) or part
                out.setdefault("color", color_hit)
                continue
            vals = index.get(key_s) or []
            exact = next((r for r, n in vals if n == norm), None)
            if exact:
                # color=black exact hit — never also keep a competing category=Black.
                if key_s in {"color", "colour"}:
                    fixed.append(exact)
                    continue
                # category=Black exact when color index also has black → prefer color.
                if key_s in _TYPE_LIKE_KEYS:
                    color_vals = list(index.get("color") or []) + list(index.get("colour") or [])
                    color_hit = next((r for r, n in color_vals if n == norm), None)
                    if color_hit is not None and (
                        _looks_like_basic_color(part) or key_s == "category"
                    ):
                        out.setdefault("color", color_hit)
                        continue
                fixed.append(exact)
                continue
            # same value under ANOTHER field (category=Zara -> brand=Zara)
            moved = [(f, r) for f, vs in index.items() if f != key_s for r, n in vs if n == norm]
            if moved:
                # Never move color → category when both exist.
                if key_s in {"color", "colour"}:
                    color_only = [(f, r) for f, r in moved if f in {"color", "colour"}]
                    if not color_only and exact is None:
                        # Keep original color spelling when index missed it.
                        fixed.append(part)
                        continue
                target_f, target_r = _pick_move_target(moved)
                out.setdefault(target_f, target_r)
                continue
            # near-miss spelling in the same field
            best = max(((_sim(norm, n), r) for r, n in vals), default=(0.0, ""))
            if best[0] >= 0.85:
                fixed.append(best[1])
                continue
            # Strict facets: never invent. Other fields with empty index may be free-text.
            if key_s in STRICT_METADATA_FIELDS or vals:
                unresolved.append(f"{key_s}={part}")
            else:
                fixed.append(part)
        if fixed:
            out[key_s] = " | ".join(dict.fromkeys(fixed))
    out = demote_color_words_from_category(out, index)
    return out, unresolved


def ground_filters_against_live_metadata(
    platform,
    filters: dict | None,
    *,
    question: str = "",
    fields: list[str] | None = None,
    graph_id: str | None = None,
) -> dict[str, str]:
    """Drop every categorical filter value that is not in live multi-KG metadata.

    Use after any LLM / keyword extraction so Scope and Cypher never carry invented
    values (brand=mixed, color=ops, …). Freeform keys pass through. When the
    metadata index cannot be built, returns the input unchanged (callers should
    still run grammar sanitize separately).
    """

    raw = {
        str(k).strip(): str(v).strip()
        for k, v in dict(filters or {}).items()
        if k not in (None, "") and v not in (None, "")
    }
    if not raw or platform is None:
        return raw
    # Remap audience/party aliases before probing so ecom gender→section is not dropped.
    try:
        from fashion_brain.planning.semantic_field_map import apply_semantic_field_map

        mapped, _ = apply_semantic_field_map(raw, graph_id)
        raw = {
            str(k).strip(): str(v).strip()
            for k, v in dict(mapped or {}).items()
            if k not in (None, "") and v not in (None, "")
        }
    except Exception:
        pass
    # Preserve already-grounded live brands — never drop brand=zara because the
    # metadata value probe timed out or truncated the brand list.
    preserved_party: dict[str, str] = {}
    try:
        from fashion_brain.ask import (
            live_brand_names,
            live_brands_mentioned,
            match_live_party_value,
        )

        names = live_brand_names(platform, graph_id)
        for key in ("brand", "buyer"):
            if not raw.get(key):
                continue
            hit = match_live_party_value(str(raw[key]), names) if names else None
            if hit:
                preserved_party[key] = hit
                raw[key] = hit
        if question and not preserved_party.get("brand"):
            mentioned = live_brands_mentioned(question, names)
            if len(mentioned) == 1:
                preserved_party["brand"] = mentioned[0]
                raw["brand"] = mentioned[0]
    except Exception:
        preserved_party = {}
    # Prefer local kg_stats value shards (fast) over /api/metadata/values fan-out.
    index: dict[str, list[tuple[str, str]]] = {}
    try:
        from knowledge_graph.retrieval.kg_stats import load_graph_stats

        gids = [graph_id] if graph_id else []
        if not gids:
            gids = _product_graph_ids(platform) or []
        for gid in gids:
            stats = load_graph_stats(str(gid))
            if not stats:
                continue
            vc = dict(stats.get("value_counts") or {})
            for attr, counts in vc.items():
                if not isinstance(counts, dict):
                    continue
                bucket = index.setdefault(str(attr), [])
                seen = {n for _, n in bucket}
                for raw_val in counts:
                    norm = normalize_name(str(raw_val))
                    if not norm or norm in seen:
                        continue
                    seen.add(norm)
                    bucket.append((str(raw_val), norm))
    except Exception:
        LOGGER.debug("kg_stats grounding index unavailable", exc_info=True)
    if not index:
        probe = list(
            dict.fromkeys(
                [
                    *(fields or []),
                    *raw.keys(),
                    *CORE_METADATA_FIELDS,
                ]
            )
        )
        try:
            index = build_value_index(platform, probe)
        except Exception:
            LOGGER.debug("live metadata index unavailable for filter grounding", exc_info=True)
            return {**raw, **preserved_party}
    if not index:
        return {**raw, **preserved_party}
    repaired, unresolved = repair_filters(raw, index)
    repaired = demote_color_words_from_category(repaired, index)
    repaired.update(preserved_party)
    if unresolved:
        leftover = [
            u
            for u in unresolved
            if not any(u.startswith(f"{k}=") for k in preserved_party)
        ]
        if leftover:
            LOGGER.info(
                "dropped filters not in live metadata: %s",
                ", ".join(leftover[:12]),
            )
    # Stock-status mix / vs asks talk ABOUT availability labels — do not pin one side
    # (also undoes repair_filters remapping brand=mixed → availability=Mixed).
    try:
        from fashion_brain.ask import mentions_stock_status_comparison

        if question and mentions_stock_status_comparison(question):
            repaired.pop("availability", None)
    except Exception:
        pass
    return repaired


# ---------------------------------------------------------------- span matching
@dataclass
class Span:
    phrase: str
    start: int
    end: int
    field: str
    value: str
    score: float
    rivals: list[tuple[str, str, float]] = field(default_factory=list)  # other fields within margin


def match_spans(question: str, index: dict) -> list[Span]:
    toks = normalize_name(question).split()
    found: list[Span] = []
    for n in range(MAX_NGRAM, 0, -1):
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i + n])
            if len(phrase) < 3:
                continue
            per_field: list[tuple[float, str, str]] = []
            for f, vals in index.items():
                best = max(((_sim(phrase, nrm), raw) for raw, nrm in vals), default=(0.0, ""))
                if best[0] >= MATCH_MIN:
                    per_field.append((best[0], f, best[1]))
            if not per_field:
                continue
            # Prefer brand/color over category when scores are tied ("black").
            per_field.sort(
                key=lambda item: (
                    -item[0],
                    _FIELD_MOVE_PRIORITY.get(item[1], 50),
                    item[1],
                )
            )
            top = per_field[0]
            rivals = [(f, v, s) for s, f, v in per_field[1:] if top[0] - s <= AMBIGUOUS_MARGIN and v != top[2]]
            found.append(Span(phrase, i, i + n, top[1], top[2], top[0], rivals))
    found.sort(key=lambda s: (-s.score, -(s.end - s.start)))   # best, then longest; keep non-overlapping
    taken: set[int] = set()
    chosen: list[Span] = []
    for s in found:
        rng = set(range(s.start, s.end))
        if rng & taken:
            continue
        taken |= rng
        chosen.append(s)
    return sorted(chosen, key=lambda s: s.start)


# ---------------------------------------------------------------- decision
@dataclass
class Understanding:
    proceed: bool = True
    filters: dict = field(default_factory=dict)
    clarify_text: str = ""
    options: list[dict] = field(default_factory=list)   # validated interpretations
    reasons: list[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)  # multi-KG column cross-check snapshot


def check_understanding(platform, question: str, history: list | None, plan: dict,
                        state: dict | None = None) -> Understanding:
    """Validate plan filters against live metadata from EVERY product KG.

    Proceed only with values that exist in metadata (or freeform keys).
    Unclear / unknown / ambiguous asks → related full-question options for the user.
    Column presence is cross-checked on ecom + plmex (+ any other product graphs).
    """
    fields = list(
        dict.fromkeys(
            [
                *(plan.get("available_attributes") or []),
                *(plan.get("filters") or {}).keys(),
                *CORE_METADATA_FIELDS,
            ]
        )
    )
    meta = build_meta_index(platform, fields)
    index = meta.values
    coverage = {
        "product_graphs": list(meta.product_graphs),
        "field_graphs": dict(meta.field_graphs),
        "missing_important": dict(meta.missing_important),
    }
    # Drop strict categorical filters when we have no index at all for them.
    raw_filters = dict(plan.get("filters") or {})
    if not index:
        kept = {
            k: v
            for k, v in raw_filters.items()
            if str(k) in FREEFORM_FILTER_KEYS
        }
        dropped = [f"{k}={v}" for k, v in raw_filters.items() if str(k) not in FREEFORM_FILTER_KEYS]
        if not dropped:
            return Understanding(True, kept, coverage=coverage)
        return Understanding(
            False,
            kept,
            "I could not verify those values against the catalog metadata. "
            "Could you name a brand, product type, or attribute that exists in the data?",
            [],
            [f"metadata unavailable for: {', '.join(dropped)}"],
            coverage,
        )

    repaired, unresolved = repair_filters(raw_filters, index)
    # If a filter field exists only on some KGs, still OK — value must be in merged index.
    # But if the field is missing everywhere, treat as unresolved for strict keys.
    for key, raw in list(repaired.items()):
        if key in FREEFORM_FILTER_KEYS or key not in STRICT_METADATA_FIELDS:
            continue
        graphs_with = meta.field_graphs.get(key) or []
        # field_graphs may be empty when overview was thin but values still discovered
        if not graphs_with and key not in index:
            unresolved.append(f"{key}={raw}")
            repaired.pop(key, None)
    spans = match_spans(question, index)
    reasons: list[str] = []
    if unresolved:
        reasons.append("not found in live metadata: " + ", ".join(unresolved))
    for s in spans:
        if s.rivals:
            reasons.append(f'"{s.phrase}" fits {s.field} and also ' + ", ".join(f"{f}" for f, _, _ in s.rivals))
        elif s.score < CONFIDENT:
            reasons.append(f'"{s.phrase}" is close to {s.field}="{s.value}" (not exact)')
    empty_guess = (
        not repaired
        and not plan.get("group_by")
        and not plan.get("analysis")
        and not spans
        and plan.get("source") == "keywords"
        and len(str(question or "").split()) >= 3
    )
    if empty_guess:
        reasons.append("no recognisable product, brand or attribute in the question")

    # Clean count/total with no filters may proceed (e.g. "how many products total").
    if not reasons:
        return Understanding(True, repaired, coverage=coverage)

    options = propose_interpretations(platform, question, history, spans, index, state or {})
    if not options:                       # LLM unavailable: still confirm the best metadata guess
        options = [
            {
                "restated_question": f'Did you mean {s.field} = "{s.value}"?',
                "filters": {s.field: s.value},
            }
            for s in spans[:3]
        ]
    # Still no options but unknown values → ask to rephrase using real catalog terms.
    text = _clarify_text(question, reasons, options)
    return Understanding(False, repaired, text, options, reasons, coverage)


# ---------------------------------------------------------------- interpretations
_PROPOSE_SYSTEM = (
    "You help Fashion Brain understand an unclear question so it can answer from REAL catalog data "
    "across every product knowledge graph (e-commerce, PLMEx, and any other registered product KG). "
    "Using ONLY the provided fields and candidate values (live metadata merged from those graphs), "
    "propose up to 3 distinct interpretations the user most likely meant. If the previous user "
    "question / active filters are provided, prefer related follow-up interpretations on that prior "
    "scope (e.g. price trend on the same brand). Write each restated_question as a complete, natural "
    "question in the SAME LANGUAGE as the user. Never invent field names or values that are not in "
    "the candidate lists. "
    "Return JSON only: "
    '{"lead":"<one short sentence in the user language saying you want to be sure>",'
    '"interpretations":[{"restated_question":"...","filters":{"<field>":"<value>"},"group_by":null}]}'
)


def propose_interpretations(platform, question, history, spans, index, state) -> list[dict]:
    from fashion_brain.ask import _extract_json_object, _plan_timeout
    from fashion_brain.llm import build_model_provider

    cands = []
    for s in spans[:6]:
        cands.append({"phrase": s.phrase, "field": s.field, "value": s.value,
                      "alternatives": [{"field": f, "value": v} for f, v, _ in s.rivals]})
    # nearest values per field for words we could not place (typos / indirect wording)
    q = normalize_name(question)
    near = {}
    for f, vals in index.items():
        top = sorted(((_sim(q, n), r) for r, n in vals), reverse=True)[:3]
        near[f] = [r for sc, r in top if sc > 0.4]
    prev = ""
    for item in reversed(history or []):
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", "")
        if role == "user":
            prev = str(item.get("content") if isinstance(item, dict) else getattr(item, "content", ""))
            break
    prompt = (f"Fields: {list(index)}\nCandidates: {cands}\nNearest values: {near}\n"
              f"Active filters in chat: {state.get('active_filters') or {}}\n"
              f"Previous user question: {prev}\nLatest user message: {question}")
    try:
        raw = build_model_provider(platform.settings).complete(
            prompt, timeout=max(_plan_timeout(platform.settings.request_timeout_seconds), 20.0),
            system=_PROPOSE_SYSTEM, temperature=0.2)
    except Exception:
        LOGGER.warning("interpretation proposal failed", exc_info=True)
        return []
    data = _extract_json_object(raw or "") or {}
    out = []
    for it in (data.get("interpretations") or [])[:3]:
        flt, bad = repair_filters(it.get("filters") or {}, index)
        if bad:                           # never offer an option containing a value not in the data
            continue
        rq = str(it.get("restated_question") or "").strip()
        if rq:
            out.append({"restated_question": rq, "filters": flt, "group_by": it.get("group_by"),
                        "lead": str(data.get("lead") or "")})
    return out


def _clarify_text(question: str, reasons: list[str], options: list[dict]) -> str:
    lead = (options[0].get("lead") if options else "") or "I want to be sure I understood you correctly."
    lines = [lead]
    for i, o in enumerate(options, 1):
        lines.append(f"{i}. {o['restated_question']}")
    if options:
        lines.append("Reply with the number, or tell me in other words.")
    else:
        lines.append("Could you rephrase it with a brand, product type, or attribute?")
    return "\n".join(lines)
