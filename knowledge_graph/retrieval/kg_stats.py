"""Per-KG rollup stats written at sync/rebuild time for fast basic queries.

On-disk layout (filter-wise directories)::

    data/generated/kg_stats/<kg_name>/
        _meta.json                 # totals + unique_counts + filter list
        <filter_name>.json         # all unique values with product counts
        …

Each ``<filter_name>.json``::

    {
      "graph_id", "filter", "unique_count",
      "values": {normalized_key: count, …},          # match keys (normalize_name)
      "display_values": {normalized_key: "Real Name"} # reader-facing surfaces
    }

Matching always uses ``normalize_name`` / ``values``. Answers always prefer
``display_values`` (then Neo4j ``name``, then light title-case). Never hardcode
PLM catalogs in the writer.

A legacy monolith ``<kg_name>.json`` may still exist for full dumps; ask/retrieve
must prefer the per-filter file and never load the whole KG meta for a single
facet answer. Multi-filter / uncached dims → metadata cache-miss (live Cypher).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT, Settings
from knowledge_graph.logging_config import get_logger

LOGGER = get_logger("knowledge_graph.kg_stats")

# Categorical / faceted columns from master_products (and matching Neo4j props).
_ECOM_COUNT_COLUMNS = (
    "brand",
    "category",
    "family",
    "subfamily",
    "section",
    "availability",
    "currency",
    "domain",
    "construction",
    "fit",
    "country_of_origin",
    "catalog_id",
    "enriched_from",
)

# Relationship-backed ecom facets not stored as scalar Postgres columns — filled from Neo4j.
_ECOM_NEO4J_REL_FACETS = ("color", "material", "gender")

# Prefer these as first-class ``products_by_*`` mirrors (others still get mirrors when present).
_PRIMARY_PRODUCTS_BY_DIMS = ("brand", "buyer", "category", "family", "availability", "section")


def stats_root(generated: Path | None = None) -> Path:
    root = Path(generated) if generated is not None else Path(DATA_ROOT) / "generated"
    return root / "kg_stats"


def stats_path(graph_id: str, generated: Path | None = None) -> Path:
    """Legacy monolith file (full dump). Prefer ``graph_stats_dir`` + filter shards."""

    return stats_root(generated) / f"{graph_id}.json"


def graph_stats_dir(graph_id: str, generated: Path | None = None) -> Path:
    """Per-KG directory: ``kg_stats/<kg_name>/``."""

    return stats_root(generated) / str(graph_id)


def graph_meta_path(graph_id: str, generated: Path | None = None) -> Path:
    return graph_stats_dir(graph_id, generated) / "_meta.json"


def filter_stats_path(graph_id: str, attribute: str, generated: Path | None = None) -> Path:
    """Direct path: ``kg_stats/<kg_name>/<filter_name>.json``."""

    return graph_stats_dir(graph_id, generated) / f"{_norm_attr(attribute)}.json"


def index_path(generated: Path | None = None) -> Path:
    return stats_root(generated) / "index.json"


def load_filter_stats(
    graph_id: str,
    attribute: str,
    generated: Path | None = None,
) -> dict[str, Any] | None:
    """Load one filter shard only — no full-KG meta scan."""

    path = filter_stats_path(graph_id, attribute, generated)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_filter_bucket(
    graph_id: str,
    attribute: str,
    generated: Path | None = None,
) -> dict[str, int] | None:
    """All unique values+counts for one filter from its shard file (or None)."""

    payload = load_filter_stats(graph_id, attribute, generated)
    if not payload:
        return None
    values = payload.get("values")
    if isinstance(values, dict):
        return {str(k): int(v) for k, v in values.items()}
    # Tolerate a bare {value: count} shard.
    skip = {"graph_id", "filter", "unique_count", "updated_at", "values", "value_counts"}
    if any(k in payload for k in ("graph_id", "filter", "unique_count")):
        return {}
    return {str(k): int(v) for k, v in payload.items() if k not in skip and isinstance(v, (int, float))}


def write_filter_shards(graph_id: str, payload: dict[str, Any], generated: Path | None = None) -> Path:
    """Write ``kg_stats/<kg>/_meta.json`` + one JSON per filter with values + display map."""

    root = graph_stats_dir(graph_id, generated)
    root.mkdir(parents=True, exist_ok=True)
    value_counts = dict(payload.get("value_counts") or {})
    unique_counts = dict(payload.get("unique_counts") or {})
    display_maps = dict(payload.get("display_values") or {})
    # Also accept products_by_* as source when value_counts is sparse.
    for key, value in payload.items():
        key_s = str(key)
        if key_s.startswith("products_by_") and isinstance(value, dict):
            attr = _norm_attr(key_s[len("products_by_") :])
            value_counts.setdefault(attr, {str(k): int(v) for k, v in value.items()})
    filter_names: list[str] = []
    for attr, bucket in sorted(value_counts.items(), key=lambda kv: str(kv[0])):
        if not isinstance(bucket, dict):
            continue
        key = _norm_attr(str(attr))
        if not key:
            continue
        hint = display_maps.get(key) or display_maps.get(attr) or {}
        if not isinstance(hint, dict):
            hint = {}
        values, display_values = collapse_value_bucket(bucket, display_hint=hint)
        n = unique_counts.get(attr)
        if n is None:
            n = unique_counts.get(key)
        if n is None:
            usable = [name for name in values if str(name).strip().casefold() not in _EMPTY_KEYS]
            n = len(usable) if usable else len(values)
        shard = {
            "graph_id": graph_id,
            "filter": key,
            "unique_count": int(n),
            "values": values,
            "display_values": display_values,
            "updated_at": payload.get("updated_at") or _utc_now(),
        }
        filter_stats_path(graph_id, key, generated).write_text(
            json.dumps(shard, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        filter_names.append(key)
    meta = {
        "graph_id": graph_id,
        "graph_name": payload.get("graph_name") or graph_id,
        "totals": dict(payload.get("totals") or {}),
        "unique_counts": {str(k): int(v) for k, v in unique_counts.items()},
        "filters": filter_names,
        "updated_at": payload.get("updated_at") or _utc_now(),
        "version": int(payload.get("version") or 4),
    }
    if payload.get("description"):
        meta["description"] = payload.get("description")
    graph_meta_path(graph_id, generated).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return root


def load_graph_stats(graph_id: str, generated: Path | None = None) -> dict[str, Any] | None:
    path = stats_path(graph_id, generated)
    payload: dict[str, Any] | None = None
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload = raw if isinstance(raw, dict) else None
        except (OSError, json.JSONDecodeError):
            payload = None
    # Rebuild a lightweight payload from filter shards when monolith is missing.
    if payload is None:
        meta_path = graph_meta_path(graph_id, generated)
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(meta, dict):
            return None
        value_counts: dict[str, dict[str, int]] = {}
        for name in list(meta.get("filters") or []) or list((meta.get("unique_counts") or {}).keys()):
            bucket = load_filter_bucket(graph_id, str(name), generated)
            if bucket is not None:
                value_counts[_norm_attr(str(name))] = bucket
        payload = {
            "graph_id": graph_id,
            "graph_name": meta.get("graph_name") or graph_id,
            "totals": dict(meta.get("totals") or {}),
            "unique_counts": dict(meta.get("unique_counts") or {}),
            "value_counts": value_counts,
            "updated_at": meta.get("updated_at"),
            "version": meta.get("version") or 4,
        }
        if meta.get("description"):
            payload["description"] = meta.get("description")
    # Ensure every value_counts filter has a products_by_<filter> mirror for direct lookup.
    return attach_products_by_breakdowns(payload)


def product_total(graph_id: str, generated: Path | None = None) -> int | None:
    stats = load_graph_stats(graph_id, generated)
    if not stats:
        return None
    totals = stats.get("totals") or {}
    value = totals.get("products")
    return int(value) if value is not None else None


def _norm_attr(attribute: str) -> str:
    return str(attribute or "").strip().casefold().replace(" ", "_")


def _products_by_key(attribute: str) -> str:
    return f"products_by_{_norm_attr(attribute)}"


# Shared empty / junk facet tokens (also referenced by collapse_value_bucket).
_EMPTY_KEYS = {"", "(empty)", "(unspecified)", "n/a", "na", "none", "null", "-", "—"}


def _display_surface_score(label: str) -> tuple[int, int, int, str]:
    """Rank a surface form: punctuation + mixed case beat scrubbed lowercase keys."""

    text = str(label or "").strip()
    marks = sum(1 for ch in text if ch in "&/-()'’")
    mixed = 1 if any(ch.isupper() for ch in text) and any(ch.islower() for ch in text) else 0
    upper = 1 if any(ch.isupper() for ch in text) else 0
    return (marks, mixed, upper, text)


def collapse_value_bucket(
    bucket: dict[str, Any] | None,
    *,
    display_hint: dict[str, str] | None = None,
) -> tuple[dict[str, int], dict[str, str]]:
    """Split a raw value→count bucket into match keys + real display surfaces.

    Counts merge under ``normalize_name``. ``display_values`` keeps the richest
    original label for each key (e.g. ``Cut & Sew-Jersey`` over ``cut sew jersey``).
    """

    from knowledge_graph.ontology import normalize_name

    values: dict[str, int] = {}
    display: dict[str, str] = {}
    hints = {str(k): str(v) for k, v in (display_hint or {}).items() if v}

    for raw_key, raw_count in (bucket or {}).items():
        surface = str(raw_key or "").strip()
        if not surface:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        key = normalize_name(surface) or surface.casefold()
        if key in _EMPTY_KEYS:
            continue
        values[key] = int(values.get(key, 0)) + count
        # Prefer an explicit hint, else the richest surface seen for this key.
        candidates = [surface]
        if key in hints:
            candidates.append(hints[key])
        if key in display:
            candidates.append(display[key])
        display[key] = max(candidates, key=_display_surface_score)

    # Apply leftover hints that didn't appear in the bucket (rare).
    for key, surface in hints.items():
        nn = normalize_name(key) or str(key).casefold()
        if nn and nn not in _EMPTY_KEYS:
            display.setdefault(nn, surface)
    return values, display


def load_display_map(
    graph_id: str,
    attribute: str,
    generated: Path | None = None,
) -> dict[str, str]:
    """normalized_key → real display label from the filter shard (or {})."""

    shard = load_filter_stats(graph_id, attribute, generated)
    if not shard:
        return {}
    raw = dict(shard.get("display_values") or {})
    if not raw:
        # Legacy shards: if values still hold real surfaces, derive the map.
        values = dict(shard.get("values") or {})
        _, derived = collapse_value_bucket(values)
        return derived
    from knowledge_graph.ontology import normalize_name

    out: dict[str, str] = {}
    for key, surface in raw.items():
        nn = normalize_name(str(key)) or str(key).casefold()
        label = str(surface or "").strip()
        if nn and label and nn not in _EMPTY_KEYS:
            out[nn] = label
    return out


def resolve_facet_display(
    value: str,
    *,
    graph_id: str | None = None,
    attribute: str | None = None,
    generated: Path | None = None,
    display_map: dict[str, str] | None = None,
) -> str | None:
    """Return the stored real label for a scrubbed/normalized facet value, if known."""

    from knowledge_graph.ontology import normalize_name

    raw = str(value or "").strip()
    if not raw:
        return None
    nn = normalize_name(raw) or raw.casefold()
    mapping = dict(display_map or {})
    if not mapping and graph_id and attribute:
        mapping = load_display_map(str(graph_id), str(attribute), generated)
    hit = mapping.get(nn)
    if hit:
        return hit
    # Compact fallback (brand initials etc.) against display map values' norms.
    compact = re.sub(r"[^a-z0-9]+", "", nn)
    for key, surface in mapping.items():
        if re.sub(r"[^a-z0-9]+", "", key) == compact:
            return surface
    return None


def dimension_bucket(
    graph_id: str,
    attribute: str,
    generated: Path | None = None,
    *,
    stats: dict[str, Any] | None = None,
) -> dict[str, int] | None:
    """Return the pre-computed products_by_<attr> / value_counts bucket, or None if missing.

    Prefers the per-filter shard ``kg_stats/<kg>/<filter>.json`` so callers never
    need the full KG meta for a single-filter answer.
    """

    key = _norm_attr(attribute)
    if not key:
        return None
    # Direct filter-wise file first.
    if stats is None:
        shard = load_filter_bucket(graph_id, key, generated)
        if shard is not None:
            return shard
        # Shard path exists as empty marker?
        if filter_stats_path(graph_id, key, generated).is_file():
            return {}
    payload = stats if stats is not None else load_graph_stats(graph_id, generated)
    if not payload:
        return None
    # Prefer explicit products_by_* mirror (new snapshot shape).
    by_key = _products_by_key(key)
    bucket = payload.get(by_key)
    if isinstance(bucket, dict) and bucket:
        return {str(k): int(v) for k, v in bucket.items()}
    # Case-insensitive lookup in value_counts (legacy + build source of truth).
    value_counts = dict(payload.get("value_counts") or {})
    for attr_name, values in value_counts.items():
        if _norm_attr(str(attr_name)) == key and isinstance(values, dict):
            return {str(k): int(v) for k, v in values.items()}
    # Empty dict means dimension was indexed but had no values — treat as present empty.
    if by_key in payload and isinstance(payload.get(by_key), dict):
        return {}
    if any(_norm_attr(str(a)) == key for a in value_counts):
        return {}
    return None


def has_dimension(graph_id: str, attribute: str, generated: Path | None = None) -> bool:
    """True when ``attribute`` is a pre-computed breakdown dimension in kg_stats."""

    if filter_stats_path(graph_id, attribute, generated).is_file():
        return True
    return dimension_bucket(graph_id, attribute, generated) is not None


def _plural_forms(name: str) -> set[str]:
    """Singular/plural variants so user 'sweater' maps to catalog 'Sweaters'."""
    n = str(name or "").strip()
    if not n:
        return set()
    out = {n}
    if n.endswith("ies") and len(n) > 3:
        out.add(n[:-3] + "y")
    elif n.endswith("ses") and len(n) > 3:
        out.add(n[:-2])
    elif n.endswith("s") and len(n) > 2:
        out.add(n[:-1])
    else:
        out.add(n + "s")
        if n.endswith("y") and len(n) > 1 and n[-2] not in "aeiou":
            out.add(n[:-1] + "ies")
    return out


def _value_match_score(want: str, candidate: str) -> float:
    """Score user wording vs a catalog value (case/punct/plural/near-miss).

    Users do not know exact catalog casing or plural forms — matching must be soft.
    """
    from difflib import SequenceMatcher

    from knowledge_graph.ontology import normalize_name

    w = normalize_name(want)
    c = normalize_name(candidate)
    if not w or not c or w in _EMPTY_KEYS or c in _EMPTY_KEYS:
        return 0.0
    if w == c:
        return 1.0
    if _plural_forms(w) & _plural_forms(c):
        return 0.96
    # Whole-token containment: sweater ⊂ cardigans sweaters
    # Reject short single-token queries inside longer multi-word labels
    # ("fashion" ⊂ "young fashion" / "girls fashion") — too weak to invent filters.
    wt, ct = set(w.split()), set(c.split())
    if wt and wt <= ct:
        if len(ct) > len(wt) and (" " not in w) and len(w) < 8:
            pass
        else:
            return 0.88
    if ct and ct <= wt and len(c) >= 3:
        return 0.84
    return float(SequenceMatcher(None, w, c).ratio())


def value_count(graph_id: str, attribute: str, value: str, generated: Path | None = None) -> int | None:
    """Count for one attribute=value from pre-computed breakdowns.

    Returns None when the dimension is not indexed (cache-miss) OR the stats file
    is missing. Returns 0 when the dimension exists but the value is absent.
    Soft-matches casing and singular/plural (sweater ↔ Sweaters).
    """

    bucket = dimension_bucket(graph_id, attribute, generated)
    if bucket is None:
        return None
    best: tuple[float, int] | None = None
    for key, count in bucket.items():
        score = _value_match_score(value, str(key))
        if score < 0.85:
            continue
        n = int(count)
        if best is None or score > best[0] or (score == best[0] and n > best[1]):
            best = (score, n)
    return 0 if best is None else best[1]


# Prefer these dims when resolving a free garment/type token against kg_stats values.
_VALUE_RESOLVE_ATTR_ORDER = (
    "category",
    "garment_type",
    "family",
    "subfamily",
    "plm_main_category",
    "plm_sub_category",
    "plm_sub_sub_category",
    "section",
    "brand",
    "buyer",
)

# Explicit planner keys that must not silently rebind to another facet (e.g. color→category).
_EXPLICIT_FILTER_KEYS = frozenset(
    {
        "brand",
        "buyer",
        "color",
        "gender",
        "material",
        "availability",
        "season",
        "section",
        "family",
        "subfamily",
        "category",
        "garment_type",
    }
)


def find_value_in_stats(
    graph_id: str,
    value: str,
    *,
    generated: Path | None = None,
    prefer_attrs: tuple[str, ...] | None = None,
    min_score: float = 0.85,
) -> tuple[str, str, int] | None:
    """Locate ``value`` in filter shards. Returns ``(attr, canonical_value, count)``.

    Matches exact, case-insensitive, singular/plural, and high-similarity near-misses
    so user wording can ground onto live catalog values without knowing exact labels.
    """

    from knowledge_graph.ontology import normalize_name

    want = normalize_name(str(value or ""))
    if not want or want in _EMPTY_KEYS:
        return None
    prefer = list(prefer_attrs or _VALUE_RESOLVE_ATTR_ORDER)
    # Scan preferred attrs first, then any remaining indexed filters.
    ordered = list(prefer)
    for name in listed_filters(graph_id, generated):
        if name not in ordered:
            ordered.append(name)
    hits: list[tuple[float, str, str, int]] = []
    for attr in ordered:
        bucket = dimension_bucket(graph_id, attr, generated)
        if not bucket:
            continue
        best_for_attr: tuple[float, str, int] | None = None
        for raw, count in bucket.items():
            score = _value_match_score(want, str(raw))
            if score < min_score:
                continue
            n = int(count)
            if best_for_attr is None or score > best_for_attr[0] or (
                score == best_for_attr[0] and n > best_for_attr[2]
            ):
                best_for_attr = (score, str(raw), n)
        if best_for_attr is not None:
            hits.append((best_for_attr[0], _norm_attr(attr), best_for_attr[1], best_for_attr[2]))
    if not hits:
        return None
    # Prefer stronger match, then preferred-attr order, then highest count.
    rank = {a: i for i, a in enumerate(prefer)}
    hits.sort(key=lambda item: (-item[0], rank.get(item[1], 999), -item[3], item[1]))
    return hits[0][1], hits[0][2], hits[0][3]


def resolve_filters_against_stats(
    graph_id: str,
    filters: dict | None,
    *,
    generated: Path | None = None,
) -> dict[str, str]:
    """Map planner filters onto real kg_stats dimensions for meta answers.

    ``garment=sweater`` → ``category=sweater`` when that value exists in the
    category shard. Drops values that match no indexed filter (e.g. buyer=dataset).
    """

    from knowledge_graph.ontology import normalize_name

    # Graph-aware semantic aliases before shard lookup (gender→section on ecom, …).
    try:
        from fashion_brain.planning.semantic_field_map import apply_semantic_field_map

        mapped, _ = apply_semantic_field_map(filters, graph_id)
        filters = mapped
    except Exception:
        pass

    out: dict[str, str] = {}
    for raw_key, raw_val in (filters or {}).items():
        key = _norm_attr(str(raw_key))
        if not key or raw_val in (None, ""):
            continue
        value = str(raw_val).strip()
        if not value:
            continue
        # Alias keys that are not real stats columns.
        alias_key = {
            "garment": None,  # resolve by value across type dims
            "garments": None,
            "type": None,
            "product_type": None,
        }.get(key, key)

        if alias_key and has_dimension(graph_id, alias_key, generated):
            n = value_count(graph_id, alias_key, value, generated)
            if n is not None and n > 0:
                # Keep canonical casing from the shard when possible.
                hit = find_value_in_stats(
                    graph_id, value, generated=generated, prefer_attrs=(alias_key,)
                )
                # Prefer-attrs can still leak to another dim when the preferred
                # shard has a weak/zero hit — force the requested key.
                if hit and hit[0] == alias_key:
                    out[alias_key] = hit[1]
                else:
                    out[alias_key] = value
                continue
            if n == 0 and alias_key in {"brand", "buyer", "color", "gender", "material", "availability"}:
                # Honest zero / keep for Neo4j — never rebind brand→category etc.
                out[alias_key] = value
                continue
            if n == 0 and alias_key not in {"brand", "buyer"}:
                # Wrong dim for this value — try cross-dim resolve below.
                pass

        # Explicit facet on a graph that does not index it → keep key (cache-miss /
        # Neo4j). Never remaps color=white → category=White on ecom.
        if alias_key and alias_key in _EXPLICIT_FILTER_KEYS:
            out[alias_key] = value
            continue

        prefer = _VALUE_RESOLVE_ATTR_ORDER
        if key in {"garment", "garments", "type", "product_type", "garment_type", "category"}:
            prefer = (
                "category",
                "garment_type",
                "family",
                "subfamily",
                "plm_main_category",
                "plm_sub_category",
                "plm_sub_sub_category",
            )
        hit = find_value_in_stats(graph_id, value, generated=generated, prefer_attrs=prefer)
        if hit is None:
            # Unknown token (dataset, name, …) — drop; do not invent a zero count.
            LOGGER.debug(
                "dropped ungrounded filter %s=%s for %s (not in kg_stats values)",
                key,
                value,
                graph_id,
            )
            continue
        out[hit[0]] = hit[1]
    # Keep ALL grounded keys. Multi-filter is a meta cache-miss (answer_simple_count
    # returns None when len>=2). Collapsing brand+sweater→category alone caused
    # false single-filter kg_stats hits on multi-filter Neo4j asks.
    return out


def attach_products_by_breakdowns(payload: dict[str, Any]) -> dict[str, Any]:
    """Mirror ``value_counts`` into top-level ``products_by_<attr>`` maps.

    Always writes primary dims (brand/buyer/category/…) when present in value_counts,
    and mirrors every other indexed facet as well so single-filter lookups stay fast.
    """

    out = dict(payload or {})
    value_counts = dict(out.get("value_counts") or {})
    # Preserve any manually set products_by_* then refresh from value_counts.
    for attr, bucket in value_counts.items():
        if not isinstance(bucket, dict):
            continue
        key = _norm_attr(str(attr))
        if not key:
            continue
        out[_products_by_key(key)] = {str(k): int(v) for k, v in bucket.items()}
    # Ensure primary keys exist as empty maps when the facet was never indexed —
    # callers can distinguish missing file vs missing dim via has_dimension / dimension_bucket.
    for attr in _PRIMARY_PRODUCTS_BY_DIMS:
        by_key = _products_by_key(attr)
        if by_key not in out and _norm_attr(attr) in {_norm_attr(a) for a in value_counts}:
            out[by_key] = dict(value_counts.get(attr) or {})
    totals = dict(out.get("totals") or {})
    if "products" not in totals and value_counts:
        # Best-effort: leave totals.products alone if already set.
        pass
    out["totals"] = totals
    return out


def _rel_types_for_attr(attr: str) -> tuple[str, ...]:
    """Relationship types for a facet — from the shared ecom/_REL_FIELDS map (full system)."""

    try:
        from knowledge_graph.retrieval.ecom import _REL_FIELDS

        rels = _REL_FIELDS.get(str(attr or "").strip().casefold()) or ()
        return tuple(str(r) for r in rels if r)
    except Exception:
        return ()


def _discover_stats_attributes(platform, graph_id: str) -> list[str]:
    """Facet keys for this graph from live schema / mapping — never a fixed short list."""

    try:
        from knowledge_graph.retrieval.ecom import resolve_analysis_fields

        keys = list(resolve_analysis_fields(platform, graph_id) or [])
    except Exception:
        keys = []
    try:
        from knowledge_graph.sources.mapping_store import get_mapping, mapping_exists

        if mapping_exists(graph_id):
            mapping = get_mapping(graph_id).get("mapping") or {}
            from knowledge_graph.pipeline.mapping_sql_sync import _stats_attributes_from_mapping

            for name in _stats_attributes_from_mapping(mapping):
                if name not in keys:
                    keys.append(name)
    except Exception:
        pass
    # Include every relationship-backed facet key known to the retrieve layer.
    try:
        from knowledge_graph.retrieval.ecom import _REL_FIELDS

        for name in _REL_FIELDS:
            if name not in keys:
                keys.append(name)
    except Exception:
        pass
    return keys


def unique_count(graph_id: str, attribute: str, generated: Path | None = None) -> int | None:
    """Distinct non-empty values for one facet from ``kg_stats`` (None if unknown/empty-only).

    Reads ``kg_stats/<kg>/<filter>.json`` directly when present.
    """

    key = _norm_attr(attribute)
    shard = load_filter_stats(graph_id, key, generated)
    if shard is not None:
        n = shard.get("unique_count")
        values = shard.get("values") if isinstance(shard.get("values"), dict) else None
        if values is not None:
            usable = [name for name in values if str(name).strip().casefold() not in _EMPTY_KEYS]
            if not usable:
                return None
            return int(n) if n is not None else len(usable)
        if n is not None:
            return int(n)
    bucket = dimension_bucket(graph_id, key, generated)
    if bucket is None:
        # Fallback: _meta.json unique_counts without loading every filter.
        meta_path = graph_meta_path(graph_id, generated)
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for attr_name, count in (meta.get("unique_counts") or {}).items():
                    if _norm_attr(str(attr_name)) == key:
                        return int(count)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        stats = load_graph_stats(graph_id, generated)
        if not stats:
            return None
        n = (stats.get("unique_counts") or {}).get(key)
        if n is None:
            for attr_name, count in (stats.get("unique_counts") or {}).items():
                if _norm_attr(str(attr_name)) == key:
                    return int(count)
            return None
        return int(n)
    usable = [name for name in bucket if str(name).strip().casefold() not in _EMPTY_KEYS]
    if not usable:
        return None
    return len(usable)


def normalize_filter_dict(filters: dict | None) -> dict[str, str]:
    """Flatten ``filters`` to ``{attr: scalar_str}``; drop empties. Lists become a single joined value."""

    out: dict[str, str] = {}
    for raw_key, raw_val in (filters or {}).items():
        key = _norm_attr(str(raw_key))
        if not key or raw_val in (None, ""):
            continue
        if isinstance(raw_val, (list, tuple, set)):
            parts = [str(v).strip() for v in raw_val if v not in (None, "")]
            if not parts:
                continue
            out[key] = parts[0] if len(parts) == 1 else " | ".join(parts)
        else:
            out[key] = str(raw_val).strip()
    return out


def resolve_entity_to_filter(
    graph_id: str,
    entity: str,
    generated: Path | None = None,
) -> dict[str, str] | None:
    """Map a bare entity token to exactly one pre-computed ``{dimension: value}`` filter.

    Generic across all indexed dimensions (brand, buyer, category, season, …). Returns
    None when the token matches zero or multiple dimensions (ambiguous / unknown).
    """

    token = str(entity or "").strip()
    if not token:
        return None
    ent = _norm_attr(token)
    if ent in {"product", "products", "item", "items"}:
        # Bare product nouns are count_entity asks, not product filters.
        return None
    stats = load_graph_stats(graph_id, generated)
    if not stats:
        return None
    value_counts = dict(stats.get("value_counts") or {})
    unique_counts = dict(stats.get("unique_counts") or {})
    # Indexed filter names themselves are count_entity asks ("how many buyers"),
    # never {buyer: "buyer"} value lookups.
    indexed_filters = {_norm_attr(str(a)) for a in value_counts} | {
        _norm_attr(str(a)) for a in unique_counts
    }
    for key in stats:
        key_s = str(key)
        if key_s.startswith("products_by_"):
            indexed_filters.add(_norm_attr(key_s[len("products_by_") :]))
    if ent in indexed_filters:
        return None
    from knowledge_graph.ontology import normalize_name

    want = normalize_name(token)
    hits: list[tuple[str, str]] = []
    # Direct per-filter value scan only — no full-meta dump to callers.
    dims: dict[str, dict[str, int]] = {}
    for attr, bucket in value_counts.items():
        if isinstance(bucket, dict):
            dims[_norm_attr(str(attr))] = {str(k): int(v) for k, v in bucket.items()}
    for key, value in stats.items():
        key_s = str(key)
        if key_s.startswith("products_by_") and isinstance(value, dict):
            dims.setdefault(key_s[len("products_by_") :], {str(k): int(v) for k, v in value.items()})
    for attr, bucket in dims.items():
        for name in bucket:
            if normalize_name(str(name)) == want:
                hits.append((attr, str(name)))
                break
    if len(hits) == 1:
        attr, name = hits[0]
        return {attr: name}
    return None


def filters_match(requested: dict | None, applied: dict | None) -> bool:
    """True when applied filters cover the same normalized keys/values as requested."""

    req = normalize_filter_dict(requested)
    got = normalize_filter_dict(applied)
    if not req:
        return True
    from knowledge_graph.ontology import normalize_name

    for key, value in req.items():
        if key not in got:
            return False
        if normalize_name(got[key]) != normalize_name(value):
            return False
    return True


def facet_inventory(graph_id: str, generated: Path | None = None) -> dict[str, int]:
    """Compact field → distinct-value map for LLM/schema grounding."""

    stats = load_graph_stats(graph_id, generated)
    if not stats:
        return {}
    out: dict[str, int] = {}
    uniques = dict(stats.get("unique_counts") or {})
    buckets = dict(stats.get("value_counts") or {})
    for attr in sorted(set(uniques) | set(buckets)):
        n = unique_count(graph_id, attr, generated)
        if n is not None and n > 0:
            out[str(attr)] = n
    products = (stats.get("totals") or {}).get("products")
    if products is not None:
        out["products"] = int(products)
    return out


# Context dims offered to the response LLM alongside a party/product answer.
_META_CONTEXT_DIMS = (
    "category",
    "section",
    "family",
    "garment_type",
    "plm_main_category",
    "gender",
    "availability",
)


def listed_filters(graph_id: str, generated: Path | None = None) -> list[str]:
    """Filter names available as shards / unique_counts for this KG."""

    meta_path = graph_meta_path(graph_id, generated)
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            names = [str(n) for n in (meta.get("filters") or []) if n]
            if names:
                return [_norm_attr(n) for n in names]
            return [_norm_attr(str(k)) for k in (meta.get("unique_counts") or {})]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    root = graph_stats_dir(graph_id, generated)
    if root.is_dir():
        return sorted(
            p.stem
            for p in root.glob("*.json")
            if p.name != "_meta.json" and not p.name.startswith("_")
        )
    return list(facet_inventory(graph_id, generated).keys())


def select_answer_filters(
    graph_id: str,
    *,
    filters: dict | None = None,
    count_entity: str | None = None,
    group_by: str | None = None,
    question: str = "",
    generated: Path | None = None,
    max_filters: int = 5,
) -> list[str]:
    """Pick only the filter shards needed for this ask (never the full KG meta)."""

    available = set(listed_filters(graph_id, generated))
    selected: list[str] = []

    def _add(name: str | None) -> None:
        key = _norm_attr(str(name or ""))
        if not key or key in {"product", "products", "item", "items"}:
            return
        if available and key not in available:
            # Soft-match: stem against available keys.
            hit = next((a for a in available if a == key or a.endswith("_" + key) or key.endswith("_" + a)), None)
            if not hit:
                return
            key = hit
        if key not in selected:
            selected.append(key)

    for raw_key in (filters or {}):
        _add(str(raw_key))
    _add(count_entity)
    _add(group_by)

    # Question tokens that match indexed filter names (buyer/brand/…).
    # Require meaningful tokens (len>=4) so "for" does not hit design_for.
    text = str(question or "").casefold()
    tokens = {t for t in re.findall(r"[a-z0-9_]+", text) if len(t) >= 4}
    for key in sorted(available):
        stem = key.split("_")[-1]
        variants = {key, key + "s", key.replace("_", "")}
        if len(stem) >= 4:
            variants |= {stem, stem + "s"}
            if stem.endswith("y") and len(stem) > 4:
                variants.add(stem[:-1] + "ies")
        if key.endswith("y") and len(key) > 4:
            variants.add(key[:-1] + "ies")
        if tokens & {str(v).casefold() for v in variants if v}:
            _add(key)

    # Complementary mix dims only when the ask is unfiltered.
    # Filtered shards are single-dimension globals — attaching them beside a
    # brand/buyer count would mis-attribute the whole-KG mix to that party.
    if not filters:
        if {"brand", "buyer"} & set(selected) or not selected:
            for dim in _META_CONTEXT_DIMS:
                if len(selected) >= max_filters:
                    break
                _add(dim)

    return selected[:max_filters]


def _monolith_filter_bucket(
    graph_id: str,
    attribute: str,
    generated: Path | None = None,
) -> tuple[dict[str, int], int] | None:
    """Fallback: value/unique counts from legacy monolith ``kg_stats/<kg>.json``."""

    key = _norm_attr(attribute)
    if not key or key in {"product", "products", "item", "items"}:
        return None
    stats = load_graph_stats(graph_id, generated)
    if not stats:
        return None
    values: dict[str, int] = {}
    for source_key in (
        key,
        f"products_by_{key}",
    ):
        raw = None
        if source_key.startswith("products_by_"):
            raw = (stats.get(source_key) if source_key in stats else None) or (
                (stats.get("value_counts") or {}).get(key)
            )
        else:
            raw = (stats.get("value_counts") or {}).get(source_key) or stats.get(
                f"products_by_{source_key}"
            )
        if isinstance(raw, dict) and raw:
            values = {str(k): int(v) for k, v in raw.items()}
            break
    unique = int((stats.get("unique_counts") or {}).get(key) or 0)
    if not values and unique <= 0:
        return None
    if unique <= 0:
        unique = len(values)
    return values, unique


def top_facet_examples(
    graph_id: str,
    attribute: str,
    *,
    generated: Path | None = None,
    limit: int = 5,
) -> list[tuple[str, int]]:
    """Top ``(display_name, count)`` pairs for a facet — for enriching short counts."""

    shard = load_filter_stats(graph_id, attribute, generated)
    key = _norm_attr(attribute)
    values: dict = {}
    display_map: dict = {}
    if shard:
        values = dict(shard.get("values") or {})
        display_map = {
            str(k): str(v)
            for k, v in dict(shard.get("display_values") or {}).items()
            if v
        }
        key = _norm_attr(str(shard.get("filter") or attribute))
    else:
        bucket = load_filter_bucket(graph_id, attribute, generated)
        if bucket is None:
            mono = _monolith_filter_bucket(graph_id, attribute, generated)
            if mono is None:
                return []
            values, _unique = mono
        else:
            values = dict(bucket)
    if not display_map:
        display_map = load_display_map(graph_id, key, generated) or {}
    ranked = sorted(
        (
            (str(k), int(v))
            for k, v in values.items()
            if str(k).strip().casefold() not in _EMPTY_KEYS
        ),
        key=lambda kv: (-kv[1], kv[0].casefold()),
    )
    out: list[tuple[str, int]] = []
    for name, count in ranked[: max(int(limit), 1)]:
        shown = resolve_facet_display(
            name,
            graph_id=graph_id,
            attribute=key,
            generated=generated,
            display_map=display_map,
        )
        if not shown:
            try:
                from knowledge_graph.retrieval.ecom import _display_label

                shown = _display_label(name)
            except Exception:
                shown = name
        if shown:
            out.append((str(shown), int(count)))
    return out


def format_filter_reference(
    graph_id: str,
    attribute: str,
    *,
    generated: Path | None = None,
    top_n: int = 5,
) -> str | None:
    """One-line summary from filter shard or monolith kg_stats (unique + top values)."""

    shard = load_filter_stats(graph_id, attribute, generated)
    if not shard:
        bucket = load_filter_bucket(graph_id, attribute, generated)
        if bucket is None:
            mono = _monolith_filter_bucket(graph_id, attribute, generated)
            if mono is None:
                return None
            values, unique = mono
            key = _norm_attr(attribute)
        else:
            values = bucket
            unique = len(values)
            key = _norm_attr(attribute)
    else:
        values = dict(shard.get("values") or {})
        unique = int(shard.get("unique_count") or len(values))
        key = _norm_attr(str(shard.get("filter") or attribute))
    if not values and unique <= 0:
        return None
    examples = top_facet_examples(
        graph_id, key, generated=generated, limit=max(int(top_n), 1)
    )
    if examples:
        bits = [f"{name}={count}" for name, count in examples]
        return f"{key}: unique={unique}; top: {', '.join(bits)}"
    return f"{key}: unique={unique}"


def build_meta_reference_lines(
    graph_id: str,
    *,
    filters: dict | None = None,
    count_entity: str | None = None,
    group_by: str | None = None,
    question: str = "",
    generated: Path | None = None,
    max_filters: int = 5,
    top_n: int = 5,
) -> list[str]:
    """Compact meta lines for the response LLM — selected filters only."""

    selected = select_answer_filters(
        graph_id,
        filters=filters,
        count_entity=count_entity,
        group_by=group_by,
        question=question,
        generated=generated,
        max_filters=max_filters,
    )
    lines: list[str] = []
    # Totals from filter-wise _meta, else monolith dump.
    totals: dict[str, Any] = {}
    meta_path = graph_meta_path(graph_id, generated)
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            totals = dict(meta.get("totals") or {})
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            totals = {}
    if totals.get("products") is None:
        stats = load_graph_stats(graph_id, generated) or {}
        totals = dict(stats.get("totals") or {}) or totals
    if totals.get("products") is not None:
        try:
            lines.append(f"products_total={int(totals['products'])}")
        except (TypeError, ValueError):
            pass
    for attr in selected:
        line = format_filter_reference(graph_id, attr, generated=generated, top_n=top_n)
        if line:
            lines.append(line)
    return lines


def stats_available(graph_id: str, generated: Path | None = None) -> bool:
    """True when monolith or filter-wise ``_meta.json`` exists for this graph."""

    return stats_path(graph_id, generated).is_file() or graph_meta_path(graph_id, generated).is_file()


def _humanize_attr(key: str) -> str:
    return str(key or "").replace("_", " ").strip() or "attribute"


def _column_definitions(platform, graph_id: str) -> dict[str, str]:
    """Map attribute → definition from mapping notes / ontology entity descriptions."""

    defs: dict[str, str] = {}
    try:
        from knowledge_graph.sources.mapping_store import get_mapping, mapping_exists

        if mapping_exists(graph_id):
            mapping = get_mapping(graph_id).get("mapping") or {}
            for key, spec in (mapping.get("product_properties") or {}).items():
                if not isinstance(spec, dict):
                    continue
                note = (
                    str(spec.get("description") or spec.get("note") or spec.get("label") or "")
                    .strip()
                )
                if note:
                    defs[str(key)] = note
            for key, spec in (mapping.get("relationships") or {}).items():
                if not isinstance(spec, dict):
                    continue
                note = str(spec.get("description") or spec.get("note") or "").strip()
                attr = str(spec.get("attribute") or key or "").strip()
                if note and attr:
                    defs.setdefault(attr, note)
    except Exception:
        pass
    try:
        registry = getattr(platform, "kg_registry", None)
        if registry is not None:
            graph = registry.get_graph(graph_id)
            ontology = registry.load_ontology_object(graph.ontology_id)
            # Entity type names often match facet keys (Brand → brand).
            for entity in getattr(ontology, "entity_types", None) or []:
                name = str(entity).strip()
                if not name:
                    continue
                key = name.casefold().replace(" ", "_")
                desc = ""
                # Ontology may expose entity metadata as objects or bare strings.
                meta = getattr(ontology, "entity_metadata", None) or {}
                if isinstance(meta, dict):
                    entry = meta.get(name) or meta.get(key) or {}
                    if isinstance(entry, dict):
                        desc = str(entry.get("description") or "").strip()
                if desc:
                    defs.setdefault(key, desc)
                    # common aliases
                    if key.endswith("category") or key in {"garmentcategory"}:
                        defs.setdefault("category", desc)
                    if key == "garmenttype":
                        defs.setdefault("garment_type", desc)
            graph_desc = str(getattr(ontology, "description", "") or "").strip()
            if graph_desc:
                defs.setdefault("__graph__", graph_desc)
    except Exception:
        pass
    return defs


def _graph_identity(platform, graph_id: str) -> tuple[str, str]:
    """Return (graph_name, description) from the registry when available."""

    name = graph_id
    description = ""
    try:
        registry = getattr(platform, "kg_registry", None)
        if registry is not None:
            graph = registry.get_graph(graph_id)
            name = str(getattr(graph, "name", None) or graph_id)
            description = str(getattr(graph, "description", None) or "").strip()
    except Exception:
        pass
    if not description:
        defs = _column_definitions(platform, graph_id) if platform is not None else {}
        description = str(defs.get("__graph__") or "")
    return name, description


def enrich_stats_payload(
    platform,
    graph_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Attach graph_name, description, columns[], and products_by_* breakdowns."""

    out = dict(payload or {})
    name, description = _graph_identity(platform, graph_id)
    out["graph_id"] = graph_id
    out["graph_name"] = name
    if description:
        out["description"] = description
    defs = _column_definitions(platform, graph_id) if platform is not None else {}
    value_counts = dict(out.get("value_counts") or {})
    unique_counts = dict(out.get("unique_counts") or {})
    columns: list[dict[str, Any]] = []
    for key in sorted(set(value_counts) | set(unique_counts)):
        bucket = dict(value_counts.get(key) or {})
        n = unique_counts.get(key)
        if n is None:
            usable = [k for k in bucket if str(k).strip().casefold() not in _EMPTY_KEYS]
            n = len(usable) if usable else len(bucket)
        columns.append(
            {
                "key": key,
                "definition": defs.get(key) or f"Product {_humanize_attr(key)} values in {name}.",
                "unique_count": int(n or 0),
                "value_counts": bucket,
            }
        )
    out["columns"] = columns
    out = attach_products_by_breakdowns(out)
    out["version"] = max(int(out.get("version") or 1), 3)
    return out


def meta_overview(graph_id: str, generated: Path | None = None) -> dict[str, Any] | None:
    """Agent-facing metadata: name, totals, columns with definitions and value counts."""

    stats = load_graph_stats(graph_id, generated)
    if not stats:
        return None
    columns = list(stats.get("columns") or [])
    if not columns:
        # Backward-compatible older stats files without columns[].
        value_counts = dict(stats.get("value_counts") or {})
        unique_counts = dict(stats.get("unique_counts") or {})
        columns = [
            {
                "key": key,
                "definition": f"Product {_humanize_attr(key)}.",
                "unique_count": int(unique_counts.get(key) or len(value_counts.get(key) or {})),
                "value_counts": dict(value_counts.get(key) or {}),
            }
            for key in sorted(set(value_counts) | set(unique_counts))
        ]
    return {
        "graph_id": stats.get("graph_id") or graph_id,
        "graph_name": stats.get("graph_name") or graph_id,
        "description": stats.get("description") or "",
        "totals": dict(stats.get("totals") or {}),
        "updated_at": stats.get("updated_at"),
        "columns": columns,
        "unique_counts": dict(stats.get("unique_counts") or {}),
        "value_counts": dict(stats.get("value_counts") or {}),
    }


def answer_simple_count(
    graph_id: str,
    *,
    filters: dict | None = None,
    count_entity: str | None = None,
    generated: Path | None = None,
) -> dict[str, Any] | None:
    """Answer a 0–1 filter / distinct-facet count from kg_stats only.

    Routing contract:
      - no filters → global ``totals.products``
      - one filter on a pre-computed dimension → that breakdown count
      - multi-filter OR uncached dimension → ``None`` (``cache_miss``)

    Never returns a global total when ``filters`` is non-empty.
    Returns ``{n, kind, attribute, value, filter, requested_filters, applied_filters, source}``
    or None on cache-miss.
    """

    if not stats_available(graph_id, generated):
        return None
    cleaned = normalize_filter_dict(filters)
    requested = dict(cleaned)
    # Combined filters cannot be answered from single-dimension breakdowns.
    if len(cleaned) >= 2:
        return None
    entity = _norm_attr(str(count_entity or ""))
    if entity in {"", "product", "products", "item", "items"}:
        entity = "product"
    if not cleaned:
        if entity != "product":
            n = unique_count(graph_id, entity, generated)
            if n is None:
                n = 0
            return {
                "n": int(n),
                "kind": "distinct_facet",
                "attribute": entity,
                "value": None,
                "filter": None,
                "requested_filters": {},
                "applied_filters": {},
                "source": "kg_stats",
            }
        total = product_total(graph_id, generated)
        if total is None:
            return None
        return {
            "n": int(total),
            "kind": "product_total",
            "attribute": None,
            "value": None,
            "filter": None,
            "requested_filters": {},
            "applied_filters": {},
            "source": "kg_stats",
        }
    attr, value = next(iter(cleaned.items()))
    attr_key = _norm_attr(str(attr))
    # Uncached dimension → cache-miss (caller should use Text2Cypher / live Cypher).
    if not has_dimension(graph_id, attr_key, generated):
        return None
    hit = find_value_in_stats(
        graph_id, str(value), generated=generated, prefer_attrs=(attr_key,)
    )
    if hit is None or hit[0] != attr_key:
        # Keep honest zero when the dimension exists but no soft match.
        n = value_count(graph_id, attr_key, str(value), generated)
        if n is None:
            return None
        canon = str(value)
    else:
        canon = hit[1]
        n = int(hit[2])
    applied = {attr_key: canon}
    filter_label = f"filter: {attr_key}={canon}"
    return {
        "n": int(n),
        "kind": "value_count",
        "attribute": attr_key,
        "value": canon,
        "filter": filter_label,
        "requested_filters": requested,
        "applied_filters": applied,
        "source": "kg_stats",
    }


def save_graph_stats(graph_id: str, payload: dict[str, Any], generated: Path | None = None) -> Path:
    """Persist full dump + filter-wise directory ``kg_stats/<kg>/<filter>.json``."""

    root = stats_root(generated)
    root.mkdir(parents=True, exist_ok=True)
    enriched = attach_products_by_breakdowns(dict(payload or {}))
    path = stats_path(graph_id, generated)
    path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_filter_shards(graph_id, enriched, generated)
    _update_index(graph_id, enriched, generated)
    return path


def shard_existing_graph_stats(graph_id: str, generated: Path | None = None) -> Path | None:
    """Split an existing monolith ``<kg>.json`` into filter-wise files (idempotent)."""

    payload = None
    path = stats_path(graph_id, generated)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload = raw if isinstance(raw, dict) else None
        except (OSError, json.JSONDecodeError):
            payload = None
    if not payload:
        return None
    return write_filter_shards(graph_id, attach_products_by_breakdowns(payload), generated)


def _update_index(graph_id: str, payload: dict[str, Any], generated: Path | None) -> None:
    path = index_path(generated)
    index: dict[str, Any] = {"version": 1, "graphs": {}}
    if path.is_file():
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    graphs = dict(index.get("graphs") or {})
    totals = payload.get("totals") or {}
    graphs[graph_id] = {
        "products": totals.get("products"),
        "brands": totals.get("brands"),
        "buyers": totals.get("buyers"),
        "updated_at": payload.get("updated_at"),
        "path": stats_path(graph_id, generated).name,
    }
    index["graphs"] = graphs
    index["updated_at"] = payload.get("updated_at")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _merge_ecom_neo4j_rel_facets(
    platform,
    *,
    value_counts: dict[str, dict[str, int]],
    unique_counts: dict[str, int],
) -> None:
    """Fill color/material/gender shards from Neo4j relationships when missing."""

    if platform is None:
        return
    session_factory = getattr(getattr(platform, "graph", None), "_session", None)
    if not callable(session_factory):
        return
    try:
        with session_factory() as session:
            for attr in _ECOM_NEO4J_REL_FACETS:
                if value_counts.get(attr):
                    continue
                bucket: dict[str, int] = {}
                for rel in _rel_types_for_attr(attr):
                    try:
                        rel_rows = list(
                            session.run(
                                f"MATCH (p:Entity:Product {{graph: $g}})-[:{rel}]->(b:Entity) "
                                "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
                                "RETURN coalesce(b.name, '(empty)') AS value, count(*) AS n "
                                "ORDER BY n DESC, value ASC",
                                {"g": "ecom_products"},
                            )
                        )
                    except Exception:
                        continue
                    if rel_rows:
                        bucket = {str(r["value"]): int(r["n"]) for r in rel_rows}
                        break
                if bucket:
                    value_counts[attr] = bucket
                    unique_counts[attr] = len(
                        [k for k in bucket if str(k).strip().casefold() not in _EMPTY_KEYS]
                    ) or len(bucket)
                    LOGGER.info("merged neo4j ecom facet=%s unique=%s", attr, unique_counts[attr])
    except Exception as exc:
        LOGGER.warning("neo4j ecom facet merge skipped: %s", exc)


def build_ecom_stats_from_postgres(
    dsn: str,
    *,
    generated: Path | None = None,
    platform=None,
) -> dict[str, Any]:
    """Aggregate master_products column value counts — source of truth for ecom_products."""

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("psycopg is required to build ecom stats") from exc

    value_counts: dict[str, dict[str, int]] = {}
    unique_counts: dict[str, int] = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT count(*)::int AS n FROM master_products")
            products = int(cur.fetchone()["n"])
            # Discover which count columns actually exist.
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'master_products'
                """
            )
            existing = {str(row["column_name"]) for row in cur.fetchall()}
            columns = [col for col in _ECOM_COUNT_COLUMNS if col in existing]
            for column in columns:
                cur.execute(
                    f"""
                    SELECT COALESCE(NULLIF(TRIM({column}::text), ''), '(empty)') AS value,
                           count(*)::int AS n
                    FROM master_products
                    GROUP BY 1
                    ORDER BY n DESC, value ASC
                    """
                )
                bucket = {str(row["value"]): int(row["n"]) for row in cur.fetchall()}
                value_counts[column] = bucket
                unique_counts[column] = len(bucket)

    _merge_ecom_neo4j_rel_facets(
        platform, value_counts=value_counts, unique_counts=unique_counts
    )

    brands = unique_counts.get("brand") or len(value_counts.get("brand") or {})
    payload = {
        "version": 1,
        "graph_id": "ecom_products",
        "source": "postgres.master_products",
        "updated_at": _utc_now(),
        "totals": {
            "products": products,
            "brands": brands,
            "buyers": 0,
            "columns_indexed": len(value_counts),
        },
        "unique_counts": unique_counts,
        "value_counts": value_counts,
    }
    payload = enrich_stats_payload(platform, "ecom_products", payload)
    save_graph_stats("ecom_products", payload, generated)
    LOGGER.info(
        "wrote ecom kg stats products=%s brands=%s columns=%s",
        products,
        brands,
        len(value_counts),
    )
    return payload


def build_product_graph_stats_from_neo4j(
    platform,
    graph_id: str,
    *,
    attributes: list[str] | None = None,
    generated: Path | None = None,
) -> dict[str, Any]:
    """Aggregate distinct facet values for one product graph from Neo4j.

    Attribute set comes from that graph's mapping/schema (or an explicit override) —
    never a fixed short fashion list.
    """

    settings: Settings = platform.settings
    out_dir = generated or settings.generated
    attrs = list(attributes) if attributes else _discover_stats_attributes(platform, graph_id)
    products = 0
    value_counts: dict[str, dict[str, int]] = {}
    unique_counts: dict[str, int] = {}

    session_factory = getattr(platform.graph, "_session", None)
    if callable(session_factory):
        with session_factory() as session:
            row = session.run(
                "MATCH (n:Entity:Product {graph: $g}) "
                "WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                "RETURN count(n) AS c",
                {"g": graph_id},
            ).single()
            products = int(row["c"]) if row else 0
            for attr in attrs:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(attr)):
                    continue
                # Prefer scalar toString; on TypeError (array props) fall back to UNWIND.
                try:
                    rows = list(
                        session.run(
                            f"MATCH (n:Entity:Product {{graph: $g}}) "
                            f"WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                            f"WITH coalesce(toString(n.`{attr}`), '(empty)') AS value "
                            f"RETURN value, count(*) AS n ORDER BY n DESC, value ASC",
                            {"g": graph_id},
                        )
                    )
                except Exception:
                    rows = list(
                        session.run(
                            f"MATCH (n:Entity:Product {{graph: $g}}) "
                            f"WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                            f"WITH n.`{attr}` AS raw "
                            f"UNWIND CASE "
                            f"  WHEN raw IS NULL THEN ['(empty)'] "
                            f"  ELSE raw "
                            f"END AS item "
                            f"WITH coalesce(toString(item), '(empty)') AS value "
                            f"RETURN value, count(*) AS n ORDER BY n DESC, value ASC",
                            {"g": graph_id},
                        )
                    )
                # Property may be blank when the facet lives on a relationship (e.g. HAS_BUYER).
                bucket = {str(r["value"]): int(r["n"]) for r in rows}
                empty_only = not bucket or (
                    len(bucket) == 1 and "(empty)" in bucket and products > 0
                )
                if empty_only:
                    for rel in _rel_types_for_attr(str(attr)):
                        rel_rows = list(
                            session.run(
                                f"MATCH (p:Entity:Product {{graph: $g}})-[:{rel}]->(b:Entity) "
                                "WHERE coalesce(p.lifecycle,'active') <> 'deleted' "
                                "RETURN coalesce(b.name, '(empty)') AS value, count(*) AS n "
                                "ORDER BY n DESC, value ASC",
                                {"g": graph_id},
                            )
                        )
                        if rel_rows:
                            bucket = {str(r["value"]): int(r["n"]) for r in rel_rows}
                            break
                if bucket:
                    value_counts[attr] = bucket
                    unique_counts[attr] = len(
                        [k for k in bucket if str(k).strip().casefold() not in _EMPTY_KEYS]
                    ) or len(bucket)
    else:
        nodes = platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False)
        products = len(nodes)
        for attr in attrs:
            bucket: dict[str, int] = {}
            for node in nodes:
                raw = node.properties.get(attr)
                key = str(raw).strip() if raw not in (None, "") else "(empty)"
                bucket[key] = bucket.get(key, 0) + 1
            if bucket:
                value_counts[attr] = dict(sorted(bucket.items(), key=lambda item: (-item[1], item[0])))
                unique_counts[attr] = len(bucket)

    payload = {
        "version": 1,
        "graph_id": graph_id,
        "source": "neo4j",
        "updated_at": _utc_now(),
        "totals": {
            "products": products,
            "columns_indexed": len(value_counts),
            # Keep common shortcuts when present; full map is unique_counts.
            **{
                f"{key}s" if not key.endswith("s") else key: int(n)
                for key, n in unique_counts.items()
                if key in {"brand", "buyer"} and n
            },
        },
        "unique_counts": unique_counts,
        "value_counts": value_counts,
    }
    payload = enrich_stats_payload(platform, graph_id, payload)
    save_graph_stats(graph_id, payload, out_dir)
    LOGGER.info("wrote neo4j kg stats graph=%s products=%s", graph_id, products)
    return payload


def refresh_ecom_stats(platform) -> dict[str, Any]:
    """Build ecom_products stats from Postgres when DSN is set, else Neo4j."""

    settings: Settings = platform.settings
    dsn = settings.resolve_ecom_dsn()
    if dsn:
        try:
            return build_ecom_stats_from_postgres(
                dsn, generated=settings.generated, platform=platform
            )
        except Exception as exc:
            LOGGER.warning("postgres ecom stats failed, falling back to neo4j: %s", exc)
    return build_product_graph_stats_from_neo4j(platform, "ecom_products", generated=settings.generated)


def refresh_product_graph_stats(
    platform,
    graph_id: str,
    *,
    attributes: list[str] | None = None,
) -> dict[str, Any]:
    """Write ``kg_stats/{graph_id}.json`` from the live Neo4j product graph.

    Used by mapped SQL syncs (PLMEX and any other authorized_sql product KG).
    E-commerce prefers Postgres via ``refresh_ecom_stats``; this is the
    graph-agnostic path for every other product graph.
    """

    settings: Settings = platform.settings
    return build_product_graph_stats_from_neo4j(
        platform,
        graph_id,
        attributes=attributes,
        generated=settings.generated,
    )
