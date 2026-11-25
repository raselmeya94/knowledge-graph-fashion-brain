"""Persistent runtime catalog artifacts + reverse value→schema index + term grounding.

On KG startup (or TTL refresh), profile Neo4j structure + categorical values, persist
JSON under ``settings.runtime_catalog``, and build an in-memory reverse index used to
ground user query words before Text2Cypher.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_graph.ontology import normalize_name

LOGGER = logging.getLogger("knowledge_graph.runtime_catalog")

_LOCK = threading.Lock()
_CACHE: RuntimeCatalog | None = None
_CACHE_TS: float = 0.0

_SKIP_PROPS = frozenset(
    {
        "embedding",
        "embeddings",
        "description",
        "image",
        "product_url",
        "content_hash",
        "raw",
        "id",
        "graph",
        "lifecycle",
        "classification",
        "prov_record_id",
        "prov_source_id",
        "prov_source_type",
        "source",
        "master_id",
        "reference",
        "display_reference",
        "scraped_at",
        "created_at",
        "updated_at",
        "normalized_name",
    }
)

_STOP_GRAMS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "which",
        "how",
        "many",
        "show",
        "list",
        "have",
        "does",
        "about",
        "most",
        "common",
        "are",
        "is",
        "in",
        "of",
        "to",
        "a",
        "an",
        "do",
        "we",
        "our",
        "any",
        "all",
        "by",
        "per",
    }
)

DEFAULT_FUZZY_THRESHOLD = 85.0


@dataclass
class ValueSchemaHit:
    label: str
    property: str
    exact_value: str
    graph_id: str = ""
    ontology_domain: str = ""
    count: int | None = None
    source: str = "db"  # db | ontology

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TermGroundingMap:
    """Compact user-term → schema hits for LLM prompts."""

    terms: dict[str, list[ValueSchemaHit]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    def to_prompt_dict(self, *, max_hits_per_term: int = 4) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for term, hits in self.terms.items():
            rows: list[dict[str, Any]] = []
            for hit in hits[:max_hits_per_term]:
                row = {
                    "label": hit.label,
                    "property": hit.property,
                    "exact_value": hit.exact_value,
                }
                if hit.graph_id:
                    row["graph_id"] = hit.graph_id
                if hit.ontology_domain:
                    row["ontology_domain"] = hit.ontology_domain
                if hit.count is not None:
                    row["count"] = hit.count
                if hit.source:
                    row["source"] = hit.source
                rows.append(row)
            if rows:
                out[term] = rows
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": self.to_prompt_dict(),
            "unmatched": list(self.unmatched),
        }


@dataclass
class RuntimeCatalog:
    """In-memory catalog snapshot loaded from disk or freshly profiled."""

    manifest: dict[str, Any] = field(default_factory=dict)
    live_schema: dict[str, Any] = field(default_factory=dict)
    values_by_graph: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    reverse_index: dict[str, list[ValueSchemaHit]] = field(default_factory=dict)
    built_at: float = 0.0
    source: str = "memory"
    root: str = ""

    def index_size(self) -> int:
        return len(self.reverse_index)

    def hit_count(self) -> int:
        return sum(len(v) for v in self.reverse_index.values())

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.reverse_index or self.live_schema else "empty",
            "source": self.source,
            "built_at": self.manifest.get("built_at") or self.built_at,
            "graph_ids": list(self.manifest.get("graph_ids") or list(self.values_by_graph)),
            "index_keys": self.index_size(),
            "index_hits": self.hit_count(),
            "root": self.root,
            "ttl_seconds": self.manifest.get("ttl_seconds"),
        }


def catalog_root(platform=None, settings=None) -> Path:
    cfg = settings or getattr(platform, "settings", None)
    if cfg is not None and hasattr(cfg, "runtime_catalog"):
        return Path(cfg.runtime_catalog)
    from knowledge_graph.config import DATA_ROOT

    return Path(DATA_ROOT) / "generated" / "runtime_catalog"


def _ttl_seconds(platform=None, settings=None) -> int:
    cfg = settings or getattr(platform, "settings", None)
    if cfg is not None and hasattr(cfg, "runtime_catalog_ttl"):
        return int(cfg.runtime_catalog_ttl())
    return 600


def _max_values(platform=None, settings=None) -> int:
    cfg = settings or getattr(platform, "settings", None)
    if cfg is None:
        return 500
    return max(int(getattr(cfg, "runtime_catalog_max_values_per_key", 500) or 500), 50)


def _force_refresh(platform=None, settings=None) -> bool:
    cfg = settings or getattr(platform, "settings", None)
    return bool(getattr(cfg, "runtime_catalog_force_refresh", False)) if cfg else False


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def invalidate_runtime_catalog() -> None:
    global _CACHE, _CACHE_TS
    with _LOCK:
        _CACHE = None
        _CACHE_TS = 0.0


def get_runtime_catalog(platform=None, *, force: bool = False) -> RuntimeCatalog:
    """Return cached catalog, loading/profiling when missing or stale."""

    global _CACHE, _CACHE_TS
    now = time.monotonic()
    ttl = _ttl_seconds(platform)
    with _LOCK:
        if (
            not force
            and not _force_refresh(platform)
            and _CACHE is not None
            and (now - _CACHE_TS) < ttl
        ):
            return _CACHE
    catalog = warm_runtime_catalog(platform, force=force or _force_refresh(platform), _as_object=True)
    assert isinstance(catalog, RuntimeCatalog)
    return catalog


def warm_runtime_catalog(
    platform=None,
    *,
    force: bool = False,
    _as_object: bool = False,
) -> dict[str, Any] | RuntimeCatalog:
    """Load fresh disk artifacts or profile Neo4j, persist, build reverse index."""

    global _CACHE, _CACHE_TS
    root = catalog_root(platform)
    root.mkdir(parents=True, exist_ok=True)
    ttl = _ttl_seconds(platform)
    force = bool(force or _force_refresh(platform))

    catalog: RuntimeCatalog | None = None
    if not force:
        catalog = _load_from_disk(root, ttl=ttl)
    if catalog is None:
        catalog = _profile_and_persist(platform, root=root, ttl=ttl)

    with _LOCK:
        _CACHE = catalog
        _CACHE_TS = time.monotonic()

    report = catalog.status()
    report["force"] = force
    LOGGER.info(
        "runtime catalog warm source=%s keys=%s graphs=%s",
        report.get("source"),
        report.get("index_keys"),
        report.get("graph_ids"),
    )
    if _as_object:
        return catalog
    return report


# --------------------------------------------------------------------------------------
# Disk I/O
# --------------------------------------------------------------------------------------


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _schema_path(root: Path) -> Path:
    return root / "live_schema.json"


def _values_dir(root: Path) -> Path:
    return root / "values"


def _values_path(root: Path, graph_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", graph_id or "unknown")
    return _values_dir(root) / f"{safe}.json"


def _load_from_disk(root: Path, *, ttl: int) -> RuntimeCatalog | None:
    man_path = _manifest_path(root)
    schema_path = _schema_path(root)
    if not man_path.is_file() or not schema_path.is_file():
        return None
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        built_at = float(manifest.get("built_epoch") or 0)
        if built_at <= 0:
            return None
        age = time.time() - built_at
        if age > max(ttl, 60):
            LOGGER.info("runtime catalog disk stale age=%.0fs ttl=%s", age, ttl)
            return None
        live_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        values_by_graph: dict[str, dict[str, list[dict[str, Any]]]] = {}
        vdir = _values_dir(root)
        if vdir.is_dir():
            for path in vdir.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                gid = str(payload.get("graph_id") or path.stem)
                values_by_graph[gid] = dict(payload.get("values") or {})
        catalog = RuntimeCatalog(
            manifest=manifest,
            live_schema=live_schema,
            values_by_graph=values_by_graph,
            built_at=built_at,
            source="disk",
            root=str(root),
        )
        catalog.reverse_index = _build_reverse_index(catalog, platform=None)
        return catalog
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        LOGGER.warning("runtime catalog load failed: %s", exc)
        return None


def _persist(catalog: RuntimeCatalog, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _values_dir(root).mkdir(parents=True, exist_ok=True)
    _schema_path(root).write_text(
        json.dumps(catalog.live_schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for graph_id, values in catalog.values_by_graph.items():
        payload = {"graph_id": graph_id, "values": values, "updated_at": _utc_now()}
        _values_path(root, graph_id).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    catalog.manifest = {
        **dict(catalog.manifest or {}),
        "built_at": _utc_now(),
        "built_epoch": catalog.built_at or time.time(),
        "ttl_seconds": catalog.manifest.get("ttl_seconds"),
        "graph_ids": list(catalog.values_by_graph),
        "index_keys": catalog.index_size(),
        "index_hits": catalog.hit_count(),
        "source": catalog.source,
        "version": 1,
    }
    _manifest_path(root).write_text(
        json.dumps(catalog.manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Debug artifact (optional); memory remains source of truth.
    try:
        compact = {
            key: [h.to_dict() for h in hits[:8]]
            for key, hits in list(catalog.reverse_index.items())[:2000]
        }
        (root / "reverse_index_sample.json").write_text(
            json.dumps({"keys": catalog.index_size(), "sample": compact}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


# --------------------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------------------


def _profile_and_persist(platform, *, root: Path, ttl: int) -> RuntimeCatalog:
    live_schema = _profile_live_schema(platform)
    graph_ids = list(live_schema.get("graph_values") or [])
    if not graph_ids and platform is not None:
        try:
            from knowledge_graph.retrieval.runtime_schema import build_runtime_schema

            ctx = build_runtime_schema(platform, None)
            graph_ids = [g.graph_id for g in (ctx.graphs or []) if g.graph_id and g.status == "active"]
        except Exception:
            graph_ids = []
    if not graph_ids:
        try:
            from knowledge_graph.retrieval.ecom import registry_product_graphs

            graph_ids = list(registry_product_graphs(platform) or [])
        except Exception:
            graph_ids = []
    if not graph_ids:
        # Last resort: live schema graph property values already tried above.
        LOGGER.warning("runtime_catalog: no product graph ids discovered; skipping value profile")
        graph_ids = []

    values_by_graph: dict[str, dict[str, list[dict[str, Any]]]] = {}
    max_n = _max_values(platform)
    product_props = {
        str(p).strip()
        for p in (live_schema.get("node_properties") or {}).get("Product") or []
        if str(p).strip()
    }
    for gid in graph_ids:
        values_by_graph[gid] = _profile_graph_values(
            platform, gid, max_n=max_n, product_props=product_props
        )

    catalog = RuntimeCatalog(
        manifest={"ttl_seconds": ttl, "graph_ids": graph_ids},
        live_schema=live_schema,
        values_by_graph=values_by_graph,
        built_at=time.time(),
        source="neo4j",
        root=str(root),
    )
    catalog.reverse_index = _build_reverse_index(catalog, platform=platform)
    try:
        _persist(catalog, root)
    except OSError as exc:
        LOGGER.warning("runtime catalog persist failed: %s", exc)
    return catalog


def _profile_live_schema(platform) -> dict[str, Any]:
    snap_dict: dict[str, Any] = {}
    try:
        from knowledge_graph.retrieval.live_schema import build_live_schema

        snap = build_live_schema(platform) if platform is not None else None
        if snap is not None:
            snap_dict = snap.to_dict()
    except Exception as exc:
        LOGGER.warning("live schema profile failed: %s", exc)

    directions = _relationship_directions(platform)
    snap_dict["relationship_directions"] = directions
    snap_dict["profiled_at"] = _utc_now()
    return snap_dict


def _relationship_directions(platform) -> list[dict[str, str]]:
    """Ontology allowlist pairs first; sample Neo4j when empty."""

    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        from knowledge_graph.ontology_registry import get_ontology_registry

        reg = get_ontology_registry(platform)
        allow = reg.allowlist_for_domains() if reg else {}
        for rel, pairs in (allow or {}).items():
            for pair in pairs or []:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                start, end = str(pair[0]), str(pair[1])
                key = (start, str(rel).upper(), end)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"start": start, "type": str(rel).upper(), "end": end, "source": "ontology"})
    except Exception as exc:
        LOGGER.debug("ontology directions skipped: %s", exc)

    if out or platform is None:
        return out[:200]

    session_factory = getattr(getattr(platform, "graph", None), "_session", None)
    if not callable(session_factory):
        return out
    try:
        with session_factory() as session:
            rows = list(
                session.run(
                    "MATCH (a)-[r]->(b) "
                    "WITH type(r) AS t, labels(a)[0] AS start, labels(b)[0] AS end, count(*) AS c "
                    "WHERE start IS NOT NULL AND end IS NOT NULL "
                    "RETURN t, start, end, c ORDER BY c DESC LIMIT 50"
                )
            )
            for row in rows:
                start = str(row.get("start") or "")
                end = str(row.get("end") or "")
                rel = str(row.get("t") or "")
                key = (start, rel, end)
                if not start or not end or not rel or key in seen:
                    continue
                seen.add(key)
                out.append({"start": start, "type": rel, "end": end, "source": "neo4j_sample"})
    except Exception as exc:
        LOGGER.warning("neo4j direction sample failed: %s", exc)
    return out[:200]


def _profile_graph_values(
    platform,
    graph_id: str,
    *,
    max_n: int,
    product_props: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = {}

    # Prefer kg_stats value_counts when available.
    try:
        from knowledge_graph.retrieval.kg_stats import load_graph_stats

        generated = getattr(getattr(platform, "settings", None), "generated", None) if platform else None
        stats = load_graph_stats(graph_id, generated)
        buckets = dict((stats or {}).get("value_counts") or {})
        for attr, bucket in buckets.items():
            key = f"Product.{attr}"
            rows = _bucket_to_rows(bucket, max_n=max_n)
            if rows:
                values[key] = rows
    except Exception as exc:
        LOGGER.debug("kg_stats values skipped for %s: %s", graph_id, exc)

    if platform is None:
        return values

    session_factory = getattr(getattr(platform, "graph", None), "_session", None)
    if not callable(session_factory):
        return values

    # Product categorical props not already covered — only query props that exist
    # on Product in the live schema (avoids Neo4j UnknownPropertyKeyWarning noise).
    prop_candidates = [
        "brand",
        "buyer",
        "availability",
        "currency",
        "category_path",
        "fit",
        "construction",
        "country_of_origin",
        "section",
        "family",
        "subfamily",
        "composition",
    ]
    known = {p for p in (product_props or set()) if p}
    if known:
        prop_candidates = [p for p in prop_candidates if p in known]
    try:
        with session_factory() as session:
            for prop in prop_candidates:
                key = f"Product.{prop}"
                if key in values and len(values[key]) >= min(20, max_n // 5):
                    continue
                try:
                    rows = list(
                        session.run(
                            f"MATCH (p:Product) WHERE p.graph = $g "
                            f"AND coalesce(p.`{prop}`, '') <> '' "
                            f"RETURN toString(p.`{prop}`) AS value, count(*) AS c "
                            f"ORDER BY c DESC LIMIT $lim",
                            {"g": graph_id, "lim": max_n},
                        )
                    )
                except Exception:
                    continue
                packed = [
                    {"value": str(r["value"]), "count": int(r["c"] or 0)}
                    for r in rows
                    if r.get("value") not in (None, "")
                ]
                if packed:
                    values[key] = packed

            # Facet entity names via relationship map.
            try:
                from knowledge_graph.retrieval.ecom import _REL_FIELDS
            except Exception:
                _REL_FIELDS = {}
            label_guess = {
                "brand": "Brand",
                "buyer": "Buyer",
                "material": "Material",
                "color": "Color",
                "garment_type": "GarmentType",
                "category": "Category",
                "gender": "Gender",
                "availability": "Availability",
                "plm_main_category": "PlmMainCategory",
                "plm_sub_category": "PlmSubCategory",
                "plm_sub_sub_category": "PlmSubSubCategory",
            }
            for attr, rels in (_REL_FIELDS or {}).items():
                label = label_guess.get(attr) or attr.replace("_", " ").title().replace(" ", "")
                key = f"{label}.name"
                if key in values and len(values[key]) >= min(20, max_n // 5):
                    continue
                for rel in rels or ():
                    try:
                        rows = list(
                            session.run(
                                "MATCH (p:Product)-[r]->(x:Entity) "
                                "WHERE p.graph = $g AND type(r) = $rel "
                                "AND coalesce(x.name, '') <> '' "
                                "RETURN toString(x.name) AS value, count(*) AS c "
                                "ORDER BY c DESC LIMIT $lim",
                                {"g": graph_id, "rel": str(rel), "lim": max_n},
                            )
                        )
                    except Exception:
                        continue
                    packed = [
                        {"value": str(r["value"]), "count": int(r["c"] or 0)}
                        for r in rows
                        if r.get("value") not in (None, "")
                    ]
                    if packed:
                        values[key] = packed
                        break
    except Exception as exc:
        LOGGER.warning("neo4j value profile failed for %s: %s", graph_id, exc)

    return values


def _bucket_to_rows(bucket: dict[str, Any], *, max_n: int) -> list[dict[str, Any]]:
    items = []
    for name, count in (bucket or {}).items():
        text = str(name or "").strip()
        if not text or text.casefold() in {"", "(empty)", "(unspecified)", "n/a", "na", "none", "null"}:
            continue
        try:
            n = int(count)
        except (TypeError, ValueError):
            n = 0
        items.append((n, text))
    items.sort(key=lambda item: (-item[0], item[1].casefold()))
    return [{"value": text, "count": n} for n, text in items[:max_n]]


# --------------------------------------------------------------------------------------
# Reverse index
# --------------------------------------------------------------------------------------


def _build_reverse_index(catalog: RuntimeCatalog, *, platform=None) -> dict[str, list[ValueSchemaHit]]:
    index: dict[str, list[ValueSchemaHit]] = {}

    def add(hit: ValueSchemaHit) -> None:
        key = normalize_name(hit.exact_value)
        if not key or len(key) < 2:
            return
        bucket = index.setdefault(key, [])
        sig = (hit.label, hit.property, normalize_name(hit.exact_value), hit.graph_id, hit.source)
        for existing in bucket:
            esig = (
                existing.label,
                existing.property,
                normalize_name(existing.exact_value),
                existing.graph_id,
                existing.source,
            )
            if esig == sig:
                if hit.count is not None and (existing.count is None or hit.count > existing.count):
                    existing.count = hit.count
                return
        bucket.append(hit)

    for graph_id, bags in (catalog.values_by_graph or {}).items():
        for field_key, rows in (bags or {}).items():
            label, prop = _split_field_key(field_key)
            if prop in _SKIP_PROPS:
                continue
            for row in rows or []:
                value = str((row or {}).get("value") or "").strip()
                if not value:
                    continue
                count = (row or {}).get("count")
                try:
                    count_i = int(count) if count is not None else None
                except (TypeError, ValueError):
                    count_i = None
                add(
                    ValueSchemaHit(
                        label=label,
                        property=prop,
                        exact_value=value,
                        graph_id=graph_id,
                        count=count_i,
                        source="db",
                    )
                )

    # Ontology canonical + aliases.
    try:
        from knowledge_graph.ontology_registry import get_ontology_registry

        reg = get_ontology_registry(platform)
        for surface, entries in (reg.canonical_index or {}).items():
            for entry in entries or []:
                add(
                    ValueSchemaHit(
                        label=str(getattr(entry, "entity_type", "") or "Entity"),
                        property="name",
                        exact_value=str(getattr(entry, "canonical_name", "") or surface),
                        graph_id="",
                        ontology_domain=str(
                            getattr(entry, "ontology_name", "") or getattr(entry, "ontology_id", "") or ""
                        ),
                        source="ontology",
                    )
                )
                alias = str(getattr(entry, "surface", "") or "").strip()
                if alias and normalize_name(alias) != normalize_name(
                    str(getattr(entry, "canonical_name", "") or "")
                ):
                    add(
                        ValueSchemaHit(
                            label=str(getattr(entry, "entity_type", "") or "Entity"),
                            property="alias",
                            exact_value=alias,
                            graph_id="",
                            ontology_domain=str(
                                getattr(entry, "ontology_name", "")
                                or getattr(entry, "ontology_id", "")
                                or ""
                            ),
                            source="ontology",
                        )
                    )
    except Exception as exc:
        LOGGER.debug("ontology reverse index merge skipped: %s", exc)

    return index


def _split_field_key(field_key: str) -> tuple[str, str]:
    text = str(field_key or "").strip()
    if "." in text:
        label, prop = text.split(".", 1)
        return label or "Entity", prop or "name"
    return "Product", text or "name"


# --------------------------------------------------------------------------------------
# Term grounding
# --------------------------------------------------------------------------------------


def _question_tokens(question: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (question or "").casefold()) if len(t) >= 2]


def _ngrams(tokens: list[str], *, max_n: int = 3) -> list[str]:
    out: list[str] = []
    n = len(tokens)
    for size in range(min(max_n, n), 0, -1):
        for i in range(0, n - size + 1):
            gram = " ".join(tokens[i : i + size])
            if len(gram) >= 2:
                out.append(gram)
    seen: set[str] = set()
    unique: list[str] = []
    for g in out:
        if g in seen:
            continue
        seen.add(g)
        unique.append(g)
    return unique


def _score(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz

        return float(fuzz.WRatio(a, b))
    except Exception:
        pass
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def ground_query_terms(
    question: str,
    *,
    platform=None,
    graph_ids: list[str] | None = None,
    catalog: RuntimeCatalog | None = None,
    limit: int = 40,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> TermGroundingMap:
    """Match query n-grams against the reverse index (exact then fuzzy)."""

    cat = catalog or get_runtime_catalog(platform)
    index = cat.reverse_index or {}
    if not index:
        return TermGroundingMap(unmatched=_question_tokens(question)[:12])

    want_graphs = {str(g) for g in (graph_ids or []) if g}
    tokens = _question_tokens(question)
    grams = _ngrams(tokens, max_n=3)
    terms: dict[str, list[ValueSchemaHit]] = {}
    matched: set[str] = set()
    keys = list(index.keys())

    def prefer(hits: list[ValueSchemaHit]) -> list[ValueSchemaHit]:
        if not want_graphs:
            return hits
        scoped = [h for h in hits if not h.graph_id or h.graph_id in want_graphs]
        return scoped or hits

    for gram in grams:
        if gram in _STOP_GRAMS and " " not in gram:
            continue
        nn = normalize_name(gram)
        if not nn or nn in matched:
            continue
        hits = list(index.get(nn) or [])
        score = 100.0
        if not hits:
            scored: list[tuple[float, str]] = []
            for key in keys:
                sc = _score(nn, key)
                if sc >= threshold:
                    scored.append((sc, key))
            scored.sort(key=lambda item: (-item[0], item[1]))
            for sc, key in scored[:3]:
                hits.extend(index.get(key) or [])
                score = max(score, sc) if hits else sc
        hits = prefer(hits)
        if not hits:
            continue
        # Dedup within term
        uniq: list[ValueSchemaHit] = []
        seen: set[tuple] = set()
        for hit in hits:
            sig = (hit.label, hit.property, normalize_name(hit.exact_value), hit.graph_id)
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(hit)
        terms[gram] = uniq[:6]
        matched.add(nn)
        if len(terms) >= limit:
            break

    unmatched = [
        g
        for g in grams
        if " " not in g and g not in _STOP_GRAMS and normalize_name(g) not in matched
    ]
    return TermGroundingMap(terms=terms, unmatched=unmatched[:20])


__all__ = [
    "RuntimeCatalog",
    "TermGroundingMap",
    "ValueSchemaHit",
    "catalog_root",
    "get_runtime_catalog",
    "ground_query_terms",
    "invalidate_runtime_catalog",
    "warm_runtime_catalog",
]
