"""Runtime schema context built from kg_registry + ontology (no source hardcodes)."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from knowledge_graph.ontology import STRUCTURAL_LABELS, normalize_name

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, "RuntimeSchemaContext"]] = {}
_CACHE_TTL_SEC = 60.0


@dataclass
class GraphRuntimeInfo:
    graph_id: str
    name: str = ""
    kind: str = ""
    ontology_id: str = ""
    description: str = ""
    status: str = "active"
    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)


@dataclass
class RuntimeSchemaContext:
    """Everything agents need for one (or all) product graphs — API-derived only."""

    graphs: list[GraphRuntimeInfo] = field(default_factory=list)
    graph_id: str | None = None
    ontology_id: str | None = None
    ontology_name: str = ""
    entity_types: list[str] = field(default_factory=list)
    relationship_allowlist: dict[str, list[list[str]]] = field(default_factory=dict)
    product_label: str = "Product"
    facet_types: list[str] = field(default_factory=list)
    filter_keys: list[str] = field(default_factory=list)
    scope_tokens: list[str] = field(default_factory=list)
    canonical_hints: list[dict[str, Any]] = field(default_factory=list)
    brand_label: str | None = None
    product_properties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graphs": [asdict(g) for g in self.graphs],
            "graph_id": self.graph_id,
            "ontology_id": self.ontology_id,
            "ontology_name": self.ontology_name,
            "entity_types": self.entity_types,
            "relationship_allowlist": self.relationship_allowlist,
            "product_label": self.product_label,
            "facet_types": self.facet_types,
            "filter_keys": self.filter_keys,
            "scope_tokens": self.scope_tokens,
            "canonical_hints": self.canonical_hints,
            "brand_label": self.brand_label,
            "product_properties": self.product_properties,
        }

    def schema_card(self) -> str:
        rel_lines = []
        for name, pairs in (self.relationship_allowlist or {}).items():
            for pair in pairs or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    rel_lines.append(f"- (:{pair[0]})-[:{name}]->(:{pair[1]})")
        brands = [h for h in self.canonical_hints if str(h.get("type") or "").casefold() == "brand"]
        brand_block = ""
        if brands:
            brand_block = "Canonical brands (from ontology):\n" + "\n".join(
                f"- {b.get('name')} aliases={b.get('aliases') or []}" for b in brands[:20]
            )
        return (
            f"Target graph_id: {self.graph_id}\n"
            f"Ontology: {self.ontology_name or self.ontology_id}\n"
            f"Product label: {self.product_label}\n"
            f"Facet types: {', '.join(self.facet_types)}\n"
            f"Filter keys: {', '.join(self.filter_keys)}\n"
            f"Product properties: {', '.join(self.product_properties) or 'id, name, graph, lifecycle'}\n"
            "Relationships:\n"
            + ("\n".join(rel_lines) if rel_lines else "- (none registered)")
            + "\n"
            + (brand_block + "\n" if brand_block else "")
            + "Match facets by id IN $…_ids from resolved entities. Never invent names."
        )


def invalidate_runtime_schema_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def default_product_graph_id(platform) -> str | None:
    """First active product-kind graph from the registry."""

    for info in _registry_graphs(platform):
        if info.kind == "product" and info.status == "active":
            return info.graph_id
    return None


def build_runtime_schema(platform, graph_id: str | None = None) -> RuntimeSchemaContext:
    """Build (or return cached) runtime schema for a graph, or all product graphs overview."""

    graphs = _registry_graphs(platform)
    target = graph_id or default_product_graph_id(platform)
    cache_key = target or "__all__"
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(cache_key)
        if hit and (now - hit[0]) < _CACHE_TTL_SEC:
            return hit[1]

    ctx = RuntimeSchemaContext(graphs=graphs, graph_id=target)
    info = next((g for g in graphs if g.graph_id == target), None) if target else None
    ontology_id = info.ontology_id if info else None
    ontology = _load_ontology(platform, ontology_id) if ontology_id else None

    if ontology:
        ctx.ontology_id = ontology_id
        ctx.ontology_name = str(ontology.get("name") or ontology_id or "")
        entity_types = [str(x) for x in (ontology.get("entity_types") or []) if x]
        allow = ontology.get("relationship_allowlist") or {}
        if not isinstance(allow, dict):
            allow = {}
        ctx.entity_types = entity_types
        ctx.relationship_allowlist = {
            str(k): [list(p) for p in (v or []) if isinstance(p, (list, tuple)) and len(p) == 2]
            for k, v in allow.items()
        }
        ctx.product_label = _infer_product_label(entity_types, ctx.relationship_allowlist)
        structural = {x.casefold() for x in STRUCTURAL_LABELS}
        ctx.facet_types = [
            t for t in entity_types if t.casefold() not in structural and t != ctx.product_label
        ]
        ctx.brand_label = next((t for t in ctx.facet_types if t.casefold() == "brand"), None)
        if ctx.brand_label is None:
            # Infer from HAS_BRAND end type if present.
            for rel, pairs in ctx.relationship_allowlist.items():
                if "brand" in rel.casefold():
                    for start, end in pairs:
                        if end in ctx.facet_types:
                            ctx.brand_label = end
                            break
                if ctx.brand_label:
                    break
        ctx.filter_keys = sorted({_facet_filter_key(t) for t in ctx.facet_types})
        canonical = ontology.get("canonical_entities") or []
        ctx.canonical_hints = []
        for item in canonical:
            if hasattr(item, "model_dump"):
                ctx.canonical_hints.append(item.model_dump())
            elif isinstance(item, dict):
                ctx.canonical_hints.append(dict(item))
        ctx.product_properties = [
            "id",
            "name",
            "brand",
            "price",
            "price_num",
            "original_price",
            "original_price_num",
            "currency",
            "availability",
            "category_path",
            "graph",
            "lifecycle",
            "source_id",
            "normalized_name",
        ]
    elif info:
        ctx.entity_types = list(info.entities or [])
        ctx.facet_types = [t for t in ctx.entity_types if t not in STRUCTURAL_LABELS and t != "Product"]
        ctx.product_label = "Product" if "Product" in ctx.entity_types else (ctx.entity_types[0] if ctx.entity_types else "Product")
        ctx.filter_keys = sorted({_facet_filter_key(t) for t in ctx.facet_types})
        ctx.brand_label = next((t for t in ctx.facet_types if t.casefold() == "brand"), None)

    ctx.scope_tokens = _scope_tokens(graphs, target)
    with _CACHE_LOCK:
        _CACHE[cache_key] = (now, ctx)
    return ctx


def ensure_facet_fulltext_indexes(platform, facet_types: list[str]) -> list[str]:
    """Create FULLTEXT + RANGE(normalized_name) indexes for facet labels."""

    from knowledge_graph.graph.constraints import apply_schema_statements, facet_index_statements

    session_factory = getattr(getattr(platform, "graph", None), "_session", None)
    if not callable(session_factory):
        return []
    statements = facet_index_statements(facet_types)
    if not statements:
        return []
    with session_factory() as session:
        return apply_schema_statements(session, statements)


def _facet_filter_key(label: str) -> str:
    text = str(label or "")
    # GarmentCategory → category; common shortenings
    mapping = {
        "garmentcategory": "category",
        "garment": "category",
    }
    key = text.casefold()
    if key in mapping:
        return mapping[key]
    return key


def _infer_product_label(entity_types: list[str], allowlist: dict[str, list[list[str]]]) -> str:
    if "Product" in entity_types:
        return "Product"
    counts: dict[str, int] = {t: 0 for t in entity_types}
    for pairs in allowlist.values():
        for pair in pairs:
            if pair and pair[0] in counts:
                counts[pair[0]] += 1
    if counts:
        return max(counts.items(), key=lambda item: item[1])[0]
    return entity_types[0] if entity_types else "Product"


def _scope_tokens(graphs: list[GraphRuntimeInfo], target: str | None) -> list[str]:
    """Tokens from graph names/descriptions/ids — used to ignore catalog-scope words."""

    tokens: set[str] = set()
    generic = {
        "how",
        "many",
        "the",
        "a",
        "an",
        "and",
        "or",
        "in",
        "of",
        "to",
        "for",
        "from",
        "with",
        "our",
        "is",
        "are",
        "found",
        "product",
        "products",
        "knowledge",
        "graph",
        "graphs",
        "source",
        "sources",
        "data",
        "contains",
        "information",
        "multiple",
    }
    selected = [g for g in graphs if not target or g.graph_id == target or g.kind == "product"]
    for g in selected:
        blob = " ".join([g.graph_id, g.name, g.description, " ".join(g.source_ids or [])])
        nn = normalize_name(blob)
        for tok in nn.split():
            if len(tok) >= 2 and tok not in generic:
                tokens.add(tok)
        # Hyphen/underscore collapsed forms
        compact = nn.replace(" ", "")
        if compact:
            tokens.add(compact)
        # id slug parts: ecom_products → ecom, products
        for part in re.split(r"[_\-\s]+", g.graph_id):
            p = normalize_name(part)
            if p and p not in generic:
                tokens.add(p)
    return sorted(tokens)


def _registry_graphs(platform) -> list[GraphRuntimeInfo]:
    out: list[GraphRuntimeInfo] = []
    try:
        rows = list(platform.kg_registry.list_graphs())
    except Exception:
        return out
    for item in rows:
        out.append(
            GraphRuntimeInfo(
                graph_id=str(getattr(item, "id", "") or ""),
                name=str(getattr(item, "name", "") or ""),
                kind=str(getattr(item, "kind", "") or ""),
                ontology_id=str(getattr(item, "ontology_id", "") or ""),
                description=str(getattr(item, "description", "") or ""),
                status=str(getattr(item, "status", "active") or "active"),
                entities=[str(x) for x in (getattr(item, "entities", None) or [])],
                relationships=[str(x) for x in (getattr(item, "relationships", None) or [])],
                capabilities=[str(x) for x in (getattr(item, "capabilities", None) or [])],
                source_ids=[str(x) for x in (getattr(item, "source_ids", None) or [])],
            )
        )
    return out


def _load_ontology(platform, ontology_id: str | None) -> dict[str, Any] | None:
    if not ontology_id:
        return None
    try:
        obj = platform.kg_registry.load_ontology_object(ontology_id)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None
