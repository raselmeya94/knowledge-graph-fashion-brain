"""Per-KG field catalog and value grounding for question study.

Stops one-off filter patches: the planner and router use each product graph's
unique attributes and live metadata values (brand vs buyer, category columns, …).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT
from knowledge_graph.ontology import normalize_name
from knowledge_graph.retrieval.metadata import discover_values, metadata_overview
from knowledge_graph.retrieval.products import CATALOG_FIELDS

# High-signal attributes probed when resolving free tokens in a question.
_PROBE_ATTRIBUTES = (
    "brand",
    "buyer",
    "category",
    "garment_type",
    "family",
    "subfamily",
    "material",
    "color",
    "season",
    "gender",
    "department",
    "availability",
    "section",
)
_SYSTEM_PROBE_ATTRIBUTES = (
    "topic",
    "graph_id",
    "graph_name",
    "kind",
    "capability",
    "entity",
    "attribute",
    "source_id",
)
_PRODUCT_HIT_ATTRS = frozenset(
    {"brand", "buyer", "category", "garment_type", "family", "subfamily", "material", "color", "season", "gender"}
)
SYSTEM_GRAPH_ID = "kg_system"

_TOKEN_STOP = frozenset(
    """
    analysis analyze analysing analyzing analyse profile review overview study data
    product products catalog catalogue how many are is a an the of in our with and or
    for from that this be any all number count total by per each past previous prior
    historic historical history earlier old existing type types kind kinds top bottom
    show give want please which what who when where list has have had made using use
    buyer buyers brand brands ecom ecomm ecommerce commerce plm plmex plm-ex system
    database graph graphs source sources neo4j postgres sql kg comm com
    """.split()
)

_ECOM_MAPPING = DATA_ROOT / "schemas" / "ecom_master_products_mapping.json"


@dataclass
class ValueHit:
    token: str
    attribute: str
    graph_id: str
    matched_value: str


@dataclass
class GraphFields:
    graph_id: str
    kind: str = "product"
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    analysis_fields: list[str] = field(default_factory=list)


@dataclass
class FieldCatalog:
    graphs: dict[str, GraphFields] = field(default_factory=dict)
    hits: list[ValueHit] = field(default_factory=list)

    def preferred_graphs(self) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()
        # Brand hits prefer commerce graphs; buyer hits prefer PLM graphs.
        # System hits are considered only when prefers_system_meta() is true.
        ranked = sorted(
            self.hits,
            key=lambda hit: (
                0 if hit.attribute == "brand" else 1 if hit.attribute == "buyer" else 2,
                hit.graph_id,
            ),
        )
        for hit in ranked:
            if hit.graph_id == SYSTEM_GRAPH_ID:
                continue
            if hit.graph_id not in seen:
                seen.add(hit.graph_id)
                order.append(hit.graph_id)
        return order

    def prefers_system_meta(self) -> bool:
        """True when the question grounds on kg_system and not on product party/material values.

        Driven by the system ontology aliases + live registry value index — not by
        hard-coded English question templates in the ask planner.
        """

        system_hits = [hit for hit in self.hits if hit.graph_id == SYSTEM_GRAPH_ID]
        if not system_hits:
            return False
        product_hits = [
            hit
            for hit in self.hits
            if hit.graph_id != SYSTEM_GRAPH_ID and hit.attribute in _PRODUCT_HIT_ATTRS
        ]
        return not product_hits

    def all_attributes(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for graph in self.graphs.values():
            if graph.kind == "system":
                continue
            for attr in graph.attributes:
                if attr not in seen:
                    seen.add(attr)
                    keys.append(attr)
        return keys


def build_field_catalog(platform, question: str = "") -> FieldCatalog:
    """Load unique columns per active product + system KG, then match question tokens to values."""

    try:
        from fashion_brain.trace import trace_debug, trace_step
    except Exception:  # pragma: no cover
        def trace_step(phase: str, message: str, **fields):  # type: ignore[misc]
            return None

        def trace_debug(message: str, **fields):  # type: ignore[misc]
            return None

    # Keep the system value index fresh so topic/graph_id grounding stays current.
    try:
        from knowledge_graph.retrieval.system_meta import build_system_meta

        build_system_meta(platform, persist=True)
    except Exception:
        pass

    catalog = FieldCatalog()
    rows = _registry_rows(platform)
    for row in rows:
        graph_id = str(row.get("graph_id") or row.get("id") or "")
        kind = str(row.get("kind") or "")
        # Metadata payloads may omit kind; treat Brand/buyer/product_search graphs as product KGs.
        if not kind:
            caps = " ".join(str(x) for x in (row.get("capabilities") or [])).casefold()
            ents = {str(x) for x in (row.get("entities") or [])}
            if "Product" in ents or "product_search" in caps or "brand_comparison" in caps or "buyer_analysis" in caps:
                kind = "product"
            if "system_metadata" in caps or "source_inventory" in caps or graph_id == SYSTEM_GRAPH_ID:
                kind = "system"
        if not graph_id or kind not in {"product", "system"}:
            continue
        if str(row.get("status") or "active") != "active":
            continue
        if kind == "system":
            attrs = list(_SYSTEM_PROBE_ATTRIBUTES)
            catalog.graphs[graph_id] = GraphFields(
                graph_id=graph_id,
                kind="system",
                description=str(row.get("description") or ""),
                capabilities=list(row.get("capabilities") or []),
                entities=list(row.get("entities") or []),
                attributes=attrs,
                analysis_fields=[],
            )
        else:
            attrs = _unique_attributes(platform, graph_id, row)
            catalog.graphs[graph_id] = GraphFields(
                graph_id=graph_id,
                kind="product",
                description=str(row.get("description") or ""),
                capabilities=list(row.get("capabilities") or []),
                entities=list(row.get("entities") or []),
                attributes=attrs,
                analysis_fields=_analysis_fields(attrs, row),
            )
        trace_debug("Catalog graph columns", graph_id=graph_id, kind=kind, attributes=attrs[:12])
    if question.strip():
        catalog.hits = match_question_values(platform, question, catalog)
    trace_step(
        "catalog",
        "Field catalog ready",
        graphs=list(catalog.graphs.keys()),
        hits=[f"{h.token}->{h.attribute}@{h.graph_id}" for h in (catalog.hits or [])[:8]],
        preferred=catalog.preferred_graphs(),
        system_meta=catalog.prefers_system_meta(),
    )
    return catalog


def match_question_values(platform, question: str, catalog: FieldCatalog) -> list[ValueHit]:
    """Resolve free tokens against live metadata values for each graph's attributes."""

    tokens = _candidate_tokens(question)
    hits: list[ValueHit] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(token: str, attribute: str, graph_id: str, matched: str) -> None:
        key = (graph_id, attribute, normalize_name(matched))
        if key in seen:
            return
        seen.add(key)
        hits.append(
            ValueHit(
                token=token,
                attribute=attribute,
                graph_id=graph_id,
                matched_value=matched,
            )
        )

    for graph_id, graph in catalog.graphs.items():
        if graph.kind == "system":
            # Ontology aliases (source/sources/graph/…) live in the system ontology — match phrases.
            for hit in _system_ontology_alias_hits(platform, question, graph_id):
                _add(hit.token, hit.attribute, hit.graph_id, hit.matched_value)
            probe = [attr for attr in _SYSTEM_PROBE_ATTRIBUTES if attr in graph.attributes] or list(
                _SYSTEM_PROBE_ATTRIBUTES
            )
            # Include registry words that product token stop-lists normally drop.
            system_tokens = _system_candidate_tokens(question)
            for token in system_tokens:
                for attribute in probe:
                    matched = _lookup_value(platform, graph_id, attribute, token)
                    if matched:
                        _add(token, attribute, graph_id, matched)
            continue

        if not tokens:
            continue
        probe = [attr for attr in _PROBE_ATTRIBUTES if attr in graph.attributes]
        if not probe:
            probe = list(_PROBE_ATTRIBUTES)[:6]
        for token in tokens:
            for attribute in probe:
                matched = _lookup_value(platform, graph_id, attribute, token)
                if matched:
                    _add(token, attribute, graph_id, matched)
    return hits


def catalog_selects_system_meta(catalog: FieldCatalog) -> bool:
    return bool(catalog.prefers_system_meta())


def _system_ontology_alias_hits(platform, question: str, graph_id: str) -> list[ValueHit]:
    """Match system ontology entity names/aliases inside the question text."""

    text = normalize_name(question)
    if not text:
        return []
    try:
        ontology = platform.kg_registry.load_ontology_object("system")
    except Exception:
        return []
    hits: list[ValueHit] = []
    padded = f" {text} "
    for entity in ontology.canonical_entities:
        for name in (entity.name, *(entity.aliases or [])):
            key = normalize_name(name)
            if len(key) < 2:
                continue
            if padded.find(f" {key} ") < 0 and text != key:
                continue
            hits.append(
                ValueHit(
                    token=name,
                    attribute="topic",
                    graph_id=graph_id,
                    matched_value=name,
                )
            )
    return hits


def _system_candidate_tokens(question: str) -> list[str]:
    """Tokens for system grounding — keep registry words that product stop-lists remove."""

    grammar = frozenset(
        """
        how many are is a an the of in our with and or for from that this be any all
        do we have what which who when where please could would me you your
        """.split()
    )
    tokens = re.findall(r"[a-z0-9][a-z0-9_&/-]*", (question or "").casefold())
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in grammar or len(token) < 2 or token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:16]


def ground_filters_with_catalog(filters: dict, catalog: FieldCatalog, *, question: str = "") -> dict:
    """Rewrite filters using value hits: primark as Brand → brand, never garment."""

    cleaned = {key: value for key, value in (filters or {}).items() if value not in (None, "")}
    # Drop filter values that are graph/source words, not catalog parties.
    for key in ("brand", "buyer", "garment", "garment_type", "category"):
        value = cleaned.get(key)
        if value and normalize_name(str(value)) in _TOKEN_STOP:
            cleaned.pop(key, None)

    text = (question or "").casefold()
    asks_all_parties = bool(re.search(r"\ball\s+(buyers?|brands?)\b", text))

    if not catalog.hits:
        return cleaned

    # Exact token → preferred attribute (brand beats buyer beats category/garment).
    by_token: dict[str, list[ValueHit]] = {}
    for hit in catalog.hits:
        by_token.setdefault(normalize_name(hit.token), []).append(hit)

    priority = {"brand": 0, "buyer": 1, "category": 2, "garment_type": 3, "family": 4}

    def best_hit(token_key: str) -> ValueHit | None:
        choices = by_token.get(token_key) or []
        if not choices:
            return None
        return sorted(choices, key=lambda hit: (priority.get(hit.attribute, 9), hit.graph_id))[0]

    # Remap garment/category that are actually party values in the KG.
    for key in ("garment", "garment_type", "category"):
        value = cleaned.get(key)
        if not value:
            continue
        hit = best_hit(normalize_name(str(value)))
        if hit and hit.attribute in {"brand", "buyer"}:
            cleaned.pop(key, None)
            cleaned[hit.attribute] = hit.matched_value

    # Explicit brand/buyer: prefer the catalog spelling and drop the other when exclusive.
    for attr in ("brand", "buyer"):
        value = cleaned.get(attr)
        if not value:
            continue
        hit = best_hit(normalize_name(str(value)))
        if hit and hit.attribute == attr:
            cleaned[attr] = hit.matched_value

    brand_hit = next((hit for hit in catalog.hits if hit.attribute == "brand"), None)
    buyer_hit = next((hit for hit in catalog.hits if hit.attribute == "buyer"), None)

    # Analysis party with both brand+buyer set: keep the attribute that actually exists.
    if cleaned.get("brand") and cleaned.get("buyer"):
        brand_key = normalize_name(str(cleaned["brand"]))
        buyer_key = normalize_name(str(cleaned["buyer"]))
        if brand_key == buyer_key:
            if brand_hit and not buyer_hit:
                cleaned.pop("buyer", None)
                cleaned["brand"] = brand_hit.matched_value
            elif buyer_hit and not brand_hit:
                cleaned.pop("brand", None)
                cleaned["buyer"] = buyer_hit.matched_value
            elif brand_hit:
                cleaned.pop("buyer", None)
                cleaned["brand"] = brand_hit.matched_value

    # Never invent a single-party filter for "all buyers/brands" questions.
    if asks_all_parties:
        cleaned.pop("brand", None)
        cleaned.pop("buyer", None)
        return cleaned

    # Fill missing party filter from hits when analysis left filters empty or wrong.
    if brand_hit and "brand" not in cleaned and "buyer" not in cleaned:
        cleaned["brand"] = brand_hit.matched_value
    if buyer_hit and "buyer" not in cleaned and "brand" not in cleaned:
        cleaned["buyer"] = buyer_hit.matched_value

    return cleaned


def planner_system_prompt(catalog: FieldCatalog) -> str:
    """Dynamic LLM study prompt: only keys that exist on registered product graphs."""

    lines = [
        "Convert the product question into one JSON object and nothing else.",
        "Use only filter keys that exist on the knowledge graphs below.",
        "Never put a brand or buyer name under garment, category, or garment_type.",
        "When a token matches Brand values, use filter key brand and prefer that graph.",
        "When a token matches Buyer values, use filter key buyer and prefer that graph.",
        "Words like ecom, ecomm, e-comm, ecommerce, plm, plmex name a knowledge graph — "
        "put them in preferred_graphs, never in filters.brand/buyer/garment.",
        "Questions about registered sources/graphs/attributes/shared interconnect belong to "
        f"{SYSTEM_GRAPH_ID} (system metadata). Prefer that graph and leave product filters empty.",
        '"all buyers" or "all brands" means group_by buyer or brand with empty filters '
        "(no single party value). On ecom_products use group_by brand.",
        'Shape: {"intent":"count","filters":{},"group_by":null,"limit":null,"season_name":null,'
        '"upcoming":false,"references_prior_chat":false,"preferred_graphs":[]}.'
        ' intent is "count", "analysis", "specification", or "unsupported".',
        "Use analysis for analysis/analyze/profile/review/overview/study of stored products.",
        "preferred_graphs lists graph_id values that own the matched filter values "
        "or that the user named (ecom_products, own_products, kg_system).",
        "Copy filter values only from the user's words (or the matched catalog spelling).",
        "Set references_prior_chat true only for follow-ups that depend on earlier turns.",
    ]
    if catalog.graphs:
        lines.append("Registered knowledge graphs and their unique filter / index columns:")
        for graph in catalog.graphs.values():
            attrs = ", ".join(graph.attributes[:40]) or "(none discovered)"
            caps = ", ".join(graph.capabilities) or "none"
            lines.append(
                f"- {graph.graph_id} [{graph.kind}]: columns=[{attrs}]; capabilities=[{caps}]; "
                f"entities={graph.entities}; {graph.description}".strip()
            )
    if catalog.hits:
        lines.append("Live value matches for tokens in the latest question:")
        for hit in catalog.hits[:24]:
            lines.append(
                f'- token "{hit.token}" → {hit.attribute}="{hit.matched_value}" on graph {hit.graph_id}'
            )
    return "\n".join(lines)


def analysis_fields_for_graph(catalog: FieldCatalog, graph_id: str) -> list[str]:
    graph = catalog.graphs.get(graph_id)
    if graph and graph.analysis_fields:
        return list(graph.analysis_fields)
    return []


def _registry_rows(platform) -> list[dict]:
    from fashion_brain.execution.kg_client import list_registry, uses_kg_api

    if uses_kg_api(platform.settings):
        payload = list_registry(platform.settings)
        rows = payload.get("registry") or payload.get("graphs") or []
        if rows:
            return [dict(row) for row in rows]
    try:
        overview = metadata_overview(platform)
        rows = overview.get("graphs") if isinstance(overview, dict) else None
        if rows:
            return [dict(row) for row in rows]
    except Exception:
        pass
    try:
        graphs = list(platform.kg_registry.list_graphs())
    except Exception:
        return []
    out = []
    for item in graphs:
        out.append(
            {
                "graph_id": item.id,
                "id": item.id,
                "kind": item.kind,
                "status": getattr(item, "status", "active"),
                "description": item.description,
                "capabilities": list(item.capabilities or []),
                "entities": list(item.entities or []),
                "source_ids": list(item.source_ids or []),
                "schema": {},
            }
        )
    return out


def _unique_attributes(platform, graph_id: str, row: dict) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        key = str(name or "").strip().lower().replace(" ", "_")
        if not key or key in seen or key in {"id", "name", "labels", "graph"}:
            return
        seen.add(key)
        keys.append(key)

    schema = row.get("schema") or {}
    product_schema = schema.get("Product") if isinstance(schema, dict) else None
    if isinstance(product_schema, dict):
        for attr in (product_schema.get("attributes") or {}):
            add(attr)

    for entity in row.get("entities") or []:
        mapped = {
            "Brand": "brand",
            "Material": "material",
            "GarmentCategory": "category",
            "Color": "color",
            "Season": "season",
            "Supplier": "supplier",
            "Company": "buyer",
            "Availability": "availability",
            "Section": "section",
            "Currency": "currency",
        }.get(str(entity))
        if mapped:
            add(mapped)

    for cap in row.get("capabilities") or []:
        text = str(cap).casefold()
        if "brand" in text:
            add("brand")
        if "buyer" in text:
            add("buyer")
        if "price" in text:
            add("price")
        if "material" in text:
            add("material")

    if graph_id == "ecom_products" or "ecom" in graph_id:
        for key in _ecom_mapping_keys():
            add(key)
        for key in ("brand", "category", "family", "subfamily", "color", "material", "availability", "section", "price"):
            add(key)
    if graph_id == "own_products" or "plm" in graph_id or "own" in graph_id:
        for field in CATALOG_FIELDS:
            add(field.key)

    # Sample live product property keys when the local graph is loaded.
    try:
        nodes = platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False)
        for node in nodes[:80]:
            for key in (node.properties or {}):
                add(key)
    except Exception:
        pass

    # Remote metadata overview may already expose schema attributes.
    try:
        from fashion_brain.execution.kg_client import kg_request, uses_kg_api

        if uses_kg_api(platform.settings):
            remote = kg_request(
                platform.settings,
                "GET",
                "/api/metadata",
                params={"graph_id": graph_id},
            )
            remote_schema = (remote or {}).get("schema") or {}
            product = remote_schema.get("Product") if isinstance(remote_schema, dict) else None
            if isinstance(product, dict):
                for attr in (product.get("attributes") or {}):
                    add(attr)
    except Exception:
        pass

    return keys


def _ecom_mapping_keys() -> list[str]:
    if not _ECOM_MAPPING.exists():
        return []
    try:
        payload = json.loads(_ECOM_MAPPING.read_text(encoding="utf-8"))
    except Exception:
        return []
    props = payload.get("product_properties") or {}
    return [str(key) for key in props.keys()]


def _analysis_fields(attributes: list[str], row: dict) -> list[str]:
    preferred = [
        "brand",
        "buyer",
        "category",
        "family",
        "subfamily",
        "garment_type",
        "material",
        "color",
        "gender",
        "season",
        "department",
        "availability",
        "section",
        "price",
        "gauge",
        "gsm",
    ]
    present = set(attributes)
    chosen = [key for key in preferred if key in present]
    if not chosen:
        return [key for key in attributes if key not in {"product_id", "name", "content_hash"}][:12]
    return chosen


def _candidate_tokens(question: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9_&/-]*", (question or "").casefold())
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _TOKEN_STOP or len(token) < 2 or token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:12]


def _lookup_value(platform, graph_id: str, attribute: str, token: str) -> str | None:
    needle = normalize_name(token)
    if not needle:
        return None
    values = _values_for(platform, graph_id, attribute, token)
    exact = next((value for value in values if normalize_name(value) == needle), None)
    if exact:
        return exact
    # Partial contains only for longer tokens (avoid "an" matching "organic").
    if len(needle) >= 4:
        soft = next((value for value in values if needle in normalize_name(value)), None)
        if soft:
            return soft
    return None


def _values_for(platform, graph_id: str, attribute: str, token: str) -> list[str]:
    from fashion_brain.execution.kg_client import metadata_values, uses_kg_api

    if uses_kg_api(platform.settings):
        payload = metadata_values(
            platform.settings,
            attribute=attribute,
            graph_id=graph_id,
            q=token,
            limit=50,
        )
        return [str(item) for item in (payload.get("values") or []) if str(item).strip()]
    try:
        payload = discover_values(
            platform,
            graph_id=graph_id,
            attribute=attribute,
            q=token,
            limit=50,
        )
        return [str(item) for item in (payload.get("values") or []) if str(item).strip()]
    except Exception:
        return []
