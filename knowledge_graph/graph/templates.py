"""Approved graph query templates. Callers cannot submit Cypher."""

from __future__ import annotations

import re

from knowledge_graph.access import Principal, allows
from knowledge_graph.errors import NotFoundError, PermissionDenied, ValidationFailed
from knowledge_graph.models import GraphNode
from knowledge_graph.graph.snapshots import loaded_source_ids
from knowledge_graph.pipeline.service import Platform
from knowledge_graph.retrieval.queries import product_materials, products_sharing_material
from knowledge_graph.retrieval.news import search_news

_PRODUCT_TEMPLATES = {"list_products", "get_product", "product_materials", "products_sharing_material", "trace"}
TEMPLATES = {
    "own_products": set(_PRODUCT_TEMPLATES),
    "ecom_products": set(_PRODUCT_TEMPLATES),
    "plmex": set(_PRODUCT_TEMPLATES),
    "plmex_products": set(_PRODUCT_TEMPLATES),
    "news": {"trace", "news_facts"},
    "documents": {"trace"},
    "customers": {"trace"},
    "shared": {"review_queue", "products_sharing_material", "canonical_materials", "trace"},
    "master": {"sources", "trace", "news_facts", "review_queue", "products_sharing_material", "canonical_materials"},
}
NODE_ID_RE = re.compile(r"^[A-Za-z0-9:._-]{1,140}$")
RECORD_ID_RE = re.compile(r"^[A-Za-z0-9:_.-]{1,120}$")

# Provenance / internal keys omitted from product detail payloads.
_HIDDEN_PROP_PREFIXES = ("prov_",)
_HIDDEN_PROP_KEYS = {"id"}


def _page(items: list, limit: int, offset: int) -> dict:
    return {"items": items[offset : offset + limit], "total": len(items), "limit": limit, "offset": offset}


def _guard_node(node: GraphNode, principal: Principal) -> None:
    classification = node.properties.get("classification") or node.properties.get("prov_classification")
    if classification and not allows(principal, classification):
        raise PermissionDenied("The current clearance cannot view this record.")


def _templates_for(platform: Platform, graph_name: str) -> set[str]:
    """Built-in templates plus trace (and news_facts when relevant) for registry graphs."""

    known = set(TEMPLATES.get(graph_name) or ())
    if known:
        return known
    try:
        graph = platform.kg_registry.get_graph(graph_name)
    except Exception:
        return set()
    allowed = {"trace"}
    if getattr(graph, "kind", None) == "product":
        allowed.update({"list_products", "get_product", "product_materials", "products_sharing_material"})
    try:
        ontology = platform.kg_registry.load_ontology_object(graph.ontology_id)
        if "EXPORTS_TO" in ontology.relationship_allowlist or graph.ontology_id == "news":
            allowed.add("news_facts")
    except Exception:
        pass
    return allowed


def run_template(
    platform: Platform,
    graph_name: str,
    template: str,
    principal: Principal,
    *,
    limit: int,
    offset: int,
    params: dict,
) -> dict:
    allowed_templates = _templates_for(platform, graph_name)
    if template not in allowed_templates:
        raise ValidationFailed(f"Template '{template}' is not approved for graph '{graph_name}'.")
    allowed = {"record_id", "node_id"}
    if template == "news_facts":
        allowed.add("q")
    unexpected = set(params) - allowed
    if unexpected:
        raise ValidationFailed(f"Unexpected query parameters: {sorted(unexpected)}.")
    record_id = params.get("record_id")
    node_id = params.get("node_id")
    if record_id is not None and not RECORD_ID_RE.match(str(record_id)):
        raise ValidationFailed("record_id contains unsupported characters.")
    if node_id is not None and not NODE_ID_RE.match(str(node_id)):
        raise ValidationFailed("node_id contains unsupported characters.")
    if platform.degraded_reason:
        from knowledge_graph.errors import GraphBackendUnavailable

        raise GraphBackendUnavailable(platform.degraded_reason)

    if template == "list_products":
        nodes = _visible_records(platform, graph_name, "Product", principal)
        return _page([_product_summary(node) for node in nodes], limit, offset)
    if template == "get_product":
        if not node_id:
            raise ValidationFailed("get_product requires params.node_id (product id).")
        node = platform.graph.get_node(str(node_id), include_deleted=False)
        if node is None or (graph_name not in {"master", "shared"} and node.graph != graph_name):
            raise NotFoundError(f"Product '{node_id}' was not found in graph '{graph_name}'.")
        if "Product" not in (node.labels or []) and "Record" not in (node.labels or []):
            raise ValidationFailed(f"Node '{node_id}' is not a Product record.")
        _guard_node(node, principal)
        return _product_detail(platform, node)
    if template == "product_materials":
        rows = []
        for row in product_materials(platform, principal):
            if record_id and row["product"].properties.get("prov_record_id") != record_id:
                continue
            rows.append(
                {
                    "product_id": row["product"].properties.get("prov_record_id"),
                    "material": row["material"].properties.get("name"),
                    "field_ref": row["edge"].properties.get("prov_field_ref"),
                    "canonical_id": row["canonical_id"],
                    "review": row["review"],
                    "content_hash": row["edge"].properties.get("prov_content_hash"),
                }
            )
        return _page(rows, limit, offset)
    if template == "review_queue":
        rows = []
        for edge in platform.graph.find_edges(rel_type="RESOLVES_TO", include_deleted=False):
            if edge.properties.get("prov_verification_status") != "review":
                continue
            rows.append(
                {
                    "source_name": edge.properties.get("source_name"),
                    "canonical_id": edge.end_id,
                    "method": edge.properties.get("prov_match_method"),
                    "similarity": edge.properties.get("prov_similarity"),
                    "evidence": edge.properties.get("prov_evidence_text"),
                    "status": "review",
                }
            )
        return _page(rows, limit, offset)
    if template == "products_sharing_material":
        return _page(products_sharing_material(platform, principal), limit, offset)
    if template == "canonical_materials":
        nodes = platform.graph.find_nodes(graph="shared", label="Material", include_deleted=False)
        return _page([_public_node(node) for node in nodes], limit, offset)
    if template == "sources":
        return {"query_graph": "master", "sources": loaded_source_ids(platform.graph)}
    if template == "news_facts":
        rows = search_news(platform, principal, str(params.get("q") or ""), limit=offset + limit)
        if rows and rows[0].get("hidden_only"):
            return {"items": [], "total": 0, "limit": limit, "offset": offset, "note": rows[0]["line"]}
        visible = [
            {
                "relationship": row["edge"].type,
                "start": row["start_name"],
                "end": row["end_name"],
                "line": row["line"],
                "evidence": row["edge"].properties.get("prov_evidence_text"),
                "graph": "news",
            }
            for row in rows
        ]
        return _page(visible, limit, offset)
    if template == "trace":
        if not node_id:
            raise ValidationFailed("trace requires node_id.")
        node = platform.graph.get_node(str(node_id), include_deleted=True)
        if node is None:
            raise NotFoundError(f"Node {node_id} was not found.")
        if graph_name not in {"master", "shared"} and node.graph != graph_name:
            raise NotFoundError(f"Node {node_id} is not in graph {graph_name}.")
        _guard_node(node, principal)
        edges = []
        for edge in platform.graph.find_edges(include_deleted=True):
            if edge.start_id != node.id and edge.end_id != node.id:
                continue
            edges.append(
                {
                    "id": edge.id,
                    "type": edge.type,
                    "start_id": edge.start_id,
                    "end_id": edge.end_id,
                    "lifecycle": edge.properties.get("lifecycle"),
                    "verification_status": edge.properties.get("prov_verification_status"),
                    "evidence": edge.properties.get("prov_evidence_text"),
                    "source_path": edge.properties.get("prov_source_path"),
                    "content_hash": edge.properties.get("prov_content_hash"),
                    "record_id": edge.properties.get("prov_record_id"),
                }
            )
        return {"node": _public_node(node), "edges": edges}
    raise ValidationFailed(f"Template '{template}' is not implemented.")


def _visible_records(platform: Platform, graph_name: str, label: str, principal: Principal) -> list[GraphNode]:
    nodes = []
    for node in platform.graph.find_nodes(graph=graph_name, label=label, include_deleted=False):
        if "Record" in node.labels and allows(principal, node.properties.get("classification")):
            nodes.append(node)
    return nodes


def _public_node(node: GraphNode) -> dict:
    props = node.properties or {}
    return {
        "id": node.id,
        "labels": node.labels,
        "graph": node.graph,
        "name": props.get("name") or props.get("title"),
        "classification": props.get("classification"),
        "lifecycle": props.get("lifecycle"),
        # File/PLM envelopes use prov_*; e-com SQL sync stores content_hash directly.
        "source_path": props.get("prov_source_path") or props.get("source_path"),
        "content_hash": props.get("prov_content_hash") or props.get("content_hash"),
        "synthetic": props.get("synthetic", props.get("prov_synthetic")),
    }


def _product_summary(node: GraphNode) -> dict:
    """List-card fields for Product nodes (e-com + PLM)."""

    props = node.properties or {}
    base = _public_node(node)
    base.update(
        {
            "brand": props.get("brand"),
            "master_id": props.get("master_id"),
            "price": props.get("price"),
            "price_num": props.get("price_num"),
            "currency": props.get("currency"),
            "availability": props.get("availability"),
            "image": props.get("image"),
            "product_url": props.get("product_url"),
            "category_path": props.get("category_path"),
            "reference": props.get("reference") or props.get("display_reference"),
        }
    )
    return base


def _product_detail(platform: Platform, node: GraphNode) -> dict:
    """Full product properties + outgoing relationship facets for UI detail pages."""

    props = {
        key: value
        for key, value in (node.properties or {}).items()
        if key not in _HIDDEN_PROP_KEYS and not any(key.startswith(prefix) for prefix in _HIDDEN_PROP_PREFIXES)
    }
    # Keep a stable provenance block without dumping every prov_* key inline.
    provenance = {
        "source_id": (node.properties or {}).get("prov_source_id"),
        "record_id": (node.properties or {}).get("prov_record_id"),
        "source_type": (node.properties or {}).get("prov_source_type"),
        "source_path": (node.properties or {}).get("prov_source_path") or (node.properties or {}).get("source_path"),
        "content_hash": (node.properties or {}).get("prov_content_hash") or (node.properties or {}).get("content_hash"),
    }
    relationships = []
    for edge in platform.graph.find_edges(graph=node.graph, include_deleted=False):
        if edge.start_id != node.id:
            continue
        end = platform.graph.get_node(edge.end_id, include_deleted=False)
        end_props = (end.properties if end else {}) or {}
        relationships.append(
            {
                "id": edge.id,
                "type": edge.type,
                "end_id": edge.end_id,
                "end_name": end_props.get("name") or edge.end_id,
                "end_type": end_props.get("entity_type")
                or (next((label for label in (end.labels if end else []) if label not in {"Entity", "SourceEntity"}), None)),
                "properties": {
                    key: value
                    for key, value in (edge.properties or {}).items()
                    if not any(key.startswith(prefix) for prefix in _HIDDEN_PROP_PREFIXES)
                },
            }
        )
    return {
        "id": node.id,
        "labels": node.labels,
        "graph": node.graph,
        "name": props.get("name") or props.get("title") or node.id,
        "properties": props,
        "provenance": provenance,
        "relationships": relationships,
        "relationship_count": len(relationships),
    }
