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

TEMPLATES = {
    "own_products": {"list_products", "product_materials", "products_sharing_material", "trace"},
    "ecom_products": {"list_products", "product_materials", "products_sharing_material", "trace"},
    "news": {"trace", "news_facts"},
    "documents": {"trace"},
    "customers": {"trace"},
    "shared": {"review_queue", "products_sharing_material", "canonical_materials", "trace"},
    "master": {"sources", "trace", "news_facts", "review_queue", "products_sharing_material", "canonical_materials"},
}
NODE_ID_RE = re.compile(r"^[A-Za-z0-9:._-]{1,140}$")
RECORD_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


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
        return _page([_public_node(node) for node in nodes], limit, offset)
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
    return {
        "id": node.id,
        "labels": node.labels,
        "graph": node.graph,
        "name": node.properties.get("name") or node.properties.get("title"),
        "classification": node.properties.get("classification"),
        "lifecycle": node.properties.get("lifecycle"),
        "source_path": node.properties.get("prov_source_path"),
        "content_hash": node.properties.get("prov_content_hash"),
        "synthetic": node.properties.get("synthetic", node.properties.get("prov_synthetic")),
    }
