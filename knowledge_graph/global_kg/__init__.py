"""Global / shared interconnect facade (canonical entities + RESOLVES_TO)."""

from __future__ import annotations

from typing import Any


def global_info(platform) -> dict[str, Any]:
    """Describe the Global KG (shared + master interconnect)."""

    shared_nodes = platform.graph.find_nodes(graph="shared", include_deleted=False)
    shared_edges = platform.graph.find_edges(graph="shared", include_deleted=False)
    master_nodes = platform.graph.find_nodes(graph="master", include_deleted=False)
    master_edges = platform.graph.find_edges(graph="master", include_deleted=False)
    return {
        "graph_id": "global",
        "layers": {
            "shared": {
                "description": "Canonical entities and RESOLVES_TO links across individual KGs",
                "node_count": len(shared_nodes),
                "edge_count": len(shared_edges),
            },
            "master": {
                "description": "Persisted shared interconnect snapshot for warm query load",
                "node_count": len(master_nodes),
                "edge_count": len(master_edges),
            },
        },
        "capabilities": [
            "canonical_materials",
            "products_sharing_material",
            "cross_source_resolve",
        ],
        "note": "Query via POST /api/graphs/cross-query or POST /graphs/shared/query with approved templates.",
    }
