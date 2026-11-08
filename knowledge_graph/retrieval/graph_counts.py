"""Fast node/edge counts for registry listings — never load every Entity into Python."""

from __future__ import annotations

from typing import Any


def graph_node_edge_counts(platform, graph_id: str) -> tuple[int, int]:
    """Return (nodes, edges) using kg_stats / Cypher when possible."""

    # Product graphs: prefer sync-time product total as a cheap signal for listings.
    try:
        from knowledge_graph.retrieval.kg_stats import load_graph_stats

        generated = getattr(getattr(platform, "settings", None), "generated", None)
        stats = load_graph_stats(graph_id, generated)
        if stats and (stats.get("totals") or {}).get("products") is not None:
            products = int(stats["totals"]["products"])
            # Full node/edge counts still come from Neo4j when available.
            nodes, edges = _cypher_counts(platform, graph_id)
            if nodes > 0:
                return nodes, edges
            return products, 0
    except Exception:
        pass
    nodes, edges = _cypher_counts(platform, graph_id)
    if nodes or edges:
        return nodes, edges
    try:
        return (
            len(platform.graph.find_nodes(graph=graph_id, include_deleted=False)),
            len(platform.graph.find_edges(graph=graph_id, include_deleted=False)),
        )
    except Exception:
        return 0, 0


def _cypher_counts(platform, graph_id: str) -> tuple[int, int]:
    session_factory = getattr(platform.graph, "_session", None)
    if not callable(session_factory):
        return 0, 0
    try:
        with session_factory() as session:
            node_row = session.run(
                "MATCH (n:Entity {graph: $g}) "
                "WHERE coalesce(n.lifecycle,'active') <> 'deleted' "
                "RETURN count(n) AS c",
                {"g": graph_id},
            ).single()
            edge_row = session.run(
                "MATCH ()-[r {graph: $g}]->() "
                "WHERE coalesce(r.lifecycle,'active') <> 'deleted' "
                "RETURN count(r) AS c",
                {"g": graph_id},
            ).single()
            return int(node_row["c"] if node_row else 0), int(edge_row["c"] if edge_row else 0)
    except Exception:
        return 0, 0


def registry_count_map(platform) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    try:
        graphs = list(platform.kg_registry.list_graphs())
    except Exception:
        return counts
    for item in graphs:
        counts[item.id] = graph_node_edge_counts(platform, item.id)
    return counts
