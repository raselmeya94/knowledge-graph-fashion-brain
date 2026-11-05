"""In-memory graph used by tests and the default local backend."""

from __future__ import annotations

from knowledge_graph.errors import RelationshipValidationError
from knowledge_graph.graph.constraints import SCHEMA_STATEMENTS
from knowledge_graph.models import GraphEdge, GraphNode
from knowledge_graph.ontology import Ontology, validate_relationship


class MemoryGraphRepository:
    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self.edges_from: dict[str, set[str]] = {}
        self.edges_to: dict[str, set[str]] = {}
        self.applied_schema: list[str] = []

    def ping(self) -> str:
        return "ok"

    def ensure_constraints(self) -> list[str]:
        if not self.applied_schema:
            self.applied_schema = list(SCHEMA_STATEMENTS)
        ids = [node.id for node in self.nodes.values()]
        if len(ids) != len(set(ids)):
            raise RelationshipValidationError("Duplicate node ids violate the entity_id constraint.")
        return list(self.applied_schema)

    def upsert_node(self, node: GraphNode) -> GraphNode:
        stored = node.model_copy(deep=True)
        if "lifecycle" not in stored.properties:
            stored.properties["lifecycle"] = "active"
        self.nodes[stored.id] = stored
        return stored.model_copy(deep=True)

    def get_node(self, node_id: str, *, include_deleted: bool = True) -> GraphNode | None:
        node = self.nodes.get(node_id)
        if node is None:
            return None
        if not include_deleted and node.properties.get("lifecycle") == "deleted":
            return None
        return node.model_copy(deep=True)

    def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        start = self.nodes.get(edge.start_id)
        end = self.nodes.get(edge.end_id)
        if start is None or end is None:
            raise RelationshipValidationError(
                f"Cannot create {edge.type} from {edge.start_id} to {edge.end_id}: both nodes must exist."
            )
        validate_relationship(edge.type, start.labels, end.labels, self.ontology)
        stored = edge.model_copy(deep=True)
        if "lifecycle" not in stored.properties:
            stored.properties["lifecycle"] = "active"
        previous = self.edges.get(stored.id)
        if previous is not None:
            self._unindex_edge(previous)
        self.edges[stored.id] = stored
        self._index_edge(stored)
        return stored.model_copy(deep=True)

    def _index_edge(self, edge: GraphEdge) -> None:
        self.edges_from.setdefault(edge.start_id, set()).add(edge.id)
        self.edges_to.setdefault(edge.end_id, set()).add(edge.id)

    def _unindex_edge(self, edge: GraphEdge) -> None:
        for index, node_id in ((self.edges_from, edge.start_id), (self.edges_to, edge.end_id)):
            ids = index.get(node_id)
            if not ids:
                continue
            ids.discard(edge.id)
            if not ids:
                del index[node_id]

    def _rebuild_edge_index(self) -> None:
        self.edges_from = {}
        self.edges_to = {}
        for edge in self.edges.values():
            self._index_edge(edge)

    def find_nodes(
        self,
        *,
        graph: str | None = None,
        label: str | None = None,
        source_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[GraphNode]:
        found: list[GraphNode] = []
        for node in self.nodes.values():
            if graph is not None and node.graph != graph:
                continue
            if label is not None and label not in node.labels:
                continue
            if source_id is not None and node.properties.get("source_id") != source_id:
                continue
            if not include_deleted and node.properties.get("lifecycle") == "deleted":
                continue
            found.append(node.model_copy(deep=True))
        return sorted(found, key=lambda item: item.id)

    def find_edges(
        self,
        *,
        graph: str | None = None,
        rel_type: str | None = None,
        start_id: str | None = None,
        end_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[GraphEdge]:
        if start_id is not None and end_id is not None:
            candidates = self.edges_from.get(start_id, set()) & self.edges_to.get(end_id, set())
            pool = (self.edges[edge_id] for edge_id in candidates)
        elif start_id is not None:
            pool = (self.edges[edge_id] for edge_id in self.edges_from.get(start_id, ()))
        elif end_id is not None:
            pool = (self.edges[edge_id] for edge_id in self.edges_to.get(end_id, ()))
        else:
            pool = self.edges.values()
        found: list[GraphEdge] = []
        for edge in pool:
            if graph is not None and edge.graph != graph:
                continue
            if rel_type is not None and edge.type != rel_type:
                continue
            if start_id is not None and edge.start_id != start_id:
                continue
            if end_id is not None and edge.end_id != end_id:
                continue
            if not include_deleted and edge.properties.get("lifecycle") == "deleted":
                continue
            found.append(edge.model_copy(deep=True))
        return sorted(found, key=lambda item: item.id)

    def retire_record(self, source_id: str, record_id: str) -> None:
        for edge in self.edges.values():
            if edge.type == "RESOLVES_TO":
                continue
            if edge.properties.get("prov_source_id") == source_id and edge.properties.get("prov_record_id") == record_id:
                edge.properties["lifecycle"] = "deleted"
        for node in self.nodes.values():
            if (
                node.properties.get("prov_source_id") == source_id
                and node.properties.get("prov_record_id") == record_id
                and "Record" in node.labels
            ):
                node.properties["lifecycle"] = "deleted"

    def refresh_orphans(self, graph: str) -> None:
        for node in self.nodes.values():
            if node.graph != graph or "SourceEntity" not in node.labels:
                continue
            node.properties["lifecycle"] = "active" if self._has_supporting_edge(node.id) else "deleted"

    def _has_supporting_edge(self, node_id: str) -> bool:
        for edge_id in (*self.edges_from.get(node_id, ()), *self.edges_to.get(node_id, ())):
            edge = self.edges[edge_id]
            if edge.type == "RESOLVES_TO" or edge.properties.get("lifecycle") == "deleted":
                continue
            return True
        return False

    def get_edge(self, edge_id: str, *, include_deleted: bool = True) -> GraphEdge | None:
        edge = self.edges.get(edge_id)
        if edge is None:
            return None
        if not include_deleted and edge.properties.get("lifecycle") == "deleted":
            return None
        return edge.model_copy(deep=True)

    def soft_delete_edge(self, edge_id: str) -> GraphEdge:
        edge = self.edges.get(edge_id)
        if edge is None:
            raise RelationshipValidationError(f"Edge {edge_id} was not found.")
        edge.properties["lifecycle"] = "deleted"
        return edge.model_copy(deep=True)

    def patch_edge(self, edge_id: str, properties: dict) -> GraphEdge:
        edge = self.edges.get(edge_id)
        if edge is None:
            raise RelationshipValidationError(f"Edge {edge_id} was not found.")
        blocked = {"lifecycle", "id", "type", "start_id", "end_id", "graph"}
        for key, value in properties.items():
            if key in blocked:
                continue
            edge.properties[key] = value
        return edge.model_copy(deep=True)

    def soft_delete_graph(self, graph_id: str) -> int:
        count = 0
        for edge in self.edges.values():
            if edge.graph == graph_id and edge.properties.get("lifecycle") != "deleted":
                edge.properties["lifecycle"] = "deleted"
                count += 1
        for node in self.nodes.values():
            if node.graph == graph_id and node.properties.get("lifecycle") != "deleted":
                node.properties["lifecycle"] = "deleted"
        return count

    def delete_edges_of_type(self, rel_type: str) -> None:
        self.edges = {key: edge for key, edge in self.edges.items() if edge.type != rel_type}
        self._rebuild_edge_index()

    def snapshot(self) -> dict:
        return {
            "nodes": [node.model_dump() for node in self.find_nodes(include_deleted=True)],
            "edges": [edge.model_dump() for edge in self.find_edges(include_deleted=True)],
            "schema": self.applied_schema,
        }

    def load_snapshot(self, payload: dict) -> None:
        self.nodes = {}
        self.edges = {}
        self.merge_snapshot(payload)
        schema = payload.get("schema")
        self.applied_schema = list(schema) if schema else list(SCHEMA_STATEMENTS)

    def merge_snapshot(self, payload: dict) -> None:
        for raw in payload.get("nodes", []):
            node = GraphNode.model_validate(raw)
            self.nodes[node.id] = node
        for raw in payload.get("edges", []):
            edge = GraphEdge.model_validate(raw)
            self.edges[edge.id] = edge
        self._rebuild_edge_index()
        if not self.applied_schema and payload.get("schema"):
            self.applied_schema = list(payload["schema"])
