"""Neo4j repository. User input is never concatenated into Cypher."""

from __future__ import annotations

from knowledge_graph.config import Settings
from knowledge_graph.errors import GraphBackendUnavailable, RelationshipValidationError
from knowledge_graph.graph.constraints import SCHEMA_STATEMENTS
from knowledge_graph.logging_config import get_logger
from knowledge_graph.models import GraphEdge, GraphNode
from knowledge_graph.ontology import Ontology, validate_relationship

LOGGER = get_logger("knowledge_graph.graph.neo4j")
_BATCH = 500


def _clean(properties: dict) -> dict:
    cleaned = {}
    for key, value in properties.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            cleaned[key] = value
    return cleaned


class Neo4jGraphRepository:
    def __init__(self, driver, ontology: Ontology, database: str = "neo4j"):
        self.driver = driver
        self.ontology = ontology
        self.database = database
        self._allowed_rel_types = set(ontology.relationship_allowlist) | {"RESOLVES_TO"}
        self._allowed_labels = set(ontology.entity_types) | {"Entity", "SourceEntity", "Canonical", "Record"}

    def _session(self):
        return self.driver.session(database=self.database)

    def ping(self) -> str:
        with self._session() as session:
            session.run("RETURN 1 AS ok")
        return "ok"

    def close(self) -> None:
        driver = getattr(self, "driver", None)
        if driver is not None:
            driver.close()
            self.driver = None

    def ensure_constraints(self) -> list[str]:
        with self._session() as session:
            for statement in SCHEMA_STATEMENTS:
                session.run(statement)
        return list(SCHEMA_STATEMENTS)

    def upsert_node(self, node: GraphNode) -> GraphNode:
        for label in node.labels:
            if label not in self._allowed_labels:
                raise RelationshipValidationError(f"Label {label} is not allowlisted.")
        props = _clean(node.properties)
        props["id"] = node.id
        props["graph"] = node.graph
        props["label_names"] = list(node.labels)
        props.setdefault("lifecycle", "active")
        assignments = " ".join(f"SET n:{label}" for label in node.labels)
        query = f"MERGE (n:Entity {{id: $id}}) SET n += $props {assignments}"
        with self._session() as session:
            session.run(query, {"id": node.id, "props": props})
        return node

    def get_node(self, node_id: str, *, include_deleted: bool = True) -> GraphNode | None:
        query = "MATCH (n:Entity {id: $id}) RETURN n, labels(n) AS labels"
        with self._session() as session:
            rows = list(session.run(query, {"id": node_id}))
        if not rows:
            return None
        return self._node_from_row(rows[0], include_deleted=include_deleted)

    def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        if edge.type not in self._allowed_rel_types:
            raise RelationshipValidationError(f"Relationship type {edge.type} is not allowlisted.")
        start = self.get_node(edge.start_id)
        end = self.get_node(edge.end_id)
        if start is None or end is None:
            raise RelationshipValidationError(
                f"Cannot create {edge.type} from {edge.start_id} to {edge.end_id}: both nodes must exist."
            )
        validate_relationship(edge.type, start.labels, end.labels, self.ontology)
        props = _clean(edge.properties)
        props["id"] = edge.id
        props["graph"] = edge.graph
        props.setdefault("lifecycle", "active")
        query = (
            "MATCH (a:Entity {id: $start_id}) "
            "MATCH (b:Entity {id: $end_id}) "
            f"MERGE (a)-[r:{edge.type} {{id: $edge_id}}]->(b) "
            "SET r += $props"
        )
        with self._session() as session:
            session.run(
                query,
                {"start_id": edge.start_id, "end_id": edge.end_id, "edge_id": edge.id, "props": props},
            )
        return edge

    def find_nodes(
        self,
        *,
        graph: str | None = None,
        label: str | None = None,
        source_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[GraphNode]:
        if label is not None and label not in self._allowed_labels:
            raise RelationshipValidationError(f"Label {label} is not allowlisted.")
        label_clause = f":{label}" if label else ""
        query = (
            f"MATCH (n:Entity{label_clause}) "
            "WHERE ($graph IS NULL OR n.graph = $graph) "
            "AND ($source_id IS NULL OR n.source_id = $source_id) "
            "RETURN n, labels(n) AS labels"
        )
        with self._session() as session:
            rows = list(session.run(query, {"graph": graph, "source_id": source_id}))
        nodes = []
        for row in rows:
            node = self._node_from_row(row, include_deleted=include_deleted)
            if node is not None:
                nodes.append(node)
        return sorted(nodes, key=lambda item: item.id)

    def find_edges(
        self,
        *,
        graph: str | None = None,
        rel_type: str | None = None,
        start_id: str | None = None,
        end_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[GraphEdge]:
        if rel_type is not None and rel_type not in self._allowed_rel_types:
            raise RelationshipValidationError(f"Relationship type {rel_type} is not allowlisted.")
        rel = f":{rel_type}" if rel_type else ""
        query = (
            f"MATCH (a:Entity)-[r{rel}]->(b:Entity) "
            "WHERE ($graph IS NULL OR r.graph = $graph) "
            "AND ($start_id IS NULL OR a.id = $start_id) "
            "AND ($end_id IS NULL OR b.id = $end_id) "
            "RETURN a.id AS start_id, b.id AS end_id, type(r) AS rel_type, properties(r) AS props"
        )
        with self._session() as session:
            rows = list(
                session.run(query, {"graph": graph, "start_id": start_id, "end_id": end_id})
            )
        edges: list[GraphEdge] = []
        for row in rows:
            props = dict(row["props"])
            if not include_deleted and props.get("lifecycle") == "deleted":
                continue
            edges.append(
                GraphEdge(
                    id=str(props.get("id")),
                    type=row["rel_type"],
                    start_id=row["start_id"],
                    end_id=row["end_id"],
                    graph=str(props.get("graph", "")),
                    properties=props,
                )
            )
        return sorted(edges, key=lambda item: item.id)

    def retire_record(self, source_id: str, record_id: str) -> None:
        query = (
            "MATCH ()-[r]->() "
            "WHERE r.prov_source_id = $source_id AND r.prov_record_id = $record_id "
            "AND type(r) <> 'RESOLVES_TO' "
            "SET r.lifecycle = 'deleted'"
        )
        node_query = (
            "MATCH (n:Entity) "
            "WHERE n.prov_source_id = $source_id AND n.prov_record_id = $record_id "
            "AND 'Record' IN labels(n) "
            "SET n.lifecycle = 'deleted'"
        )
        with self._session() as session:
            session.run(query, {"source_id": source_id, "record_id": record_id})
            session.run(node_query, {"source_id": source_id, "record_id": record_id})

    def refresh_orphans(self, graph: str) -> None:
        query = (
            "MATCH (n:Entity) "
            "WHERE n.graph = $graph AND 'SourceEntity' IN labels(n) "
            "OPTIONAL MATCH (n)-[r]-() "
            "WHERE coalesce(r.lifecycle, 'active') <> 'deleted' AND type(r) <> 'RESOLVES_TO' "
            "WITH n, count(r) AS live "
            "SET n.lifecycle = CASE WHEN live = 0 THEN 'deleted' ELSE 'active' END"
        )
        with self._session() as session:
            session.run(query, {"graph": graph})

    def get_edge(self, edge_id: str, *, include_deleted: bool = True) -> GraphEdge | None:
        query = (
            "MATCH (a:Entity)-[r]->(b:Entity) WHERE r.id = $edge_id "
            "RETURN a.id AS start_id, b.id AS end_id, type(r) AS rel_type, properties(r) AS props"
        )
        with self._session() as session:
            rows = list(session.run(query, {"edge_id": edge_id}))
        if not rows:
            return None
        props = dict(rows[0]["props"])
        if not include_deleted and props.get("lifecycle") == "deleted":
            return None
        return GraphEdge(
            id=str(props.get("id") or edge_id),
            type=rows[0]["rel_type"],
            start_id=rows[0]["start_id"],
            end_id=rows[0]["end_id"],
            graph=str(props.get("graph", "")),
            properties=props,
        )

    def soft_delete_edge(self, edge_id: str) -> GraphEdge:
        edge = self.get_edge(edge_id, include_deleted=True)
        if edge is None:
            raise RelationshipValidationError(f"Edge {edge_id} was not found.")
        query = "MATCH ()-[r]->() WHERE r.id = $edge_id SET r.lifecycle = 'deleted'"
        with self._session() as session:
            session.run(query, {"edge_id": edge_id})
        edge.properties["lifecycle"] = "deleted"
        return edge

    def patch_edge(self, edge_id: str, properties: dict) -> GraphEdge:
        edge = self.get_edge(edge_id, include_deleted=True)
        if edge is None:
            raise RelationshipValidationError(f"Edge {edge_id} was not found.")
        blocked = {"lifecycle", "id", "type", "start_id", "end_id", "graph"}
        props = {key: value for key, value in properties.items() if key not in blocked}
        query = "MATCH ()-[r]->() WHERE r.id = $edge_id SET r += $props"
        with self._session() as session:
            session.run(query, {"edge_id": edge_id, "props": props})
        edge.properties.update(props)
        return edge

    def soft_delete_graph(self, graph_id: str) -> int:
        query = (
            "MATCH (n:Entity {graph: $graph_id}) SET n.lifecycle = 'deleted' "
            "WITH count(n) AS nodes "
            "MATCH ()-[r {graph: $graph_id}]->() SET r.lifecycle = 'deleted' "
            "RETURN count(r) AS edges"
        )
        with self._session() as session:
            row = session.run(query, {"graph_id": graph_id}).single()
        return int(row["edges"]) if row else 0

    def delete_edges_of_type(self, rel_type: str) -> None:
        if rel_type not in self._allowed_rel_types:
            raise RelationshipValidationError(f"Relationship type {rel_type} is not allowlisted.")
        query = f"MATCH ()-[r:{rel_type}]->() DELETE r"
        with self._session() as session:
            session.run(query)

    def clear(self) -> None:
        """Remove all nodes/relationships in batches (avoids Neo4j tx memory caps)."""

        with self._session() as session:
            while True:
                row = session.run(
                    "MATCH (n) WITH n LIMIT $limit DETACH DELETE n RETURN count(*) AS c",
                    {"limit": _BATCH * 10},
                ).single()
                if not row or int(row["c"]) == 0:
                    break

    def counts(self) -> dict[str, int]:
        with self._session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": int(nodes), "edges": int(edges)}

    def distinct_source_graphs(self) -> list[str]:
        query = (
            "MATCH (n:Entity) "
            "WHERE n.graph IS NOT NULL AND n.graph <> 'shared' "
            "RETURN DISTINCT n.graph AS g ORDER BY g"
        )
        with self._session() as session:
            return [str(row["g"]) for row in session.run(query) if row["g"]]

    def load_snapshot(self, payload: dict) -> None:
        """Replace DB contents with a snapshot (empty payload clears)."""

        self.clear()
        self.merge_snapshot(payload)
        schema = payload.get("schema")
        self.applied_schema = list(schema) if schema else list(SCHEMA_STATEMENTS)

    def merge_snapshot(self, payload: dict) -> None:
        """Upsert snapshot nodes/edges into Neo4j via Cypher MERGE (parameterized)."""

        nodes = [GraphNode.model_validate(raw) for raw in payload.get("nodes", [])]
        edges = [GraphEdge.model_validate(raw) for raw in payload.get("edges", [])]
        for index in range(0, len(nodes), _BATCH):
            chunk = nodes[index : index + _BATCH]
            for node in chunk:
                self.upsert_node(node)
        for index in range(0, len(edges), _BATCH):
            chunk = edges[index : index + _BATCH]
            for edge in chunk:
                try:
                    self.upsert_edge(edge)
                except RelationshipValidationError as exc:
                    LOGGER.warning("skip edge %s: %s", edge.id, exc)
        if not getattr(self, "applied_schema", None) and payload.get("schema"):
            self.applied_schema = list(payload["schema"])
        LOGGER.info("neo4j merge_snapshot nodes=%s edges=%s", len(nodes), len(edges))

    def _node_from_row(self, row, *, include_deleted: bool) -> GraphNode | None:
        raw = row["n"]
        props = dict(raw)
        if not include_deleted and props.get("lifecycle") == "deleted":
            return None
        labels = list(row["labels"])
        return GraphNode(id=props["id"], labels=labels, graph=props.get("graph", ""), properties=props)


class UnavailableGraphRepository:
    def __init__(self, reason: str):
        self.reason = reason

    def ping(self) -> str:
        raise GraphBackendUnavailable(self.reason)

    def ensure_constraints(self) -> list[str]:
        raise GraphBackendUnavailable(self.reason)

    def upsert_node(self, node: GraphNode) -> GraphNode:
        raise GraphBackendUnavailable(self.reason)

    def get_node(self, node_id: str, *, include_deleted: bool = True) -> GraphNode | None:
        raise GraphBackendUnavailable(self.reason)

    def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        raise GraphBackendUnavailable(self.reason)

    def find_nodes(self, **kwargs) -> list[GraphNode]:
        raise GraphBackendUnavailable(self.reason)

    def find_edges(self, **kwargs) -> list[GraphEdge]:
        raise GraphBackendUnavailable(self.reason)

    def retire_record(self, source_id: str, record_id: str) -> None:
        raise GraphBackendUnavailable(self.reason)

    def refresh_orphans(self, graph: str) -> None:
        raise GraphBackendUnavailable(self.reason)

    def delete_edges_of_type(self, rel_type: str) -> None:
        raise GraphBackendUnavailable(self.reason)

    def get_edge(self, edge_id: str, *, include_deleted: bool = True):
        raise GraphBackendUnavailable(self.reason)

    def soft_delete_edge(self, edge_id: str):
        raise GraphBackendUnavailable(self.reason)

    def patch_edge(self, edge_id: str, properties: dict):
        raise GraphBackendUnavailable(self.reason)

    def soft_delete_graph(self, graph_id: str) -> int:
        raise GraphBackendUnavailable(self.reason)

    def clear(self) -> None:
        raise GraphBackendUnavailable(self.reason)

    def counts(self) -> dict[str, int]:
        raise GraphBackendUnavailable(self.reason)

    def distinct_source_graphs(self) -> list[str]:
        raise GraphBackendUnavailable(self.reason)

    def load_snapshot(self, payload: dict) -> None:
        raise GraphBackendUnavailable(self.reason)

    def merge_snapshot(self, payload: dict) -> None:
        raise GraphBackendUnavailable(self.reason)


def connect_neo4j(settings: Settings, ontology: Ontology) -> Neo4jGraphRepository:
    if not settings.neo4j_password:
        raise GraphBackendUnavailable("Neo4j password is empty. Refusing to connect.")
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise GraphBackendUnavailable("The neo4j Python driver is not installed.") from exc
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=5,
        )
        driver.verify_connectivity()
    except Exception as exc:
        raise GraphBackendUnavailable(f"Neo4j is unavailable ({exc.__class__.__name__}).") from exc
    repo = Neo4jGraphRepository(driver, ontology, database=settings.neo4j_database)
    repo.applied_schema = []
    repo.ensure_constraints()
    return repo
