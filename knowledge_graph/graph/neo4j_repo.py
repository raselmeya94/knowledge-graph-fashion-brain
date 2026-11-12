"""Neo4j repository. User input is never concatenated into Cypher."""

from __future__ import annotations

from knowledge_graph.access import CLEARANCE_RANK
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
        start = "(a:Entity)" if start_id is None else "(a:Entity {id: $start_id})"
        end = "(b:Entity)" if end_id is None else "(b:Entity {id: $end_id})"
        filters = []
        params = {}
        if graph is not None:
            filters.append("r.graph = $graph")
            params["graph"] = graph
        if start_id is not None:
            params["start_id"] = start_id
        if end_id is not None:
            params["end_id"] = end_id
        query = (
            f"MATCH {start}-[r{rel}]->{end} "
            + (f"WHERE {' AND '.join(filters)} " if filters else "")
            + "RETURN a.id AS start_id, b.id AS end_id, type(r) AS rel_type, properties(r) AS props"
        )
        with self._session() as session:
            rows = list(session.run(query, params))
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

    def product_material_rows(self, graph_ids: list[str]) -> list[dict]:
        """Fetch product-to-material edges and their resolutions in one query."""

        if not graph_ids:
            return []
        query = (
            "MATCH (p:Entity:Product)-[u:USES_MATERIAL]->(m:Entity) "
            "WHERE p.graph IN $graph_ids "
            "AND u.graph = p.graph "
            "AND coalesce(p.lifecycle, 'active') <> 'deleted' "
            "AND coalesce(u.lifecycle, 'active') <> 'deleted' "
            "AND coalesce(m.lifecycle, 'active') <> 'deleted' "
            "OPTIONAL MATCH (m)-[r:RESOLVES_TO]->(c:Entity) "
            "WHERE coalesce(r.lifecycle, 'active') <> 'deleted' "
            "AND coalesce(c.lifecycle, 'active') <> 'deleted' "
            "WITH p, u, m, collect(CASE WHEN r IS NULL THEN null "
            "ELSE {id: c.id, status: r.prov_verification_status} END) AS resolutions "
            "RETURN p.id AS product_id, p.graph AS graph_id, properties(p) AS product_props, "
            "labels(p) AS product_labels, properties(u) AS edge_props, m.id AS material_id, "
            "labels(m) AS material_labels, properties(m) AS material_props, "
            "resolutions ORDER BY product_id, material_id"
        )
        with self._session() as session:
            rows = list(session.run(query, {"graph_ids": graph_ids}))

        result = []
        for row in rows:
            resolutions = [item for item in row["resolutions"] if item and item.get("id")]
            verified = next((item for item in resolutions if item.get("status") == "verified"), None)
            review = any(item.get("status") == "review" for item in resolutions)
            product_props = dict(row["product_props"])
            material_props = dict(row["material_props"])
            edge_props = dict(row["edge_props"])
            product_id = str(row["product_id"])
            material_id = str(row["material_id"])
            graph_id = str(row["graph_id"] or "")
            result.append(
                {
                    "product": GraphNode(
                        id=product_id,
                        labels=list(row["product_labels"]),
                        graph=graph_id,
                        properties=product_props,
                    ),
                    "edge": GraphEdge(
                        id=str(edge_props.get("id") or f"{product_id}:{material_id}"),
                        type="USES_MATERIAL",
                        start_id=product_id,
                        end_id=material_id,
                        graph=str(edge_props.get("graph") or graph_id),
                        properties=edge_props,
                    ),
                    "material": GraphNode(
                        id=material_id,
                        labels=list(row["material_labels"]),
                        graph=str(material_props.get("graph") or ""),
                        properties=material_props,
                    ),
                    "graph_id": graph_id,
                    "canonical_id": str(verified["id"]) if verified else None,
                    "material_key": str(verified["id"]) if verified else material_id,
                    "review": review and verified is None,
                }
            )
        return result

    def shared_material_groups(
        self,
        graph_ids: list[str],
        *,
        clearance: str,
        across_graphs: bool = False,
        limit: int | None = None,
    ) -> list[dict]:
        """Aggregate shared materials in Neo4j instead of transferring every product edge.

        ``limit`` caps returned groups for top-N asks only. None returns the full
        aggregated set (analysis). Do not use this as a product-detail page size.
        """

        if not graph_ids:
            return []
        rank = CLEARANCE_RANK.get(clearance)
        if rank is None:
            raise ValueError(f"Unknown clearance '{clearance}'.")
        allowed_classifications = [
            level for level, level_rank in CLEARANCE_RANK.items() if level_rank <= rank
        ]
        # Across-graph asks: start from verified RESOLVES_TO canonicals (shared layer).
        # Same-graph asks: count products per material key without collecting every product id.
        limit_clause = " LIMIT $limit" if limit is not None else ""
        if across_graphs:
            query = (
                "MATCH (src:Entity)-[r:RESOLVES_TO]->(c:Entity) "
                "WHERE r.prov_verification_status = 'verified' "
                "AND coalesce(r.lifecycle, 'active') <> 'deleted' "
                "AND coalesce(c.lifecycle, 'active') <> 'deleted' "
                "AND (c.graph = 'shared' OR 'Canonical' IN labels(c) OR 'Material' IN labels(c)) "
                "MATCH (p:Entity:Product)-[u:USES_MATERIAL]->(src) "
                "WHERE p.graph IN $graph_ids AND u.graph = p.graph "
                "AND coalesce(p.classification, p.prov_classification, 'internal') IN $allowed_classifications "
                "AND coalesce(p.lifecycle, 'active') <> 'deleted' "
                "AND coalesce(u.lifecycle, 'active') <> 'deleted' "
                "WITH c, p.graph AS graph_id, count(DISTINCT p.id) AS per_graph "
                "WITH c, collect(DISTINCT graph_id) AS graph_ids, sum(per_graph) AS product_count "
                "WHERE size(graph_ids) > 1 AND product_count > 1 "
                "RETURN c.id AS canonical_id, c.id AS material_key, "
                "coalesce(c.name, c.id) AS material, graph_ids, product_count "
                "ORDER BY product_count DESC, material"
                + limit_clause
            )
        else:
            query = (
                "MATCH (p:Entity:Product)-[u:USES_MATERIAL]->(m:Entity) "
                "WHERE p.graph IN $graph_ids AND u.graph = p.graph "
                "AND coalesce(p.classification, p.prov_classification, 'internal') IN $allowed_classifications "
                "AND coalesce(p.lifecycle, 'active') <> 'deleted' "
                "AND coalesce(u.lifecycle, 'active') <> 'deleted' "
                "AND coalesce(m.lifecycle, 'active') <> 'deleted' "
                "OPTIONAL MATCH (m)-[r:RESOLVES_TO]->(c:Entity) "
                "WHERE coalesce(r.lifecycle, 'active') <> 'deleted' "
                "AND coalesce(c.lifecycle, 'active') <> 'deleted' "
                "AND r.prov_verification_status = 'verified' "
                "WITH p, m, head(collect(DISTINCT c)) AS canonical "
                "WITH coalesce(canonical.id, m.id) AS material_key, "
                "coalesce(canonical.name, m.name, m.id) AS material_name, "
                "canonical.id AS canonical_id, "
                "p.graph AS graph_id "
                "WITH material_key, "
                "head(collect(DISTINCT material_name)) AS material, "
                "head(collect(DISTINCT canonical_id)) AS canonical_id, "
                "collect(DISTINCT graph_id) AS graph_ids, "
                "count(*) AS product_count "
                "WHERE product_count > 1 "
                "RETURN canonical_id, material_key, material, graph_ids, product_count "
                "ORDER BY product_count DESC, material"
                + limit_clause
            )
        params: dict = {
            "graph_ids": graph_ids,
            "allowed_classifications": allowed_classifications,
        }
        if limit is not None:
            params["limit"] = max(1, int(limit))
        with self._session() as session:
            rows = list(session.run(query, params))
            groups = []
            for row in rows:
                row_graph_ids = sorted(str(item) for item in (row["graph_ids"] or []))
                canonical_id = row["canonical_id"] or row["material_key"]
                material_key = str(row["material_key"] or canonical_id)
                # Analysis answers use counts only — skip product-id samples unless top-N listing.
                refs = []
                if limit is not None:
                    refs = self._sample_shared_material_products(
                        session,
                        graph_ids=graph_ids,
                        material_key=material_key,
                        across_graphs=across_graphs,
                        allowed_classifications=allowed_classifications,
                        limit=8,
                    )
                groups.append(
                    {
                        "canonical_id": str(canonical_id),
                        "material": str(row["material"] or canonical_id),
                        "products": [str(item.get("product_id") or "") for item in refs],
                        "product_refs": refs,
                        "product_count": int(row["product_count"]),
                        "verified": bool(row["canonical_id"]),
                        "graph_id": row_graph_ids[0] if len(row_graph_ids) == 1 else None,
                        "graph_ids": row_graph_ids,
                    }
                )
        return groups

    def _sample_shared_material_products(
        self,
        session,
        *,
        graph_ids: list[str],
        material_key: str,
        across_graphs: bool,
        allowed_classifications: list[str],
        limit: int = 8,
    ) -> list[dict]:
        """Fetch a few example products for one material without scanning the full catalog."""

        if across_graphs:
            query = (
                "MATCH (src:Entity)-[r:RESOLVES_TO]->(c:Entity {id: $material_key}) "
                "WHERE r.prov_verification_status = 'verified' "
                "AND coalesce(r.lifecycle, 'active') <> 'deleted' "
                "MATCH (p:Entity:Product)-[u:USES_MATERIAL]->(src) "
                "WHERE p.graph IN $graph_ids AND u.graph = p.graph "
                "AND coalesce(p.classification, p.prov_classification, 'internal') IN $allowed_classifications "
                "AND coalesce(p.lifecycle, 'active') <> 'deleted' "
                "RETURN p.graph AS graph_id, "
                "coalesce(p.prov_record_id, p.product_id, p.id) AS product_id "
                "LIMIT $limit"
            )
        else:
            query = (
                "MATCH (p:Entity:Product)-[u:USES_MATERIAL]->(m:Entity) "
                "WHERE p.graph IN $graph_ids AND u.graph = p.graph "
                "AND coalesce(p.classification, p.prov_classification, 'internal') IN $allowed_classifications "
                "AND coalesce(p.lifecycle, 'active') <> 'deleted' "
                "AND coalesce(u.lifecycle, 'active') <> 'deleted' "
                "OPTIONAL MATCH (m)-[r:RESOLVES_TO]->(c:Entity) "
                "WHERE coalesce(r.lifecycle, 'active') <> 'deleted' "
                "AND r.prov_verification_status = 'verified' "
                "WITH p, coalesce(c.id, m.id) AS material_key "
                "WHERE material_key = $material_key "
                "RETURN p.graph AS graph_id, "
                "coalesce(p.prov_record_id, p.product_id, p.id) AS product_id "
                "LIMIT $limit"
            )
        rows = list(
            session.run(
                query,
                {
                    "graph_ids": graph_ids,
                    "material_key": material_key,
                    "allowed_classifications": allowed_classifications,
                    "limit": max(1, int(limit)),
                },
            )
        )
        return [{"graph_id": str(row["graph_id"]), "product_id": str(row["product_id"])} for row in rows]

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

    def product_material_rows(self, graph_ids: list[str]) -> list[dict]:
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
