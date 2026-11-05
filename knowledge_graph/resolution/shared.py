"""Build the shared graph without collapsing source entities."""

from __future__ import annotations

from knowledge_graph.ids import edge_id
from knowledge_graph.models import GraphEdge, GraphNode, ReviewDecision
from knowledge_graph.ontology import Ontology, normalize_name
from knowledge_graph.resolution.matcher import match_entity


def ensure_canonical_nodes(repo, ontology: Ontology) -> None:
    for entity in ontology.canonical_entities:
        repo.upsert_node(
            GraphNode(
                id=entity.id,
                labels=["Entity", "Canonical", entity.type],
                graph="shared",
                properties={
                    "name": entity.name,
                    "normalized_name": normalize_name(entity.name),
                    "entity_type": entity.type,
                    "aliases": entity.aliases,
                    "external_ids": entity.external_ids,
                    "lifecycle": "active",
                    "synthetic": True,
                    "source_id": "ontology",
                    "prov_source_id": "ontology",
                    "prov_record_id": entity.id,
                    "prov_source_path": "schemas/ontology.json",
                    "prov_evidence_text": f"Declared in schemas/ontology.json as {entity.id}.",
                    "prov_extraction_method": "ontology",
                    "prov_verification_status": "verified",
                    "prov_synthetic": True,
                },
            )
        )


def _support_records(repo, node_id: str) -> list[str]:
    records: set[str] = set()
    seen: set[str] = set()
    incident = (
        *repo.find_edges(start_id=node_id, include_deleted=False),
        *repo.find_edges(end_id=node_id, include_deleted=False),
    )
    for edge in incident:
        if edge.id in seen or edge.type == "RESOLVES_TO":
            continue
        seen.add(edge.id)
        record_id = edge.properties.get("prov_record_id")
        if record_id:
            records.add(str(record_id))
    return sorted(records)


def build_shared_graph(repo, ontology: Ontology, decisions: list[ReviewDecision], scorer=None) -> dict:
    ensure_canonical_nodes(repo, ontology)
    repo.delete_edges_of_type("RESOLVES_TO")
    created = {"verified": 0, "review": 0, "rejected": 0}
    for node in repo.find_nodes(include_deleted=False):
        if "SourceEntity" not in node.labels:
            continue
        entity_type = node.properties.get("entity_type")
        name = node.properties.get("name")
        if not entity_type or not name:
            continue
        matches = match_entity(
            name=name,
            entity_type=entity_type,
            external_id=node.properties.get("external_id"),
            ontology=ontology,
            decisions=decisions,
            scorer=scorer,
        )
        support = _support_records(repo, node.id)
        for decision in matches:
            if decision.status == "unresolved" or not decision.canonical_id:
                continue
            properties = {
                "lifecycle": "active",
                "prov_source_id": node.properties.get("source_id"),
                "prov_source_type": node.properties.get("source_type"),
                "prov_record_id": ",".join(support),
                "prov_support_record_ids": support,
                "prov_evidence_text": decision.evidence,
                "prov_extraction_method": "deterministic_resolution",
                "prov_verification_status": decision.status,
                "prov_match_method": decision.method,
                "prov_synthetic": True,
                "prov_source_path": node.properties.get("prov_source_path"),
                "prov_content_hash": node.properties.get("prov_content_hash"),
                "source_name": decision.source_name,
                "source_id": node.properties.get("source_id"),
            }
            if decision.similarity is not None:
                properties["prov_similarity"] = decision.similarity
            repo.upsert_edge(
                GraphEdge(
                    id=edge_id("RESOLVES_TO", node.id, decision.canonical_id, node.id, decision.status),
                    type="RESOLVES_TO",
                    start_id=node.id,
                    end_id=decision.canonical_id,
                    graph="shared",
                    properties=properties,
                )
            )
            created[decision.status] = created.get(decision.status, 0) + 1
    return created
