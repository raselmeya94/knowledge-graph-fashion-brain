"""Source-graph builders. Each builder accepts only its own envelopes."""

from __future__ import annotations

from knowledge_graph.ids import edge_id, primary_id, primary_label, source_entity_id
from knowledge_graph.models import GraphEdge, GraphNode, RecordEnvelope
from knowledge_graph.ontology import Ontology, normalize_name


def _provenance(env: RecordEnvelope, **extra) -> dict:
    payload = {
        "prov_source_id": env.source_id,
        "prov_record_id": env.record_id,
        "prov_source_path": env.source_path,
        "prov_url": env.source_url,
        "prov_source_timestamp": env.source_timestamp,
        "prov_ingested_at": env.ingested_at,
        "prov_version": env.version,
        "prov_content_hash": env.content_hash,
        "prov_extraction_method": "deterministic",
        "prov_verification_status": "observed",
        "prov_synthetic": env.synthetic,
        "prov_classification": env.classification,
        "prov_source_type": env.source_type,
        "lifecycle": "active",
        "source_id": env.source_id,
        "source_type": env.source_type,
    }
    payload.update(extra)
    return {key: value for key, value in payload.items() if value is not None}


def _upsert_nodes(env: RecordEnvelope, repo, ontology: Ontology) -> dict[str, str]:
    label = primary_label(env.source_type)
    record_node_id = primary_id(env.source_type, env.record_id)
    local_map = {"record": record_node_id}
    repo.upsert_node(
        GraphNode(
            id=record_node_id,
            labels=["Entity", "Record", label],
            graph=env.source_id,
            properties=_provenance(
                env,
                name=env.title,
                normalized_name=normalize_name(env.title),
                title=env.title,
                classification=env.classification,
                document_type=env.document_type,
                publisher=env.publisher,
                synthetic=env.synthetic,
                prov_evidence_text=env.title,
                prov_field_ref="title",
                **dict(env.metadata.get("node_properties") or {}),
            ),
        )
    )
    for entity in env.entities:
        if entity.local_id == "record":
            continue
        try:
            node_id = source_entity_id(env.source_type, entity.entity_type, entity.name)
        except ValueError:
            local_map[entity.local_id] = ""
            continue
        local_map[entity.local_id] = node_id
        repo.upsert_node(
            GraphNode(
                id=node_id,
                labels=["Entity", "SourceEntity", entity.entity_type],
                graph=env.source_id,
                properties=_provenance(
                    env,
                    name=entity.name,
                    normalized_name=normalize_name(entity.name),
                    entity_type=entity.entity_type,
                    external_id=entity.external_id,
                    synthetic=env.synthetic,
                    prov_evidence_text=entity.evidence_text,
                    prov_field_ref=entity.field_ref,
                    prov_section_ref=entity.section_ref,
                    **dict(entity.properties or {}),
                ),
            )
        )
    return local_map


def build_envelopes(envelopes: list[RecordEnvelope], repo, ontology: Ontology, expected_type: str) -> list[str]:
    errors: list[str] = []
    active = []
    for env in envelopes:
        if env.source_type != expected_type:
            raise ValueError(
                f"{expected_type} graph builder received envelope {env.record_id} from {env.source_type}."
            )
        if env.lifecycle == "active":
            active.append(env)
    maps = [(env, _upsert_nodes(env, repo, ontology)) for env in active]
    for env, local_map in maps:
        for rel in env.relationships:
            if rel.end_record_id:
                end = primary_id(env.source_type, rel.end_record_id)
            else:
                end = local_map.get(rel.end_local_id or "")
            start = local_map.get(rel.start_local_id)
            if start == "" or end == "":
                continue
            if start is None or end is None:
                errors.append(f"{env.record_id}: relationship {rel.type} is missing an endpoint.")
                continue
            try:
                repo.upsert_edge(
                    GraphEdge(
                        id=edge_id(rel.type, start, end, env.record_id, rel.field_ref),
                        type=rel.type,
                        start_id=start,
                        end_id=end,
                        graph=env.source_id,
                        properties=_provenance(
                            env,
                            prov_evidence_text=rel.evidence_text,
                            prov_field_ref=rel.field_ref,
                            prov_section_ref=rel.section_ref,
                            prov_verification_status=rel.verification_status,
                            **dict(rel.properties or {}),
                        ),
                    )
                )
            except Exception as exc:
                errors.append(f"{env.record_id}: {exc}")
    return errors


def build_own_products_graph(envelopes: list[RecordEnvelope], repo, ontology: Ontology) -> list[str]:
    return build_envelopes(envelopes, repo, ontology, "own_products")


BUILDERS = {
    "own_products": build_own_products_graph,
}


def register_builder(source_type: str, builder) -> None:
    BUILDERS[source_type] = builder


def get_builder(source_type: str):
    builder = BUILDERS.get(source_type)
    if builder is None:
        raise ValueError(
            f"No graph builder is registered for '{source_type}'. "
            "Register one with register_builder without changing the shared graph engine."
        )
    return builder
