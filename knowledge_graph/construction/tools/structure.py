"""Turn extracted content into one structured payload."""

from __future__ import annotations


class StructureDataTool:
    name = "structure_data"
    description = "Consolidate entities and relationships into the structure stored for review and graph ingestion."

    def run(self, payload: dict, ctx) -> dict:
        del ctx
        relationships = payload.get("relationships") or []
        entities: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for rel in relationships:
            for type_key, name_key in (("start_type", "start_name"), ("end_type", "end_name")):
                name = str(rel.get(name_key) or "").strip()
                entity_type = str(rel.get(type_key) or "").strip()
                if not name or entity_type == "Document":
                    continue
                key = (entity_type, name.casefold())
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    {
                        "type": entity_type,
                        "name": name,
                        "evidence": rel.get("evidence") or "",
                    }
                )
        return {
            "source_kind": payload.get("source_kind") or payload.get("kind") or "text",
            "filename": payload.get("filename") or "",
            "title": payload.get("title") or "Untitled document",
            "text": payload.get("text") or "",
            "graph_id": payload.get("graph_id") or "",
            "ontology_id": payload.get("ontology_id") or "",
            "metadata": payload.get("metadata") or {},
            "entities": entities,
            "relationships": relationships,
            "warnings": list(payload.get("warnings") or []),
        }
