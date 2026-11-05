"""Dynamic metadata and paginated attribute value discovery for agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_graph.config import DATA_ROOT
from knowledge_graph.errors import NotFoundError, ValidationFailed
from knowledge_graph.kg_registry.store import enrich_graph_record


def _product_attributes_for_graph(platform, graph_id: str, enriched) -> dict[str, str]:
    """Distinct Product columns for one KG (ontology + mapping + sampled nodes)."""

    attrs: dict[str, str] = {"name": "string", "id": "string"}
    caps = {str(item).casefold() for item in (enriched.capabilities or [])}
    entities = {str(item) for item in (enriched.entities or [])}

    if "Brand" in entities or any("brand" in cap for cap in caps):
        attrs["brand"] = "string"
    if any("buyer" in cap for cap in caps) or graph_id == "own_products":
        attrs["buyer"] = "string"
    if "GarmentCategory" in entities:
        attrs["category"] = "string"
    if "Material" in entities:
        attrs["material"] = "string"
        attrs["composition"] = "string"
    if "Color" in entities:
        attrs["color"] = "string"
    if "Season" in entities:
        attrs["season"] = "string"
    if "Availability" in entities:
        attrs["availability"] = "string"
    if "Section" in entities:
        attrs["section"] = "string"
    if any("price" in cap for cap in caps) or "Currency" in entities:
        attrs["price"] = "number"
        attrs["currency"] = "string"

    if graph_id == "ecom_products" or "ecom" in graph_id:
        mapping_path = Path(DATA_ROOT) / "schemas" / "ecom_master_products_mapping.json"
        if mapping_path.exists():
            try:
                payload = json.loads(mapping_path.read_text(encoding="utf-8"))
                for key in (payload.get("product_properties") or {}):
                    attrs[str(key)] = "string"
            except Exception:
                pass
        for key in ("family", "subfamily", "section", "availability", "brand", "category"):
            attrs.setdefault(key, "string")
    if graph_id == "own_products":
        from knowledge_graph.retrieval.products import CATALOG_FIELDS

        for field in CATALOG_FIELDS:
            attrs.setdefault(field.key, "string")

    try:
        nodes = platform.graph.find_nodes(graph=graph_id, label="Product", include_deleted=False)
        for node in nodes[:100]:
            for key, value in (node.properties or {}).items():
                if key in attrs or key.startswith("prov_"):
                    continue
                attrs[str(key)] = "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else "string"
    except Exception:
        pass
    return attrs


def metadata_overview(platform, graph_id: str | None = None) -> dict[str, Any]:
    """Schema-oriented metadata for one graph or all active graphs."""

    rows = []
    graphs = platform.kg_registry.list_graphs()
    if graph_id:
        graphs = [item for item in graphs if item.id == graph_id]
        if not graphs and graph_id not in {"shared", "master", "global"}:
            raise NotFoundError(f"Graph '{graph_id}' was not found.")
        if graph_id in {"shared", "master", "global"}:
            return {
                "graph_id": graph_id,
                "kind": "interconnect",
                "capabilities": ["canonical_materials", "products_sharing_material"],
                "schema": {},
            }
    for graph in graphs:
        try:
            ontology = platform.kg_registry.load_ontology_object(graph.ontology_id)
        except Exception:
            ontology = None
        enriched = enrich_graph_record(graph, ontology)
        schema: dict[str, Any] = {}
        if ontology is not None:
            for entity in ontology.entity_types:
                schema[entity] = {"attributes": {"name": "string", "id": "string"}}
            if "Product" in schema:
                # Unique product columns for this graph only (not a shared hard-coded set).
                schema["Product"]["attributes"].update(
                    _product_attributes_for_graph(platform, enriched.id, enriched)
                )
            if enriched.kind == "system":
                schema["KnowledgeGraph"] = {
                    "attributes": {
                        "graph_id": "string",
                        "graph_name": "string",
                        "kind": "string",
                        "capability": "string",
                        "entity": "string",
                        "attribute": "string",
                        "source_id": "string",
                        "topic": "string",
                    }
                }
        rows.append(
            {
                "graph_id": enriched.id,
                "id": enriched.id,
                "kind": enriched.kind,
                "source_ids": list(enriched.source_ids),
                "description": enriched.description,
                "entities": list(enriched.entities),
                "relationships": list(enriched.relationships),
                "capabilities": list(enriched.capabilities),
                "status": enriched.status,
                "schema": schema,
            }
        )
    if graph_id and rows:
        return rows[0]
    return {"graphs": rows}


def discover_values(
    platform,
    *,
    graph_id: str | None = None,
    attribute: str,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated distinct attribute values. Never dump full catalogs into LLM prompts."""

    if not 1 <= limit <= 200:
        raise ValidationFailed("limit must be between 1 and 200.")
    if offset < 0:
        raise ValidationFailed("offset must be zero or greater.")
    key = attribute.strip().lower().replace(" ", "_")
    if not key:
        raise ValidationFailed("attribute is required.")

    graphs = [graph_id] if graph_id else [item.id for item in platform.kg_registry.list_graphs() if item.kind == "product"]
    if graph_id and graph_id not in {"shared", "master"}:
        try:
            platform.kg_registry.get_graph(graph_id)
        except NotFoundError:
            raise

    values: list[str] = []
    seen: set[str] = set()
    needle = (q or "").strip().casefold()

    # System metadata graph: values come from the persisted registry snapshot index.
    if graph_id == "kg_system" or (not graph_id and key in {
        "graph_id", "graph_name", "kind", "capability", "entity", "attribute", "source_id", "topic"
    }):
        from knowledge_graph.retrieval.system_meta import SYSTEM_GRAPH_ID, system_value_index

        index = system_value_index(platform, refresh=False)
        for value in index.get(key) or []:
            text = str(value).strip()
            if not text or text.casefold() in seen:
                continue
            if needle and needle not in text.casefold():
                continue
            seen.add(text.casefold())
            values.append(text)
        if graph_id == SYSTEM_GRAPH_ID or graph_id == "kg_system":
            values.sort(key=lambda item: item.casefold())
            page = values[offset : offset + limit]
            return {
                "graph_id": SYSTEM_GRAPH_ID,
                "attribute": key,
                "q": q,
                "values": page,
                "total": len(values),
                "limit": limit,
                "offset": offset,
            }

    # Prefer catalog index when available (own_products / PLM path).
    catalog = getattr(platform, "catalog", None) or getattr(platform, "products", None)
    if catalog is not None and hasattr(catalog, "distinct_values"):
        try:
            for value in catalog.distinct_values(key) or []:
                text = str(value).strip()
                if not text or text.casefold() in seen:
                    continue
                if needle and needle not in text.casefold():
                    continue
                seen.add(text.casefold())
                values.append(text)
        except Exception:
            pass

    # Graph node properties / labels for brand, category, material, etc.
    label_map = {
        "brand": "Brand",
        "material": "Material",
        "category": "GarmentCategory",
        "color": "Color",
        "season": "Season",
        "supplier": "Supplier",
    }
    label = label_map.get(key)
    for gid in graphs:
        if label:
            nodes = platform.graph.find_nodes(graph=gid, label=label, include_deleted=False)
            for node in nodes:
                text = str(node.properties.get("name") or node.properties.get("title") or "").strip()
                if not text or text.casefold() in seen:
                    continue
                if needle and needle not in text.casefold():
                    continue
                seen.add(text.casefold())
                values.append(text)
        else:
            products = platform.graph.find_nodes(graph=gid, label="Product", include_deleted=False)
            for node in products:
                raw = node.properties.get(key) or node.properties.get(attribute)
                if raw is None:
                    continue
                text = str(raw).strip()
                if not text or text.casefold() in seen:
                    continue
                if needle and needle not in text.casefold():
                    continue
                seen.add(text.casefold())
                values.append(text)

    values.sort(key=lambda item: item.casefold())
    page = values[offset : offset + limit]
    return {
        "graph_id": graph_id,
        "attribute": key,
        "q": q,
        "values": page,
        "total": len(values),
        "limit": limit,
        "offset": offset,
    }
